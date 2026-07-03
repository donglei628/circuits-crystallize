"""
N10 核生长轨迹 n(t)(growing nucleus)。②-5 的 Avrami n≈1、N3 的 CV(τ)=0.13、N1 的生长主导都**推断**了
「势垒定均值、确定性生长定分散」;N10 **直接看生长**:toy K=3 从零训练,在点火附近高时间分辨(每 5 步)记录
conj_score(t),对上升段拟合 KJMA X(t)=1−exp(−(K·(t−t0))^n) 读 Avrami n。判据:
  n≈1 = 单核前沿确定性生长(与 ②-5 一致);n≫1 = 多点同时成核+生长;生长段窄且各种子形状塌缩 = 生长确定性。

  python nucleus_growth.py --seeds 8
  python nucleus_growth.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from real_lm_f2 import make_markov, build
from real_lm_f2_K import gen_batch_K, conj_score
import real_lm_f2_K as RK

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def growth_curve(d, H, V, T, M, K, nu, lr, max_steps, ee, seed, device):
    """train from scratch; record conj_score every `ee` steps until saturation (>0.9) + a tail -> (steps, scores)."""
    torch.manual_seed(1000 + seed); m = build(d, H, V, T, device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(seed)
    curve = []; sat_at = None
    for s in range(max_steps + 1):
        if s % ee == 0:
            m.eval(); cs = conj_score(m, M, T, V, K, device); m.train()
            curve.append((s, cs))
            if cs > 0.9 and sat_at is None:
                sat_at = s
            if sat_at is not None and s >= sat_at + 20 * ee:                 # tail after saturation, then stop
                return curve
        x = gen_batch_K(M, T, V, 64, nu, K, rng, device)
        out = m(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    return curve


def kjma_fit(steps, xs, floor, ceil):
    """fit X(t)=1-exp(-(k(t-t0))^n) on the rising segment via grid+least squares in log-log (Avrami plot)."""
    x = (np.asarray(xs) - floor) / max(ceil - floor, 1e-9)
    t = np.asarray(steps, float)
    seg = (x > 0.03) & (x < 0.97)
    if seg.sum() < 3:
        return None
    ts, xs2 = t[seg], np.clip(x[seg], 1e-4, 1 - 1e-4)
    best = None
    for t0 in np.linspace(ts.min() - 3 * (ts[1] - ts[0] if len(ts) > 1 else 5), ts.min() - 1e-6, 25):
        u = np.log(ts - t0); y = np.log(-np.log(1 - xs2))                    # Avrami plot: y = n*ln k + n*ln(t-t0)
        n, b = np.polyfit(u, y, 1)
        r2 = 1 - ((y - (n * u + b)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        if best is None or r2 > best[2]:
            best = (float(n), float(t0), float(r2))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=8000); ap.add_argument("--ee", type=int, default=5)
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out", default="nucleus_growth.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = 2; args.max_steps = 3000
    RK.OFFSETS = args.offsets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = make_markov(args.vocab, 0)
    print(f"device={device} K={args.K} nu={args.nu} ee={args.ee} seeds={args.seeds} (N10 核生长轨迹 n(t), KJMA)", flush=True)
    res = []
    for sd in range(args.seeds):
        curve = growth_curve(args.d, args.heads, args.vocab, args.T, M, args.K, args.nu, args.lr,
                             args.max_steps, args.ee, sd, device)
        st = [c[0] for c in curve]; xs = [c[1] for c in curve]
        floor = float(np.median(xs[:max(3, len(xs) // 10)])); ceil = float(max(xs))
        # growth width: steps from 10% to 90% of the rise
        x = (np.asarray(xs) - floor) / max(ceil - floor, 1e-9)
        try:
            t10 = st[int(np.argmax(x >= 0.1))]; t90 = st[int(np.argmax(x >= 0.9))]
        except Exception:
            t10 = t90 = None
        fit = kjma_fit(st, xs, floor, ceil)
        res.append(dict(seed=sd, curve=curve, floor=floor, ceil=ceil, t10=t10, t90=t90,
                        width=(t90 - t10) if (t10 is not None and t90 is not None) else None,
                        kjma=dict(n=fit[0], t0=fit[1], r2=fit[2]) if fit else None))
        w = res[-1]["width"]; n_s = f"{fit[0]:.2f}(R²{fit[2]:.2f})" if fit else "-"
        print(f"  seed{sd}: 点火 t10={t10} 生长宽 t90−t10={w}  Avrami n={n_s}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
    ns = [r["kjma"]["n"] for r in res if r["kjma"] and r["kjma"]["r2"] > 0.8]
    ws = [r["width"] for r in res if r["width"]]
    t10s = [r["t10"] for r in res if r["t10"] is not None]
    print(f"\n=== N10 核生长 ===", flush=True)
    if ns: print(f"  Avrami n = {np.median(ns):.2f}(中位,n={len(ns)};②-5 曾测 n≈1)", flush=True)
    if ws and t10s:
        print(f"  点火 t10: 均值 {np.mean(t10s):.0f} CV={np.std(t10s)/np.mean(t10s):.2f} | 生长宽: 均值 {np.mean(ws):.0f} CV={np.std(ws)/np.mean(ws):.2f}", flush=True)
        print(f"  判据: 生长宽 ≪ 点火时刻 且 各种子生长形状一致 = 势垒定均值、确定性生长定分散(合 N1/N3)", flush=True)
    print("N10_DONE", flush=True)


if __name__ == "__main__":
    main()
