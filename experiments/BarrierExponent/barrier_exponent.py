"""
N8 势垒指数诊断:形成率 ∝ exp(−b/ΔL*^φ),φ=1(普通 Arrhenius) vs φ=2(经典 CNT 几何势垒)。旧坎:可形成的
ΔL* 范围只 ~3× → 1/ΔL* 与 1/ΔL*² 共线 0.99,分不出(progress_map)。本实验用 make_markov 的 **peak** 当干净 ΔL*
旋钮(peak 高=基础语言可预测=拷贝电路省 loss 少=ΔL* 小;peak 低=均匀=ΔL* 大),固定 V/K/nu,把 ΔL* 尽量拉宽。
每个 peak:训练测 t*(conj_score 形成步)+ 亚稳平台 loss(形成前)− 地板 loss(形成后)= ΔL*(实测,nats)。
拟合 ln(1/t*) 对 1/ΔL* 和 1/ΔL*²,比 R²;并报两者的共线度(诚实:范围不够宽就分不出,如实写)。

  python barrier_exponent.py --peaks 0.3 0.6 1 2 4 8 16 --seeds 5
  python barrier_exponent.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from real_lm_f2 import make_markov, build
from real_lm_f2_K import gen_batch_K, conj_score
import real_lm_f2_K as RK

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def run_one(d, H, V, T, M, K, nu, lr, max_steps, ee, thr, seed, device):
    """train; return (t*, plateau_loss just before formation, floor_loss after) — ΔL* proxy = plateau − floor."""
    torch.manual_seed(1000 + seed); m = build(d, H, V, T, device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(seed)
    tstar = None; loss_hist = []                                          # (step, loss)
    for s in range(max_steps + 1):
        if s % ee == 0:
            m.eval()
            if conj_score(m, M, T, V, K, device) > thr and tstar is None:
                tstar = s
                # train a bit more to reach floor
                for _ in range(300):
                    x = gen_batch_K(M, T, V, 64, nu, K, rng, device)
                    o = m(x, labels=x); o.loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
                m.eval()
                with torch.no_grad():
                    xf = gen_batch_K(M, T, V, 256, nu, K, rng, device); floor = float(m(xf, labels=xf).loss)
                m.train()
                pre = [l for st, l in loss_hist if st >= tstar * 0.4]     # metastable plateau = loss over pre-formation window
                plateau = float(np.median(pre)) if pre else (loss_hist[-1][1] if loss_hist else floor)
                return tstar, plateau, floor
            m.train()
        x = gen_batch_K(M, T, V, 64, nu, K, rng, device)
        o = m(x, labels=x)
        if s % ee == 0: loss_hist.append((s, float(o.loss)))
        o.loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None, None, None


def fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m, b = np.polyfit(xs, ys, 1)
    r2 = 1 - ((ys - (m * xs + b)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9)
    return float(m), float(b), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", type=float, nargs="+", default=[0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--V", type=int, default=64); ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=8000); ap.add_argument("--ee", type=int, default=25)
    ap.add_argument("--thresh", type=float, default=0.5); ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out", default="barrier_exponent.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.peaks = [0.6, 8.0]; args.seeds = 2; args.max_steps = 3000
    RK.OFFSETS = args.offsets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} peaks={args.peaks} V={args.V} K={args.K} nu={args.nu} (N8 势垒指数 φ: 1/ΔL* vs 1/ΔL*²)", flush=True)
    res = []
    for peak in args.peaks:
        M = make_markov(args.V, 0, peak)
        rows = [run_one(args.d, args.heads, args.V, args.T, M, args.K, args.nu, args.lr, args.max_steps, args.ee, args.thresh, sd, device)
                for sd in range(args.seeds)]
        ts = [t for t, _, _ in rows if t is not None]
        dls = [p - f for t, p, f in rows if t is not None and p is not None]
        med_t = float(np.median(ts)) if ts else None
        med_dl = float(np.median(dls)) if dls else None
        res.append(dict(peak=peak, med_tstar=med_t, dLstar=med_dl, n=len(ts)))
        print(f"  peak={peak}: t*={med_t} ΔL*={med_dl if med_dl is None else round(med_dl,3)} (n={len(ts)}/{args.seeds})", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
    good = [(r["dLstar"], r["med_tstar"]) for r in res if r["med_tstar"] and r["dLstar"] and r["dLstar"] > 0]
    print(f"\n=== N8 势垒指数诊断(n={len(good)}) ===", flush=True)
    if len(good) >= 3:
        dl = np.array([g[0] for g in good]); t = np.array([g[1] for g in good])
        y = np.log(t)                                                    # ln t* ∝ b/ΔL*^φ
        m1, b1, r1 = fit(1.0 / dl, y); m2, b2, r2 = fit(1.0 / dl ** 2, y)
        collin = float(abs(np.corrcoef(1.0 / dl, 1.0 / dl ** 2)[0, 1]))
        print(f"  ΔL* 范围: {dl.min():.3f}–{dl.max():.3f}(跨度 {dl.max()/dl.min():.1f}×)", flush=True)
        print(f"  φ=1 (Arrhenius 1/ΔL*):  ln t* 斜率={m1:.2f}  R²={r1:.3f}", flush=True)
        print(f"  φ=2 (CNT 1/ΔL*²):       ln t* 斜率={m2:.2f}  R²={r2:.3f}", flush=True)
        print(f"  两模型共线度 |corr(1/ΔL*,1/ΔL*²)|={collin:.3f}", flush=True)
        if collin > 0.97:
            print(f"  => ⚠️ ΔL* 范围仍不够宽,两模型共线({collin:.3f})分不出 φ —— 诚实:toy 天花板(同 progress_map)", flush=True)
        else:
            win = "φ=1 Arrhenius" if r1 > r2 else "φ=2 CNT"
            print(f"  => 可分辨:{win} 拟合更好(R² {max(r1,r2):.3f} vs {min(r1,r2):.3f})", flush=True)
    print("N8_DONE", flush=True)


if __name__ == "__main__":
    main()
