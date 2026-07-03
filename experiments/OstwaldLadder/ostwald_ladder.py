"""
N12 Ostwald 阶梯(多级亚稳)。经典 Ostwald step rule:相变不直接进最稳相,而是逐级经过亚稳中间相。
我们造一个有内置中间相的任务:K=3 LUT 合取,但 LUT 的一半表项(按 (x2,x3) 对折半)只依赖第一个 offset
(target=g(x1))——所以存在便宜的 K=1 捷径电路(≈50%+ 准确率,中间亚稳相),完整 K=3 电路=100%(稳定相)。
分开追踪:acc_easy(捷径位,target=g(x1))与 acc_hard(必须 3-合取的位)。
  Ostwald 台阶 = acc_easy 先点火并平台(K=1 相),隔一段 acc_hard 再点火(K=3 相)—— 两级分离的一阶台阶;
  一步到位   = 两者同时起跳。量:两次点火时刻 t*_easy / t*_hard 与间隔;逐种子。

  python ostwald_ladder.py --seeds 8
  python ostwald_ladder.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from real_lm_f2 import make_markov, markov_step, build

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OFFS = [1, 2, 3]                                                          # K=3 conjunction positions


def make_ladder_lut(V, easy_frac, seed=777):
    """LUT3[x1,x2,x3]: for a fixed easy_frac of (x2,x3) pairs the target is g(x1) (K=1 shortcut phase);
    otherwise a full random 3-tuple function (needs the K=3 conjunction). Returns (lut, easy_mask, g)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, V, V, dtype=np.int64)                              # the K=1 shortcut map
    easy = rng.random((V, V)) < easy_frac                                  # (x2,x3) pairs where shortcut holds
    lut = rng.integers(0, V, (V, V, V), dtype=np.int64)                    # full random 3-tuple function
    lut[:, easy] = np.broadcast_to(g[:, None], (V, easy.sum()))            # easy pairs: target = g(x1), any (x2,x3) in mask
    return lut, easy, g


def gen_batch(M, T, V, B, nu, lut, rng, device, with_mask=False):
    """nu<1 REQUIRED in spirit: at nu=1 the closed-loop deterministic LUT dynamics collapse into period-3 orbits
    (measured 58%!) and a trivial copy-from-offset-3 predicts everything -- nu<=0.6 breaks the feedback loop."""
    seq = np.zeros((B, T), dtype=np.int64); seq[:, 0] = rng.integers(0, V, B)
    is_cj = np.zeros((B, T), dtype=bool)
    mo = max(OFFS)
    for t in range(1, T):
        mk = markov_step(seq[:, t - 1], M, rng)
        if t >= mo:
            cj = lut[seq[:, t - 1], seq[:, t - 2], seq[:, t - 3]]
            take = rng.random(B) < nu
            seq[:, t] = np.where(take, cj, mk); is_cj[:, t] = take
        else:
            seq[:, t] = mk
    out = torch.from_numpy(seq).to(device)
    return (out, is_cj) if with_mask else out


@torch.no_grad()
def probe(model, M, T, V, lut, easy, device, nu=0.5, seed=999, B=256):
    """distribution-matched probe at the TRAINING nu: score only true conjunction positions (mask), split easy
    (target=g(x1)) vs hard (needs the full 3-tuple). Avoids the nu=1 period-3 collapse artifact."""
    rng = np.random.default_rng(seed)
    x, is_cj = gen_batch(M, T, V, B, nu, lut, rng, device, with_mask=True)
    mo = max(OFFS)
    pred = model(x).logits[:, mo - 1:T - 1].argmax(-1)
    tgt = x[:, mo:T]
    xn = x.cpu().numpy()
    x2 = xn[:, mo - 2:T - 2]; x3 = xn[:, mo - 3:T - 3]
    cm = torch.from_numpy(is_cj[:, mo:T]).to(device)                       # true conjunction positions only
    em = torch.from_numpy(easy[x2, x3]).to(device) & cm
    hm = (~torch.from_numpy(easy[x2, x3]).to(device)) & cm
    ok = (pred == tgt)
    a_easy = float(ok[em].float().mean()) if em.any() else float("nan")
    a_hard = float(ok[hm].float().mean()) if hm.any() else float("nan")
    return a_easy, a_hard


def run_seed(d, H, V, T, M, lut, easy, nu, lr, max_steps, ee, seed, device):
    torch.manual_seed(1000 + seed); m = build(d, H, V, T, device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(seed)
    curve = []; t_easy = None; t_hard = None
    for s in range(max_steps + 1):
        if s % ee == 0:
            m.eval(); ae, ah = probe(m, M, T, V, lut, easy, device, nu=nu); m.train()
            curve.append((s, ae, ah))
            if t_easy is None and ae > 0.5: t_easy = s
            if t_hard is None and ah > 0.5: t_hard = s
            if t_hard is not None and s >= t_hard + 20 * ee:
                break
        x = gen_batch(M, T, V, 64, nu, lut, rng, device)
        out = m(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    return curve, t_easy, t_hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--easy_frac", type=float, default=0.5); ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=12000); ap.add_argument("--ee", type=int, default=10)
    ap.add_argument("--out", default="ostwald_ladder.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = 2; args.max_steps = 6000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = args.vocab
    M = make_markov(V, 0)
    lut, easy, g = make_ladder_lut(V, args.easy_frac)
    print(f"device={device} V={V} easy_frac={args.easy_frac} (N12 Ostwald 阶梯: K=1 捷径亚稳相 vs K=3 稳定相)", flush=True)
    res = []
    for sd in range(args.seeds):
        curve, te, th = run_seed(args.d, args.heads, V, args.T, M, lut, easy, args.nu, args.lr,
                                 args.max_steps, args.ee, sd, device)
        gap = (th - te) if (te is not None and th is not None) else None
        res.append(dict(seed=sd, t_easy=te, t_hard=th, gap=gap, curve=curve))
        print(f"  seed{sd}: K=1捷径相点火 t*_easy={te}  K=3全相点火 t*_hard={th}  间隔={gap}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
    tes = [r["t_easy"] for r in res if r["t_easy"] is not None]
    ths = [r["t_hard"] for r in res if r["t_hard"] is not None]
    gaps = [r["gap"] for r in res if r["gap"] is not None]
    print(f"\n=== N12 Ostwald 阶梯 ===", flush=True)
    if tes and ths:
        print(f"  t*_easy 中位 {np.median(tes):.0f} | t*_hard 中位 {np.median(ths):.0f} | 间隔中位 {np.median(gaps):.0f}", flush=True)
        ladder = np.median(gaps) > 3 * args.ee and np.median(ths) > 1.5 * np.median(tes)
        print(f"  判定: {'✅ 两级分离台阶 = Ostwald step rule 成立(先亚稳 K=1 相、后稳定 K=3 相)' if ladder else '同时起跳/间隔不显著 = 一步到位,Ostwald 不成立(如实报)'}", flush=True)
    print("N12_DONE", flush=True)


if __name__ == "__main__":
    main()
