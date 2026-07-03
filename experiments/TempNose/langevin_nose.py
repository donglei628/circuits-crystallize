"""
N1 温度鼻子(Langevin TTT):注入可控权重噪声(温度浴),扫温度,测形成时间 t*。
预言:t* 呈 U 形(鼻子)—— 太冷(T→0)势垒难越、太热(T 大)熔化不形成、中间有最优温度。这是与 saddle-to-saddle
(零噪声确定性)的分水岭:纯 GD 是 T=0 那端,真温度浴有熔点。基于 real_lm_f2_K 的 from-scratch tiny GPTNeoX 学
K-conjunction;Langevin:每步 opt.step 后 w += sqrt(2·T·lr)·ξ(过阻尼 Langevin,T=浴温度)。

  python langevin_nose.py --Ts 0 0.003 0.01 0.03 0.1 0.3 --K 2 --seeds 6
  python langevin_nose.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from real_lm_f2 import make_markov, build
from real_lm_f2_K import gen_batch_K, conj_score
import real_lm_f2_K as RK

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def t_star_bath(d, H, V, T_len, M, K, nu, lr, T_bath, max_steps, ee, thr, seed, device):
    """train tiny GPTNeoX on the K-conjunction with an injected weight-noise bath at temperature T_bath -> t* (or None)."""
    torch.manual_seed(1000 + seed); m = build(d, H, V, T_len, device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(seed); scale = (2.0 * T_bath * lr) ** 0.5 if T_bath > 0 else 0.0
    for s in range(0, max_steps + 1):
        if s % ee == 0:
            m.eval()
            if conj_score(m, M, T_len, V, K, device) > thr:
                m.train(); return s
            m.train()
        x = gen_batch_K(M, T_len, V, 64, nu, K, rng, device)
        out = m(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
        if scale > 0:                                                        # Langevin bath: inject weight noise ~ sqrt(2 T lr)
            with torch.no_grad():
                for p in m.parameters(): p.add_(torch.randn_like(p) * scale)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", type=float, nargs="+", default=[0.0, 0.003, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--K", type=int, default=2); ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=8000); ap.add_argument("--ee", type=int, default=25)
    ap.add_argument("--thresh", type=float, default=0.5); ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out", default="langevin_nose.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.Ts = [0.0, 0.03]; args.seeds = 2; args.max_steps = 3000
    RK.OFFSETS = args.offsets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = make_markov(args.vocab, 0)
    print(f"device={device} Ts={args.Ts} K={args.K} lr={args.lr} (N1 温度鼻子: 形成时间 t* vs 浴温度 T)", flush=True)
    res = []
    for T_bath in args.Ts:
        ts = [t_star_bath(args.d, args.heads, args.vocab, args.T, M, args.K, args.nu, args.lr, T_bath,
                          args.max_steps, args.ee, args.thresh, sd, device) for sd in range(args.seeds)]
        formed = [t for t in ts if t is not None]; med = float(np.median(formed)) if formed else None
        res.append(dict(T_bath=T_bath, med_tstar=med, n_formed=len(formed), all=ts))
        print(f"  T={T_bath}: t*={med} formed {len(formed)}/{args.seeds}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
    formed_ts = [(r["T_bath"], r["med_tstar"]) for r in res if r["med_tstar"]]
    print(f"\n=== N1 温度鼻子: t* vs 浴温度 ===", flush=True)
    for T, t in formed_ts: print(f"  T={T}: t*={t:.0f}", flush=True)
    if len(formed_ts) >= 3:
        ts_only = [t for _, t in formed_ts]; imin = int(np.argmin(ts_only))
        nose = 0 < imin < len(ts_only) - 1
        print(f"  最小 t* 在 T={formed_ts[imin][0]};{'✅ 鼻子(U形,中间最优温度=TTT nose)' if nose else '单调(冷端最快→无鼻子,浴纯熔化)'}", flush=True)
    print("NOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
