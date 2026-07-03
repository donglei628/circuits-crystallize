"""
Test FORMULA PART 2 (combinatorial barrier ~ K) on a REAL pretrained Pythia. The regeneration-restore axis is broken on
Pythia, so instead we CONTROL the conjunction depth K directly: take pretrained Pythia-160m and continue-train it to
learn a NEW controllable-K skill (offset-sum of K spaced offsets) on data of density nu. K is ours to set. Measure the
formation time t*(K, nu) and test, with attempt-frequency exponent FIXED at 1 (the CNT correspondence, already verified):
    ln t* = c0 + 1*|ln nu| + K*(|ln p| + a*|ln nu|)     -- barrier linear in K (combinatorial), per-component nu-coupling a
If this fits (high R2) on the real Pythia model, the combinatorial-barrier part holds on a real architecture.

  python pythia_barrier.py --model EleutherAI/pythia-160m --Ks 2 3 4 --nus 0.5 0.7 1.0 --bf16
"""
from __future__ import annotations
import argparse, copy, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import real_lm_f2_K as RK
from real_lm_f2_K import make_markov, gen_batch_K, conj_score

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def train_tstar(model, base_state, M, T, V, nu, K, lr, max_steps, ee, thr, batch, seed, device):
    model.load_state_dict(base_state)                       # reset to pretrained each run
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for s in range(0, max_steps + 1):
        if s % ee == 0:
            model.eval()
            if conj_score(model, M, T, V, K, device) > thr:
                model.train(); return s
            model.train()
        x = gen_batch_K(M, T, V, batch, nu, K, rng, device)
        out = model(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    ap.add_argument("--offsets", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--ee", type=int, default=10); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--bf16", action="store_true"); ap.add_argument("--out", default="pythia_barrier.json")
    args = ap.parse_args()
    RK.OFFSETS = args.offsets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = args.vocab; M = make_markov(V, 0, peak=4.0)
    dt = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision="step143000", dtype=dt,
                                                 attn_implementation="eager").to(device)
    base_state = copy.deepcopy(model.state_dict())
    print(f"device={device} model={args.model} Ks={args.Ks} nus={args.nus} offsets={args.offsets} "
          f"(PART-2 on real Pythia: barrier ~ K, attempt-freq exponent fixed at 1)", flush=True)

    grid = {}
    for K in args.Ks:
        for nu in args.nus:
            ts = [t for sd in range(args.seeds)
                  if (t := train_tstar(model, base_state, M, args.T, V, nu, K, args.lr, args.max_steps, args.ee,
                                       args.thresh, args.batch, 100 + sd, device)) is not None]
            grid[(K, nu)] = float(np.median(ts)) if ts else None
            print(f"  K={K} nu={nu}: t*={grid[(K, nu)]} (n={len(ts)}/{args.seeds})", flush=True)
            json.dump({str(k): v for k, v in grid.items()}, open(os.path.join(RESULTS, args.out), "w"), indent=2)

    # fit with b1 FIXED = 1
    pts = [(K, nu, v) for (K, nu), v in grid.items() if v is not None]
    if len(pts) >= 4:
        Kx = np.array([p[0] for p in pts], float); nux = np.array([p[1] for p in pts], float); tx = np.array([p[2] for p in pts], float)
        u = -np.log(nux); y = np.log(tx)
        y2 = y - u; X = np.vstack([np.ones_like(Kx), Kx, Kx * u]).T
        sol, *_ = np.linalg.lstsq(X, y2, rcond=None); c0, A, a = sol
        yh = u + X @ sol; r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        print(f"\n=== PART-2 on real Pythia (b1 fixed=1) ===", flush=True)
        print(f"  |ln p|={A:.2f} (p={np.exp(-A):.3f}, intrinsic per-component barrier)  coupling a={a:.2f}  R2={r2:.4f}", flush=True)
        X2 = np.vstack([np.ones_like(Kx), u, Kx, Kx * u]).T; s2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        print(f"  (free attempt-freq exponent b1={s2[1]:.2f}, should be ~1)", flush=True)
        print(f"  => {'barrier IS linear in K on real Pythia (part 2 holds)' if r2 > 0.85 and A > 0 else 'check'}", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
