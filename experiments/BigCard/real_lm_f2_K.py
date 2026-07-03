"""
FORMULA-2 end-to-end on a from-scratch real LM, BARRIER measured the CLEAN way: control the conjunction depth K directly
in the data (instead of the flaky seed-dissect). A tiny GPTNeoX is trained from scratch on a Markov background (the
learnable 'real-LM' competitor) into which a K-CONJUNCTION skill is injected at density nu: at conjunction positions the
token is (sum of K fixed earlier offsets) mod V -- the model must attend to and combine K positions, conjunction depth K.

Combinatorial nucleation: t* = t0 + 1/(N_sites * nu * p^K). So at fixed nu, ln(t*-t0) = ln(1/(N_sites*nu)) + K*|ln p|:
  - slope over K  = |ln p|   (the per-conjunct barrier, measured from FORMATION -- no regeneration, no floor)
  - intercept     = C = 1/N_sites   (the prefactor)
Sweep (K, nu); fit the whole grid to one (C, p, t0). C constant across width => formula 2 CLOSES end-to-end, with
every constant measured from clean timing on a real-architecture LM we grew. Anyone with (nu,K,p,C) can compute t*.

  python real_lm_f2_K.py --smoke
  python real_lm_f2_K.py --ds 128 256 --Ks 1 2 3 --nus 0.5 1.0 --seeds 3
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from real_lm_f2 import make_markov, markov_step, build

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OFFSETS = [1, 2, 3, 4, 5, 6]          # use the first K of these as the conjunction positions
CONJ_WEIGHTS = None                   # if a list, target = (sum of w_i * offset_i) mod V -- DISTINCT coeffs forbid the
                                      # "attend to all K positions and sum" merge, forcing K truly independent components
CONJ_MODE = "sum"                     # "sum" = reducible weighted-sum mod V (associative -> incremental shortcut exists)
                                      # "lut" = NON-reducible: target = LUT[ base-V index of the K-tuple ], a FIXED random
                                      #         function of the joint K-tuple. Partial index -> no info (random table), so
                                      #         there is no incremental/sum shortcut; model must bind all K jointly.
_LUT = {}
def _lut_for(V, K):
    if (V, K) not in _LUT:
        rng = np.random.default_rng(777)                 # FIXED seed -> the SAME target function for every run/seed
        _LUT[(V, K)] = rng.integers(0, V, V ** K, dtype=np.int64)
    return _LUT[(V, K)]


def gen_batch_K(M, T, V, B, nu, K, rng, device=None):
    """Markov background; at each position (>= max offset) the token is, w.p. nu, the (weighted sum of K earlier offsets)
    mod V (conjunction depth K), else the Markov continuation. Plain sum by default; distinct CONJ_WEIGHTS = no-merge."""
    offs = OFFSETS[:K]; mo = max(offs)
    wts = CONJ_WEIGHTS[:K] if CONJ_WEIGHTS else [1] * K
    seq = np.zeros((B, T), dtype=np.int64); seq[:, 0] = rng.integers(0, V, B)
    for t in range(1, T):
        mk = markov_step(seq[:, t - 1], M, rng)
        if t >= mo:
            if CONJ_MODE == "lut":                       # NON-reducible: random lookup over the joint K-tuple
                lut = _lut_for(V, K)
                idx = np.zeros(B, dtype=np.int64)
                for d in offs:
                    idx = idx * V + seq[:, t - d]         # encode K-tuple as a unique base-V integer in [0, V^K)
                cj = lut[idx]
            else:                                        # reducible: weighted sum mod V
                cj = np.zeros(B, dtype=np.int64)
                for w, d in zip(wts, offs):
                    cj += w * seq[:, t - d]
                cj %= V
            seq[:, t] = np.where(rng.random(B) < nu, cj, mk)
        else:
            seq[:, t] = mk
    out = torch.from_numpy(seq)
    return out.to(device) if device is not None else out


@torch.no_grad()
def conj_score(model, M, T, V, K, device, seed=999):
    """STRICT: probe with nu=1 (every position >= max offset IS the K-conjunction) -> accuracy = did the K-conjunction form."""
    rng = np.random.default_rng(seed); mo = max(OFFSETS[:K])
    x = gen_batch_K(M, T, V, 256, 1.0, K, rng, device)
    pred = model(x).logits[:, mo - 1:T - 1].argmax(-1)
    return float((pred == x[:, mo:T]).float().mean())


def train_tstar(model, M, T, V, nu, K, lr, max_steps, ee, thr, seed, device):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for s in range(0, max_steps + 1):
        if s % ee == 0:
            model.eval()
            if conj_score(model, M, T, V, K, device) > thr:
                model.train(); return s
            model.train()
        x = gen_batch_K(M, T, V, 64, nu, K, rng, device)
        out = model(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None


def fit_line(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m, b = np.polyfit(xs, ys, 1)
    r2 = 1 - ((ys - (m * xs + b)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9)
    return float(m), float(b), float(r2)


def measure_width(d, M, args, device):
    V, T, H = args.vocab, args.T, args.heads
    nu_bar = max(args.nus)                          # barrier read off t*(K) at the largest nu (cleanest, K=1 on the line)
    tK = {}; aK = {}; tstar_grid = {}
    for K in args.Ks:
        pts = []; per_nu = {}
        for nu in args.nus:
            ts = []
            for sd in range(args.seeds):
                torch.manual_seed(1000 + sd); m = build(d, H, V, T, device)
                t = train_tstar(m, M, T, V, nu, K, args.lr, args.max_steps, args.ee, args.thresh, sd, device)
                if t is not None: ts.append(t)
            tm = float(np.median(ts)) if ts else None
            tstar_grid[f"K{K}_nu{nu}"] = tm; per_nu[nu] = tm
            if tm is not None: pts.append((1.0 / nu, tm))
            print(f"    [d{d}] K={K} nu={nu}: t*={tm} (n={len(ts)}/{args.seeds})", flush=True)
        if per_nu.get(nu_bar): tK[K] = per_nu[nu_bar]               # barrier point: t*(K) at nu=1
        if len(pts) >= 2:                                          # 1/nu confirmation (K>=2; K=1 is nu-independent)
            a, t0, r2 = fit_line([p[0] for p in pts], [p[1] for p in pts]); aK[K] = a
            print(f"      -> 1/nu check a(K={K}) = {a:.1f} (t0={t0:.0f}, R2={r2:.2f})", flush=True)
    # barrier + prefactor from ln t*(K, nu=1) vs K  (t* = t0 + C/(nu*p^K); at nu=1, ln t* ~ ln C + K|ln p|)
    Ks = sorted(tK)
    res = dict(d=d, tK_nu1=tK, aK=aK, tstar_grid=tstar_grid)
    if len(Ks) >= 2 and all(tK[K] > 0 for K in Ks):
        lnp, b, r2 = fit_line(Ks, [np.log(tK[K]) for K in Ks])
        res.update(lnp=float(lnp), C=float(np.exp(b)), barrier_r2=float(r2))
        print(f"  ==[d{d}]== ln t*(K,nu=1) = {b:.2f} + {lnp:.3f}*K  (R2={r2:.2f}) => |ln p|={lnp:.3f}, C={np.exp(b):.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--ee", type=int, default=25); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--markov_seed", type=int, default=0); ap.add_argument("--out", default="real_lm_f2_K.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.ds = [128]; args.Ks = [1, 2, 3]; args.nus = [1.0]; args.seeds = 1; args.max_steps = 6000
    M = make_markov(args.vocab, args.markov_seed)
    print(f"device={device} ds={args.ds} Ks={args.Ks} (formula-2 barrier via CONTROLLED K, real-LM from scratch)", flush=True)

    out = []
    for d in args.ds:
        out.append(measure_width(d, M, args, device))
        json.dump({"args": vars(args), "results": out}, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
    print(f"\n  ===== VERDICT (formula 2, barrier via controlled K) =====", flush=True)
    Cs = [r["C"] for r in out if "C" in r]
    for r in out:
        if "C" in r:
            print(f"   d={r['d']:>4}: |ln p|={r['lnp']:.3f}, C={r['C']:.2f} (barrier R2={r['barrier_r2']:.2f})", flush=True)
    if len(Cs) >= 2:
        cv = float(np.std(Cs) / (np.mean(Cs) + 1e-9))
        print(f"   C across widths: {[round(c,2) for c in Cs]} CV={cv:.2f} => "
              f"{'CONSTANT -- formula 2 CLOSES (clean barrier)' if cv < 0.25 else 'STRUCTURED'}", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
