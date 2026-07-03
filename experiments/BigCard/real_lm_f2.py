"""
FORMULA-2 END-TO-END on a minimal, fully-controlled, FROM-SCRATCH real LM (the first-principles fix for "can't sweep nu
on a frozen pretrained model": don't do archaeology on a shattered apple -- grow your own apple). A tiny GPTNeoX
(Pythia's architecture) is trained from scratch on a GENUINE language task where induction is INCIDENTAL, not forced:
the data is a fixed random bigram/Markov background (a learnable "competitor" skill) into which induction-predictable
repeats are injected at a density nu WE CONTROL. The model first learns the Markov (plateau) then nucleates induction
(snap) -- like a real LM -- and because we control nu we can measure the full rate law end-to-end:
  - sweep nu -> t*(nu), fit t* = t0 + a/nu               (timing axis)
  - seed-dissect (transplant components) -> |ln p|, K     (barrier axis, independent of nu)
  - C = a * p^K  -> the prefactor; constant across width  => formula 2 CLOSES on a real-architecture LM we grew.
Then anyone with (nu, p, K, C) can COMPUTE t*. No pretrained checkpoint, every condition ours.

  python real_lm_f2.py --smoke
  python real_lm_f2.py --ds 128 192 256 --seeds 3
"""
from __future__ import annotations
import argparse, copy, json, os
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from end2end_gptneox import build, comp_state, set_components

RESULTS = os.path.join(os.path.dirname(__file__), "results")
COMPS = ["IND_QK", "IND_OV"]          # rate-limiting components (checked at smoke); K set from crater
K = 2


def make_markov(V, seed, peak=4.0):
    """a fixed random bigram transition matrix (the learnable background 'language')."""
    g = np.random.default_rng(seed)
    M = g.random((V, V)) ** peak                         # peaked -> genuinely predictable (not uniform)
    return M / M.sum(1, keepdims=True)


def markov_step(prev, M, rng):
    """vectorised: sample next token for each item in `prev` (B,) from rows of M -> (B,)."""
    cum = M[prev].cumsum(1)                              # (B, V)
    r = rng.random((len(prev), 1))
    return (cum > r).argmax(1).astype(np.int64)


def gen_batch(M, T, V, B, nu, rng, device):
    """first half: pure Markov. second half: copy the aligned first-half token w.p. nu (induction-predictable),
    else continue the Markov background. -> induction is one incidental skill at density nu."""
    h = T // 2; seq = np.zeros((B, T), dtype=np.int64)
    seq[:, 0] = rng.integers(0, V, B)
    for t in range(1, h):
        seq[:, t] = markov_step(seq[:, t - 1], M, rng)
    A = seq[:, :h].copy()
    for j in range(h):
        t = h + j
        cp = rng.random(B) < nu
        seq[:, t] = np.where(cp, A[:, j], markov_step(seq[:, t - 1], M, rng))
    return torch.from_numpy(seq).to(device)


@torch.no_grad()
def ind_score(model, T, V, device, seed=12345):
    """STRICT induction: a fully-repeated UNIFORM-random probe (Markov can't predict it -> isolates induction)."""
    rng = np.random.default_rng(seed); h = T // 2
    R = rng.integers(0, V, (256, h)); x = np.concatenate([R, R], 1).astype(np.int64)
    xt = torch.from_numpy(x).to(device)
    pred = model(xt).logits[:, h:2 * h - 1].argmax(-1)
    tgt = torch.from_numpy(x[:, h + 1:2 * h].astype(np.int64)).to(device)
    return float((pred == tgt).float().mean())


def train_tstar(model, M, T, V, nu, lr, max_steps, ee, thr, seed, device):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for s in range(0, max_steps + 1):
        if s % ee == 0:
            model.eval()
            if ind_score(model, T, V, device) > thr:
                model.train(); return s
            model.train()
        x = gen_batch(M, T, V, 64, nu, rng, device)
        out = model(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None


def fit_line(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m, b = np.polyfit(xs, ys, 1)
    r2 = 1 - ((ys - (m * xs + b)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9)
    return float(m), float(b), float(r2)


def measure_width(d, M, args, device):
    V, T, H = args.vocab, args.T, args.heads; hd = d // H
    # ---- timing axis: t*(nu) ----
    tstars = {}
    for nu in args.nus:
        ts = []
        for sd in range(args.seeds):
            torch.manual_seed(1000 + sd); m = build(d, H, V, T, device)
            t = train_tstar(m, M, T, V, nu, args.lr, args.max_steps, args.ee, args.thresh, sd, device)
            if t is not None: ts.append(t)
        tstars[nu] = float(np.median(ts)) if ts else None
        print(f"    [d{d}] nu={nu}: t*={tstars[nu]} (n={len(ts)}/{args.seeds})", flush=True)
    pts = [(1.0 / nu, tstars[nu]) for nu in args.nus if tstars[nu] is not None]
    a, t0, r2t = fit_line([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else (None, None, None)

    # ---- barrier axis: seed-dissect (donor -> lesion -> restore subset) ----
    torch.manual_seed(7); donor = build(d, H, V, T, device)
    train_tstar(donor, M, T, V, 1.0, args.lr, args.max_steps, args.ee, 0.85, 0, device)
    donor_state = comp_state(donor, H, hd)
    subsets = [()] + [c for r in range(1, K + 1) for c in combinations(COMPS, r)]
    rate = {}
    for sub in subsets:
        rs = []
        for sd in range(args.seeds):
            m = copy.deepcopy(donor)
            torch.manual_seed(3000 + sd); rnd = build(d, H, V, T, device)
            set_components(m, comp_state(rnd, H, hd), set(COMPS), H, hd)        # lesion
            if sub: set_components(m, donor_state, set(sub), H, hd)             # restore subset
            t = train_tstar(m, M, T, V, 1.0, args.restore_lr, args.restore_steps, args.restore_ee, args.thresh, sd, device)
            rs.append(1.0 / (args.restore_steps * 2) if t is None else 1.0 / max(t, args.restore_ee))
        rate[sub] = float(np.median(rs))
        print(f"    [d{d}] restore {('+'.join(sub) or 'none'):>16}: rate={rate[sub]:.5f}", flush=True)
    nx = np.array([len(s) for s in subsets]); ly = np.log([rate[s] for s in subsets])
    lnp, b, r2b = fit_line(nx, ly)
    pK = float(np.exp(-K * lnp)); C = float(a * pK) if a else None
    res = dict(d=d, tstars=tstars, a=a, t0=t0, t_r2=r2t, lnp=float(lnp), barrier_r2=r2b, pK=pK, C=C,
               restore_rate={('+'.join(s) or 'none'): rate[s] for s in subsets})
    print(f"  ==[d{d}]== a={a:.1f}(t0={t0:.0f},R2={r2t:.2f}) | |ln p|={lnp:.3f}(R2={r2b:.2f}) | p^K={pK:.4f} | C=a*p^K={C:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[128, 192, 256])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.25, 0.4, 0.6, 1.0])
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--restore_steps", type=int, default=4000); ap.add_argument("--ee", type=int, default=25)
    ap.add_argument("--restore_ee", type=int, default=5); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--restore_lr", type=float, default=1.5e-4)   # slower restore so the seed-dissect rate stays above the floor
    ap.add_argument("--markov_seed", type=int, default=0); ap.add_argument("--out", default="real_lm_f2.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.ds = [128]; args.nus = [0.5, 1.0]; args.seeds = 1; args.max_steps = 4000; args.restore_steps = 2000
    M = make_markov(args.vocab, args.markov_seed)
    print(f"device={device} ds={args.ds} (formula-2 end-to-end on a from-scratch real LM; Markov background + induction@nu)", flush=True)

    out = []
    for d in args.ds:
        out.append(measure_width(d, M, args, device))
        json.dump({"args": vars(args), "results": out}, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
    Cs = [r["C"] for r in out if r["C"]]
    print(f"\n  ===== VERDICT (formula 2, real-LM from scratch) =====", flush=True)
    for r in out:
        print(f"   d={r['d']:>4}: C={r['C']:.2f} (a={r['a']:.1f}, |ln p|={r['lnp']:.3f})", flush=True)
    if len(Cs) >= 2:
        cv = float(np.std(Cs) / (np.mean(Cs) + 1e-9))
        print(f"   C across widths: {[round(c,2) for c in Cs]} CV={cv:.2f} => "
              f"{'CONSTANT -- formula 2 CLOSES end-to-end on a real-architecture LM we grew' if cv < 0.25 else 'STRUCTURED'}", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
