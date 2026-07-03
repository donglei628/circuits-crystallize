"""
SUPP-LAW (northstar ③ — competition MAGNITUDE / suppression law). Substrate A: IN-WEIGHT associative memory (the
only substrate with a CONSERVED per-circuit stored strength that can trade off — in-context recall cannot, because its
binding is recomputed each forward pass and the head is key-agnostic/reused). Two sets of fixed key->value bindings
must be STORED IN WEIGHTS; they compete for the d-dim storage. We set the cross-set address overlap rho and the payoff
ratio R a-priori, and measure each set's recall strength S_i vs its solo baseline, testing the suppression law

    S_i / S_i^solo  =  ( 1 - rho^2 * g(R_j / R_i) )_+ ,     g(x) = x^beta / (1 + x^beta)

with beta = beta(regime): WTA (beta->inf) in the rich/small-init regime, graded sharing (beta->1) in the lazy one.
NOVEL beyond Toy-Models-of-Superposition (static geometry) and grokking (same-task substitution): (a) the GRADED law
vs the importance RATIO, and (b) the FORMATION-RATE depletion -- does the winner deplete the loser's growth rate
(solute-depletion / nucleation), tracked via S_i(t).

Model: fixed key embeddings E (frozen, controlled rho) -> trainable MLP (d -> d_h -> n_vals) -> value logits.
Keys set1 = orthonormal u_i; set2 = rho*u_i + sqrt(1-rho^2)*w_i (controlled pairwise cross-overlap rho).

  python supp_law.py --smoke
  python supp_law.py --rhos 0 0.3 0.6 0.9 --R2 4 --init 0.5 --seeds 8
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def make_keys(m, d, rho, rng):
    """2m fixed unit key directions (works for ANY m,d incl. over-capacity 2m>d): set1=u_i random unit; set2_i has
    EXACT cross-overlap cos(u_i, s2_i)=rho (w_i is the u_i-orthogonal part). Within-set overlaps are random ~1/sqrt(d)."""
    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    u = unit(rng.standard_normal((m, d)))
    w = rng.standard_normal((m, d))
    w = w - np.sum(w * u, axis=1, keepdims=True) * u                           # make w_i orthogonal to u_i
    w = unit(w)
    s2 = unit(rho * u + np.sqrt(max(1 - rho ** 2, 0.0)) * w)
    return np.concatenate([u, s2], 0).astype(np.float32)                       # (2m, d), set1=[0:m], set2=[m:2m]


class MemMLP(nn.Module):
    """trainable associative-memory readout over FROZEN key embeddings. d_h=0 -> LINEAR readout (the clean substrate:
    with weight decay, mapping two cos-rho keys to different values costs norm ~1/(1-rho^2), so the bottleneck is
    WEIGHT NORM and overlapping keys genuinely compete). d_h>0 -> 1-hidden MLP (for the rich/lazy regime knob)."""
    def __init__(self, E, d_h, n_vals, init):
        super().__init__()
        self.E = nn.Parameter(torch.tensor(E), requires_grad=False)            # frozen key directions (2m, d)
        d = E.shape[1]; self.d_h = d_h
        if d_h > 0:
            self.W1 = nn.Parameter(torch.randn(d, d_h) * init)
            self.W2 = nn.Parameter(torch.randn(d_h, n_vals) * init)
        else:
            self.U = nn.Parameter(torch.randn(d, n_vals) * init)

    def forward(self, idx):
        k = self.E[idx]
        if self.d_h > 0:
            return F.relu(k @ self.W1) @ self.W2
        return k @ self.U


@torch.no_grad()
def set_recall(model, idx, vals):
    """fraction of keys in idx whose argmax value-logit equals the stored value."""
    pred = model(idx).argmax(-1)
    return float((pred == vals).float().mean())


def train_mem(E, vals, set_of, active_sets, R, d_h, n_vals, init, wd, steps, batch, lr, seed, device,
              track=None, eval_every=20):
    """train the memory on keys from `active_sets` with per-set sampling weight R; return final per-set recall + S(t)."""
    rng = np.random.default_rng(seed)
    model = MemMLP(E, d_h, n_vals, init).to(device)
    valst = torch.tensor(vals, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # sampling pool: key indices weighted by their set's payoff R
    idxs = [i for i in range(len(vals)) if set_of[i] in active_sets]
    wts = np.array([R[set_of[i]] for i in idxs], float); wts /= wts.sum()
    idx_all = {s: torch.tensor([i for i in range(len(vals)) if set_of[i] == s], device=device) for s in (0, 1)}
    curve = {0: [], 1: []}
    for s in range(steps + 1):
        if s % eval_every == 0 and track is not None:
            for ss in active_sets:
                curve[ss].append(set_recall(model, idx_all[ss], valst[idx_all[ss]]))
        if s >= steps:
            break
        bi = rng.choice(idxs, size=batch, p=wts)
        bidx = torch.tensor(bi, device=device)
        logits = model(bidx)
        loss = F.cross_entropy(logits, valst[bidx])
        opt.zero_grad(); loss.backward(); opt.step()
    S = {ss: set_recall(model, idx_all[ss], valst[idx_all[ss]]) for ss in (0, 1)}
    return S, curve


def run_config(rho, R2, m, d, d_h, n_vals, init, wd, steps, batch, lr, seeds, device):
    """joint (both sets) vs solo (each alone). return mean S_i, S_i^solo, suppression, and (seed 0) BOTH the joint S(t)
    and the solo S(t) of each set -> lets us test formation-RATE depletion (joint rate < solo rate = solute depletion)."""
    R = {0: 1.0, 1: float(R2)}
    Sj = {0: [], 1: []}; Ssolo = {0: [], 1: []}; curve0 = None; solo_curve = {0: None, 1: None}
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        E = make_keys(m, d, rho, rng)
        vals = rng.integers(0, n_vals, 2 * m).astype(np.int64)
        set_of = np.array([0] * m + [1] * m)
        Sjoint, c = train_mem(E, vals, set_of, (0, 1), R, d_h, n_vals, init, wd, steps, batch, lr, seed, device,
                              track=(seed == 0))
        if seed == 0:
            curve0 = c
        for ss in (0, 1):
            Sj[ss].append(Sjoint[ss])
        for ss in (0, 1):
            Ssol, cs = train_mem(E, vals, set_of, (ss,), R, d_h, n_vals, init, wd, steps, batch, lr, seed, device,
                                 track=(seed == 0))
            Ssolo[ss].append(Ssol[ss])
            if seed == 0:
                solo_curve[ss] = cs[ss]                                         # S_ss(t) trained ALONE
    out = {}
    for ss in (0, 1):
        sj = float(np.mean(Sj[ss])); ssolo = float(np.mean(Ssolo[ss]))
        out[ss] = dict(S=sj, Ssolo=ssolo, supp=1.0 - sj / max(ssolo, 1e-9),
                       S_seeds=[float(x) for x in Sj[ss]])               # per-seed (for WTA-vs-graded index)
    return out, curve0, solo_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--R2s", type=float, nargs="+", default=None)  # if set, 2D sweep rho x R2 (maps g(R_j/R_i))
    ap.add_argument("--R2", type=float, default=4.0)              # payoff ratio R2/R1 (set0 is the weaker if R2>1)
    ap.add_argument("--inits", type=float, nargs="+", default=None)  # if set, REGIME mode: sweep init at R2/rho fixed
    ap.add_argument("--m", type=int, default=12)                  # keys per set (2m total; 2m<=d)
    ap.add_argument("--d", type=int, default=32)                  # residual/storage dim (the conserved resource)
    ap.add_argument("--d_h", type=int, default=0)                # 0 = LINEAR readout (clean substrate); >0 = MLP
    ap.add_argument("--n_vals", type=int, default=64)
    ap.add_argument("--init", type=float, default=0.1)           # init scale (rich/lazy regime knob, matters for MLP)
    ap.add_argument("--wd", type=float, default=0.05)            # weight decay = the storage cost that makes overlap compete
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="supp_law.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.rhos = [0.0, 0.3, 0.6, 0.9]; args.seeds = 4; args.steps = 1500
    print(f"device={device} rhos={args.rhos} R2/R1={args.R2} m={args.m} d={args.d} d_h={args.d_h} init={args.init} "
          f"wd={args.wd} (SUPP-LAW: in-weight assoc-mem competition; set1=weaker if R2>1)", flush=True)

    # REGIME mode: at symmetric payoff (R2=1) + high overlap, sweep init scale -> WTA (symmetry-broken) vs graded sharing
    if args.inits is not None:
        rho = args.rhos[0]
        print(f"\n  ===== REGIME (WTA vs graded): R2={args.R2} rho={rho} d_h={args.d_h}, sweep init =====", flush=True)
        print(f"  {'init':>6} {'meanS0':>7} {'meanS1':>7} {'WTA_idx':>8} {'WTA_frac':>9}  (WTA_idx=<|S0-S1|>; frac= one>0.7 & other<0.3)", flush=True)
        reg = []
        for ini in args.inits:
            out, _, _ = run_config(rho, args.R2, args.m, args.d, args.d_h, args.n_vals, ini, args.wd,
                                   args.steps, args.batch, args.lr, args.seeds, device)
            s0 = np.array(out[0]['S_seeds']); s1 = np.array(out[1]['S_seeds'])
            wta_idx = float(np.mean(np.abs(s0 - s1)))
            wta_frac = float(np.mean((np.maximum(s0, s1) > 0.7) & (np.minimum(s0, s1) < 0.3)))
            print(f"  {ini:>6.2f} {s0.mean():>7.2f} {s1.mean():>7.2f} {wta_idx:>8.2f} {wta_frac:>9.2f}", flush=True)
            reg.append(dict(init=float(ini), meanS0=float(s0.mean()), meanS1=float(s1.mean()),
                            wta_idx=wta_idx, wta_frac=wta_frac, s0=s0.tolist(), s1=s1.tolist()))
            json.dump(reg, open(os.path.join(RESULTS, args.out), "w"), default=float)
        idx = np.array([r['wta_idx'] for r in reg]); ins = np.array([r['init'] for r in reg])
        print(f"\n  => WTA_idx vs init: {dict(zip(np.round(ins,2), np.round(idx,2)))}", flush=True)
        print(f"  => {'CROSSOVER seen (WTA_idx changes with init -> regime sets WTA-vs-graded)' if idx.max()-idx.min()>0.25 else 'flat -- report honestly'}", flush=True)
        print(f"\n  saved results/{args.out}", flush=True)
        return

    R2s = args.R2s if args.R2s is not None else [args.R2]
    res = []
    gfit = {}
    for R2 in R2s:
        print(f"\n  ===== R2/R1 = {R2} =====", flush=True)
        print(f"  {'rho':>5} {'S0':>6} {'S0solo':>7} {'supp0':>6} | {'S1':>6} {'S1solo':>7} {'supp1':>6}", flush=True)
        sub = []
        for rho in args.rhos:
            out, curve0, solo_curve = run_config(rho, R2, args.m, args.d, args.d_h, args.n_vals, args.init, args.wd,
                                                 args.steps, args.batch, args.lr, args.seeds, device)
            print(f"  {rho:>5.2f} {out[0]['S']:>6.2f} {out[0]['Ssolo']:>7.2f} {out[0]['supp']:>6.2f} | "
                  f"{out[1]['S']:>6.2f} {out[1]['Ssolo']:>7.2f} {out[1]['supp']:>6.2f}", flush=True)
            rec = dict(R2=float(R2), rho=float(rho), set0=out[0], set1=out[1], curve0=curve0, solo_curve=solo_curve)
            sub.append(rec); res.append(rec)
            json.dump(res, open(os.path.join(RESULTS, args.out), "w"), default=float)
        rs = np.array([r["rho"] for r in sub]); sup = np.array([r["set0"]["supp"] for r in sub])
        if len(rs) >= 3 and np.std(sup) > 1e-6:
            a2 = np.polyfit(rs ** 2, sup, 1); pred = a2[0] * rs ** 2 + a2[1]
            r2 = 1 - ((sup - pred) ** 2).sum() / max(((sup - sup.mean()) ** 2).sum(), 1e-9)
            gfit[R2] = dict(slope=float(a2[0]), intercept=float(a2[1]), r2=float(r2))
            print(f"    supp(set0) ~= {a2[1]:+.2f} + {a2[0]:.2f}*rho^2   R2={r2:.2f}", flush=True)

    # g(R_j/R_i): the rho^2-slope (interference strength) should RISE with the payoff ratio toward 1 (WTA)
    if len(gfit) >= 2:
        print(f"\n  ===== g(R_j/R_i): rho^2-slope vs payoff ratio (winner-take-all as ratio grows) =====", flush=True)
        for R2 in sorted(gfit):
            print(f"    R2/R1={R2:>4}: rho^2-slope(=g·)={gfit[R2]['slope']:+.2f}  intercept={gfit[R2]['intercept']:+.2f}  R2fit={gfit[R2]['r2']:.2f}", flush=True)
        ratios = np.array(sorted(gfit)); slopes = np.array([gfit[r]['slope'] for r in sorted(gfit)])
        mono = np.all(np.diff(slopes) >= -0.05)
        print(f"  => slope rises with payoff ratio: {'YES (g monotone -> stronger winner suppresses more) SUPPORTED' if mono else 'NO -- report'}", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
