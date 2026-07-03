"""
COMBINATORIAL NUCLEATION THEOREM (the direct quantitative test of our combinatorial sub-type) — the modified nucleation
theorem for a combinatorial (non-geometric) barrier: if formation requires K sub-circuits to CO-ALIGN and the rate is
J ∝ p^K (p = per-component alignment probability), then seeding j of the K components leaves J ∝ p^(K-j), so

    ln(rate)  =  const  +  j · |ln p|        (LINEAR in the number of components seeded; slope = |ln p|)

and the rate saturates when j = K. This is the combinatorial analog of the classical nucleation theorem
(∂lnJ/∂lnS = n*+1), which FAILS for us (gives n*~=-0.6 vs ΔL*, because the barrier does NOT contain the driving force).
Here we read off the per-component log-speedup |ln p| and confirm the conjunction depth K = 3 (QK ∧ OV ∧ EMBED).

Method: transplant the FULL 2^3 = 8 subsets of {QK, OV, EMBED} (full strength) into a fresh recipient, train, measure t*.
Then fit ln(1/t*) vs |subset|. Reuses Seed Dissect's transplant machinery. (POS excluded — Seed Dissect showed it is a
REVERSE nucleus.)

  python comb_nuc_theorem.py --seeds 12
  python comb_nuc_theorem.py --smoke
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np
import torch
from seed_tool import _data, _new_model, _train, KEY_LEN, KEY_POOL, VAL_POOL
from seed_dissect import transplant
from run_expA import RESULTS

COMPS = ["QK", "OV", "EMBED"]
SUBSETS = [list(c) for k in range(4) for c in itertools.combinations(COMPS, k)]   # all 8 subsets, by size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n_donor", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--out", default="comb_nuc.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.seeds, args.n_donor, args.max_steps = 4, 1, 3500
    print(f"device={device} subsets={['+'.join(s) or 'de-novo' for s in SUBSETS]} seeds={args.seeds} "
          f"(kl{KEY_LEN}/kp{KEY_POOL}/vp{VAL_POOL})", flush=True)

    donors = []; cand = 0
    while len(donors) < args.n_donor and cand < args.n_donor + 5:
        pool, ev, vocab, rng = _data(1000 + cand, device)
        m = _new_model(vocab, 1000 + cand, device); r = _train(m, pool, ev, device, rng)
        cand += 1
        if not r["censored"] and np.isfinite(r["tstar"]):
            donors.append({k: v.detach().clone() for k, v in m.state_dict().items()})
            print(f"  donor {len(donors)}: formed@{r['tstar']:.0f}", flush=True)
    if not donors:
        print("  NO donor formed; abort"); return

    rows = []
    for subset in SUBSETS:
        for seed in range(args.seeds):
            pool, ev, vocab, rng = _data(seed, device)
            rec = _new_model(vocab, seed, device)
            if subset:
                transplant(rec, donors[seed % len(donors)], subset)
            r = _train(rec, pool, ev, device, rng, max_steps=args.max_steps)
            rows.append(dict(subset="+".join(subset) or "de-novo", n=len(subset),
                             seed=seed, tstar=float(r["tstar"]), censored=bool(r["censored"])))
            json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        sub = [x["tstar"] for x in rows if x["subset"] == ("+".join(subset) or "de-novo") and not x["censored"]]
        print(f"  {('+'.join(subset) or 'de-novo'):>14} (n={len(subset)}): formed {len(sub)}/{args.seeds} "
              f"t*med={np.median(sub) if sub else float('nan'):.0f}", flush=True)
    analyze(rows)
    print(f"\n  saved results/{args.out}")


def analyze(rows):
    print("\n  ===========  COMBINATORIAL NUCLEATION THEOREM: ln(rate) vs # components seeded  ===========")
    print(f"  {'subset':>14} {'#comp':>5} {'formed':>7} {'t*med':>7} {'ln(rate)':>9}")
    js, lrs = [], []
    bycomp = {}
    for subset in SUBSETS:
        name = "+".join(subset) or "de-novo"
        sub = [x["tstar"] for x in rows if x["subset"] == name and not x["censored"]]
        if not sub:
            print(f"  {name:>14} {len(subset):>5}  (none formed)"); continue
        med = float(np.median(sub)); lr = float(np.log(1.0 / med))
        js.append(len(subset)); lrs.append(lr); bycomp[name] = (len(subset), lr)
        print(f"  {name:>14} {len(subset):>5} {len(sub):>7} {med:>7.0f} {lr:>9.2f}")
    js = np.array(js, float); lrs = np.array(lrs)
    if len(set(js)) >= 3:
        slope, b = np.polyfit(js, lrs, 1)
        pred = slope * js + b; ss_res = ((lrs - pred) ** 2).sum(); ss_tot = ((lrs - lrs.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        print(f"\n  FIT ln(rate) = {b:.2f} + {slope:+.2f}*(#comp)   R2={r2:.2f}")
        print(f"  => slope = |ln p| = {slope:.2f}  (per-component multiplicative speedup x{np.exp(slope):.2f})")
        print(f"  => p (per-component alignment prob) ~= {np.exp(-slope):.3f}")
        print(f"  => {'COMBINATORIAL NUCLEATION THEOREM HOLDS: ln(rate) linear in #components, K=3 conjunction (QK,OV,EMBED), J~p^K' if r2 > 0.6 and slope > 0.2 else 'not clean -- report honestly (components non-equivalent)'}")
        # per-component marginal contribution (does each of QK/OV/EMBED multiplicatively speed up?)
        print("\n  per-component marginal speedup (median over subsets with vs without it):")
        for c in COMPS:
            wi = [lr for nm, (n, lr) in bycomp.items() if c in nm.split("+")]
            wo = [lr for nm, (n, lr) in bycomp.items() if c not in nm.split("+") and nm != "de-novo" or nm == "de-novo"]
            if wi and wo:
                d = np.mean(wi) - np.mean(wo)
                print(f"    {c:>6}: Δln(rate)={d:+.2f}  (x{np.exp(d):.2f})")


if __name__ == "__main__":
    main()
