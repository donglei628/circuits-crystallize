"""
SEED DISSECTION — what EXACTLY is the transplantable nucleus? (reviewer r7's "most original" question)

The whole-head seed (s=0.18) is ambiguous: is it 18% of structure / function / norm? r7's idea: transplant DIFFERENT
PARTS of a formed circuit (full strength, rest random) and see which part(s) suffice to nucleate fast. The minimal
sufficient component set = the structural nucleus. This turns "a seed accelerates formation" into "THIS specific
sub-structure IS the nucleus", which is far harder for a functional-threshold story to explain.

Components of the attention-only toy (per layer): QK = qkv.weight[0:2d] (matcher); OV = qkv.weight[2d:3d] + out.weight
(copy); EMBED = tok + unembed; POS = pos. Transplant a component SET into a fresh random recipient at full strength,
train, measure t*. Compare across sets: which is the nucleus.

  python seed_dissect.py --seeds 10
  python seed_dissect.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from seed_tool import _data, _new_model, _train, KEY_LEN, KEY_POOL, VAL_POOL
from run_expA import RESULTS

# component -> predicate(name, d_model) -> slice spec (or None). Slice given as (dim0_lo, dim0_hi) or None=whole tensor.
def comp_slices(name, comp, d):
    """return list of slice-objects (row ranges) of `name` that belong to component `comp`, or None if not involved."""
    if comp == "QK" and name.endswith("attn.qkv.weight"):
        return (0, 2 * d)
    if comp == "OV" and name.endswith("attn.qkv.weight"):
        return (2 * d, 3 * d)
    if comp == "OV" and name.endswith("attn.out.weight"):
        return (0, None)
    if comp == "EMBED" and (name == "tok.weight" or name == "unembed.weight"):
        return (0, None)
    if comp == "POS" and name == "pos.weight":
        return (0, None)
    if comp == "LN" and ("ln" in name):
        return (0, None)
    return None


def transplant(recipient, donor_sd, components):
    """set recipient's weights = donor's, for the slices belonging to any component in `components`; rest stays random."""
    d = recipient.cfg.d_model
    with torch.no_grad():
        for name, p in recipient.named_parameters():
            for comp in components:
                sl = comp_slices(name, comp, d)
                if sl is None:
                    continue
                lo, hi = sl
                hi = p.shape[0] if hi is None else hi
                p.data[lo:hi] = donor_sd[name][lo:hi]


# component-sets to test (the dissection menu)
SETS = {
    "de-novo": [],
    "QK": ["QK"],
    "OV": ["OV"],
    "EMBED": ["EMBED"],
    "POS": ["POS"],
    "QK+OV": ["QK", "OV"],
    "QK+OV+EMBED": ["QK", "OV", "EMBED"],
    "ALL": ["QK", "OV", "EMBED", "POS", "LN"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=None)         # subset of SETS keys
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_donor", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--out", default="seed_dissect.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = args.sets or list(SETS)
    if args.smoke:
        sets = ["de-novo", "OV", "QK", "QK+OV", "ALL"]; args.seeds, args.n_donor, args.max_steps = 3, 1, 3000
    print(f"device={device} sets={sets} seeds={args.seeds} (config kl{KEY_LEN} pool{KEY_POOL} vp{VAL_POOL})", flush=True)

    # donors (must form)
    donors = []; cand = 0
    while len(donors) < args.n_donor and cand < args.n_donor + 5:
        pool, ev, vocab, rng = _data(1000 + cand, device)
        m = _new_model(vocab, 1000 + cand, device)
        r = _train(m, pool, ev, device, rng)
        cand += 1
        if not r["censored"] and np.isfinite(r["tstar"]):
            donors.append({k: v.detach().clone() for k, v in m.state_dict().items()})
            print(f"  donor {len(donors)}: formed@{r['tstar']:.0f}", flush=True)
    if not donors:
        print("  NO donor formed; abort"); return

    rows = []
    for sname in sets:
        comps = SETS[sname]
        for seed in range(args.seeds):
            pool, ev, vocab, rng = _data(seed, device)
            rec = _new_model(vocab, seed, device)
            if comps:
                transplant(rec, donors[seed % len(donors)], comps)
            r = _train(rec, pool, ev, device, rng, max_steps=args.max_steps)
            rows.append(dict(set=sname, seed=seed, tstar=float(r["tstar"]), censored=bool(r["censored"])))
            json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        sub = [x["tstar"] for x in rows if x["set"] == sname and not x["censored"]]
        print(f"  {sname:>14}: formed {len(sub)}/{args.seeds}  t*med={np.median(sub) if sub else float('nan'):.0f}", flush=True)

    analyze(rows, sets)
    print(f"\n  saved results/{args.out}")


def analyze(rows, sets):
    print("\n  ===========  SEED DISSECTION: which component is the nucleus?  ===========")
    denovo = [x["tstar"] for x in rows if x["set"] == "de-novo" and not x["censored"]]
    base = float(np.median(denovo)) if denovo else float("nan")
    print(f"  de-novo (no seed) t* = {base:.0f}")
    print(f"  {'component set':>14} {'formed':>7} {'t*med':>7} {'speedup vs de-novo':>20}")
    res = {}
    for sname in sets:
        sub = [x["tstar"] for x in rows if x["set"] == sname and not x["censored"]]
        med = float(np.median(sub)) if sub else float("nan")
        res[sname] = med
        sp = base / med if (sub and med > 0 and np.isfinite(base)) else float("nan")
        print(f"  {sname:>14} {len(sub):>7} {med:>7.0f} {sp:>18.1f}x")
    print("\n  Reads: a component that ALONE gives a large speedup is (part of) the nucleus; if no single component")
    print("  suffices but a COMBINATION does, the nucleus is that minimal combination. A pure functional threshold")
    print("  would not care WHICH sub-structure is transplanted, only how much -- so component-specificity is the signal.")


if __name__ == "__main__":
    main()
