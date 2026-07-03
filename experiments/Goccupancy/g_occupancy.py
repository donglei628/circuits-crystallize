"""
G-occupancy — the FREE ENERGY ΔG(n), done right: ΔG(n) = -log P(n), where P(n) is the OCCUPANCY (how often the order
parameter sits at completeness n along the dynamics). This is the operational free energy / potential of mean force.

Why this works where M2 (drift) and Surface Cost (transplant) failed: the free energy IS -log(probability) (Boltzmann).
A metastable plateau (where the system dwells) is a deep well (high occupancy); the half-built circuit (transited fast,
rarely occupied) is the BARRIER. M1's bimodal histogram already hinted at this; here we measure it cleanly.

Clean version: order parameter n = q_mass (leading-head induction mass, the cleanest two-state variable from M1);
track densely THROUGH the snap AND long PAST it (so the formed state is a proper, deep well comparable to the plateau);
many seeds for a smooth histogram. Output: ΔG(n) with its two wells (plateau, formed) and the interior barrier; the
barrier HEIGHT is the (combinatorial) surface-tension scale σ.

  python g_occupancy.py --seeds 24
  python g_occupancy.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from seed_tool import _data, _new_model, KEY_LEN, KEY_POOL, VAL_POOL
from data import IGNORE
from run_F1 import acc_and_ce
from run_expA import RESULTS


@torch.no_grad()
def q_mass(model, ev_tok, ev_tgt):
    _, attns = model(ev_tok, want_attn=True)
    qmask = (ev_tgt != IGNORE); match = (ev_tok[:, None, :] == ev_tgt[:, :, None]).float(); qm = qmask[:, None].float()
    best = 0.0
    for att in attns:
        mass = (att * match[:, None]).sum(-1); ph = (mass * qm).sum(dim=(0, 2)) / qm.sum().clamp(min=1)
        best = max(best, float(ph.max()))
    return best


def one_seed(seed, device, max_steps, eval_every, post):
    pool, ev, vocab, rng = _data(seed, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps, qm, accs = [], [], []; formed = None; stop = max_steps
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            a, _ = acc_and_ce(model, *ev)
            steps.append(st); qm.append(q_mass(model, ev_tok, ev_tgt)); accs.append(float(a))
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; stop = min(max_steps, st + post)
        if st >= stop:
            break
        model.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(seed=seed, tstar=formed, steps=steps, q_mass=qm, acc=accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--post", type=int, default=1500)     # LONG post-formation tracking -> a proper formed well
    ap.add_argument("--bins", type=int, default=22)
    ap.add_argument("--out", default="g_occupancy.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 3 if args.smoke else args.seeds
    if args.smoke:
        args.max_steps, args.post, args.eval_every = 3500, 600, 8
    print(f"device={device} seeds={n} eval_every={args.eval_every} post={args.post}", flush=True)
    rows = []
    for s in range(n):
        r = one_seed(s, device, args.max_steps, args.eval_every, args.post)
        rows.append(r); json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        print(f"  seed {s}: t*={r['tstar']} ({len(r['steps'])} evals)", flush=True)
    analyze(rows, args.bins)
    print(f"\n  saved results/{args.out}")


def analyze(rows, bins):
    vals = []
    for r in rows:
        if r["tstar"] is not None:
            vals += r["q_mass"]
    vals = np.array(vals, float)
    lo, hi = 0.0, float(np.percentile(vals, 99.5))
    edges = np.linspace(lo, hi, bins + 1); hist, _ = np.histogram(vals, bins=edges)
    p = hist / hist.sum(); centers = (edges[:-1] + edges[1:]) / 2
    G = -np.log(np.where(p > 0, p, np.nan)); G = G - np.nanmin(G)
    print(f"\n  ===========  ΔG(n) = -log occupancy   (n = q_mass, {len([r for r in rows if r['tstar']])} formed)  ===========")
    print(f"  {'n':>6} {'occ':>8} {'ΔG':>6}")
    for c, pp, g in zip(centers, p, G):
        bar = "#" * int(g * 3) if np.isfinite(g) else "x" * 6
        print(f"  {c:>6.3f} {pp:>8.4f} {g:>6.2f}  {bar}")
    # wells = local minima of G; barrier = the interior max between the lowest two wells
    fin = np.where(np.isfinite(G))[0]
    if len(fin) >= 5:
        # plateau well ~ lowest-n minimum; formed well ~ highest-n minimum
        gv = G[fin]
        i_lowwell = fin[int(np.argmin(gv[: len(gv) // 2]))]
        i_highwell = fin[len(gv) // 2 + int(np.argmin(gv[len(gv) // 2:]))]
        between = [i for i in fin if i_lowwell < i < i_highwell]
        if between:
            ipk = max(between, key=lambda i: G[i])
            barrier = G[ipk] - max(G[i_lowwell], G[i_highwell])
            print(f"\n  plateau well  @n={centers[i_lowwell]:.3f} (ΔG={G[i_lowwell]:.2f})")
            print(f"  formed well   @n={centers[i_highwell]:.3f} (ΔG={G[i_highwell]:.2f})")
            print(f"  BARRIER       @n={centers[ipk]:.3f} (ΔG={G[ipk]:.2f})  ->  barrier height above wells = {barrier:.2f}")
            print(f"  ==> {'CLEAN DOUBLE-WELL: ΔG(n) has two minima + an interior barrier (the free energy of circuit nucleation)' if barrier > 0.5 else 'barrier shallow/absent'}")
            print(f"      barrier height ({barrier:.2f} in -log units) = the surface-tension scale σ (rarity of the half-built circuit)")


if __name__ == "__main__":
    main()
