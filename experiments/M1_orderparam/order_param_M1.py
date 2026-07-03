"""
M1 — DISCOVER THE ORDER PARAMETER (which model quantity behaves like the nucleation order parameter n).

A true (first-order) order parameter has defining properties we can TEST, instead of assuming the mapping:
  (P1) it JUMPS sharply at the snap (small 10%->90% width)  -- a two-state variable, not a sliding quality
  (P2) across the transition it is BIMODAL: pooled over (seed, time), values cluster at LOW and HIGH with a GAP in the
       middle (the system is rarely caught half-way). An intensive quality (e.g. accuracy) slides through the middle.
We track several candidates through de-novo formation and score them on P1+P2. The winner is the order parameter n.

Candidates:
  acc      = repeat-position accuracy            (intensive; the formation signal itself)
  q_mass   = leading-head induction mass         (intensive 0-1 quality -- our old, suspect choice)
  q_neff   = participation ratio of induction mass over heads = (Σs)^2/Σs^2  (EXTENSIVE: a count of heads)
  q_active = number of heads with induction mass > thr  (EXTENSIVE: a count)
  q_total  = total induction mass summed over heads      (EXTENSIVE: a sum)

  python order_param_M1.py --seeds 16
  python order_param_M1.py --smoke
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
def per_head(model, ev_tok, ev_tgt):
    _, attns = model(ev_tok, want_attn=True)
    qmask = (ev_tgt != IGNORE); match = (ev_tok[:, None, :] == ev_tgt[:, :, None]).float(); qm = qmask[:, None].float()
    out = []
    for att in attns:
        mass = (att * match[:, None]).sum(-1); ph = (mass * qm).sum(dim=(0, 2)) / qm.sum().clamp(min=1)
        out.extend(ph.tolist())
    return np.clip(np.array(out, float), 0, None)


def candidates(model, ev, ev_tok, ev_tgt, thr=0.25):
    a, _ = acc_and_ce(model, *ev)
    h = per_head(model, ev_tok, ev_tgt)
    q_mass = float(h.max())
    q_neff = float((h.sum() ** 2) / (h ** 2).sum()) if (h ** 2).sum() > 0 else 0.0
    q_active = float((h > thr).sum())
    q_total = float(h.sum())
    return dict(acc=float(a), q_mass=q_mass, q_neff=q_neff, q_active=q_active, q_total=q_total)


def one_seed(seed, device, max_steps, eval_every):
    pool, ev, vocab, rng = _data(seed, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    traj = []; formed = None; stop = max_steps; buffer = 400
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            c = candidates(model, ev, ev_tok, ev_tgt); c["step"] = st; traj.append(c)
            if formed is None and not np.isnan(c["acc"]) and c["acc"] >= 0.80:
                formed = st; stop = min(max_steps, st + buffer)
        if st >= stop:
            break
        model.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(seed=seed, tstar=formed, traj=traj)


CANDS = ["acc", "q_mass", "q_neff", "q_active", "q_total"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--out", default="order_param_M1.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 3 if args.smoke else args.seeds
    if args.smoke:
        args.max_steps = 2500
    print(f"device={device} seeds={n} eval_every={args.eval_every} (kl{KEY_LEN} pool{KEY_POOL} vp{VAL_POOL})", flush=True)
    rows = []
    for s in range(n):
        r = one_seed(s, device, args.max_steps, args.eval_every)
        rows.append(r)
        print(f"  seed {s}: t*={r['tstar']} ({len(r['traj'])} evals)", flush=True)
        json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
    analyze(rows)
    print(f"\n  saved results/{args.out}")


def analyze(rows):
    print("\n  ===========  M1: which candidate is the order parameter?  ===========")
    formed = [r for r in rows if r["tstar"] is not None]
    print(f"  {len(formed)}/{len(rows)} formed")
    print(f"  {'candidate':>9} {'jump_width':>11} {'mid_frac':>9}  (sharp jump + low mid_frac = two-state = order param)")
    scores = {}
    for cand in CANDS:
        widths, mids = [], []
        for r in formed:
            st = np.array([t["step"] for t in r["traj"]]); v = np.array([t[cand] for t in r["traj"]], float)
            lo, hi = np.percentile(v, 5), np.percentile(v, 95)
            if hi - lo < 1e-6:
                continue
            vn = (v - lo) / (hi - lo)
            # jump width: steps from first crossing 0.1 to first crossing 0.9 (around the rise)
            up10 = np.where(vn >= 0.1)[0]; up90 = np.where(vn >= 0.9)[0]
            if up10.size and up90.size and up90[0] > up10[0]:
                widths.append(st[up90[0]] - st[up10[0]])
            # bimodality proxy: fraction of timepoints in the middle band [0.33,0.67]
            mids.append(float(((vn >= 0.33) & (vn <= 0.67)).mean()))
        jw = float(np.median(widths)) if widths else float("nan")
        mf = float(np.median(mids)) if mids else float("nan")
        scores[cand] = (jw, mf)
        print(f"  {cand:>9} {jw:>11.0f} {mf:>9.3f}")
    # rank: smallest mid_frac (most two-state) wins; tie-break by jump width
    valid = {k: v for k, v in scores.items() if np.isfinite(v[1])}
    if valid:
        winner = min(valid, key=lambda k: (valid[k][1], valid[k][0]))
        print(f"\n  => most two-state (order-parameter-like): {winner}  (mid_frac={valid[winner][1]:.3f})")
        print(f"     intensive (acc,q_mass) vs extensive (q_neff,q_active,q_total): does an EXTENSIVE one win?")
        print(f"     if yes -> the order parameter is a COUNT (heads), not a quality (induction mass) -> remap n.")


if __name__ == "__main__":
    main()
