"""
N-growth — POST-NUCLEATION GROWTH LAW (the "stochastic ignition -> deterministic growth" crossover; independent of M2).

Nucleation theory: BEFORE the critical nucleus, ignition is STOCHASTIC (the waiting time t* is random/Poisson, high
variance across seeds). AFTER the critical nucleus, growth is ~DETERMINISTIC (the nucleus grows in a reproducible way,
low variance). So aligning trajectories at the snap, the order parameter n(t) should:
  - have HIGH cross-seed variance in WHEN the snap happens (stochastic ignition)
  - but LOW cross-seed variance in HOW it grows AFTER the snap (deterministic growth)
and the post-snap growth should follow a clean kinetic curve.

We train de-novo, track the order parameter (q_active heads, q_mass) densely through AND well past the snap, align at
t*, and measure: (a) post-snap growth determinism (variance collapse after alignment), (b) the growth shape.

  python n_growth.py --seeds 16
  python n_growth.py --smoke
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
def order_params(model, ev, ev_tok, ev_tgt, thr=0.25):
    a, _ = acc_and_ce(model, *ev)
    _, attns = model(ev_tok, want_attn=True)
    qmask = (ev_tgt != IGNORE); match = (ev_tok[:, None, :] == ev_tgt[:, :, None]).float(); qm = qmask[:, None].float()
    h = []
    for att in attns:
        mass = (att * match[:, None]).sum(-1); ph = (mass * qm).sum(dim=(0, 2)) / qm.sum().clamp(min=1)
        h.extend(ph.tolist())
    h = np.clip(np.array(h, float), 0, None)
    return float(a), float(h.max()), float((h > thr).sum())


def one_seed(seed, device, max_steps, eval_every, post):
    pool, ev, vocab, rng = _data(seed, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps, accs, qmass, qact = [], [], [], []; formed = None; stop = max_steps
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            a, qm, qa = order_params(model, ev, ev_tok, ev_tgt)
            steps.append(st); accs.append(a); qmass.append(qm); qact.append(qa)
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; stop = min(max_steps, st + post)
        if st >= stop:
            break
        model.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(seed=seed, tstar=formed, steps=steps, acc=accs, q_mass=qmass, q_active=qact)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--post", type=int, default=800)     # steps to keep tracking AFTER the snap
    ap.add_argument("--out", default="n_growth.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 3 if args.smoke else args.seeds
    if args.smoke:
        args.max_steps, args.post = 3000, 400
    print(f"device={device} seeds={n} eval_every={args.eval_every} post={args.post}", flush=True)
    rows = []
    for s in range(n):
        r = one_seed(s, device, args.max_steps, args.eval_every, args.post)
        rows.append(r); json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        print(f"  seed {s}: t*={r['tstar']}", flush=True)
    analyze(rows, args.eval_every)
    print(f"\n  saved results/{args.out}")


def analyze(rows, ev):
    formed = [r for r in rows if r["tstar"] is not None]
    print(f"\n  ===========  N-growth: stochastic ignition -> deterministic growth  ===========")
    print(f"  {len(formed)}/{len(rows)} formed")
    # ignition stochasticity: spread of t*
    ts = np.array([r["tstar"] for r in formed], float)
    print(f"  IGNITION (when the snap happens): t* = {np.median(ts):.0f} +- {ts.std():.0f}  (CV={ts.std()/ts.mean():.2f})  <- stochastic")
    # post-snap growth: align each trajectory at its t*, sample q_active at offsets after the snap
    for cand in ["q_active", "q_mass"]:
        offs = [0, 50, 100, 200, 400]
        print(f"\n  GROWTH of {cand} after the snap (aligned at t*):")
        print(f"    {'offset':>7} {'mean':>7} {'std':>6} {'CV':>6}")
        for off in offs:
            vals = []
            for r in formed:
                st = np.array(r["steps"]); v = np.array(r[cand], float)
                idx = np.argmin(np.abs(st - (r["tstar"] + off)))
                if abs(st[idx] - (r["tstar"] + off)) <= ev * 2:
                    vals.append(v[idx])
            if len(vals) >= 2:
                m, sd = np.mean(vals), np.std(vals)
                print(f"    +{off:>5} {m:>7.3f} {sd:>6.3f} {sd/m if m>0 else float('nan'):>6.2f}")
        print(f"    => if CV stays LOW and shrinks after the snap = DETERMINISTIC growth (nucleation: ignition random, growth not)")


if __name__ == "__main__":
    main()
