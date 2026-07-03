"""
sigma-K — IS THE SURFACE TENSION COMBINATORIAL? (does the free-energy barrier grow with the number of sub-circuits K
that must conjoin?). This nails what sigma IS: not a geometric n^(2/3) cost, but the cost of co-aligning K components.

K knob = composition depth (key_len): kl1 = match-and-copy (the basic induction circuit); kl2 = compose a bigram key,
THEN match, THEN copy (one extra required sub-circuit); kl3 = one more. So larger key_len = more components that must
conjoin = larger K. The nucleus from Seed Dissect was QK+OV+EMBED at kl1; each composition step adds a required piece.

For each key_len we measure the free-energy barrier height from ΔG(n)=-log occupancy (n = q_mass induction mass). If the
barrier height GROWS with key_len (K), then sigma is the combinatorial conjunction cost: more parts to align -> taller
barrier -> rarer half-built state. That is the defining measurement of a COMBINATORIAL (not geometric) surface tension.

  python sigma_K.py --key_lens 1 2 3 --seeds 16
  python sigma_K.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from seed_tool import _new_model, KEY_POOL, VAL_POOL, FREQ, NOISE, MAX_KEYS, T
from data import Regularity, build_vocab, generate_batch, IGNORE
from run_F1 import acc_and_ce
from run_expA import RESULTS


def data_kl(seed, key_len, kp, vp, device):
    rng = np.random.default_rng(seed)
    regs = [Regularity("k", key_len=key_len, freq=FREQ, noise=NOISE, max_keys=MAX_KEYS)]
    vocab = build_vocab(regs, key_pool_size=kp, val_pool_size=vp)
    pool, _, _, _ = generate_batch(4096, T, regs, vocab, rng, device=device)
    ev4 = generate_batch(256, T, regs, vocab, np.random.default_rng(seed + 9999), device=device)
    return pool, ev4[:3], vocab, rng


@torch.no_grad()
def q_mass(model, ev_tok, ev_tgt):
    _, attns = model(ev_tok, want_attn=True)
    qmask = (ev_tgt != IGNORE); match = (ev_tok[:, None, :] == ev_tgt[:, :, None]).float(); qm = qmask[:, None].float()
    best = 0.0
    for att in attns:
        mass = (att * match[:, None]).sum(-1); ph = (mass * qm).sum(dim=(0, 2)) / qm.sum().clamp(min=1)
        best = max(best, float(ph.max()))
    return best


def one_seed(seed, key_len, device, max_steps, eval_every, post):
    pool, ev, vocab, rng = data_kl(seed, key_len, KEY_POOL, VAL_POOL, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    qm = []; formed = None; stop = max_steps
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            a, _ = acc_and_ce(model, *ev); qm.append(q_mass(model, ev_tok, ev_tgt))
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; stop = min(max_steps, st + post)
        if st >= stop:
            break
        model.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(seed=seed, key_len=key_len, tstar=formed, q_mass=qm)


def barrier_height(qmass_lists, bins=20):
    """ΔG(n)=-log occupancy from pooled q_mass; return barrier height (interior max above the two wells)."""
    vals = np.concatenate([np.array(x, float) for x in qmass_lists]) if qmass_lists else np.array([])
    if vals.size < 30:
        return None
    edges = np.linspace(0.0, float(np.percentile(vals, 99.5)), bins + 1)
    hist, _ = np.histogram(vals, bins=edges); p = hist / hist.sum()
    G = -np.log(np.where(p > 0, p, np.nan)); G = G - np.nanmin(G)
    fin = np.where(np.isfinite(G))[0]
    if len(fin) < 5:
        return None
    gv = G[fin]
    lo = fin[int(np.argmin(gv[: len(gv) // 2]))]; hi = fin[len(gv) // 2 + int(np.argmin(gv[len(gv) // 2:]))]
    between = [i for i in fin if lo < i < hi]
    if not between:
        return None
    ipk = max(between, key=lambda i: G[i])
    return float(G[ipk] - max(G[lo], G[hi]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key_lens", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--post", type=int, default=1200)
    ap.add_argument("--out", default="sigma_K.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.key_lens, args.seeds, args.max_steps, args.post = [1, 2], 4, 5000, 600
    print(f"device={device} key_lens(K)={args.key_lens} seeds={args.seeds}", flush=True)
    rows = []
    for kl in args.key_lens:
        for s in range(args.seeds):
            r = one_seed(s, kl, device, args.max_steps, args.eval_every, args.post)
            rows.append(r); json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        nf = sum(1 for x in rows if x["key_len"] == kl and x["tstar"] is not None)
        print(f"  kl{kl}: formed {nf}/{args.seeds}", flush=True)
    analyze(rows, args.key_lens)
    print(f"\n  saved results/{args.out}")


def analyze(rows, key_lens):
    print("\n  ===========  sigma-K: free-energy barrier height vs conjunction depth K (key_len)  ===========")
    print(f"  {'key_len(K)':>10} {'#formed':>8} {'barrier_height(σ)':>18}")
    pts = []
    for kl in key_lens:
        formed = [x["q_mass"] for x in rows if x["key_len"] == kl and x["tstar"] is not None]
        bh = barrier_height(formed)
        pts.append((kl, len(formed), bh))
        print(f"  {kl:>10} {len(formed):>8} {bh if bh is None else round(bh, 2):>18}")
    valid = [(kl, bh) for kl, nf, bh in pts if bh is not None]
    if len(valid) >= 2:
        kls = np.array([v[0] for v in valid]); bhs = np.array([v[1] for v in valid])
        slope = np.polyfit(kls, bhs, 1)[0]
        print(f"\n  barrier height vs K: slope = {slope:+.2f} per composition step")
        print(f"  ==> {'COMBINATORIAL surface tension: barrier GROWS with K (more parts to conjoin = taller barrier = rarer half-built state). sigma = the conjunction cost.' if slope > 0.3 else 'barrier does not clearly grow with K -- report honestly'}")


if __name__ == "__main__":
    main()
