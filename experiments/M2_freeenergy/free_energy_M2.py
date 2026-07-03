"""
M2 — FREE ENERGY FROM THE DYNAMICS (derive ΔG(n) operationally, do NOT assume ΔG = loss).

The free energy is, by definition, the potential the order parameter relaxes in: dn/dt = -dΔG/dn + noise. So we MEASURE
the drift of n at many values and integrate:  ΔG(n) = -∫ drift(n) dn. A real nucleation barrier shows up as a drift
that is NEGATIVE below the critical size (the partial nucleus is pulled back / dissolves) and POSITIVE above it (it
grows) -> ΔG(n) has an interior MAXIMUM (the barrier) between two minima (metastable plateau, formed). We do this for
EVERY candidate order parameter; the one whose reconstructed ΔG is a clean double-well is the right n AND gives the
true (dynamics-defined) free energy -- from which Δμ, σ follow.

Setup (persistent metastable phase needed): recipients START at the shortcut plateau of a HARD config (kp48, where
de-novo mostly fails), then we transplant the formed nucleus (attention QK+OV) at fraction s to SET the initial n, and
measure the short-horizon drift.

  python free_energy_M2.py --seeds 10
  python free_energy_M2.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from seed_tool import _new_model, _train, KEY_LEN, KEY_POOL, VAL_POOL, FREQ, NOISE, MAX_KEYS, T
from data import IGNORE, Regularity, build_vocab, generate_batch
from run_F1 import acc_and_ce
from run_expA import RESULTS

CANDS = ["acc", "q_mass", "q_neff", "q_active", "q_total"]


def data_cfg(seed, kp, vp, device):
    rng = np.random.default_rng(seed)
    regs = [Regularity("k", key_len=KEY_LEN, freq=FREQ, noise=NOISE, max_keys=MAX_KEYS)]
    vocab = build_vocab(regs, key_pool_size=kp, val_pool_size=vp)
    pool, _, _, _ = generate_batch(4096, T, regs, vocab, rng, device=device)
    ev4 = generate_batch(256, T, regs, vocab, np.random.default_rng(seed + 9999), device=device)
    return pool, ev4[:3], vocab, rng


def transplant_attn(rec, donor_sd, s):
    with torch.no_grad():
        for name, p in rec.named_parameters():
            if name.endswith("attn.qkv.weight") or name.endswith("attn.out.weight"):
                p.data.mul_(1 - s).add_(donor_sd[name].to(p.device), alpha=s)


@torch.no_grad()
def per_head(model, ev_tok, ev_tgt):
    _, attns = model(ev_tok, want_attn=True)
    qmask = (ev_tgt != IGNORE); match = (ev_tok[:, None, :] == ev_tgt[:, :, None]).float(); qm = qmask[:, None].float()
    out = []
    for att in attns:
        mass = (att * match[:, None]).sum(-1); ph = (mass * qm).sum(dim=(0, 2)) / qm.sum().clamp(min=1)
        out.extend(ph.tolist())
    return np.clip(np.array(out, float), 0, None)


def cand_vals(model, ev, ev_tok, ev_tgt, thr=0.25):
    a, _ = acc_and_ce(model, *ev); h = per_head(model, ev_tok, ev_tgt)
    return dict(acc=float(a), q_mass=float(h.max()),
                q_neff=float((h.sum() ** 2) / (h ** 2).sum()) if (h ** 2).sum() > 0 else 0.0,
                q_active=float((h > thr).sum()), q_total=float(h.sum()))


def make_plateau(seed, kp, vp, device, steps):
    pool, ev, vocab, rng = data_cfg(seed, kp, vp, device)
    net = _new_model(vocab, seed, device); opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for st in range(steps):
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(net(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach().clone() for k, v in net.state_dict().items()}, pool, ev, vocab, rng


def probe(plateau_sd, donor_sd, s, kp, vp, seed, device, K_short, eval_every):
    """start at the plateau, transplant nucleus at s, train K_short steps; return n0 and drift for each candidate."""
    pool, ev, vocab, rng = data_cfg(seed, kp, vp, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    rec = _new_model(vocab, seed, device); rec.load_state_dict(plateau_sd)
    if s > 0:
        transplant_attn(rec, donor_sd, s)
    n0 = cand_vals(rec, ev, ev_tok, ev_tgt)
    opt = torch.optim.AdamW(rec.parameters(), lr=1e-3)
    for st in range(K_short):
        rec.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(rec(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    nK = cand_vals(rec, ev, ev_tok, ev_tgt)
    return {c: dict(n0=n0[c], drift=nK[c] - n0[c]) for c in CANDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard_kp", type=int, default=48)
    ap.add_argument("--vp", type=int, default=VAL_POOL)
    ap.add_argument("--svals", type=float, nargs="+",
                    default=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--plateau_steps", type=int, default=500)
    ap.add_argument("--K_short", type=int, default=120)
    ap.add_argument("--out", default="free_energy_M2.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.svals, args.seeds, args.plateau_steps = [0.0, 0.2, 0.4, 0.6], 3, 400
    print(f"device={device} hard_kp={args.hard_kp} svals={args.svals} seeds={args.seeds} K_short={args.K_short}", flush=True)

    # easy donor = nucleus source
    pool, ev, vocab, rng = data_cfg(1000, KEY_POOL, VAL_POOL, device)
    donor = _new_model(vocab, 1000, device); rd = _train(donor, pool, ev, device, rng)
    donor_sd = {k: v.detach().clone() for k, v in donor.state_dict().items()}
    print(f"  easy donor formed@{rd['tstar']:.0f}", flush=True)

    rows = []
    for seed in range(args.seeds):
        plateau_sd, *_ = make_plateau(seed, args.hard_kp, args.vp, device, args.plateau_steps)
        for s in args.svals:
            res = probe(plateau_sd, donor_sd, s, args.hard_kp, args.vp, seed, device, args.K_short, 0)
            rows.append(dict(seed=seed, s=s, **{c: res[c] for c in CANDS}))
            json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        print(f"  seed {seed} done", flush=True)

    analyze(rows)
    print(f"\n  saved results/{args.out}")


def analyze(rows):
    print("\n  ===========  M2: reconstructed free energy ΔG(n) per candidate  ===========")
    for c in CANDS:
        pts = sorted([(r[c]["n0"], r[c]["drift"]) for r in rows])
        n0 = np.array([p[0] for p in pts]); dr = np.array([p[1] for p in pts])
        # bin n0 into ~8 bins, mean drift per bin
        bins = np.linspace(n0.min(), n0.max(), 9)
        idx = np.clip(np.digitize(n0, bins) - 1, 0, 7)
        bn, bd = [], []
        for b in range(8):
            m = idx == b
            if m.sum() >= 2:
                bn.append(n0[m].mean()); bd.append(dr[m].mean())
        if len(bn) < 4:
            print(f"  {c:>9}: too few bins"); continue
        bn, bd = np.array(bn), np.array(bd)
        # ΔG(n) = -∫ drift dn  (cumulative)
        G = -np.concatenate([[0], np.cumsum((bd[1:] + bd[:-1]) / 2 * np.diff(bn))])
        ipk = int(np.argmax(G)); interior = 0 < ipk < len(G) - 1
        # drift sign change neg->pos = unstable fixed point (critical size)
        sign_change = any(bd[i] < 0 and bd[i + 1] > 0 for i in range(len(bd) - 1))
        print(f"  {c:>9}: ΔG interior max(barrier)={interior} at n0={bn[ipk]:.3f}; "
              f"drift neg->pos crossing(critical size)={sign_change}  "
              f"{'<-- DOUBLE-WELL: order-parameter-like' if interior and sign_change else ''}")
    print("\n  the candidate with a clean double-well ΔG (barrier + drift sign-change) is the true order parameter n,")
    print("  and its ΔG is the dynamics-defined free energy -> Δμ (well depth), σ (barrier) follow.")


if __name__ == "__main__":
    main()
