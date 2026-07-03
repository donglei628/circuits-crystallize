"""
SURFACE COST — the functional "surface tension": is a HALF-BUILT circuit PENALIZED (raises loss above the shortcut),
the way a sub-critical crystal nucleus pays a surface-energy cost? (the n^(2/3) leg of the nucleation correspondence.)

Physics: ΔG(n) = -n·Δμ + σ·n^(2/3); the surface term makes PARTIAL nuclei cost free energy, so ΔG humps before it
falls. Circuit analog: a partially-assembled matcher+copier doesn't retrieve, so it removes no loss and may even raise
it -> the gradient doesn't pull it -> noise dissolves it. So along the assembly path of the REAL nucleus (QK+OV+EMBED,
from the dissection), the (frozen) loss L(s) should HUMP -- rise above the shortcut-plateau loss for intermediate s,
then fall to the formed state -- exactly the surface-energy barrier.

This is the STATIC view of the same barrier whose DYNAMIC view is the CSD grow-or-die. Inference-light (mostly frozen
eval), runs on 184.

  python surface_cost.py --seeds 6
  python surface_cost.py --smoke
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

NUCLEUS = ["QK", "OV", "EMBED"]   # the real nucleus from the dissection


def data_cfg(seed, kp, vp, device):
    rng = np.random.default_rng(seed)
    regs = [Regularity("k", key_len=KEY_LEN, freq=FREQ, noise=NOISE, max_keys=MAX_KEYS)]
    vocab = build_vocab(regs, key_pool_size=kp, val_pool_size=vp)
    pool, _, _, _ = generate_batch(4096, T, regs, vocab, rng, device=device)
    ev4 = generate_batch(256, T, regs, vocab, np.random.default_rng(seed + 9999), device=device)
    return pool, ev4[:3], vocab, rng


def comp_slice(name, comp, d):
    if comp == "QK" and name.endswith("attn.qkv.weight"): return (0, 2 * d)
    if comp == "OV" and name.endswith("attn.qkv.weight"): return (2 * d, 3 * d)
    if comp == "OV" and name.endswith("attn.out.weight"): return (0, None)
    if comp == "EMBED" and name in ("tok.weight", "unembed.weight"): return (0, None)
    return None


def transplant(recipient, donor_sd, comps, s):
    """blend the nucleus component slices at fraction s: theta = (1-s)*rand + s*donor."""
    d = recipient.cfg.d_model
    with torch.no_grad():
        for name, p in recipient.named_parameters():
            for comp in comps:
                sl = comp_slice(name, comp, d)
                if sl is None: continue
                lo, hi = sl; hi = p.shape[0] if hi is None else hi
                p.data[lo:hi] = (1 - s) * p.data[lo:hi] + s * donor_sd[name][lo:hi]


@torch.no_grad()
def nucleus_mass(model, ev_tok, ev_tgt):
    _, attns = model(ev_tok, want_attn=True)
    qmask = (ev_tgt != IGNORE); match = (ev_tok[:, None, :] == ev_tgt[:, :, None]).float(); qm = qmask[:, None].float()
    best = 0.0
    for att in attns:
        mass = (att * match[:, None]).sum(-1); ph = (mass * qm).sum(dim=(0, 2)) / qm.sum().clamp(min=1)
        best = max(best, float(ph.max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svals", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--kp", type=int, default=KEY_POOL)
    ap.add_argument("--vp", type=int, default=VAL_POOL)
    ap.add_argument("--plateau_steps", type=int, default=600)
    ap.add_argument("--out", default="surface_cost.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.svals, args.seeds = [0.0, 0.3, 0.6, 1.0], 3
    print(f"device={device} svals={args.svals} seeds={args.seeds} nucleus={NUCLEUS} (kp{args.kp} vp{args.vp})", flush=True)

    # donor = the nucleus source (a formed circuit)
    pool, ev, vocab, rng = data_cfg(1000, args.kp, args.vp, device)
    donor = _new_model(vocab, 1000, device)
    rd = _train(donor, pool, ev, device, rng)
    donor_sd = {k: v.detach().clone() for k, v in donor.state_dict().items()}
    print(f"  donor formed@{rd['tstar']:.0f}", flush=True)

    # recipients START AT THE SHORTCUT PLATEAU (trained on the shortcut, before they form). The surface-cost test:
    # add the donor nucleus to a plateau net -> does a PARTIAL nucleus RAISE the loss ABOVE the plateau (incompatible
    # half-built structure) before the formed circuit lowers it? That hump = the functional surface tension.
    plateaus = []
    for seed in range(args.seeds):
        poolk, evk, vocabk, rngk = data_cfg(seed, args.kp, args.vp, device)
        pnet = _new_model(vocabk, seed, device); opt = torch.optim.AdamW(pnet.parameters(), lr=1e-3)
        for st in range(args.plateau_steps):
            idx = torch.from_numpy(rngk.integers(poolk.shape[0], size=64)).to(device); tok = poolk[idx]
            loss = F.cross_entropy(pnet(tok[:, :-1]).reshape(-1, vocabk.size), tok[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sd = {k: v.detach().clone() for k, v in pnet.state_dict().items()}
        _, lp = acc_and_ce(pnet, *evk)
        plateaus.append((sd, evk, vocabk, seed, float(lp)))
    L_plateau = float(np.mean([p[4] for p in plateaus]))
    print(f"  shortcut-plateau loss L_plateau = {L_plateau:.3f} (mean of {args.seeds} plateau nets)", flush=True)

    rows = []
    for s in args.svals:
        Ls, ns = [], []
        for sd, evk, vocabk, seed, lp in plateaus:
            rec = _new_model(vocabk, seed, device); rec.load_state_dict(sd)     # START at the plateau
            if s > 0:
                transplant(rec, donor_sd, NUCLEUS, s)
            a, c = acc_and_ce(rec, *evk)
            ns.append(nucleus_mass(rec, evk[0], evk[1])); Ls.append(float(c))
        rows.append(dict(s=s, L_mean=float(np.mean(Ls)), L_std=float(np.std(Ls)),
                         n_mean=float(np.mean(ns)), L_plateau=L_plateau))
        json.dump(dict(L_plateau=L_plateau, rows=rows), open(os.path.join(RESULTS, args.out), "w"), indent=2)
        print(f"  s={s:.2f}: L={np.mean(Ls):.3f}  nucleus={np.mean(ns):.3f}", flush=True)

    analyze(rows, float(L_plateau))
    print(f"\n  saved results/{args.out}")


def analyze(rows, L_plateau):
    s = np.array([r["s"] for r in rows]); L = np.array([r["L_mean"] for r in rows]); n = np.array([r["n_mean"] for r in rows])
    print("\n  ===========  SURFACE COST: loss along the nucleus-assembly path  ===========")
    print(f"  shortcut-plateau loss = {L_plateau:.3f}")
    print(f"  {'s':>5} {'L(loss)':>8} {'nucleus':>8}")
    for r in rows:
        bar = "#" * int(max(r["L_mean"] - L.min(), 0) / (np.ptp(L) + 1e-9) * 25)
        print(f"  {r['s']:>5.2f} {r['L_mean']:>8.3f} {r['n_mean']:>8.3f}  {bar}")
    ipk = int(np.argmax(L)); interior = 0 < ipk < len(L) - 1
    above = L.max() > L_plateau
    print(f"\n  L(s) interior maximum (hump): {interior} at s={s[ipk]:.2f} (L={L[ipk]:.3f})")
    print(f"  hump rises ABOVE the shortcut plateau ({L_plateau:.3f}): {above}")
    print(f"  ==> {'SURFACE COST present: partial circuit is penalized (loss humps above the shortcut) -> the functional surface tension of nucleation' if interior and above else 'no clear hump above plateau -- report honestly (partial-structure cost weak/absent)'}")


if __name__ == "__main__":
    main()
