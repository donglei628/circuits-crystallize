"""
CSD — CRITICAL SLOWING DOWN near the critical seed s* (the dynamical signature separating nucleation from a functional
threshold; reviewer r7). v2: HARD-CONFIG version.

v1 (easy config kl1/pool16/vp24) gave a NEGATIVE: de-novo ALWAYS forms (P(form)~1), so there is no persistent
metastable phase and no grow-or-die bifurcation -- the seed is a smooth accelerator, t* variance just declines with s,
no critical-slowing-down peak. Lesson (and it is the physics): critical slowing down only appears when the metastable
phase PERSISTS, i.e. de-novo FAILS, so the seed is the only route to formation -- then subcritical seeds DIE (stay
metastable) and supercritical seeds GROW, with maximal dwell/variance at s*.

v2: run on a HARD config where de-novo P(form) is low (~0.2-0.4). The donor (the seed crystal) is trained at an EASY
config and transplanted -- like a seed crystal made elsewhere dropped into a supercooled (metastable) liquid. We
transplant only the config-invariant ATTENTION structure (QK+OV) so the donor's circuit applies across pool sizes,
leaving the recipient's own (correct-vocab) embeddings.

  # find a hard config (de-novo P(form) low) on 184:
  python csd_critical_slowing.py --find_config --kps 16 32 48 64 --seeds 8
  # run CSD on the chosen hard config:
  python csd_critical_slowing.py --hard_kp 48 --seeds 16
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


def data_cfg(seed, key_pool, val_pool, device):
    rng = np.random.default_rng(seed)
    regs = [Regularity("k", key_len=KEY_LEN, freq=FREQ, noise=NOISE, max_keys=MAX_KEYS)]
    vocab = build_vocab(regs, key_pool_size=key_pool, val_pool_size=val_pool)
    pool, _, _, _ = generate_batch(4096, T, regs, vocab, rng, device=device)
    ev4 = generate_batch(256, T, regs, vocab, np.random.default_rng(seed + 9999), device=device)
    return pool, ev4[:3], vocab, rng


def transplant_attn(recipient, donor_sd, s):
    """blend ONLY the config-invariant attention weights (qkv + out) at strength s; leave embeddings (vocab-specific)
       at the recipient's random init. theta_attn = (1-s)*rand + s*donor."""
    with torch.no_grad():
        for name, p in recipient.named_parameters():
            if name.endswith("attn.qkv.weight") or name.endswith("attn.out.weight"):
                p.data.mul_(1.0 - s).add_(donor_sd[name].to(p.device), alpha=s)


@torch.no_grad()
def nucleus_size(model, ev_tok, ev_tgt):
    _, attns = model(ev_tok, want_attn=True)
    qmask = (ev_tgt != IGNORE); match = (ev_tok[:, None, :] == ev_tgt[:, :, None]).float(); qm = qmask[:, None].float()
    best = 0.0
    for att in attns:
        mass = (att * match[:, None]).sum(-1)
        per_head = (mass * qm).sum(dim=(0, 2)) / qm.sum().clamp(min=1)
        best = max(best, float(per_head.max()))
    return best


def one_run(donor_sd, s, seed, kp, vp, device, max_steps, eval_every):
    pool, ev, vocab, rng = data_cfg(seed, kp, vp, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    rec = _new_model(vocab, seed, device)
    if s > 0 and donor_sd is not None:
        transplant_attn(rec, donor_sd, s)
    opt = torch.optim.AdamW(rec.parameters(), lr=1e-3)
    steps, accs, nuc = [], [], []; formed = None; stop = max_steps; buffer = 500
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            a, _ = acc_and_ce(rec, *ev)
            steps.append(st); accs.append(float(a)); nuc.append(nucleus_size(rec, ev_tok, ev_tgt))
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; stop = min(max_steps, st + buffer)
        if st >= stop:
            break
        rec.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(rec(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(s=s, seed=seed, tstar=formed, steps=steps, acc=accs, nuc=nuc)


def find_config(kps, vp, seeds, device, max_steps):
    """measure de-novo P(form) at several key-pools to locate a HARD config (P(form) ~ 0.2-0.4)."""
    print(f"  finding hard config: de-novo P(form) at kps={kps} vp={vp} ({seeds} seeds, max_steps={max_steps})")
    for kp in kps:
        nform = 0; ts = []
        for seed in range(seeds):
            r = one_run(None, 0.0, seed, kp, vp, device, max_steps, 25)
            if r["tstar"] is not None:
                nform += 1; ts.append(r["tstar"])
        print(f"    kp{kp:>3}: de-novo P(form)={nform/seeds:.2f}  t*med={np.median(ts) if ts else float('nan'):.0f}", flush=True)


def train_easy_donor(device, n_donor):
    """donors trained at the EASY config (pool16/vp24) -- the seed crystal made elsewhere."""
    donors = []; cand = 0
    while len(donors) < n_donor and cand < n_donor + 6:
        pool, ev, vocab, rng = data_cfg(1000 + cand, KEY_POOL, VAL_POOL, device)
        m = _new_model(vocab, 1000 + cand, device)
        r = _train(m, pool, ev, device, rng)
        cand += 1
        if not r["censored"] and np.isfinite(r["tstar"]):
            donors.append({k: v.detach().clone() for k, v in m.state_dict().items()})
            print(f"  easy donor {len(donors)}: formed@{r['tstar']:.0f}", flush=True)
    return donors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find_config", action="store_true")
    ap.add_argument("--kps", type=int, nargs="+", default=[16, 32, 48, 64])
    ap.add_argument("--hard_kp", type=int, default=48)
    ap.add_argument("--vp", type=int, default=VAL_POOL)
    ap.add_argument("--svals", type=float, nargs="+",
                    default=[0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--n_donor", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--out", default="csd_hard.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.find_config:
        if args.smoke:
            args.kps, args.seeds, args.max_steps = [16, 32, 48], 6, 4000
        find_config(args.kps, args.vp, args.seeds, device, args.max_steps)
        return

    if args.smoke:
        args.svals, args.seeds, args.n_donor, args.max_steps = [0.0, 0.2, 0.4, 0.6], 4, 1, 4000
    print(f"device={device} HARD kp={args.hard_kp} vp={args.vp} svals={args.svals} seeds={args.seeds} "
          f"(donor from easy pool{KEY_POOL}/vp{VAL_POOL})", flush=True)

    donors = train_easy_donor(device, args.n_donor)
    if not donors:
        print("  NO easy donor formed; abort"); return

    rows = []
    for s in args.svals:
        for seed in range(args.seeds):
            r = one_run(donors[seed % len(donors)], s, seed, args.hard_kp, args.vp, device, args.max_steps, args.eval_every)
            rows.append(r)
            json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        sub = [x["tstar"] for x in rows if x["s"] == s and x["tstar"] is not None]
        nf = len(sub)
        print(f"  s={s:.2f}: formed {nf}/{args.seeds}  t*med={np.median(sub) if sub else float('nan'):.0f}  "
              f"std={np.std(sub) if len(sub)>=2 else float('nan'):.0f}", flush=True)

    analyze(rows, args.svals)
    print(f"\n  saved results/{args.out}")


def analyze(rows, svals):
    print("\n  ===========  CSD (hard config): grow-or-die + critical slowing down  ===========")
    print(f"  {'s':>5} {'P(form)':>8} {'t*med':>7} {'std(t*)':>8} {'CV':>6}")
    stats = []
    for s in svals:
        sub = [x for x in rows if x["s"] == s]
        formed = [x["tstar"] for x in sub if x["tstar"] is not None]
        n = len(sub); pf = len(formed) / n if n else 0
        med = float(np.median(formed)) if formed else float("nan")
        std = float(np.std(formed)) if len(formed) >= 2 else float("nan")
        cv = std / med if (formed and med > 0) else float("nan")
        stats.append((s, pf, med, std, cv))
        print(f"  {s:>5.2f} {pf:>8.2f} {med:>7.0f} {std:>8.0f} {cv:>6.2f}")
    # s* = P(form) crosses 0.5 (the grow-or-die bifurcation midpoint)
    sstar = None
    for i in range(1, len(stats)):
        if stats[i - 1][1] < 0.5 <= stats[i][1]:
            a, b = stats[i - 1], stats[i]
            sstar = a[0] + (b[0] - a[0]) * (0.5 - a[1]) / (b[1] - a[1] + 1e-9); break
    valid = [(s, std) for s, pf, med, std, cv in stats if np.isfinite(std)]
    if sstar is not None:
        print(f"\n  GROW-OR-DIE bifurcation: P(form) crosses 0.5 at s* ~= {sstar:.3f} (subcritical DIE, supercritical GROW)")
        if valid:
            speak = max(valid, key=lambda x: x[1])[0]
            print(f"  std(t*) peaks at s={speak:.3f}  ({'AT s* -> CRITICAL SLOWING DOWN (nucleation, threshold cannot fake)' if abs(speak-sstar)<=0.06 else 'near/away from s* -- inspect'})")
    else:
        print("\n  P(form) never crosses 0.5 -> config not hard enough (de-novo still forms) OR seed too weak; retune.")


if __name__ == "__main__":
    main()
