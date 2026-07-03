"""
qk_grow_or_die — DIRECTION B: is the true reaction coordinate the QK IGNITION MARGIN (pre-softmax addressing
confidence), where grow-or-die / a critical nucleus SHOULD live, rather than q_mass (the downstream attention mass,
which we proved has monotone drift = irreversible, no dissolution)?

First-principles reason QK can be bistable where q_mass is not: q_mass ~ QK_align x OV_copy. The OV substrate is built
slowly during incubation and is STABLE (does not dissolve), so q_mass inherits its monotonicity. But the QK match is a
SOFTMAX winner-take-all: once the logit MARGIN at the correct key exceeds the distractors enough, attending there grows
the gradient that widens the margin further (positive feedback -> grow); too small a margin -> noise wins -> it relaxes
(die). That unstable threshold is a genuine critical point -- but it is MASKED in q_mass by the saturating softmax.

Coordinate  phi_QK = pre-softmax attention-logit MARGIN at the correct key position (best head over both layers):
    margin = (logit at the correct value position) - (max logit over the other causal positions),  averaged over queries.
We do NOT modify the shared model: a forward hook captures each block's ln1 output, and we recompute q,k,scores from the
qkv weights. Unbounded (can be <0 before lock, >0 after) -- the right scale to see bistability.

The clean test needs NO off-manifold preparation: IF QK makes failed sub-critical attempts during the long incubation
(margin rises a bit, falls back), then natural trajectories DENSELY sample the sub-critical region, and we can read the
Langevin structure directly (Kramers-Moyal). For each seed we track phi_QK (and q_mass for contrast) densely through
incubation + ignition, then offline:
  (1) EXCURSIONS: does margin make failed up-then-back attempts before ignition?  (q_mass did NOT -- subcrit)
  (2) DRIFT(phi):  binned <d phi>/dt -- a ZERO-CROSSING with POSITIVE slope = an unstable fixed point = critical nucleus.
  (3) D(phi):      binned var(d phi)/2dt -- the noise.
  (4) OCCUPANCY:   -log P(phi) -- a BARRIER (max) should sit at the drift zero-crossing.
  (5) COMMITTOR:   from natural visits to bin b pre-ignition, P(ignite within H before returning below b) -- crosses 0.5
                   at the same phi*. If (2),(4),(5) agree -> reversible grow-or-die DOES live in QK (q_mass was wrong
                   coordinate). If margin is ALSO monotone -> B fails too (go to C).

  python qk_grow_or_die.py --smoke
  python qk_grow_or_die.py --seeds 24 --eval_every 2
"""
from __future__ import annotations
import argparse, json, math, os
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


@torch.no_grad()
def qk_margin(model, ev_tok, ev_tgt):
    """pre-softmax logit margin at the correct key position, best head over both layers. Hook captures ln1 outputs;
       q,k,scores are recomputed from the qkv weights (no model modification)."""
    caps = {}
    handles = []
    for li, blk in enumerate(model.blocks):
        handles.append(blk.ln1.register_forward_hook(lambda m, inp, out, li=li: caps.__setitem__(li, out.detach())))
    model(ev_tok)
    for h in handles:
        h.remove()
    B, T = ev_tok.shape
    qmask = (ev_tgt != IGNORE)                                  # [B,T] queries that count
    match = (ev_tok[:, None, :] == ev_tgt[:, :, None])          # [B,Tq,Tk] correct key positions (value == target)
    causal = torch.tril(torch.ones(T, T, device=ev_tok.device, dtype=torch.bool))
    correct = match & causal[None]                              # [B,Tq,Tk]
    has_corr = correct.any(-1)                                  # [B,Tq] query has a valid correct key
    valid = (qmask & has_corr)                                  # [B,Tq]
    best = -1e9
    for li, blk in enumerate(model.blocks):
        x = caps[li]; D = x.shape[-1]; h = blk.attn.h; dh = blk.attn.dh
        qkv = x @ blk.attn.qkv.weight.t()
        q, k, _ = qkv.split(D, dim=-1)
        q = q.view(B, T, h, dh).transpose(1, 2); k = k.view(B, T, h, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)     # [B,h,Tq,Tk]
        cor = correct[:, None]; cau = causal[None, None]
        sc_correct = scores.masked_fill(~cor, -1e9).amax(-1)               # [B,h,Tq] best correct-key logit
        sc_other = scores.masked_fill(cor | ~cau, -1e9).amax(-1)           # [B,h,Tq] best distractor logit
        margin = sc_correct - sc_other                                     # [B,h,Tq]
        vv = valid[:, None].expand(B, h, T).float()
        m = (margin * vv).sum(-1) / vv.sum(-1).clamp(min=1)                # [B,h]
        best = max(best, float(m.max()))
    return best


def one_seed(seed, device, max_steps, eval_every, post):
    pool, ev, vocab, rng = _data(seed, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps, marg, qm, accs = [], [], [], []; formed = None; stop = max_steps
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            a, _ = acc_and_ce(model, *ev)
            steps.append(st); marg.append(qk_margin(model, ev_tok, ev_tgt))
            qm.append(q_mass(model, ev_tok, ev_tgt)); accs.append(float(a))
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; stop = min(max_steps, st + post)
        if st >= stop:
            break
        model.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=64)).to(device); tok = pool[idx]
        loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return dict(seed=seed, tstar=formed, steps=steps, margin=marg, q_mass=qm, acc=accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--post", type=int, default=600)
    ap.add_argument("--out", default="qk_grow.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 4 if args.smoke else args.seeds
    if args.smoke:
        args.max_steps, args.post, args.eval_every = 2500, 300, 2
    print(f"device={device} seeds={n} eval_every={args.eval_every} (config kl{KEY_LEN}/kp{KEY_POOL}/vp{VAL_POOL})", flush=True)
    rows = []
    for s in range(n):
        r = one_seed(s, device, args.max_steps, args.eval_every, args.post)
        rows.append(r); json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), default=float)
        mm = np.array(r["margin"], float)
        print(f"  seed {s}: t*={r['tstar']} margin range [{mm.min():+.2f},{mm.max():+.2f}]", flush=True)
    analyze(rows, args.eval_every)
    print(f"\n  saved results/{args.out}")


def analyze(rows, dt):
    formed = [r for r in rows if r["tstar"] is not None]
    print(f"\n  ===========  QK-margin as reaction coordinate ({len(formed)}/{len(rows)} formed)  ===========")
    if not formed:
        print("  none formed"); return

    # (1) EXCURSIONS: pre-ignition failed attempts (margin rises >=delta then falls back >=delta before igniting)
    delta = 0.5; nfail = []
    for r in formed:
        st = np.array(r["steps"]); mg = np.array(r["margin"], float); pre = st < r["tstar"]
        m = mg[pre]
        if len(m) < 5:
            nfail.append(0); continue
        cnt = 0; lo = m[0]; rising = False
        for v in m[1:]:
            if v - lo >= delta:
                rising = True
            if rising and v <= (lo + delta) - delta and (max(m[:1]) or True):
                pass
            lo = min(lo, v)
        # simpler peak-counting: count local maxima that drop back by >=delta before the final ignition ramp
        peaks = 0
        for i in range(1, len(m) - 1):
            if m[i] - m[max(0, i - 3)] >= delta and m[i] - m[min(len(m) - 1, i + 3)] >= delta:
                peaks += 1
        nfail.append(peaks)
    print(f"  (1) EXCURSIONS pre-ignition (failed up-then-back attempts, delta={delta}): "
          f"median {np.median(nfail):.0f}, max {max(nfail)}  -> {'QK ATTEMPTS & FALLS BACK (bistable!)' if np.median(nfail)>=1 else 'NO failed attempts (monotone, like q_mass)'}")

    # pooled increments for Kramers-Moyal
    mvals, dphis = [], []
    for r in formed:
        mg = np.array(r["margin"], float)
        mvals += mg[:-1].tolist(); dphis += np.diff(mg).tolist()
    mvals = np.array(mvals); dphis = np.array(dphis)
    lo, hi = np.percentile(mvals, 2), np.percentile(mvals, 98)
    edges = np.linspace(lo, hi, 13); idx = np.digitize(mvals, edges)
    print(f"\n  (2,3) Kramers-Moyal: drift <dphi>/dt and noise D=var/2dt per margin bin")
    print(f"  {'margin':>8} {'n':>5} {'drift':>9} {'D':>9}")
    centers, drifts = [], []
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() < 8:
            continue
        c = float(mvals[sel].mean()); A = float(dphis[sel].mean() / dt); Dc = float(dphis[sel].var() / (2 * dt))
        centers.append(c); drifts.append(A)
        print(f"  {c:>8.2f} {int(sel.sum()):>5} {A:>+9.4f} {Dc:>9.4f}")
    centers = np.array(centers); drifts = np.array(drifts)
    # (2) unstable fixed point: drift crosses zero with POSITIVE slope
    cross = None
    for i in range(1, len(centers)):
        if drifts[i - 1] < 0 <= drifts[i]:
            cross = centers[i - 1] + (centers[i] - centers[i - 1]) * (0 - drifts[i - 1]) / (drifts[i] - drifts[i - 1] + 1e-9)
            break
    print(f"\n  (2) drift zero-crossing with +slope (unstable fixed point / critical nucleus) = "
          f"{None if cross is None else round(cross,2)}  -> {'GROW-OR-DIE in QK!' if cross is not None else 'drift monotone (no unstable point)'}")

    # (4) occupancy barrier
    allm = np.concatenate([np.array(r["margin"], float) for r in formed])
    e2 = np.linspace(np.percentile(allm, 1), np.percentile(allm, 99), 22)
    h2, _ = np.histogram(allm, bins=e2); p = h2 / h2.sum(); cc = (e2[:-1] + e2[1:]) / 2
    G = -np.log(np.where(p > 0, p, np.nan))
    fin = np.where(np.isfinite(G))[0]
    if len(fin) >= 5:
        gv = G[fin]; lo_i = fin[int(np.argmin(gv[:len(gv) // 2]))]; hi_i = fin[len(gv) // 2 + int(np.argmin(gv[len(gv) // 2:]))]
        between = [i for i in fin if lo_i < i < hi_i]
        if between:
            ipk = max(between, key=lambda i: G[i])
            print(f"  (4) occupancy -log P barrier at margin = {cc[ipk]:+.2f} (height {G[ipk]-max(G[lo_i],G[hi_i]):.2f}) "
                  f"-> compare to drift zero-crossing above (should match if reversible)")
        else:
            print(f"  (4) occupancy -log P has no interior barrier (single basin)")
    print(f"\n  VERDICT: reversible grow-or-die lives in QK IF (1) shows failed attempts AND (2) drift has a +slope")
    print(f"  zero-crossing AND (4) occupancy barrier sits at the same margin. Else B fails -> coordinate still wrong (C).")


if __name__ == "__main__":
    main()
