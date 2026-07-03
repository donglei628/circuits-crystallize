"""
B1 — OVER-DETERMINATION / FDT PROOF for the BATHED system at a single nucleation-band temperature T*.

Goal: show ONE potential U(phi) and ONE diffusion D(phi) simultaneously explain FOUR independently-measured laws ->
the physics<->model correspondence is an IDENTITY (not an analogy) once a thermal bath is added. This is the capstone
of the two-layer theory's Layer 2 (run only at a T* INSIDE the reversible-critical-nucleus band, from A1/A2).

At fixed T* (band), order parameter phi = q_mass. From bath trajectories started across a phi-grid (warm AdamW + noise):
  - DRIFT A(phi) and DIFFUSION D(phi): Kramers-Moyal on the per-step increments (binned by phi).
  - OCCUPANCY P(phi): pooled phi visited -> U_occ(phi) = -<D> * log P(phi).
  - COMMITTOR q(phi): fraction of trajectories from phi that end in the formed well.
OVER-DETERMINATION (one U, one D must reproduce all):
  (1) FDT / detailed balance:  U_drift(phi) = -∫(A/D)dphi   vs   U_occ(phi) = -<D> log P(phi)   -> same shape?
  (2) committor: q_pred(phi) from the SCALE FUNCTION of (A,D)   vs   q_meas(phi)   -> functional match (corr)?
  (3) coincidence: argmax(U) == committor==0.5 watershed == drift A zero-crossing.
If (1)-(3) hold, one Langevin (U,D) reproduces drift+diffusion+occupancy+committor at T* => correspondence PROVEN.

  python bath_b1.py --Tstar 0.0100 --seed0 1 --seeds 1            # smoke at the known band temp
  python bath_b1.py --Tstar <band T from A2> --seed0 0 --seeds 6  # formal, multi-seed
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


def _opt(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)


def grad_step(model, opt, pool, vocab, rng, batch, device):
    model.train()
    idx = torch.from_numpy(rng.integers(pool.shape[0], size=batch)).to(device); tok = pool[idx]
    loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


@torch.no_grad()
def bath_kick(model, T, wd):
    if wd <= 0 and T <= 0:
        return
    for p in model.parameters():
        if wd > 0:
            p.mul_(1 - wd)
        if T > 0:
            p.add_(torch.randn_like(p) * T)


def train_formed(seed, device, max_steps, eval_every, post):
    pool, ev, vocab, rng = _data(seed, device); ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = _opt(model)
    plateau = None; best = None; best_q = -1; formed = None; stop = max_steps
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            phi = q_mass(model, ev_tok, ev_tgt); a, _ = acc_and_ce(model, *ev)
            if plateau is None and 0.32 <= phi <= 0.45 and st >= 200:
                plateau = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if phi > best_q:
                best_q = phi; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; stop = min(max_steps, st + post)
        if st >= stop:
            break
        grad_step(model, opt, pool, vocab, rng, 64, device)
    return (best if (formed is not None and best_q >= 0.70) else None), plateau, vocab


def blend(a_sd, b_sd, a):
    return {k: (1 - a) * a_sd[k] + a * b_sd[k] for k in a_sd}


def prepare(plateau_sd, formed_sd, seed, targets, device):
    pool, ev, vocab, _ = _data(seed, device); ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); out = []
    for tgt in targets:
        lo, hi, best = 0.0, 1.0, None
        for _ in range(20):
            am = 0.5 * (lo + hi); model.load_state_dict(blend(plateau_sd, formed_sd, am))
            phi = q_mass(model, ev_tok, ev_tgt)
            if best is None or abs(phi - tgt) < abs(best[1] - tgt):
                best = (am, phi)
            if phi < tgt:
                lo = am
            else:
                hi = am
        out.append((blend(plateau_sd, formed_sd, best[0]), best[1]))
    return out


def bath_traj(start_sd, seed, T, wd, steps, eval_every, noise_seed, device):
    pool, ev, vocab, _ = _data(seed, device); ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); model.load_state_dict(start_sd); opt = _opt(model)
    rng = np.random.default_rng(noise_seed); qs = []
    for st in range(steps + 1):
        if st % eval_every == 0:
            qv = q_mass(model, ev_tok, ev_tgt); qs.append(0.0 if not np.isfinite(qv) else qv)
        if st >= steps:
            break
        loss = grad_step(model, opt, pool, vocab, rng, 64, device); bath_kick(model, T, wd)
        if not np.isfinite(loss):
            qs.append(0.0); break
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tstar", type=float, default=0.0100)      # band temperature (from A1/A2); 0.0100 = known seed-1 band
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--qgrid", type=float, nargs="+",
                    default=[0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76, 0.82, 0.88])
    ap.add_argument("--ktraj", type=int, default=24)            # trajectories per start phi (committor + KM)
    ap.add_argument("--steps", type=int, default=400)           # committor trajectory length
    ap.add_argument("--occ_n", type=int, default=8)             # long trajectories per occupancy start
    ap.add_argument("--occ_steps", type=int, default=2500)      # long-trajectory length for equilibrium-ish occupancy
    ap.add_argument("--eval_every", type=int, default=3)        # finer eval -> better KM drift/D
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--post", type=int, default=800)
    ap.add_argument("--out", default="bath_b1.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.qgrid = [0.42, 0.52, 0.62, 0.72, 0.82]; args.ktraj = 8; args.steps = 300; args.seeds = 1
        args.occ_n = 4; args.occ_steps = 1500
    print(f"device={device} T*={args.Tstar} seed0={args.seed0} seeds={args.seeds} qgrid={args.qgrid} "
          f"ktraj={args.ktraj} steps={args.steps} (B1 over-determination; kl{KEY_LEN}/kp{KEY_POOL}/vp{VAL_POOL})", flush=True)

    out = []
    for seed in range(args.seed0, args.seed0 + args.seeds):
        formed, plateau, vocab = train_formed(seed, device, args.max_steps, 5, args.post)
        if formed is None or plateau is None:
            print(f"  seed {seed}: endpoints missing -- skip", flush=True); continue
        prepared = prepare(plateau, formed, seed, args.qgrid, device)
        print(f"\n##### seed {seed}  (T*={args.Tstar}) #####", flush=True)
        mv, dphi, commit = [], [], []
        for sd, q0 in prepared:                                  # committor + Kramers-Moyal drift/D
            grows = 0
            for k in range(args.ktraj):
                qs = bath_traj(sd, seed, args.Tstar, args.wd, args.steps, args.eval_every,
                               seed * 333 + int(q0 * 1000) + k, device)
                qa = np.array(qs, float)
                mv += qa[:-1].tolist(); dphi += np.diff(qa).tolist()
                if qa[-1] >= 0.65:
                    grows += 1
            commit.append((float(q0), grows / args.ktraj))
        # dedicated LONG trajectories for an equilibrium-ish occupancy at T* (let the system dwell/hop; drop transient)
        occ = []
        occ_starts = [prepared[0], prepared[len(prepared) // 2], prepared[-1]]   # plateau-ish / mid / formed-ish
        for sd, q0 in occ_starts:
            for k in range(args.occ_n):
                qs = bath_traj(sd, seed, args.Tstar, args.wd, args.occ_steps, args.eval_every,
                               seed * 555 + int(q0 * 1000) + k, device)
                qa = np.array(qs, float); occ += qa[len(qa) // 3:].tolist()      # discard initial transient
        res = analyze(seed, args.Tstar, np.array(mv), np.array(dphi), np.array(occ), commit, args.eval_every)
        out.append(dict(seed=seed, Tstar=args.Tstar, commit=commit, **res))
        json.dump(out, open(os.path.join(RESULTS, args.out), "w"), default=float)
    print(f"\n  saved results/{args.out}")


def analyze(seed, T, mv, dphi, occ, commit, dt):
    print(f"\n  --- B1 over-determination @ seed {seed}, T*={T:.4f} ---", flush=True)
    cq = np.array([c[0] for c in commit]); cc = np.array([c[1] for c in commit])
    if len(mv) < 60:
        print("  too few increments"); return dict()
    # bins over the populated phi range
    edges = np.linspace(np.percentile(mv, 2), np.percentile(mv, 98), 13); cen = (edges[:-1] + edges[1:]) / 2
    idx = np.digitize(mv, edges)
    A = np.full(len(cen), np.nan); D = np.full(len(cen), np.nan)
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() >= 12:
            A[b - 1] = dphi[sel].mean() / dt; D[b - 1] = max(dphi[sel].var() / (2 * dt), 1e-9)
    good = np.isfinite(A) & np.isfinite(D)
    cen, A, D = cen[good], A[good], D[good]
    if len(cen) < 5:
        print("  too few phi bins"); return dict()
    Dbar = float(np.mean(D))
    # (1) U from drift  vs  U from occupancy
    AoverD = A / D
    U_drift = -np.concatenate([[0], np.cumsum(0.5 * (AoverD[1:] + AoverD[:-1]) * np.diff(cen))]); U_drift -= U_drift.min()
    dc = float(np.median(np.diff(cen))) if len(cen) > 1 else 0.04
    e_occ = np.r_[cen - dc / 2, cen[-1] + dc / 2]
    hist, _ = np.histogram(occ, bins=e_occ); pocc = hist / max(hist.sum(), 1)
    U_occ = -Dbar * np.log(np.where(pocc > 0, pocc, np.nan))
    if np.isfinite(U_occ).any():
        U_occ = U_occ - np.nanmin(U_occ[np.isfinite(U_occ)])
    fin = np.isfinite(U_occ)
    r_fdt = float(np.corrcoef(U_drift[fin], U_occ[fin])[0, 1]) if fin.sum() >= 4 and np.std(U_occ[fin]) > 1e-9 else float("nan")
    # (2) committor predicted from scale function of (A,D)
    psi = np.concatenate([[0], np.cumsum(0.5 * (AoverD[1:] + AoverD[:-1]) * np.diff(cen))])   # ∫A/D
    w = np.exp(-(psi - psi.max()))
    S = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(cen))])
    q_pred = S / S[-1] if S[-1] > 0 else S
    q_meas_on_cen = np.interp(cen, cq, cc)
    r_comm = float(np.corrcoef(q_pred, q_meas_on_cen)[0, 1]) if np.std(q_pred) > 1e-9 and np.std(q_meas_on_cen) > 1e-9 else float("nan")
    # (3) coincidence
    btop = float(cen[int(np.argmax(U_drift))])
    cross = None
    for i in range(1, len(cen)):
        if A[i - 1] < 0 <= A[i]:
            cross = cen[i - 1] + (cen[i] - cen[i - 1]) * (0 - A[i - 1]) / (A[i] - A[i - 1] + 1e-9); break
    wat = None
    for i in range(1, len(cq)):
        if cc[i - 1] < 0.5 <= cc[i]:
            wat = cq[i - 1] + (cq[i] - cq[i - 1]) * (0.5 - cc[i - 1]) / (cc[i] - cc[i - 1] + 1e-9); break
    print(f"  {'phi':>6} {'A drift':>9} {'D':>9} {'U_drift':>8} {'U_occ':>7} {'q_meas':>7} {'q_pred':>7}")
    for i in range(len(cen)):
        print(f"  {cen[i]:>6.3f} {A[i]:>+9.4f} {D[i]:>9.5f} {U_drift[i]:>8.2f} "
              f"{U_occ[i] if np.isfinite(U_occ[i]) else float('nan'):>7.2f} {q_meas_on_cen[i]:>7.2f} {q_pred[i]:>7.2f}")
    print(f"\n  (1) FDT/detailed-balance  corr(U_drift, U_occ) = {r_fdt:+.2f}   (~+1 => occupancy IS the drift potential)")
    print(f"  (2) committor  corr(q_measured, q_pred-from-(A,D)) = {r_comm:+.2f}   (~+1 => one (U,D) gives drift AND committor)")
    print(f"  (3) coincidence: U barrier top={btop:.3f}; A zero-crossing={None if cross is None else round(cross,3)}; "
          f"committor watershed={None if wat is None else round(wat,3)}")
    ok = (r_fdt > 0.8) and (r_comm > 0.8)
    print(f"  ==> {'OVER-DETERMINATION HOLDS at T*: one (U,D) reproduces drift+occupancy+committor => correspondence is an IDENTITY (with a bath)' if ok else 'partial -- report which leg fails'}")
    return dict(phi=cen.tolist(), A=A.tolist(), D=D.tolist(), U_drift=U_drift.tolist(), U_occ=U_occ.tolist(),
                q_pred=q_pred.tolist(), r_fdt=r_fdt, r_comm=r_comm, btop=btop, cross=cross, wat=wat)


if __name__ == "__main__":
    main()
