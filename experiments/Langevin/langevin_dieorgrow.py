"""
langevin_dieorgrow — measure the two MISSING Langevin quantities of the order parameter phi = q_mass (induction mass):
the mobility gamma and the diffusion/noise D, AND run the grow-or-die / committor test, ALL from the same family of
controlled-phi states. This is the first-principles PROOF of the physics<->model correspondence by OVER-DETERMINATION:
one potential U(phi) and one noise D(phi) must simultaneously satisfy several independently-measured laws.

The Langevin law (bare math): d phi/dt = drift(phi) + noise,  drift = -(1/gamma) U'(phi),  noise variance = 2 D.
From ONE thing -- the statistics of one optimizer step on phi at a controlled phi -- we get drift and D:
  - JUMP CLOUD: clone the frozen state, take ONE AdamW step with a fresh minibatch, record d phi. Repeat M times.
        mean(d phi)      = drift A(phi)        (per step)
        var (d phi) / 2  = D(phi)              (the effective temperature / noise)
  - COMMITTOR (grow-or-die): from the same frozen state, continue training a SHORT window T_release with K fresh
        noise seeds (growth-time << T_release << incubation, so supercritical states GROW and subcritical states stay
        i.e. DIE without time to spontaneously nucleate). q(phi) = fraction reaching the formed well; the step count to
        commit is the critical-slowing probe (slowest at the barrier top).
  - OCCUPANCY U_occ(phi) = -log P(phi) from the pooled q_mass dwell during harvest (same method as G-occupancy).

The proof = these must be ONE consistent gradient-Langevin system:
  (1) committor watershed phi* (q=0.5)        ==  occupancy barrier top phi*_occ (~0.64)
  (2) critical slowing: commit-time peaks at phi*
  (3) FLUCTUATION-DISSIPATION:  A(phi) == -D(phi) * U_occ'(phi)   <-- drift = D x occupancy-slope. THE consistency.
  (4) mobility gamma(phi) = -U_occ'(phi)/A(phi); Einstein eff-temp kT = D*gamma should be ~constant across phi.
If (1)-(4) hold with ONE U and ONE D, the model's q_mass dynamics ARE the physics law (identity, not analogy). If they
fail, the order parameter q_mass is not yet the true reaction coordinate (or AdamW preconditioning breaks plain Langevin).

States are harvested from REAL de-novo snap trajectories (same config, on-manifold), each carrying its AdamW state so
the local dynamics are faithful. Order parameter phi = q_mass measured exactly as in G-occupancy.

  python langevin_dieorgrow.py --smoke
  python langevin_dieorgrow.py --phis 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 --reps 4 --mjump 400 --kcommit 24
"""
from __future__ import annotations
import argparse, copy, json, os
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
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def one_step(model, opt, pool, vocab, rng, batch, device):
    model.train()
    idx = torch.from_numpy(rng.integers(pool.shape[0], size=batch)).to(device); tok = pool[idx]
    loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()


def snap_state(model, opt):
    m = {k: v.detach().clone() for k, v in model.state_dict().items()}
    o = copy.deepcopy(opt.state_dict())
    return m, o


# ---------------------------------------------------------------- harvest controlled-phi states from real trajectories
def harvest(seed, device, targets, tol, reps, max_steps, eval_every, post, got):
    pool, ev, vocab, rng = _data(seed, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = _opt(model)
    qm_dwell = []; formed = None; stop = max_steps
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            phi = q_mass(model, ev_tok, ev_tgt); qm_dwell.append(phi)
            a, _ = acc_and_ce(model, *ev)
            for tgt in targets:
                if len(got[tgt]) < reps and abs(phi - tgt) <= tol:
                    m, o = snap_state(model, opt)
                    got[tgt].append(dict(seed=seed, phi=phi, m=m, o=o))
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; stop = min(max_steps, st + post)
        if st >= stop:
            break
        one_step(model, opt, pool, vocab, rng, 64, device)
    return qm_dwell


# --------------- interpolation preparation: blend a plateau state and a formed state to hit ANY target phi (incl. the
# --------------- barrier, which natural trajectories almost never dwell at). theta(a) = (1-a)*plateau + a*formed.
def endpoints(seed, device, max_steps, eval_every, post):
    """train one de-novo run; capture a PLATEAU state (q_mass in [.32,.45], pre-formation) and a FORMED state
       (q_mass>=.92), plus the pooled q_mass dwell for the occupancy potential."""
    pool, ev, vocab, rng = _data(seed, device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); opt = _opt(model)
    plateau = None; best_state = None; best_phi = -1.0; dwell = []; formed_step = None; stop = max_steps
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            phi = q_mass(model, ev_tok, ev_tgt); dwell.append(phi)
            a, _ = acc_and_ce(model, *ev)
            if plateau is None and 0.32 <= phi <= 0.45 and st >= 200:
                plateau = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if phi > best_phi:                       # keep the highest-q_mass state seen = the formed endpoint
                best_phi = phi; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if formed_step is None and not np.isnan(a) and a >= 0.80:
                formed_step = st; stop = min(max_steps, st + post)
        if st >= stop:
            break
        one_step(model, opt, pool, vocab, rng, 64, device)
    formed_state = best_state if best_phi >= 0.88 else None
    return plateau, formed_state, dwell, vocab


def blend(plateau, formed, a):
    return {k: (1 - a) * plateau[k] + a * formed[k] for k in plateau}


def prep_interp(seed, device, targets, tol, max_steps, eval_every, post, got):
    """for each target phi, binary-search the blend coefficient a so the interpolated model has q_mass ~= target."""
    plateau, formed, dwell, vocab = endpoints(seed, device, max_steps, eval_every, post)
    if plateau is None or formed is None:
        print(f"    seed {seed}: endpoints missing (plateau={plateau is not None} formed={formed is not None}) -- skip", flush=True)
        return dwell
    _, ev, _, _ = _data(seed, device); ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device)
    for tgt in targets:
        if len(got[tgt]) >= 1e9:
            continue
        lo_a, hi_a = 0.0, 1.0; best = None
        for _ in range(22):
            a = 0.5 * (lo_a + hi_a)
            model.load_state_dict(blend(plateau, formed, a))
            phi = q_mass(model, ev_tok, ev_tgt)
            if best is None or abs(phi - tgt) < abs(best[1] - tgt):
                best = (a, phi)
            if phi < tgt:
                lo_a = a
            else:
                hi_a = a
        if abs(best[1] - tgt) <= max(tol, 0.04):
            got[tgt].append(dict(seed=seed, phi=best[1], m=blend(plateau, formed, best[0]), o=None))
    return dwell


# ----------------------------------------------------------------------------------- jump cloud: drift A(phi), D(phi)
def jump_cloud(snap, device, M, batch):
    pool, ev, vocab, _ = _data(snap["seed"], device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, snap["seed"], device)
    dphis = []
    for m in range(M):
        model.load_state_dict(snap["m"]); opt = _opt(model)
        if snap["o"] is not None:
            opt.load_state_dict(snap["o"])
        phi_b = q_mass(model, ev_tok, ev_tgt)
        rng = np.random.default_rng(snap["seed"] * 100003 + m * 7 + 1)
        one_step(model, opt, pool, vocab, rng, batch, device)
        phi_a = q_mass(model, ev_tok, ev_tgt)
        dphis.append(phi_a - phi_b)
    dphis = np.array(dphis, float)
    return float(dphis.mean()), float(dphis.var() / 2.0)   # drift A, diffusion D (per step)


# ----------------------------------------------------------------------------- committor q(phi) + critical slowing
def committor(snap, device, K, T_release, eval_step, hi, lo, batch):
    pool, ev, vocab, _ = _data(snap["seed"], device)
    ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, snap["seed"], device)
    grows = 0; times = []
    for k in range(K):
        model.load_state_dict(snap["m"]); opt = _opt(model)
        if snap["o"] is not None:
            opt.load_state_dict(snap["o"])
        rng = np.random.default_rng(snap["seed"] * 911 + k * 13 + 5)
        outcome = None
        for t in range(1, T_release + 1):
            one_step(model, opt, pool, vocab, rng, batch, device)
            if t % eval_step == 0:
                phi = q_mass(model, ev_tok, ev_tgt)
                if phi >= hi:
                    outcome = ("grow", t); break
                if phi <= lo:
                    outcome = ("die", t); break
        if outcome is None:
            phi = q_mass(model, ev_tok, ev_tgt)
            outcome = ("grow" if phi >= 0.5 * (hi + lo) else "die", T_release)
        if outcome[0] == "grow":
            grows += 1
        times.append(outcome[1])
    return grows / K, float(np.median(times))


def occupancy_potential(dwell_all, phigrid, bins=40):
    """U_occ(phi) = -log P(phi) from pooled q_mass dwell; return U on phigrid + its slope (finite diff)."""
    vals = np.array(dwell_all, float)
    hi = float(np.percentile(vals, 99.5)); edges = np.linspace(0.0, hi, bins + 1)
    hist, _ = np.histogram(vals, bins=edges); p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    G = -np.log(np.where(p > 0, p, np.nan))
    fin = np.isfinite(G)
    Ug = np.interp(phigrid, centers[fin], G[fin] - np.nanmin(G[fin]))
    dphi = (phigrid[1] - phigrid[0]) if len(phigrid) > 1 else 1.0
    slope = np.gradient(Ug, dphi)
    return Ug.tolist(), slope.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phis", type=float, nargs="+",
                    default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max_seeds", type=int, default=24)
    ap.add_argument("--mjump", type=int, default=400)
    ap.add_argument("--kcommit", type=int, default=24)
    ap.add_argument("--t_release", type=int, default=300)
    ap.add_argument("--eval_step", type=int, default=10)
    ap.add_argument("--hi", type=float, default=0.80)
    ap.add_argument("--lo", type=float, default=0.45)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--post", type=int, default=1200)
    ap.add_argument("--prep", choices=["interp", "harvest"], default="interp")
    ap.add_argument("--out", default="langevin.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.phis = [0.50, 0.65, 0.80]; args.reps = 2; args.max_seeds = 4
        args.mjump = 48; args.kcommit = 6; args.t_release = 200
        args.max_steps = 3500; args.post = 1000
    print(f"device={device} prep={args.prep} phis={args.phis} reps={args.reps} M={args.mjump} K={args.kcommit} "
          f"T_release={args.t_release} (config kl{KEY_LEN}/kp{KEY_POOL}/vp{VAL_POOL})", flush=True)

    # ---- prepare `reps` controlled-phi snapshots per target (interp covers the barrier; harvest is on-manifold) ----
    got = {t: [] for t in args.phis}; dwell_all = []
    for seed in range(args.max_seeds):
        if args.prep == "interp":
            dwell = prep_interp(seed, device, args.phis, args.tol, args.max_steps, args.eval_every, args.post, got)
        else:
            dwell = harvest(seed, device, args.phis, args.tol, args.reps,
                            args.max_steps, args.eval_every, args.post, got)
        dwell_all += dwell
        have = {t: len(got[t]) for t in args.phis}
        print(f"  prep seed {seed}: " + " ".join(f"{t:.2f}:{have[t]}" for t in args.phis), flush=True)
        if all(len(got[t]) >= args.reps for t in args.phis):
            break

    # ---- per-phi: jump cloud (drift, D) + committor (grow-or-die, slowing) ----
    rows = []
    for t in args.phis:
        snaps = got[t]
        if not snaps:
            print(f"  phi~{t:.2f}: NO snapshot harvested (config never dwelt here) -- skip", flush=True)
            continue
        drifts, Ds, qs, ctimes, phis_meas = [], [], [], [], []
        for sn in snaps:
            A, D = jump_cloud(sn, device, args.mjump, 64)
            q, ct = committor(sn, device, args.kcommit, args.t_release, args.eval_step, args.hi, args.lo, 64)
            drifts.append(A); Ds.append(D); qs.append(q); ctimes.append(ct); phis_meas.append(sn["phi"])
        row = dict(phi_target=t, phi_meas=float(np.mean(phis_meas)), n=len(snaps),
                   drift=float(np.mean(drifts)), drift_sd=float(np.std(drifts)),
                   D=float(np.mean(Ds)), D_sd=float(np.std(Ds)),
                   committor=float(np.mean(qs)), commit_time=float(np.mean(ctimes)))
        rows.append(row)
        json.dump(dict(rows=rows, dwell=dwell_all), open(os.path.join(RESULTS, args.out), "w"), default=float)
        print(f"  phi~{t:.2f} (meas {row['phi_meas']:.2f}, n={row['n']}): drift={row['drift']:+.4f} "
              f"D={row['D']:.5f} q(grow)={row['committor']:.2f} commit_t={row['commit_time']:.0f}", flush=True)

    analyze(rows, dwell_all, np.array([r["phi_target"] for r in rows], float))
    json.dump(dict(rows=rows, dwell=dwell_all), open(os.path.join(RESULTS, args.out), "w"), default=float)
    print(f"\n  saved results/{args.out}")


def analyze(rows, dwell_all, phigrid):
    if len(rows) < 2:
        print("\n  too few phi levels to analyze"); return
    phim = np.array([r["phi_meas"] for r in rows]); A = np.array([r["drift"] for r in rows])
    D = np.array([r["D"] for r in rows]); q = np.array([r["committor"] for r in rows])
    ct = np.array([r["commit_time"] for r in rows])
    Ug, slope = occupancy_potential(dwell_all, phigrid)
    Ug = np.array(Ug); slope = np.array(slope)
    print("\n  ===========  LANGEVIN over-determination check (one U, one D)  ===========")
    hdr = "-D*Uslope"
    print(f"  {'phi':>5} {'drift A':>9} {'D':>8} {'q(grow)':>8} {'commit_t':>9} {'U_occ':>7} {hdr:>10}")
    for i in range(len(rows)):
        fdt = -D[i] * slope[i]
        print(f"  {phim[i]:>5.2f} {A[i]:>+9.4f} {D[i]:>8.5f} {q[i]:>8.2f} {ct[i]:>9.0f} {Ug[i]:>7.2f} {fdt:>+10.4f}")

    # (1) committor watershed phi* (q crosses 0.5)
    phistar_c = None
    for i in range(1, len(rows)):
        if q[i - 1] < 0.5 <= q[i]:
            phistar_c = phim[i - 1] + (phim[i] - phim[i - 1]) * (0.5 - q[i - 1]) / (q[i] - q[i - 1] + 1e-9); break
    # occupancy barrier top (max of U between the two ends)
    phistar_occ = float(phim[int(np.argmax(Ug))])
    # (2) critical slowing: phi of max commit_time
    phi_slow = float(phim[int(np.argmax(ct))])
    print(f"\n  (1) committor watershed phi* (q=0.5)      = {None if phistar_c is None else round(phistar_c,3)}")
    print(f"      occupancy barrier top phi*_occ        = {phistar_occ:.3f}")
    print(f"  (2) critical slowing: commit_time peaks @ = {phi_slow:.3f}")
    # (3) FDT: corr / ratio between measured drift A and -D*U_occ'
    fdt_pred = -D * slope
    if np.std(A) > 1e-9 and np.std(fdt_pred) > 1e-9:
        r = float(np.corrcoef(A, fdt_pred)[0, 1])
        print(f"  (3) FLUCT-DISSIPATION  A vs -D*U_occ':  corr = {r:+.2f}  (=> ONE Langevin system if ~+1)")
    # (4) mobility gamma + Einstein eff-temp
    gam = np.where(np.abs(A) > 1e-6, -slope / A, np.nan)
    kT = D * gam
    print(f"  (4) mobility gamma range {np.nanmin(gam):+.2f}..{np.nanmax(gam):+.2f}; "
          f"Einstein eff-temp kT=D*gamma = {np.nanmean(kT):.4f} +- {np.nanstd(kT):.4f} (constant => consistent)")
    ok1 = (phistar_c is not None) and abs(phistar_c - phistar_occ) <= 0.08
    ok2 = abs(phi_slow - phistar_occ) <= 0.10
    print(f"\n  ==> (1) watershed==occ-top: {'YES' if ok1 else 'no'};  (2) slowing@top: {'YES' if ok2 else 'no'}")
    print(f"      If (1),(2),(3 corr~+1),(4 kT const) all hold: the physics law and the q_mass law are ONE system")
    print(f"      (correspondence PROVEN by over-determination). Any failure => q_mass not yet the true coordinate")
    print(f"      (or AdamW preconditioning breaks plain Langevin) -- report honestly which leg fails.")


if __name__ == "__main__":
    main()
