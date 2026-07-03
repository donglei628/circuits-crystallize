"""
bath_fdt — P1: measure the two missing Langevin quantities (drift A, diffusion D) of q_mass WITH the thermal bath, and
run the over-determination proof. With a bath (gradient build + injected weight noise T) the dynamics have detailed
balance, so -log(occupancy) is a TRUE free energy and the laws must be mutually consistent.

FIX vs the first version: drift/D are NOT measured by a cold-AdamW single step (that first step is sign-like and inflates
the drift); they are measured by KRAMERS-MOYAL on the ACTUAL bath trajectories (warm AdamW + noise, the same dynamics as
the committor): drift A(q)=<Δq>/dt, D(q)=var(Δq)/2dt, binned by q over many bath trajectories started across the size grid.

At a temperature near T_melt we collect bath trajectories from controlled nucleus sizes (plateau<->formed interp), then:
  - DRIFT A(q), NOISE D(q)      (Kramers-Moyal on the trajectories)
  - OCCUPANCY -logP(q)          (pooled q over the trajectories = the free energy, IF there is coexistence)
  - COMMITTOR C(q0)             (fraction of trajectories from size q0 that GROW vs DISSOLVE)
Over-determination: a barrier (A zero-crossing with +slope = unstable fixed point) must sit where the committor crosses
0.5 AND at the occupancy -logP maximum, and FDT A(q)=-D(q)*(-logP)'(q) must hold. One (U,D) reproducing all => ONE Langevin
system => correspondence proven WITH a bath. If instead the committor flips sharply & size-independently with NO barrier
=> SPINODAL (barrierless) rather than a critical-nucleus transition -- report honestly.

  python bath_fdt.py --smoke
  python bath_fdt.py --Ts 0.009 0.0102 0.011 --seeds 4
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
    # accept the endpoint if the run FUNCTIONALLY formed (acc>=0.80 reached) and its peak induction-mass is clearly
    # above the barrier (>=0.70). This admits soft-copy seeds (whose q_mass caps lower) and greatly raises the usable rate;
    # such seeds just cover sizes up to their own cap (the low/mid window watershed, which is what we need).
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
    """release from start_sd under (gradient build + bath dissolve); record q_mass every eval_every steps."""
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


def seed_summary(seed, Ts, percommit):
    """committor matrix (sizes x temps) + per-seed verdict: SPINODAL (size-independent flip) vs CRITICAL NUCLEUS
       (at some T, small sizes die & big grow = a size watershed)."""
    print(f"\n  ===== seed {seed} SUMMARY: committor q(grow), rows=size, cols=T =====", flush=True)
    sizes = [q for q, _ in percommit[Ts[0]]]
    print("    size\\T  " + " ".join(f"{T:>7.4f}" for T in Ts))
    for si, sz in enumerate(sizes):
        print(f"    {sz:>6.2f}  " + " ".join(f"{dict(percommit[T])[sz]:>7.2f}" for T in Ts), flush=True)
    # per-T: spread across sizes + monotone size-watershed?
    sized_T = 0
    for T in Ts:
        cs = np.array([c for _, c in percommit[T]])
        spread = cs.max() - cs.min()
        watershed = (cs[0] < 0.5 <= cs[-1]) and spread >= 0.5   # small die, big grow
        if watershed:
            sized_T += 1
    print(f"  => temps with a SIZE-dependent watershed (small die / big grow): {sized_T}/{len(Ts)}", flush=True)
    print(f"     {'CRITICAL NUCLEUS (reversible nucleation) at >=1 temp' if sized_T >= 1 else 'NO size-watershed at any temp -> SPINODAL (barrierless, size-independent flip)'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", type=float, nargs="+",
                    default=[0.0088, 0.0092, 0.0096, 0.0099, 0.0102, 0.0105, 0.0108, 0.0112])
    ap.add_argument("--qgrid", type=float, nargs="+",
                    default=[0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed0", type=int, default=0)   # seed offset → parallel GPUs run disjoint seed ranges
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ktraj", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--post", type=int, default=800)
    ap.add_argument("--out", default="bath_fdt.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.Ts = [0.0095, 0.0105]; args.qgrid = [0.45, 0.58, 0.70, 0.82, 0.90]
        args.seeds = 2; args.ktraj = 6; args.steps = 250
    print(f"device={device} Ts={args.Ts} qgrid={args.qgrid} seeds={args.seeds} ktraj={args.ktraj} "
          f"(Kramers-Moyal on bath trajectories; kl{KEY_LEN}/kp{KEY_POOL}/vp{VAL_POOL})", flush=True)

    out = []
    for seed in range(args.seed0, args.seed0 + args.seeds):
        formed, plateau, vocab = train_formed(seed, device, args.max_steps, 5, args.post)
        if formed is None or plateau is None:
            print(f"  seed {seed}: endpoints missing -- skip", flush=True); continue
        prepared = prepare(plateau, formed, seed, args.qgrid, device)
        print(f"\n##### seed {seed} #####", flush=True)
        percommit = {}
        for T in args.Ts:
            mv, dphi = [], []; commit = []
            for sd, q0 in prepared:
                grows = 0
                for k in range(args.ktraj):
                    qs = bath_traj(sd, seed, T, args.wd, args.steps, args.eval_every,
                                   seed * 333 + int(q0 * 1000) + k, device)
                    qa = np.array(qs, float)
                    mv += qa[:-1].tolist(); dphi += np.diff(qa).tolist()
                    if qa[-1] >= 0.65:
                        grows += 1
                commit.append((float(q0), grows / args.ktraj))
            percommit[T] = commit
            analyze(seed, T, np.array(mv), np.array(dphi), commit, args.eval_every)
            out.append(dict(seed=seed, T=T, commit=commit, mv=mv[::7], dphi=dphi[::7]))
            json.dump(out, open(os.path.join(RESULTS, args.out), "w"), default=float)   # save after EVERY (seed,T)
        seed_summary(seed, args.Ts, percommit)                                          # print verdict after EACH seed
        json.dump(out, open(os.path.join(RESULTS, args.out), "w"), default=float)
    print(f"\n  saved results/{args.out}")


def analyze(seed, T, mv, dphi, commit, dt):
    print(f"\n  --- seed {seed} T={T:.4f} ---", flush=True)
    # committor vs size
    cq = np.array([c[0] for c in commit]); cc = np.array([c[1] for c in commit])
    wat = None
    for i in range(1, len(cq)):
        if cc[i - 1] < 0.5 <= cc[i]:
            wat = cq[i - 1] + (cq[i] - cq[i - 1]) * (0.5 - cc[i - 1]) / (cc[i] - cc[i - 1] + 1e-9); break
    print("  committor C(size): " + " ".join(f"{q:.2f}:{c:.2f}" for q, c in commit))
    print(f"  committor watershed (size where C=0.5) = {None if wat is None else round(wat,3)}")
    # Kramers-Moyal drift + D + occupancy from trajectories
    if len(mv) < 40:
        print("  too few increments for KM"); return
    edges = np.linspace(np.percentile(mv, 1), np.percentile(mv, 99), 13)
    centers = (edges[:-1] + edges[1:]) / 2; idx = np.digitize(mv, edges)
    h, _ = np.histogram(mv, bins=edges); p = h / h.sum()
    G = -np.log(np.where(p > 0, p, np.nan)); G = G - np.nanmin(G[np.isfinite(G)]) if np.isfinite(G).any() else G
    print(f"  {'q':>6} {'occ':>7} {'-logP':>7} {'driftA':>9} {'D':>9} {'-D*Uslope':>10}")
    A = np.full(len(centers), np.nan); D = np.full(len(centers), np.nan)
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() >= 8:
            A[b - 1] = dphi[sel].mean() / dt; D[b - 1] = dphi[sel].var() / (2 * dt)
    slope = np.gradient(np.nan_to_num(G, nan=np.nanmax(G[np.isfinite(G)]) if np.isfinite(G).any() else 0.0),
                        centers[1] - centers[0])
    A_l, fdt_l = [], []
    for i in range(len(centers)):
        if np.isfinite(A[i]) and np.isfinite(G[i]):
            pred = -D[i] * slope[i]; A_l.append(A[i]); fdt_l.append(pred)
            print(f"  {centers[i]:>6.3f} {p[i]:>7.4f} {G[i]:>7.2f} {A[i]:>+9.4f} {D[i]:>9.5f} {pred:>+10.4f}")
    # barrier top of occupancy + drift zero-crossing
    fin = np.where(np.isfinite(G))[0]
    btop = centers[fin[int(np.argmax(G[fin]))]] if len(fin) else None
    cross = None
    Af = [(centers[i], A[i]) for i in range(len(centers)) if np.isfinite(A[i])]
    for i in range(1, len(Af)):
        if Af[i - 1][1] < 0 <= Af[i][1]:
            cross = Af[i - 1][0] + (Af[i][0] - Af[i - 1][0]) * (0 - Af[i - 1][1]) / (Af[i][1] - Af[i - 1][1] + 1e-9); break
    print(f"  occupancy barrier top = {None if btop is None else round(btop,3)}; "
          f"drift A zero-crossing(+slope) = {None if cross is None else round(cross,3)}")
    if len(A_l) >= 3 and np.std(A_l) > 1e-9 and np.std(fdt_l) > 1e-9:
        r = float(np.corrcoef(A_l, fdt_l)[0, 1])
        print(f"  FDT corr(A, -D*(-logP)') = {r:+.2f}  (~+1 => drift==free-energy gradient => ONE Langevin system)")
    print("  => REVERSIBLE critical nucleus if {committor watershed, A zero-crossing, occupancy barrier top} coincide;")
    print("     if committor flips size-INDEPENDENTLY with no barrier => SPINODAL (barrierless).")


if __name__ == "__main__":
    main()
