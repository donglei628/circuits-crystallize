"""
melt_dieorgrow — DIRECTION C: supply the MISSING DISSOLVING FORCE (a thermal bath) and test whether reversible
grow-or-die then appears. Physical nucleus is die-or-grow ONLY because it is bathed in liquid: two opposing forces
pull on it at once -- a BUILDING force (driving force Δμ) and a DISSOLVING force (thermal exchange / surface). Our
circuit feels only the BUILDING force (the loss gradient); minibatch noise is anti-thermal (does not dissolve). So with
one force it can only build-or-wait -- there is no dissolution, hence no die-or-grow. To test die-or-grow we must ADD
the dissolving force.

The dissolving force = a thermal bath: inject Gaussian WEIGHT NOISE of scale T each step (this IS thermal exchange --
random kicks that erode structure), on top of the gradient build. The loss landscape itself is the confining potential,
so gradient + noise is a Langevin sampler with stationary distribution ~ exp(-L_eff/T): now there is detailed balance,
dissolution is possible, and -log(occupancy) becomes a TRUE free energy (so the over-determination/FDT test, which
failed without a bath, can finally hold).

Three parts (interlocking):
  (A) MELT CURVE: from a FORMED circuit, ramp T; the order parameter q_mass drops from formed (~0.95) to disordered
      (~plateau) at a melting temperature T_melt (entropy of the disordered state wins over the loss advantage of the
      circuit). This PROVES dissolution now exists. Locate T_melt.
  (B) DIE-OR-GROW: at T near T_melt, prepare nuclei of controlled size n=q_mass (plateau<->formed weight interp), release
      under (gradient build + noise dissolve) with many bath seeds; committor q(n) = fraction that GROW vs DISSOLVE.
      A watershed n* where q crosses 0.5 = a genuine reversible critical nucleus (what was absent without the bath).
  (C) FDT: at fixed T, occupancy -log P(q) is now a free energy; check drift A(q) == -D(q)*(-logP)'(q) from the same
      bath trajectories (Kramers-Moyal). If it holds -> one U, one D -> the Langevin correspondence is proven WITH a bath.

  python melt_dieorgrow.py --smoke
  python melt_dieorgrow.py --part all --seeds 8
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
    """the dissolving force: confining decay (optional) + Gaussian thermal noise of scale T on every weight."""
    if wd <= 0 and T <= 0:
        return
    for p in model.parameters():
        if wd > 0:
            p.mul_(1 - wd)
        if T > 0:
            p.add_(torch.randn_like(p) * T)


def train_formed(seed, device, max_steps, eval_every, post):
    pool, ev, vocab, rng = _data(seed, device)
    ev_tok, ev_tgt = ev[0], ev[1]
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
    return (best if best_q >= 0.85 else None), plateau, vocab


def blend(a_sd, b_sd, a):
    return {k: (1 - a) * a_sd[k] + a * b_sd[k] for k in a_sd}


def bath_run(start_sd, seed, T, wd, steps, eval_every, noise_seed, device):
    """release from start_sd under gradient build + bath dissolve; track q_mass."""
    pool, ev, vocab, _ = _data(seed, device); ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device); model.load_state_dict(start_sd); opt = _opt(model)
    rng = np.random.default_rng(noise_seed)
    qs = []
    for st in range(steps + 1):
        if st % eval_every == 0:
            qv = q_mass(model, ev_tok, ev_tgt)
            qs.append(0.0 if not np.isfinite(qv) else qv)        # destroyed (NaN) = fully dissolved
        if st >= steps:
            break
        loss = grad_step(model, opt, pool, vocab, rng, 64, device)
        bath_kick(model, T, wd)
        if not np.isfinite(loss):                                # blew up -> treat as dissolved, stop
            qs.append(0.0); break
    return qs


# ------------------------------------------------------------------------------------- (A) melt curve
def part_melt(formed_sd, seed, Ts, wd, steps, eval_every, nseed, device, diss="noise"):
    knob = "weight-decay wd" if diss == "wd" else "bath temperature T (noise)"
    print(f"\n  (A) MELT CURVE: from a formed circuit, ramp the DISSOLVING FORCE = {knob}", flush=True)
    print(f"  {'val':>9} {'q_final(med)':>13} {'melted?':>8}")
    curve = []
    for T in Ts:
        finals = []
        for ns in range(nseed):
            # diss='wd': dissolving force is weight decay (noise off); else injected Gaussian noise (wd off)
            kT, kwd = (0.0, T) if diss == "wd" else (T, wd)
            qs = bath_run(formed_sd, seed, kT, kwd, steps, eval_every, seed * 7919 + int(T * 1e6) + ns, device)
            finals.append(float(np.median(qs[-max(3, len(qs) // 4):])))
        med = float(np.median(finals)); curve.append((T, med))
        print(f"  {T:>9.5f} {med:>13.3f} {'YES' if med < 0.6 else 'no':>8}", flush=True)
    # T_melt ~ where q_final crosses 0.6 from above
    Tmelt = None
    for i in range(1, len(curve)):
        if curve[i - 1][1] >= 0.6 > curve[i][1]:
            (t0, q0), (t1, q1) = curve[i - 1], curve[i]
            Tmelt = t0 + (t1 - t0) * (q0 - 0.6) / (q0 - q1 + 1e-9); break
    print(f"  => T_melt (q crosses 0.6) ~= {None if Tmelt is None else round(Tmelt,5)}", flush=True)
    return curve, Tmelt


# --------------------------------------------------------- (B,C) 2D phase map: committor q(size n0, temperature T)
def part_bc(formed_sd, plateau_sd, seed, Tbc, wd, ngrid, kbath, steps, eval_every, device, diss="noise"):
    print(f"\n  (B,C) DIE-OR-GROW phase map: committor q(grow) over size n0 x temperature T", flush=True)
    pool, ev, vocab, _ = _data(seed, device); ev_tok, ev_tgt = ev[0], ev[1]
    model = _new_model(vocab, seed, device)
    # prepare nuclei of controlled size n=q_mass by plateau<->formed interpolation
    targets = np.linspace(0.45, 0.92, ngrid)
    prepared = []
    for tgt in targets:
        lo_a, hi_a = 0.0, 1.0; best = None
        for _ in range(20):
            am = 0.5 * (lo_a + hi_a); model.load_state_dict(blend(plateau_sd, formed_sd, am))
            phi = q_mass(model, ev_tok, ev_tgt)
            if best is None or abs(phi - tgt) < abs(best[1] - tgt):
                best = (am, phi)
            if phi < tgt:
                lo_a = am
            else:
                hi_a = am
        prepared.append((blend(plateau_sd, formed_sd, best[0]), best[1]))
    n0s = [p[1] for p in prepared]
    print("  committor q(grow):  rows = n0 (size), cols = T (temperature)")
    print("      n0\\T  " + " ".join(f"{T:>7.4f}" for T in Tbc), flush=True)
    grid = np.zeros((len(prepared), len(Tbc))); alltraj = []
    for ni, (sd, n0) in enumerate(prepared):                      # n0 outer -> print a row as soon as it is done
        for ti, T in enumerate(Tbc):
            grows = 0
            for k in range(kbath):
                kT, kwd = (0.0, T) if diss == "wd" else (T, wd)   # diss='wd' sweeps weight decay (noise off)
                qs = bath_run(sd, seed, kT, kwd, steps, eval_every, seed * 333 + int(n0 * 1000) + ti * 17 + k, device)
                if abs(T - Tbc[len(Tbc) // 2]) < 1e-9:
                    alltraj.append(qs)
                fin = float(np.median(qs[-max(3, len(qs) // 4):]))
                if fin >= 0.65:
                    grows += 1
            grid[ni, ti] = grows / kbath
        print(f"  {n0:>7.3f}  " + " ".join(f"{grid[ni,ti]:>7.2f}" for ti in range(len(Tbc))), flush=True)
    # critical nucleus n*(T) = size where committor crosses 0.5, per temperature
    print("\n  critical nucleus n*(T) (size where q crosses 0.5):")
    nstar = {}
    for ti, T in enumerate(Tbc):
        col = grid[:, ti]; star = None
        for ni in range(1, len(n0s)):
            if col[ni - 1] < 0.5 <= col[ni]:
                star = n0s[ni - 1] + (n0s[ni] - n0s[ni - 1]) * (0.5 - col[ni - 1]) / (col[ni] - col[ni - 1] + 1e-9); break
        nstar[T] = star
        print(f"    T={T:.4f}: n* = {None if star is None else round(star,3)}")
    vals = [(T, s) for T, s in nstar.items() if s is not None]
    if len(vals) >= 2:
        Tv = np.array([v[0] for v in vals]); sv = np.array([v[1] for v in vals])
        slope = np.polyfit(Tv, sv, 1)[0]
        print(f"  => n*(T) slope = {slope:+.2f}  -> {'CRITICAL NUCLEUS GROWS WITH T (Gibbs-Thomson: bigger=more stable=melts hotter) => REVERSIBLE die-or-grow CONFIRMED' if slope > 0 else 'n* not increasing with T -- inspect'}")
    elif len(vals) == 1:
        print(f"  => a watershed n* exists at one T (die-or-grow appears WITH a bath); need >=2 T for the n*(T) law")
    else:
        print(f"  => no watershed at any T in this band -- retune T range around T_melt")
    fdt(alltraj, eval_every)
    return dict(n0s=n0s, Tbc=list(Tbc), grid=grid.tolist(), nstar={f"{k:.4f}": v for k, v in nstar.items()}), alltraj


def fdt(alltraj, dt):
    mv, dphi, occ = [], [], []
    for qs in alltraj:
        q = np.array(qs, float); occ += q.tolist()
        mv += q[:-1].tolist(); dphi += np.diff(q).tolist()
    mv = np.array(mv); dphi = np.array(dphi); occ = np.array(occ)
    if len(mv) < 50:
        print("  (C) too few points for FDT"); return
    edges = np.linspace(np.percentile(occ, 1), np.percentile(occ, 99), 16)
    cc = (edges[:-1] + edges[1:]) / 2; h, _ = np.histogram(occ, bins=edges); p = h / h.sum()
    G = -np.log(np.where(p > 0, p, np.nan)); slope = np.gradient(G, cc[1] - cc[0])
    idx = np.digitize(mv, edges)
    print("  (C) FDT check (bath gives a true free energy):")
    print(f"  {'q':>6} {'occ':>7} {'-logP':>7} {'driftA':>9} {'D':>8} {'-D*slope':>9}")
    A_list, fdt_list = [], []
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() < 10 or not np.isfinite(G[b - 1]):
            continue
        A = float(dphi[sel].mean() / dt); D = float(dphi[sel].var() / (2 * dt))
        pred = -D * slope[b - 1]; A_list.append(A); fdt_list.append(pred)
        print(f"  {cc[b-1]:>6.3f} {p[b-1]:>7.4f} {G[b-1]:>7.2f} {A:>+9.4f} {D:>8.5f} {pred:>+9.4f}")
    if len(A_list) >= 3 and np.std(A_list) > 1e-9 and np.std(fdt_list) > 1e-9:
        r = float(np.corrcoef(A_list, fdt_list)[0, 1])
        print(f"  => FDT corr(A, -D*slope) = {r:+.2f}  (~+1 => drift == free-energy gradient => ONE Langevin system)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["melt", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--Ts", type=float, nargs="+",
                    default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--Tbc", type=float, nargs="+", default=None)   # die-or-grow temperature band; None=auto from T_melt
    ap.add_argument("--ngrid", type=int, default=10)
    ap.add_argument("--kbath", type=int, default=16)
    ap.add_argument("--melt_steps", type=int, default=600)
    ap.add_argument("--bc_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--nseed_melt", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--post", type=int, default=800)
    ap.add_argument("--out", default="melt.json")
    ap.add_argument("--diss", choices=["noise", "wd"], default="noise")  # dissolving force: injected noise or weight decay
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.seeds = 3; args.Ts = [0.0, 0.01, 0.02, 0.04, 0.08, 0.15]; args.ngrid = 5; args.kbath = 6
        args.melt_steps = 300; args.bc_steps = 300; args.nseed_melt = 2; args.max_steps = 3000; args.post = 800
    print(f"device={device} part={args.part} Ts={args.Ts} wd={args.wd} "
          f"(config kl{KEY_LEN}/kp{KEY_POOL}/vp{VAL_POOL})", flush=True)

    out = dict(melt=[], bc=[], Tmelt=None, Tstar=None)
    for seed in range(args.seeds):
        formed, plateau, vocab = train_formed(seed, device, args.max_steps, args.eval_every, args.post)
        if formed is None or plateau is None:
            print(f"  seed {seed}: endpoints missing (formed={formed is not None} plateau={plateau is not None}) -- skip", flush=True)
            continue
        print(f"\n##### seed {seed} #####", flush=True)
        curve, Tmelt = part_melt(formed, seed, args.Ts, args.wd, args.melt_steps, args.eval_every, args.nseed_melt, device, args.diss)
        out["melt"].append(dict(seed=seed, curve=curve, Tmelt=Tmelt))
        if args.part == "all":
            if args.Tbc is not None:
                Tbc = args.Tbc
            elif Tmelt is not None:
                Tbc = [round(Tmelt * f, 5) for f in (0.55, 0.70, 0.82, 0.92, 1.02, 1.15)]
            else:
                Tbc = sorted(args.Ts)[1:7]
            bc, _ = part_bc(formed, plateau, seed, Tbc, args.wd, args.ngrid, args.kbath,
                            args.bc_steps, args.eval_every, device, args.diss)
            out["bc"].append(dict(seed=seed, Tbc=list(Tbc), **bc))
        json.dump(out, open(os.path.join(RESULTS, args.out), "w"), default=float)
        if args.smoke and len(out["melt"]) >= 1 and args.part == "all":
            break          # smoke: one good seed is enough to see the structure
    print(f"\n  saved results/{args.out}")


if __name__ == "__main__":
    main()
