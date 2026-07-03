"""
END-TO-END for FORMULA 4 (regeneration), parallel to end2end_const (formula 2). Regeneration = nucleation with the
conjunction depth reduced by the surviving/seeded components: a lesioned circuit with s intact components re-forms at
rate proportional to p^(K-s), so

    T_regrow(s, nu) = t0 + C_r / (nu * p^(K-s))                         (nu = regen data density, s = #seeded components)

This has the SAME two-axis structure as formula 2, and we close it END-TO-END by measuring the two axes INDEPENDENTLY:
  - axis nu  : at each fixed s, sweep the regeneration data density nu and fit a_r(s) = slope of T_regrow vs 1/nu.
  - axis s   : a_r(s) = C_r * p^(-(K-s)), so ln a_r(s) is linear in s with slope ln p  ->  reads off p (independent of
               the within-s nu-sweep).
Then C_r = a_r(s) * p^(K-s) must be a SINGLE prefactor: constant across s AND across width, and equal to formula 2's
formation prefactor C_form = a_form * p^K (the same N_sites governs formation and regeneration). If C_r is NOT constant,
either the p^(K-s) form is wrong or the measurement is.

Reuses end2end_const's induction toy + component machinery (PREV wedge IND_QK wedge IND_OV, K=3). Regeneration = start
from a trained donor, LESION all 3, RESTORE a cumulative s-subset (rest of model kept), re-train at density nu.

  python end2end_regen.py --smoke
  python end2end_regen.py --ds 256 384 --seeds 3 --out e2e_regen.json
"""
from __future__ import annotations
import argparse, copy, json, os
import numpy as np
import torch
from end2end_const import Tf, train_tstar, comp_state, set_components

RESULTS = os.path.join(os.path.dirname(__file__), "results")
COMPS = ["IND_OV", "IND_QK", "PREV"]          # cumulative restore order (s=1 restores IND_OV, s=2 adds IND_QK, ...)
K = 3


def fit_line(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m, b = np.polyfit(xs, ys, 1)
    r2 = 1 - ((ys - (m * xs + b)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9)
    return float(m), float(b), float(r2)


def regen_time(donor_m, rand_state, donor_state, subset, nu, d, args, device, seed):
    """lesion all 3 components of a donor copy, restore `subset`, re-train at density nu -> t*."""
    m = copy.deepcopy(donor_m)
    set_components(m, rand_state, set(COMPS), d)                 # lesion all 3
    if subset:
        set_components(m, donor_state, set(subset), d)          # restore the given subset
    return train_tstar(m, args.T, args.vocab, nu, args.regen_lr, args.regen_steps, args.regen_ee, args.thresh, seed, device)


def measure_width(d, args, device):
    V, T, H = args.vocab, args.T, args.heads
    # ---- formation reference: a_form (nu-sweep from scratch) + p^K (seed-dissect slope) -> C_form ----
    tform = {}
    for nu in args.nus:
        ts = [train_tstar(Tf(V, d, H, T + 2).to(device), T, V, nu, args.lr, args.max_steps, args.ee, args.thresh, sd, device)
              for sd in range(args.seeds)]
        ts = [t for t in ts if t is not None]
        tform[nu] = float(np.median(ts)) if ts else None
    fpts = [(1 / nu, tform[nu]) for nu in args.nus if tform[nu]]
    a_form, t0_form, _ = fit_line([p[0] for p in fpts], [p[1] for p in fpts]) if len(fpts) >= 2 else (None, None, None)

    # ---- donor + lesion/restore states ----
    torch.manual_seed(7); donor_m = Tf(V, d, H, T + 2).to(device)
    train_tstar(donor_m, T, V, 1.0, args.lr, args.max_steps, args.ee, 0.85, 0, device)
    donor_state = comp_state(donor_m, d)
    torch.manual_seed(3000); rand_state = comp_state(Tf(V, d, H, T + 2).to(device), d)

    # ---- REGENERATION two-axis: for each s, nu-sweep -> a_r(s), AVERAGING over which s-subset is restored ----
    from itertools import combinations
    a_r = {}; t0_r = {}; rows = []
    for s in range(0, K):                                        # s = 0,1,2 surviving components (s=3 is instant)
        subs = [()] if s == 0 else list(combinations(COMPS, s))  # all size-s subsets (the law is over the COUNT s)
        pts = []
        for nu in args.nus:
            tms = []                                             # median t* per subset, then average over subsets
            for sub in subs:
                ts = [regen_time(donor_m, rand_state, donor_state, sub, nu, d, args, device, sd) for sd in range(args.seeds)]
                ts = [t for t in ts if t is not None]
                if ts: tms.append(float(np.median(ts)))
            tm = float(np.mean(tms)) if tms else None
            if tm is not None: pts.append((1 / nu, tm))
            rows.append((s, nu, tm))
            print(f"    [d{d}] s={s} nu={nu}: T_regrow={tm} (avg over {len(subs)} subsets)", flush=True)
        if len(pts) >= 2:
            a_r[s], t0_r[s], r2 = fit_line([p[0] for p in pts], [p[1] for p in pts])
            print(f"      -> a_r(s={s}) = {a_r[s]:.1f}  (t0={t0_r[s]:.0f}, R2={r2:.2f})", flush=True)

    # ---- axis s: p from slope of ln a_r(s) vs s ; then C_r(s) = a_r(s) * p^(K-s) ----
    ss = sorted(a_r)
    res = dict(d=d, a_form=a_form, t0_form=t0_form, a_r=a_r, t0_r=t0_r, rows=rows)
    if len(ss) >= 2 and all(a_r[s] > 0 for s in ss):
        slope, b, r2s = fit_line(ss, [np.log(a_r[s]) for s in ss])  # slope = ln p
        p = float(np.exp(slope)); lnp = -slope
        C_r = {s: float(a_r[s] * p ** (K - s)) for s in ss}
        Cs = list(C_r.values())
        cv = float(np.std(Cs) / (np.mean(Cs) + 1e-9))
        C_form = float(a_form * p ** K) if a_form else None
        res.update(p=p, lnp=lnp, lnp_r2=r2s, C_r=C_r, C_r_mean=float(np.mean(Cs)), C_r_cv=cv, C_form=C_form)
        cf = round(C_form, 2) if C_form else None
        print(f"  ==[d{d}]== |ln p|(regen)={lnp:.3f}(R2={r2s:.2f}) | C_r per s={[round(c,2) for c in Cs]} "
              f"mean={np.mean(Cs):.2f} CV={cv:.2f} | C_form={cf}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[256, 384])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--vocab", type=int, default=256); ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--regen_steps", type=int, default=6000); ap.add_argument("--ee", type=int, default=25)
    ap.add_argument("--regen_ee", type=int, default=5); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--regen_lr", type=float, default=1.5e-4)    # slower regen so even s=K-1 stays above the time-resolution floor
    ap.add_argument("--out", default="e2e_regen.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.ds = [256]; args.nus = [0.5, 1.0]; args.seeds = 1; args.max_steps = 4000; args.regen_steps = 2500
    print(f"device={device} ds={args.ds} (formula 4 end-to-end: C_r=a_r(s)*p^(K-s) constant across s & width & =C_form?)", flush=True)

    out = []
    for d in args.ds:
        out.append(measure_width(d, args, device))
        json.dump({"args": vars(args), "results": out}, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    print(f"\n  ===== VERDICT (formula 4 end-to-end) =====", flush=True)
    Cms = []
    for r in out:
        if "C_r_mean" in r:
            Cms.append(r["C_r_mean"])
            print(f"   d={r['d']:>4}: C_r={r['C_r_mean']:.2f} (CV across s={r['C_r_cv']:.2f}) | C_form={r.get('C_form')}", flush=True)
    if len(Cms) >= 2:
        cv = float(np.std(Cms) / (np.mean(Cms) + 1e-9))
        print(f"   C_r across widths: {[round(c,2) for c in Cms]} CV={cv:.2f} "
              f"=> {'CONSTANT (formula 4 closes end-to-end)' if cv < 0.25 else 'STRUCTURED'}", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
