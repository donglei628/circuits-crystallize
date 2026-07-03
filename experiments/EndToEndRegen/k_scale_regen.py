"""
PYTHIA FORMULA-4 END-TO-END (regeneration two-axis, real LM). Same closure as the toy (end2end_regen.py) but on a real
Pythia: lesion the WHOLE redundant induction set, RESTORE a subset of the K=3 components, and re-train on data whose
induction density nu we CONTROL (the key advantage over formula 2: the regeneration retraining data is OURS, so we can
sweep nu on a pretrained model). Two independent axes:
  - axis nu : at each #restored s, sweep nu (fraction of second-half tokens that repeat) -> a_r(s) = slope of T_regrow vs 1/nu
  - axis s  : a_r(s) = C_r * p^-(K-s) -> ln a_r(s) linear in s with slope ln p  (reads off p, independent of the nu-sweep)
Then C_r = a_r(s) * p^(K-s) is the regeneration prefactor (= 1/N_sites). Predicted by the toy: C_r should be LARGE on a
real (very wide) model -- the basin-governance limit (many competing sites -> few effective nucleation sites).
Restoring the ACTUAL trained component weights anchors the rebuild (no need to force a basin: the wandering only happens
when a NEW head must be found).

  python k_scale_regen.py --model EleutherAI/pythia-160m --smoke
  python k_scale_regen.py --model EleutherAI/pythia-160m --nus 0.3 0.5 1.0 --reps 2
"""
from __future__ import annotations
import argparse, copy, json, os
from itertools import combinations
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from k_scale import (make_rep_batch, all_induction_heads, all_prevtoken_heads, induction_score, copy_score, set_comp)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
COMPS = ["PREV", "IND_QK", "IND_OV"]; K = 3


def text_batches_nu(V, L, B, n, nu, device, seed):
    """training batches with controllable induction density nu: second half copies the first with prob nu per position."""
    g = np.random.default_rng(seed); h = L // 2
    for _ in range(n):
        first = g.integers(0, V, (B, h), dtype=np.int64)
        alt = g.integers(0, V, (B, h), dtype=np.int64)
        second = np.where(g.random((B, h)) < nu, first, alt)
        seq = np.concatenate([first, second], axis=1).astype(np.int64)
        yield torch.from_numpy(seq).to(device)


def regen_time(model, base_state, restore_set, ind_heads, prev_heads, V, nu, args, rep, base, device, seed):
    """lesion all 3 components, restore `restore_set`, retrain at density nu -> steps to reach target*base (or None)."""
    model.load_state_dict(base_state)
    for c in COMPS:
        set_comp(model, c, ind_heads, prev_heads, src_state=None)            # lesion all (fresh random)
    for c in restore_set:
        set_comp(model, c, ind_heads, prev_heads, src_state=base_state)      # restore the subset
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for s, tb in enumerate(text_batches_nu(V, args.train_L, args.batch, args.steps, nu, device, seed), 1):
        out = model(tb, labels=tb); out.loss.backward(); opt.step(); opt.zero_grad()
        if s % args.eval_every == 0:
            model.eval(); ind = copy_score(model, rep, args.L); model.train()
            if ind >= args.target * base:
                return s
    return None


def fit_line(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m, b = np.polyfit(xs, ys, 1)
    r2 = 1 - ((ys - (m * xs + b)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9)
    return float(m), float(b), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--nus", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--train_L", type=int, default=256); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--ind_thresh", type=float, default=0.15); ap.add_argument("--prev_thresh", type=float, default=0.50)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", default="kscale_regen.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model = "EleutherAI/pythia-70m"; args.nus = [0.5, 1.0]; args.reps = 1; args.steps = 400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model); V = tok.vocab_size
    rep = make_rep_batch(V, args.L, max(args.batch, 4), device, 0)
    dt = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision="step143000", dtype=dt,
                                                 attn_implementation="eager").to(device)
    ind_heads = all_induction_heads(model, rep, args.L, args.ind_thresh)
    prev_heads = all_prevtoken_heads(model, rep, args.L, args.prev_thresh)
    base = copy_score(model, rep, args.L)
    base_state = copy.deepcopy(model.state_dict())
    print(f"device={device} model={args.model} copy(func)={base:.3f}  #ind_heads={len(ind_heads)} #prev_heads={len(prev_heads)} "
          f"nus={args.nus} (formula-4 end-to-end: C_r=a_r(s)*p^(K-s))", flush=True)

    # ---- two-axis: for each #restored s, average over size-s subsets, sweep nu -> a_r(s) ----
    a_r = {}; rows = []
    for s in range(0, K):                                       # s = 0,1,2 restored components (s=3 instant)
        subs = [()] if s == 0 else list(combinations(COMPS, s))
        pts = []
        for nu in args.nus:
            tms = []
            for sub in subs:
                ts = [regen_time(model, base_state, list(sub), ind_heads, prev_heads, V, nu, args, rep, base, device,
                                 seed=(s + 1) * 1000 + r) for r in range(args.reps)]
                ts = [t for t in ts if t is not None]
                if ts: tms.append(float(np.median(ts)))
            tm = float(np.mean(tms)) if tms else None
            if tm is not None: pts.append((1.0 / nu, tm))
            rows.append((s, nu, tm))
            print(f"  s={s} nu={nu}: T_regrow={tm} (avg over {len(subs)} subsets)", flush=True)
            json.dump(dict(model=args.model, base=base, n_ind=len(ind_heads), n_prev=len(prev_heads), rows=rows),
                      open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        if len(pts) >= 2:
            a_r[s], t0, r2 = fit_line([p[0] for p in pts], [p[1] for p in pts])
            print(f"    -> a_r(s={s}) = {a_r[s]:.1f} (t0={t0:.0f}, R2={r2:.2f})", flush=True)

    ss = sorted(a_r)
    out = dict(model=args.model, base=base, n_ind=len(ind_heads), n_prev=len(prev_heads), rows=rows, a_r=a_r)
    if len(ss) >= 2 and all(a_r[s] > 0 for s in ss):
        slope, b, r2s = fit_line(ss, [np.log(a_r[s]) for s in ss])      # slope = ln p
        p = float(np.exp(slope)); lnp = -slope
        C_r = {s: float(a_r[s] * p ** (K - s)) for s in ss}; Cs = list(C_r.values())
        cv = float(np.std(Cs) / (np.mean(Cs) + 1e-9))
        out.update(p=p, lnp=lnp, lnp_r2=r2s, C_r=C_r, C_r_mean=float(np.mean(Cs)), C_r_cv=cv)
        print(f"\n=== PYTHIA FORMULA-4 ({args.model}) ===", flush=True)
        print(f"  |ln p|(regen) = {lnp:.3f} (R2={r2s:.2f}) | C_r per s = {[round(c,2) for c in Cs]} "
              f"mean={np.mean(Cs):.2f} CV={cv:.2f}", flush=True)
        print(f"  => {'two-axis CLOSES (C_r constant across s)' if cv < 0.3 else 'C_r not constant across s -- report honestly'}", flush=True)
        print(f"  toy C_r was ~0.9-3 (d256-384); a real (much wider) model is predicted to have a LARGER C_r (basin limit)", flush=True)
    json.dump(out, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
