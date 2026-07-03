"""
P10/N15 (reviewer r6 / r2#3.2) — the RANDOM-INIT control for the Pythia regeneration claim.

The regeneration result (d4_regen): lesion a formed Pythia induction head -> continue training ~125 steps -> induction
recovers to ~0.86x baseline in a FRESH head. Reviewer's objection: the lesioned model still holds ALL the other
pretrained structure, which could let a new head acquire induction by ordinary fast learning -- NOT by re-nucleation.
Decisive control: SAME architecture from RANDOM INIT (step0), SAME steps, SAME data; measure induction.
  - random-init stays far below baseline ==> regeneration is NON-trivial (seeded re-nucleation on surviving structure).
  - random-init also reaches it         ==> the speedup is generic fast learning (report honestly).

32-box adaptations: --text_file (plain wikitext txt, no `datasets` dependency) and hub-fallback for revisions
not present in the local models dir (step143000 lives in the HF hub cache on 32; step0 in models/).

  python p10_randinit_control.py --local_models /path/to/workdir/models --text_file wikitext2_train.txt
"""
from __future__ import annotations
import argparse, gc, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from d4_regen import make_rep_batch, induction_score, RESULTS


def load_model(name, revision, device, local_models=None):
    """prefer the local-dir layout {local_models}/{basename}/{revision}; fall back to the HF hub cache."""
    if local_models:
        path = os.path.join(local_models, name.split("/")[-1], revision)
        if os.path.isfile(os.path.join(path, "config.json")):               # a bare mkdir'd dir is NOT a checkpoint

            return AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32, attn_implementation="eager").to(device)
    return AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.float32, attn_implementation="eager").to(device)


def text_batches_file(tok, path, L, B, n_batches, device, seed=0):
    """same contract as d4_regen.text_batches but reads a plain local txt (no `datasets` needed)."""
    txt = open(path, encoding="utf-8").read()
    ids = tok(txt, return_tensors="np")["input_ids"][0]
    need = n_batches * B * L
    if len(ids) < need:
        ids = np.tile(ids, need // len(ids) + 1)
    g = np.random.default_rng(seed)
    starts = g.integers(0, len(ids) - L - 1, size=(n_batches, B))
    for bi in range(n_batches):
        chunk = np.stack([ids[s:s + L] for s in starts[bi]])
        yield torch.from_numpy(chunk.astype(np.int64)).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--steps", type=int, default=125)          # match the regen recovery horizon
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--train_L", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--baseline_rev", default="step143000")
    ap.add_argument("--local_models", default=None)
    ap.add_argument("--text_file", required=True)              # plain wikitext txt (no datasets dependency)
    ap.add_argument("--out", default="p10_randinit.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 20; args.eval_every = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_src = os.path.join(args.local_models, args.model.split("/")[-1], "step0") if args.local_models else args.model
    tok = AutoTokenizer.from_pretrained(tok_src)
    rep = make_rep_batch(tok.vocab_size, args.L, max(args.batch, 4), device, seed=0)

    mf = load_model(args.model, args.baseline_rev, device, args.local_models)
    base, base_head = induction_score(mf, rep, args.L)
    print(f"device={device} model={args.model}  FORMED baseline induction={base:.3f} @L{base_head[0]}H{base_head[1]}", flush=True)
    del mf; gc.collect(); torch.cuda.empty_cache()

    model = load_model(args.model, "step0", device, args.local_models)
    init_ind, ih0 = induction_score(model, rep, args.L)
    print(f"  RANDOM-INIT (step0) induction={init_ind:.3f} @L{ih0[0]}H{ih0[1]} (expect ~chance)", flush=True)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    traj = [dict(step=0, ind_global=float(init_ind))]
    batches = text_batches_file(tok, args.text_file, args.train_L, args.batch, args.steps, device, seed=0)
    for s, tb in enumerate(batches, 1):
        out = model(tb, labels=tb)
        out.loss.backward(); opt.step(); opt.zero_grad()
        if s % args.eval_every == 0 or s == args.steps:
            model.eval()
            ind_g, gh = induction_score(model, rep, args.L)
            model.train()
            traj.append(dict(step=s, ind_global=float(ind_g), gh=list(gh)))
            print(f"    step {s:>4}: random-init induction={ind_g:.3f}@L{gh[0]}H{gh[1]}  "
                  f"({100*ind_g/max(base,1e-6):.0f}% of baseline)  loss={float(out.loss):.2f}", flush=True)
            json.dump(dict(model=args.model, base=base, base_head=list(base_head), init_ind=init_ind,
                           steps=args.steps, traj=traj),
                      open(os.path.join(RESULTS, args.out), "w"), indent=2)

    peak = max(traj[1:], key=lambda x: x["ind_global"]) if len(traj) > 1 else traj[-1]
    fr = lambda v: 100 * v / max(base, 1e-6)
    print(f"\n=== P10/N15: random-init control ({args.steps} steps) ===", flush=True)
    print(f"  FORMED baseline       : {base:.3f}", flush=True)
    print(f"  random-init @init      : {init_ind:.3f} ({fr(init_ind):.0f}%)", flush=True)
    print(f"  random-init @{args.steps} steps : peak {peak['ind_global']:.3f} ({fr(peak['ind_global']):.0f}% of baseline)", flush=True)
    verdict = ("NON-TRIVIAL regeneration: random init reaches only %.0f%% in %d steps, far below regen's ~86%% -> "
               "seeded re-nucleation on surviving structure" % (fr(peak['ind_global']), args.steps)) \
        if fr(peak['ind_global']) < 50 else \
        ("random init ALSO reaches %.0f%% -> regen speedup may be generic fast learning (report honestly)" % fr(peak['ind_global']))
    print(f"  ==> {verdict}", flush=True)
    print("P10_DONE", flush=True)


if __name__ == "__main__":
    main()
