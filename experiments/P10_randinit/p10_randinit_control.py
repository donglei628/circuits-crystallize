"""
P10 (reviewer r6 / r2#3.2) — the RANDOM-INIT control for the Pythia regeneration claim.

The regeneration result (d4_regen): lesion a formed Pythia induction head -> continue training ~125 steps -> induction
recovers to ~0.86x baseline in a FRESH head. Reviewer's objection: the lesioned model still holds ALL the other
pretrained structure (other heads, MLPs, embeddings, LayerNorm), which could let a new head acquire induction by
ordinary fast learning -- NOT by re-nucleation. The decisive control: take the SAME architecture from RANDOM INIT
(checkpoint step0, no pretrained structure) and train the SAME number of steps on the SAME data; measure induction.
  - random-init induction stays far below 0.86x baseline  ==> regeneration is NON-trivial (it leverages surviving
    structure = seeded re-nucleation, not generic fast learning).
  - random-init also reaches ~0.86x baseline               ==> the "regeneration" speedup is just fast learning; the
    re-nucleation interpretation is unsupported (report honestly).

  python p10_randinit_control.py --model EleutherAI/pythia-160m --steps 125 --eval_every 25
  python p10_randinit_control.py --model EleutherAI/pythia-70m --steps 20 --eval_every 10 --smoke
"""
from __future__ import annotations
import argparse, gc, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from d4_regen import make_rep_batch, induction_score, text_batches, RESULTS


def load_model(name, revision, device, local_models=None):
    """load from the HF hub (revision) OR from a pre-downloaded local-dir layout
       {local_models}/{model-basename}/{revision}."""
    if local_models:
        path = os.path.join(local_models, name.split("/")[-1], revision)
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager").to(device)
    return AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.float32, attn_implementation="eager").to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--steps", type=int, default=125)          # match the regen recovery horizon
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--train_L", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--baseline_rev", default="step143000")    # the formed model (for the denominator)
    ap.add_argument("--local_models", default=None)            # e.g. /path/to/workdir/models (184)
    ap.add_argument("--out", default="p10_randinit.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_src = os.path.join(args.local_models, args.model.split("/")[-1], args.baseline_rev) if args.local_models else args.model
    tok = AutoTokenizer.from_pretrained(tok_src)
    rep = make_rep_batch(tok.vocab_size, args.L, max(args.batch, 4), device, seed=0)

    # --- 1. baseline induction of the FORMED model (denominator), then free it ---
    mf = load_model(args.model, args.baseline_rev, device, args.local_models)
    base, base_head = induction_score(mf, rep, args.L)
    print(f"device={device} model={args.model}  FORMED baseline induction={base:.3f} @L{base_head[0]}H{base_head[1]}", flush=True)
    del mf; gc.collect(); torch.cuda.empty_cache()

    # --- 2. RANDOM INIT (step0): induction at init, then train the same horizon ---
    model = load_model(args.model, "step0", device, args.local_models)
    init_ind, ih0 = induction_score(model, rep, args.L)
    print(f"  RANDOM-INIT (step0) induction={init_ind:.3f} @L{ih0[0]}H{ih0[1]} (expect ~chance)", flush=True)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    traj = [dict(step=0, ind_global=float(init_ind))]
    batches = text_batches(tok, args.train_L, args.batch, args.steps, device, seed=0)
    for s, tb in enumerate(batches, 1):
        out = model(tb, labels=tb)
        out.loss.backward(); opt.step(); opt.zero_grad()
        if s % args.eval_every == 0 or s == args.steps:
            model.eval()
            ind_g, gh = induction_score(model, rep, args.L)         # best induction ANYWHERE
            model.train()
            traj.append(dict(step=s, ind_global=float(ind_g), gh=list(gh)))
            print(f"    step {s:>4}: random-init induction={ind_g:.3f}@L{gh[0]}H{gh[1]}  "
                  f"({100*ind_g/max(base,1e-6):.0f}% of baseline)  loss={float(out.loss):.2f}", flush=True)
            json.dump(dict(model=args.model, base=base, base_head=list(base_head), init_ind=init_ind,
                           steps=args.steps, traj=traj),
                      open(os.path.join(RESULTS, args.out), "w"), indent=2)

    peak = max(traj[1:], key=lambda x: x["ind_global"]) if len(traj) > 1 else traj[-1]
    final = traj[-1]["ind_global"]
    fr = lambda v: 100 * v / max(base, 1e-6)
    print(f"\n=== P10: random-init control ({args.steps} steps) ===")
    print(f"  FORMED baseline       : {base:.3f}")
    print(f"  random-init @init      : {init_ind:.3f} ({fr(init_ind):.0f}% of baseline)")
    print(f"  random-init @{args.steps} steps : final {final:.3f} ({fr(final):.0f}%)  peak {peak['ind_global']:.3f} ({fr(peak['ind_global']):.0f}%)")
    print(f"  regeneration reached  : ~0.86x baseline in the same horizon (d4_regen)")
    verdict = ("NON-TRIVIAL regeneration: random init reaches only %.0f%% of baseline in %d steps, far below the "
               "regen's ~86%% -> the lesioned model's surviving structure is what enables fast recovery (seeded "
               "re-nucleation)" % (fr(peak['ind_global']), args.steps)) if fr(peak['ind_global']) < 50 else \
              ("random init ALSO reaches %.0f%% -> the regen speedup may be ordinary fast learning; re-nucleation "
               "interpretation NOT supported (report honestly)" % fr(peak['ind_global']))
    print(f"  ==> {verdict}")
    print(f"  saved results/{args.out}")


if __name__ == "__main__":
    main()
