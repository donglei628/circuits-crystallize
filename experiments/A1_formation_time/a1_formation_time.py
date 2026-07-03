"""
A1 — REAL-MODEL FORMATION TIME t*  (first step of validating the rate law on a real LM; reviewer P8 / the "9th point").

Sweep the public Pythia training checkpoints (step0 .. step143000), measure the induction (prefix-matching) score at
each, and read off the formation CURVE and t* (the training step at which induction snaps in). INFERENCE ONLY (load
checkpoint -> forward pass -> measure -> free) => low GPU memory (~2-3 GB), safe next to the colleague's vLLM.

Self-contained (no cross-task imports). Loads from the pre-downloaded 184 local-dir layout via --local_models.

  # local smoke (logic check, 3 revisions of 70m, downloads via mirror):
  HF_ENDPOINT=https://hf-mirror.com python a1_formation_time.py --model EleutherAI/pythia-70m \
      --revs step0 step1000 step143000 --out a1_smoke.json
  # 184 formal (full 20 checkpoints, from local models):
  CUDA_VISIBLE_DEVICES=3 python a1_formation_time.py --model EleutherAI/pythia-160m \
      --local_models /path/to/workdir/models --out a1_pythia160m.json
"""
from __future__ import annotations
import argparse, gc, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FULL_REVS = ["step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128", "step256",
             "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000",
             "step128000", "step143000"]


def make_rep_batch(vocab, L, B, device, seed=0):
    g = np.random.default_rng(seed)
    half = g.integers(0, vocab, size=(B, L), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


@torch.no_grad()
def induction_score(model, rep, L):
    """best prefix-matching head: attention from second-half query to the token that followed the same token earlier."""
    out = model(rep, output_attentions=True)
    attns = out.attentions; device = rep.device
    dest = torch.arange(L, 2 * L - 1, device=device); src = dest - L + 1; D = dest.numel()
    best = (-1.0, None)
    for l, A in enumerate(attns):
        a = A[:, :, dest, :]; pick = a[:, :, torch.arange(D, device=device), src]; sc = pick.mean(dim=(0, 2))
        m = float(sc.max())
        if m > best[0]:
            best = (m, (l, int(sc.argmax())))
    return best


def step_of(rev):
    return int(rev.replace("step", ""))


def load_model(name, rev, device, local_models):
    if local_models:
        path = os.path.join(local_models, name.split("/")[-1], rev)
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(device)
    return AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.float32,
                                                attn_implementation="eager").to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--revs", nargs="+", default=None)            # default = FULL_REVS
    ap.add_argument("--local_models", default=None)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="a1_formation.json")
    args = ap.parse_args()
    revs = args.revs or FULL_REVS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_src = os.path.join(args.local_models, args.model.split("/")[-1], "step143000") if args.local_models else args.model
    tok = AutoTokenizer.from_pretrained(tok_src)
    rep = make_rep_batch(tok.vocab_size, args.L, args.batch, device, seed=0)
    print(f"device={device} model={args.model} revs={len(revs)} L={args.L}", flush=True)

    rows = []
    for rev in revs:
        try:
            m = load_model(args.model, rev, device, args.local_models)
        except Exception as e:
            print(f"  {rev}: load FAILED ({str(e)[:80]}) -- skip", flush=True); continue
        sc, head = induction_score(m, rep, args.L)
        rows.append(dict(rev=rev, step=step_of(rev), induction=float(sc), head=list(head)))
        del m; gc.collect(); torch.cuda.empty_cache()
        print(f"  {rev:>10} (step {step_of(rev):>6}): induction={sc:.3f} @L{head[0]}H{head[1]}", flush=True)
        json.dump(dict(model=args.model, L=args.L, rows=rows), open(args.out, "w"), indent=2)

    analyze(rows, args.model)
    print(f"\n  saved {args.out}")


def analyze(rows, model):
    rows = sorted(rows, key=lambda r: r["step"])
    if len(rows) < 3:
        print("\n  (smoke: too few points for t*; pipeline check only)")
        return
    steps = np.array([r["step"] for r in rows]); ind = np.array([r["induction"] for r in rows])
    final = ind[-1]
    thr = 0.5 * final
    # t* = first step crossing 0.5*final (snap midpoint), interpolated in log-step
    tstar = None
    for i in range(1, len(steps)):
        if ind[i - 1] < thr <= ind[i]:
            ls0, ls1 = np.log10(max(steps[i - 1], 1)), np.log10(steps[i])
            frac = (thr - ind[i - 1]) / (ind[i] - ind[i - 1] + 1e-9)
            tstar = 10 ** (ls0 + frac * (ls1 - ls0)); break
    # the snap = the largest single-interval jump
    jumps = ind[1:] - ind[:-1]
    js = int(np.argmax(jumps))
    # turnover: distinct carrying heads after formation
    formed_heads = [tuple(r["head"]) for r in rows if r["induction"] > thr]
    print(f"\n  ===========  A1: real-model formation time ({model})  ===========")
    print(f"  final induction={final:.3f}  threshold(0.5*final)={thr:.3f}")
    print(f"  t* (induction snaps in) ~ step {tstar:.0f}" if tstar else "  t*: induction never crossed 0.5*final")
    print(f"  steepest snap between step {steps[js]} -> {steps[js+1]}  (+{jumps[js]:.3f})")
    print(f"  carrying head(s) once formed: {sorted(set(formed_heads))}  (distinct={len(set(formed_heads))} -> turnover if >1)")
    print(f"  ==> this t* is the real-LM datapoint to confront with the rate law (P8 / 9th point)")


if __name__ == "__main__":
    main()
