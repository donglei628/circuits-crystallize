"""
A2 — OV vs QK TIMING on a real LM (reviewer r6#6).

Toy finding: the OV (copy) substrate is largely built DURING the incubation period (well before the snap), while the
QK (prefix-match) retrieval locks in AT the snap -- so the incubation period has a mechanistic meaning. r6 asks: is the
OV-before-QK timing a narrow-toy artifact, or does it hold in a real LM? Here we take the eventual induction head in
Pythia and, across the training checkpoints, measure BOTH its QK match-score (prefix-matching attention) and its OV
copy-score (diagonal dominance of M_OV = W_U W_O^h W_V^h W_E), then compare which rises first.

Self-contained (no cross-task imports). Inference only (forward pass + weight matmuls) => low GPU memory.

  # 184 smoke:
  CUDA_VISIBLE_DEVICES=3 python a2_ovqk_timing.py --model EleutherAI/pythia-70m --revs step0 step1000 step143000 \
      --local_models /path/to/workdir/models --nsample 300 --out a2_smoke70m.json
  # 184 formal:
  CUDA_VISIBLE_DEVICES=3 python a2_ovqk_timing.py --model EleutherAI/pythia-160m \
      --local_models /path/to/workdir/models --out a2_pythia160m.json
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
def qk_scores(model, batch, L):
    """per-(layer,head) prefix-matching (QK) induction score on the repeated sequence."""
    out = model(batch, output_attentions=True); attns = out.attentions; device = batch.device
    dest = torch.arange(L, 2 * L - 1, device=device); src = dest - L + 1; D = dest.numel()
    scores = {}
    for l, A in enumerate(attns):
        a = A[:, :, dest, :]; pick = a[:, :, torch.arange(D, device=device), src]; sc = pick.mean(dim=(0, 2))
        for h in range(sc.numel()):
            scores[(l, h)] = float(sc[h])
    return scores


@torch.no_grad()
def ov_copy_score(model, layer, head, si, fold_ln=True):
    """OV-circuit copy score: mean diagonal percentile of M_OV[a,b]=W_U[b]·W_O^h·W_V^h·W_E[a]. 0.5=chance,1=copy."""
    gn = model.gpt_neox
    Wqkv = gn.layers[layer].attention.query_key_value.weight
    Wdense = gn.layers[layer].attention.dense.weight
    H = model.config.num_attention_heads; hid = model.config.hidden_size; hd = hid // H
    Wv = Wqkv[head * 3 * hd + 2 * hd: head * 3 * hd + 3 * hd, :]
    Wo = Wdense[:, head * hd:(head + 1) * hd]
    We = gn.embed_in.weight; Wu = model.embed_out.weight
    if fold_ln:
        Wu = Wu * gn.final_layer_norm.weight; We = We * gn.layers[layer].input_layernorm.weight
    Es = We.index_select(0, si)
    OVe = Es @ Wv.t() @ Wo.t()
    logits = OVe @ Wu.index_select(0, si).t()
    diag = logits.diagonal()
    return (logits < diag[:, None]).float().mean(1).mean().item()


def step_of(rev):
    return int(rev.replace("step", ""))


def load_model(name, rev, device, local_models):
    if local_models:
        path = os.path.join(local_models, name.split("/")[-1], rev)
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    return AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.float32,
                                                attn_implementation="eager").to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--revs", nargs="+", default=None)
    ap.add_argument("--local_models", default=None)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nsample", type=int, default=800)
    ap.add_argument("--out", default="a2_ovqk.json")
    args = ap.parse_args()
    revs = args.revs or FULL_REVS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_src = os.path.join(args.local_models, args.model.split("/")[-1], "step143000") if args.local_models else args.model
    tok = AutoTokenizer.from_pretrained(tok_src)
    batch = make_rep_batch(tok.vocab_size, args.L, args.batch, device, seed=0)
    rng = np.random.default_rng(0)
    si = torch.tensor(sorted(rng.choice(tok.vocab_size, size=min(args.nsample, tok.vocab_size), replace=False).tolist())).to(device)
    print(f"device={device} model={args.model} revs={len(revs)} nsample={si.numel()}", flush=True)

    # target = the eventual induction head (best QK matcher at the final checkpoint)
    mf = load_model(args.model, "step143000", device, args.local_models)
    qk_final = qk_scores(mf, batch, args.L)
    target = max(qk_final, key=qk_final.get)
    del mf; gc.collect(); torch.cuda.empty_cache()
    print(f"  target induction head (final best QK) = L{target[0]}H{target[1]}", flush=True)

    rows = []
    for rev in revs:
        m = load_model(args.model, rev, device, args.local_models)
        qk = qk_scores(m, batch, args.L)[target]
        ov = ov_copy_score(m, target[0], target[1], si)
        rows.append(dict(rev=rev, step=step_of(rev), qk=float(qk), ov=float(ov)))
        del m; gc.collect(); torch.cuda.empty_cache()
        print(f"  {rev:>10} (step {step_of(rev):>6}): QK={qk:.3f}  OV={ov:.3f}", flush=True)
        json.dump(dict(model=args.model, target_head=list(target), rows=rows), open(args.out, "w"), indent=2)

    analyze(rows, target, args.model)
    print(f"\n  saved {args.out}")


def snap_step(steps, vals, lo, hi):
    """first step where the normalized value (val-lo)/(hi-lo) crosses 0.5, interpolated in log-step."""
    norm = (np.array(vals) - lo) / (hi - lo + 1e-9)
    for i in range(1, len(steps)):
        if norm[i - 1] < 0.5 <= norm[i]:
            ls0, ls1 = np.log10(max(steps[i - 1], 1)), np.log10(steps[i])
            frac = (0.5 - norm[i - 1]) / (norm[i] - norm[i - 1] + 1e-9)
            return 10 ** (ls0 + frac * (ls1 - ls0))
    return None


def analyze(rows, target, model):
    rows = sorted(rows, key=lambda r: r["step"])
    steps = [r["step"] for r in rows]; qk = [r["qk"] for r in rows]; ov = [r["ov"] for r in rows]
    print(f"\n  ===========  A2: OV vs QK timing ({model}, head L{target[0]}H{target[1]})  ===========")
    print(f"  {'step':>8} {'QK':>7} {'OV':>7}")
    for r in rows:
        print(f"  {r['step']:>8} {r['qk']:>7.3f} {r['ov']:>7.3f}")
    if len(rows) < 3:
        print("\n  (smoke: too few points for timing)"); return
    qk_snap = snap_step(steps, qk, qk[0], max(qk))
    ov_snap = snap_step(steps, ov, 0.5, max(ov))          # OV chance is 0.5
    print(f"\n  QK match-score snaps at step ~ {qk_snap:.0f}" if qk_snap else "  QK: no clear snap")
    print(f"  OV copy-score  snaps at step ~ {ov_snap:.0f}" if ov_snap else "  OV: no clear snap")
    if qk_snap and ov_snap:
        verdict = ("OV BEFORE QK (incubation builds the copy substrate first, as in the toy)" if ov_snap < 0.7 * qk_snap
                   else "OV and QK rise together (the toy's OV-before-QK is NOT reproduced -> narrow-model-specific)"
                   if ov_snap < 1.4 * qk_snap else "QK before OV (opposite of the toy)")
        print(f"  ==> {verdict}")


if __name__ == "__main__":
    main()
