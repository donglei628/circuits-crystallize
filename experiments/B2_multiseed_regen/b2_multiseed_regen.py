"""
B2 — MULTI-SEED / MULTI-HEAD Pythia regeneration (reviewer r5#5).

The §8 real-LM regeneration result was a SINGLE run: lesion the one top induction head (L4H6), n=1. r5 accepts it as
an existence demonstration but flags the single point. B2 turns it into a robust result: lesion EACH of the top-k
induction heads, each with several data seeds, run the lesion->regrow, and report the distribution of recovery (does
it consistently re-nucleate? does it consistently relocate to a fresh head?).

Self-contained. ~5 GB (one Pythia-160m training at a time). Loads from the 184 local-dir layout via --local_models.

  # 184 smoke:
  CUDA_VISIBLE_DEVICES=3 python b2_multiseed_regen.py --model EleutherAI/pythia-70m --topk 1 --seeds 1 --steps 40 \
      --local_models /path/to/workdir/models --out b2_smoke70m.json
  # 184 formal:
  CUDA_VISIBLE_DEVICES=3 python b2_multiseed_regen.py --model EleutherAI/pythia-160m --topk 3 --seeds 2 --steps 200 \
      --local_models /path/to/workdir/models --out b2_pythia160m.json
"""
from __future__ import annotations
import argparse, gc, json, os
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def make_rep_batch(vocab, L, B, device, seed=0):
    g = np.random.default_rng(seed)
    half = g.integers(0, vocab, size=(B, L), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


@torch.no_grad()
def induction_score(model, rep, L, head=None):
    out = model(rep, output_attentions=True); attns = out.attentions; device = rep.device
    dest = torch.arange(L, 2 * L - 1, device=device); src = dest - L + 1; D = dest.numel()
    if head is not None:
        A = attns[head[0]][:, head[1], dest, :]
        return float(A[:, torch.arange(D, device=device), src].mean().item())
    best = (-1.0, None); allsc = {}
    for l, A in enumerate(attns):
        a = A[:, :, dest, :]; pick = a[:, :, torch.arange(D, device=device), src]; sc = pick.mean(dim=(0, 2))
        for h in range(sc.numel()):
            allsc[(l, h)] = float(sc[h])
        m = float(sc.max())
        if m > best[0]:
            best = (m, (l, int(sc.argmax())))
    return best[0], best[1], allsc


def lesion_head(model, L, h, scale=0.02):
    gn = model.gpt_neox; cfg = model.config
    H = cfg.num_attention_heads; hid = cfg.hidden_size; hd = hid // H
    with torch.no_grad():
        qkv = gn.layers[L].attention.query_key_value.weight
        qkv[h * 3 * hd:(h + 1) * 3 * hd, :].normal_(0, scale)
        gn.layers[L].attention.dense.weight[:, h * hd:(h + 1) * hd].normal_(0, scale)


def text_batches(tok, L, B, n_batches, device, seed=0):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    txt = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(txt, return_tensors="np")["input_ids"][0]
    need = n_batches * B * L
    if len(ids) < need:
        ids = np.tile(ids, need // len(ids) + 1)
    g = np.random.default_rng(seed)
    starts = g.integers(0, len(ids) - L - 1, size=(n_batches, B))
    for bi in range(n_batches):
        chunk = np.stack([ids[s:s + L] for s in starts[bi]])
        yield torch.from_numpy(chunk.astype(np.int64)).to(device)


def load_model(name, rev, device, local_models):
    if local_models:
        path = os.path.join(local_models, name.split("/")[-1], rev)
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(device)
    return AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.float32,
                                                attn_implementation="eager").to(device)


def one_regen(model, tok, rep, L, target, ind_head, lr, steps, eval_every, train_L, batch, device, data_seed):
    """lesion `target`, continue training; return recovery trajectory + peak/relocation for the FIXED induction head."""
    lesion_head(model, target[0], target[1])
    crater = induction_score(model, rep, L, head=ind_head)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    traj = []; base_head = ind_head
    for s, tb in enumerate(text_batches(tok, train_L, batch, steps, device, seed=data_seed), 1):
        out = model(tb, labels=tb); out.loss.backward(); opt.step(); opt.zero_grad()
        if s % eval_every == 0 or s == steps:
            model.eval()
            g_sc, g_head, _ = induction_score(model, rep, L)
            model.train()
            traj.append(dict(step=s, ind_global=float(g_sc), gh=list(g_head)))
    peak = max(traj, key=lambda x: x["ind_global"]) if traj else dict(ind_global=float(crater), gh=list(base_head), step=0)
    relocated = peak["gh"] != list(base_head)
    return float(crater), peak, relocated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--topk", type=int, default=3)            # lesion each of the top-k induction heads
    ap.add_argument("--seeds", type=int, default=2)           # data-order seeds per head
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--train_L", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--local_models", default=None)
    ap.add_argument("--out", default="b2_regen.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_src = os.path.join(args.local_models, args.model.split("/")[-1], "step143000") if args.local_models else args.model
    tok = AutoTokenizer.from_pretrained(tok_src)
    rep = make_rep_batch(tok.vocab_size, args.L, max(args.batch, 4), device, seed=0)

    # baseline: top-k induction heads on the formed model
    mf = load_model(args.model, "step143000", device, args.local_models)
    base, base_head, allsc = induction_score(mf, rep, args.L)
    topk = sorted(allsc, key=allsc.get, reverse=True)[:args.topk]
    print(f"device={device} model={args.model} baseline induction={base:.3f}; top-{args.topk} heads={topk}", flush=True)
    del mf; gc.collect(); torch.cuda.empty_cache()

    runs = []
    for tgt in topk:
        for seed in range(args.seeds):
            model = load_model(args.model, "step143000", device, args.local_models)
            crater, peak, relocated = one_regen(model, tok, rep, args.L, tgt, tgt, args.lr, args.steps,
                                                args.eval_every, args.train_L, args.batch, device, data_seed=seed)
            rec = 100 * (peak["ind_global"] - crater) / max(base - crater, 1e-6)
            runs.append(dict(target=list(tgt), seed=seed, base=base, crater=crater,
                             peak=peak["ind_global"], peak_head=peak["gh"], peak_step=peak["step"],
                             recovery_pct=rec, relocated=bool(relocated)))
            print(f"  lesion L{tgt[0]}H{tgt[1]} seed{seed}: crater {crater:.3f} -> peak {peak['ind_global']:.3f} "
                  f"@L{peak['gh'][0]}H{peak['gh'][1]} ({rec:.0f}% recovery, {'RELOCATED' if relocated else 'same head'})", flush=True)
            json.dump(dict(model=args.model, base=base, topk=[list(t) for t in topk], runs=runs), open(args.out, "w"), indent=2)
            del model; gc.collect(); torch.cuda.empty_cache()

    analyze(runs, base, args.model)
    print(f"\n  saved {args.out}")


def analyze(runs, base, model):
    rec = np.array([r["recovery_pct"] for r in runs])
    reloc = np.mean([r["relocated"] for r in runs])
    print(f"\n  ===========  B2: multi-seed/head regeneration ({model}, n={len(runs)})  ===========")
    print(f"  recovery%: mean {rec.mean():.0f} +- {rec.std():.0f}  (min {rec.min():.0f}, max {rec.max():.0f})")
    print(f"  fraction recovering >50%: {np.mean(rec > 50):.0%}")
    print(f"  relocation rate (regrew in a DIFFERENT head): {reloc:.0%}")
    print(f"  ==> {'ROBUST regeneration across heads/seeds (no longer a single run)' if np.mean(rec>50)>=0.7 else 'regeneration inconsistent -- report distribution honestly'}")


if __name__ == "__main__":
    main()
