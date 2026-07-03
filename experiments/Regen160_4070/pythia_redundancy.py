"""
D-Pythia — is the induction function DELOCALIZED/REDUNDANT in a real LM? (cross-substrate test of toy D3)

At a trained checkpoint we rank all heads by prefix-matching induction score, then GREEDILY ablate them
top-down (zeroing each head's slice of the attention output, before the output projection) and watch the
functional readout — the ICL loss drop on a repeated-random sequence — collapse. |S| = #heads we must ablate
to destroy induction (ICL drop ≤ 20% of baseline).

  few heads (|S| small)  ⇒ LOCALIZED induction
  many heads (|S| large) ⇒ DELOCALIZED / redundant (like the toy, where |S|≈9/12)

  python pythia_redundancy.py --model EleutherAI/pythia-160m --step 8000 --L 128 --batch 8 --max_ablate 30
  python pythia_redundancy.py --model EleutherAI/pythia-70m --step 143000 --L 64 --batch 4 --max_ablate 10 --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def make_batch(vocab, L, B, device, seed=0):
    g = np.random.default_rng(seed)
    half = g.integers(0, vocab, size=(B, L), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


def get_layers_and_proj(model):
    """Return (list of attention modules, output-proj attr name, head_size, n_heads). Robust to GPTNeoX naming."""
    core = getattr(model, "gpt_neox", None) or model.base_model
    layers = core.layers
    attns = [lyr.attention for lyr in layers]
    a0 = attns[0]
    proj_name = "dense" if hasattr(a0, "dense") else ("o_proj" if hasattr(a0, "o_proj") else None)
    cfg = model.config
    nh = cfg.num_attention_heads
    hs = cfg.hidden_size // nh
    return attns, proj_name, hs, nh


# global ablation state read by the hooks: {layer_idx: set(head_idx)}
ABLATE = {}


def install_hooks(attns, proj_name, head_size):
    handles = []
    for l, attn in enumerate(attns):
        proj = getattr(attn, proj_name)

        def pre_hook(module, inputs, _l=l):
            heads = ABLATE.get(_l)
            if not heads:
                return None
            x = inputs[0].clone()                          # [B,T,hidden] concatenated heads
            for h in heads:
                x[..., h * head_size:(h + 1) * head_size] = 0.0
            return (x,) + inputs[1:]

        handles.append(proj.register_forward_pre_hook(pre_hook))
    return handles


@torch.no_grad()
def induction_per_head(model, tok_batch, L):
    out = model(tok_batch, output_attentions=True)
    attns = out.attentions
    device = tok_batch.device
    dest = torch.arange(L, 2 * L - 1, device=device); src = dest - L + 1; D = dest.numel()
    scores = {}
    for l, A in enumerate(attns):
        a = A[:, :, dest, :]
        pick = a[:, :, torch.arange(D, device=device), src]
        sc = pick.mean(dim=(0, 2))
        for h in range(sc.numel()):
            scores[(l, h)] = float(sc[h])
    return scores


@torch.no_grad()
def icl_drop(model, tok_batch, L):
    out = model(tok_batch)
    lp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
    tgt = tok_batch[:, 1:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(nll[:, :L].mean() - nll[:, L:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--step", type=int, default=8000)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_ablate", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    batch = make_batch(tok.vocab_size, args.L, args.batch, device, seed=0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=f"step{args.step}", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    attns, proj_name, head_size, nh = get_layers_and_proj(model)
    print(f"device={device} model={args.model} step={args.step} heads={len(attns)}L×{nh}H  proj={proj_name}")

    scores = induction_per_head(model, batch, args.L)
    ranked = sorted(scores, key=lambda lh: scores[lh], reverse=True)
    ABLATE.clear()
    install_hooks(attns, proj_name, head_size)

    base = icl_drop(model, batch, args.L)
    print(f"  baseline ICL drop = {base:.3f} nats; top heads by induction score:")
    for lh in ranked[:6]:
        print(f"     L{lh[0]}H{lh[1]} score={scores[lh]:.3f}")

    curve = [(0, base)]
    nS = None
    for k in range(1, args.max_ablate + 1):
        l, h = ranked[k - 1]
        ABLATE.setdefault(l, set()).add(h)
        d = icl_drop(model, batch, args.L)
        curve.append((k, d))
        if nS is None and d <= 0.20 * base:
            nS = k
        if k <= 12 or k % 5 == 0:
            print(f"  ablated top-{k:>2} heads (last L{l}H{h}): ICL drop = {d:.3f} ({d/base*100 if base else 0:.0f}% of base)")
        if nS is not None and k >= nS + 2:
            break

    total_heads = len(attns) * nh
    print(f"\n=== D-Pythia redundancy (step {args.step}) ===")
    print(f"  baseline ICL drop {base:.2f} nats; total heads {total_heads}")
    if nS:
        print(f"  |S| = {nS} heads to destroy induction (ICL≤20% base)  ⇒ "
              f"{'LOCALIZED (few heads)' if nS <= 3 else 'DELOCALIZED/redundant (many heads)'} "
              f"[{nS}/{total_heads} = {nS/total_heads*100:.1f}% of heads]")
    else:
        print(f"  induction NOT destroyed within {args.max_ablate} ablations ⇒ very redundant (|S|>{args.max_ablate})")
    res = dict(model=args.model, step=args.step, baseline_icl=base, total_heads=total_heads, nS=nS,
               top_heads=[[lh[0], lh[1], scores[lh]] for lh in ranked[:10]],
               ablation_curve=[[k, float(d)] for k, d in curve])
    json.dump(res, open(os.path.join(RESULTS, f"pythia_redundancy_{args.model.split('/')[-1]}_s{args.step}.json"), "w"), indent=2)
    print(f"  saved results/pythia_redundancy_{args.model.split('/')[-1]}_s{args.step}.json")


if __name__ == "__main__":
    main()
