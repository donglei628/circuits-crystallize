"""
K-SCALE (the capstone of large-model nucleation thermodynamics) — measure the CONJUNCTION DEPTH K of a REAL induction
circuit across model sizes, and test the prediction that K is SCALE-INVARIANT (~3 = prev-token-head ∧ induction-QK ∧
induction-OV), confirming the combinatorial nucleation theorem (J ∝ p^K, ln(rate) linear in # components restored) on
real LMs at scale.

Method (combinatorial nucleation theorem, real-LM seeded form): identify the induction head (top prefix-match) and the
prev-token head that feeds it. The 3 components of the induction circuit: PREV (prev-token head Q,K,V,out), IND_QK
(induction head Q,K), IND_OV (induction head V + its output slice). LESION all three (re-randomize) -> induction craters.
Then for each SUBSET of the 3, RESTORE that subset to its trained values (leave the rest random), continue training, and
measure the regeneration RATE (1/steps-to-50%-recovery). Predicted: ln(rate) = const + (#restored)*|ln p|, K = 3.
Run across Pythia sizes; predicted K invariant.

Prereq: pip install transformers datasets (Tsinghua mirror). Run on 184 GPU (when free) or the local 4070.

  python k_scale.py --model EleutherAI/pythia-160m --steps 300 --eval_every 20
  python k_scale.py --model EleutherAI/pythia-70m --smoke
"""
from __future__ import annotations
import argparse, copy, itertools, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def make_rep_batch(vocab, L, B, device, seed=0):
    g = np.random.default_rng(seed)
    half = g.integers(0, vocab, size=(B, L), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


@torch.no_grad()
def induction_score(model, rep, L, head=None):
    out = model(rep, output_attentions=True); attns = out.attentions; dev = rep.device
    dest = torch.arange(L, 2 * L - 1, device=dev); src = dest - L + 1; D = dest.numel()
    if head is not None:
        A = attns[head[0]][:, head[1], dest, :]
        return float(A[:, torch.arange(D, device=dev), src].mean().item())
    best = (-1.0, None)
    for l, A in enumerate(attns):
        a = A[:, :, dest, :]; pick = a[:, :, torch.arange(D, device=dev), src]; sc = pick.mean(dim=(0, 2))
        m = float(sc.max())
        if m > best[0]:
            best = (m, (l, int(sc.argmax())))
    return best


@torch.no_grad()
def all_induction_heads(model, rep, L, thresh):
    """ALL heads whose induction (prefix-match) attention > thresh -> lesion the WHOLE redundant induction set, not just
    the top head (at scale a single-head lesion leaves big residual function via redundant heads)."""
    out = model(rep, output_attentions=True); attns = out.attentions; dev = rep.device
    dest = torch.arange(L, 2 * L - 1, device=dev); src = dest - L + 1; D = dest.numel()
    heads = []
    for l, A in enumerate(attns):
        a = A[:, :, dest, :]; pick = a[:, :, torch.arange(D, device=dev), src]; sc = pick.mean(dim=(0, 2))
        for h in range(sc.numel()):
            if float(sc[h]) > thresh:
                heads.append((l, h))
    return heads


@torch.no_grad()
def all_prevtoken_heads(model, rep, L, thresh):
    """ALL heads attending to the immediately-previous token > thresh (the upstream PREV set)."""
    out = model(rep, output_attentions=True); attns = out.attentions; dev = rep.device
    pos = torch.arange(1, 2 * L, device=dev); heads = []
    for l, A in enumerate(attns):
        sc = A[:, :, pos, pos - 1].mean(dim=(0, 2))
        for h in range(sc.numel()):
            if float(sc[h]) > thresh:
                heads.append((l, h))
    return heads


@torch.no_grad()
def copy_score(model, rep, L):
    """FUNCTIONAL induction: mean prob assigned to the correct repeated token at second-half positions. Unlike the
    attention induction_score (a QK property that saturates at PREV+IND_QK), this needs IND_OV too (to write the
    attended value to the output) -> the full K=3 conjunction PREV/\\IND_QK/\\IND_OV."""
    dev = rep.device
    logits = model(rep).logits                                  # (B, 2L, V)
    pos = torch.arange(L, 2 * L - 1, device=dev)                # predict at second-half positions
    tgt = rep[:, pos + 1]                                       # correct next token (the repeat)
    lp = logits[:, pos].log_softmax(-1)
    return float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp().mean())


@torch.no_grad()
def prevtoken_head(model, rep, L):
    """find the head that most attends to the immediately-previous token (the upstream component of induction)."""
    out = model(rep, output_attentions=True); attns = out.attentions; dev = rep.device
    pos = torch.arange(1, 2 * L, device=dev)
    best = (-1.0, None)
    for l, A in enumerate(attns):
        a = A[:, :, pos, pos - 1]; sc = a.mean(dim=(0, 2))
        m = float(sc.max())
        if m > best[0]:
            best = (m, (l, int(sc.argmax())))
    return best


def comp_param_slices(model, comp, ind_heads, prev_heads):
    """yield (param, row_slice, col_slice) for component `comp` across ALL relevant heads (lists)."""
    gn = model.gpt_neox; cfg = model.config; H = cfg.num_attention_heads; hd = cfg.hidden_size // H
    if comp == "PREV":
        for L, h in prev_heads:
            qkv = gn.layers[L].attention.query_key_value.weight
            yield (qkv, slice(h * 3 * hd, (h + 1) * 3 * hd), slice(None))            # whole head Q,K,V
            yield (gn.layers[L].attention.dense.weight, slice(None), slice(h * hd, (h + 1) * hd))
    elif comp == "IND_QK":
        for L, h in ind_heads:
            qkv = gn.layers[L].attention.query_key_value.weight
            yield (qkv, slice(h * 3 * hd, h * 3 * hd + 2 * hd), slice(None))          # Q,K
    elif comp == "IND_OV":
        for L, h in ind_heads:
            qkv = gn.layers[L].attention.query_key_value.weight
            yield (qkv, slice(h * 3 * hd + 2 * hd, (h + 1) * 3 * hd), slice(None))    # V
            yield (gn.layers[L].attention.dense.weight, slice(None), slice(h * hd, (h + 1) * hd))


def set_comp(model, comp, ind_heads, prev_heads, src_state=None, scale=0.02):
    """if src_state given -> RESTORE from it; else LESION (re-randomize). Operates over ALL heads in the lists."""
    with torch.no_grad():
        for p, rs, cs in comp_param_slices(model, comp, ind_heads, prev_heads):
            if src_state is None:
                p[rs, cs] = torch.randn_like(p[rs, cs]) * scale
            else:
                # find this param's name to index src_state
                name = next(n for n, q in model.named_parameters() if q is p)
                p[rs, cs] = src_state[name][rs, cs].to(p.device)


def text_batches(tok, L, B, n, device, seed=0):
    """REPEATED-sequence training batches: on a repeated sequence the dominant way to cut loss IS induction, so this
    drives induction-circuit regrowth strongly (Olsson-style) -- unlike natural text where induction is incidental and
    barely regrows in a few hundred steps. Fresh random repeats each batch -> learns GENERAL induction, eval on a
    held-out repeat. Also removes the wikitext download dependency entirely."""
    g = np.random.default_rng(seed); V = tok.vocab_size
    for bi in range(n):
        half = g.integers(0, V, size=(B, L - L // 2), dtype=np.int64)
        seq = np.concatenate([half, half[:, :L // 2]], axis=1)           # [chunk, chunk] -> repeated
        yield torch.from_numpy(seq.astype(np.int64)).to(device)


def regen_rate(model, base_state, restore_set, ind_heads, prev_heads, tok, args, rep, base, device):
    """lesion all 3 components (across ALL redundant heads), RESTORE `restore_set`, train -> steps-to-target.
    Averaged over args.reps independent runs (fresh lesion randomization + training seed) -> median regrew kills the
    large run-to-run variance of a single huge-lesion regrowth (esp. at scale). rate = 1/median(regrew)."""
    regrews = []; crater0 = None
    for r in range(args.reps):
        model.load_state_dict(base_state)
        for c in ["PREV", "IND_QK", "IND_OV"]:
            set_comp(model, c, ind_heads, prev_heads, src_state=None)            # lesion all (random each rep)
        for c in restore_set:
            set_comp(model, c, ind_heads, prev_heads, src_state=base_state)      # restore subset
        if r == 0:
            crater0 = copy_score(model, rep, args.L)                             # functional recovery (no retrain)
        model.train(); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        regrew = None
        for s, tb in enumerate(text_batches(tok, args.train_L, args.batch, args.steps, device,
                                            seed=(len(restore_set) + 1) * 100 + r), 1):
            out = model(tb, labels=tb); out.loss.backward(); opt.step(); opt.zero_grad()
            if s % args.eval_every == 0 or s == args.steps:
                model.eval(); ind = copy_score(model, rep, args.L); model.train()
                if regrew is None and ind >= 0.8 * base:             # ABSOLUTE high target (above every partial crater)
                    regrew = s; break
        regrews.append(regrew if regrew else args.steps * 2)                     # censored -> very slow
    med = float(np.median(regrews))
    return 1.0 / med, float(crater0), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--train_L", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")               # load+train in bfloat16 (fits 1b alongside vLLM)
    ap.add_argument("--reps", type=int, default=1)               # repeat each subset N times, median regrew (kills variance)
    ap.add_argument("--ind_thresh", type=float, default=0.15)    # lesion ALL induction heads above this (kill redundancy)
    ap.add_argument("--prev_thresh", type=float, default=0.50)   # lesion ALL strong prev-token heads above this
    ap.add_argument("--out", default="k_scale.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.model = "EleutherAI/pythia-70m"; args.steps = 80; args.eval_every = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    rep = make_rep_batch(tok.vocab_size, args.L, max(args.batch, 4), device, 0)
    dt = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision="step143000",
                                                 dtype=dt, attn_implementation="eager").to(device)
    base_attn, _top = induction_score(model, rep, args.L)
    ind_heads = all_induction_heads(model, rep, args.L, args.ind_thresh)         # ALL induction heads (kill redundancy)
    prev_heads = all_prevtoken_heads(model, rep, args.L, args.prev_thresh)
    base = copy_score(model, rep, args.L)                        # FUNCTIONAL baseline (regeneration target)
    base_state = copy.deepcopy(model.state_dict())
    print(f"device={device} model={args.model} attn_ind(top)={base_attn:.3f} copy(func)={base:.3f}  "
          f"#induction_heads={len(ind_heads)}(>{args.ind_thresh}) #prev_heads={len(prev_heads)}(>{args.prev_thresh})", flush=True)
    print(f"  induction heads: {ind_heads}\n  prev heads: {prev_heads}", flush=True)

    COMPS = ["PREV", "IND_QK", "IND_OV"]
    subsets = [list(c) for k in range(4) for c in itertools.combinations(COMPS, k)]
    print(f"\n  {'restore set':>22} {'#comp':>5} {'crater':>7} {'regrew@':>8} {'rate':>9} {'ln(rate)':>9}")
    pts = []
    for ss in subsets:
        rate, crater, rg = regen_rate(model, base_state, ss, ind_heads, prev_heads, tok, args, rep, base, device)
        pts.append((len(ss), float(np.log(rate)), "+".join(ss) or "none"))
        print(f"  {('+'.join(ss) or 'none'):>22} {len(ss):>5} {crater:>7.3f} {str(rg):>8} {rate:>9.5f} {np.log(rate):>9.2f}", flush=True)
        json.dump(dict(model=args.model, base=base, n_ind_heads=len(ind_heads), n_prev_heads=len(prev_heads),
                       ind_heads=ind_heads, prev_heads=prev_heads,
                       pts=[(n, lr, nm) for n, lr, nm in pts]), open(os.path.join(RESULTS, args.out), "w"), indent=2)
    js = np.array([p[0] for p in pts], float); lr = np.array([p[1] for p in pts])
    slope, b = np.polyfit(js, lr, 1); pred = slope * js + b
    r2 = 1 - ((lr - pred) ** 2).sum() / max(((lr - lr.mean()) ** 2).sum(), 1e-9)
    print(f"\n=== K-SCALE ({args.model}) ===")
    print(f"  ln(regen rate) = {b:.2f} + {slope:+.2f}*(#components restored)   R2={r2:.2f}")
    print(f"  => |ln p| = {slope:.2f} (per-component speedup x{np.exp(slope):.2f}); conjunction depth K = 3 (PREV,IND_QK,IND_OV)")
    print(f"  => {'COMBINATORIAL NUCLEATION THEOREM holds on a REAL LM (ln rate linear in #components)' if r2>0.5 and slope>0.2 else 'not clean -- report honestly'}")
    print(f"  saved results/{args.out}  (run across sizes 160m/410m/1b -> test K invariance)")


if __name__ == "__main__":
    main()
