"""
CROSS-VALIDATION (independent re-implementation) of the core claim: the token FREQUENCY distribution (Zipf vs uniform),
not the embedding geometry, drives induction formation speed. The worry being checked: with Zipf, a few tokens are very
common, so a model could fake a high copy-score by a UNIGRAM shortcut (always predict the common token) without learning
real induction. We guard against it with TWO probes:
  - ind_same   : copy-score on a same-frequency (Zipf or uniform) repeated probe   [can be inflated by the unigram shortcut]
  - ind_STRICT : copy-score on a UNIFORM-token repeated probe                       [unigram shortcut fails here -> real induction only]
plus qk_prev (the attention look-back, frequency-independent) and a unigram baseline. If a Zipf-trained model is fast on
ind_STRICT too, the conclusion is robust. This file shares NO code with micro_compare.py (fresh data gen + measures).

  python xval_freq.py --freq zipf    --out xv_zipf.json
  python xval_freq.py --freq uniform --out xv_uni.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoModelForCausalLM

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def freq_probs(vocab, kind, s=1.07):
    if kind == "uniform":
        return None
    p = 1.0 / np.power(np.arange(1, vocab + 1), s); return p / p.sum()


def sample_tokens(shape, vocab, probs, rng):
    if probs is None:
        return rng.integers(0, vocab, shape, dtype=np.int64)
    return rng.choice(vocab, shape, p=probs).astype(np.int64)


def make_seq(B, T, vocab, nu, probs, rng):
    half = T // 2
    R = sample_tokens((B, half), vocab, probs, rng)
    rand2 = sample_tokens((B, half), vocab, probs, rng)
    copy = rng.random((B, half)) < nu
    return np.concatenate([R, np.where(copy, R, rand2)], axis=1)


@torch.no_grad()
def copy_score(model, T, vocab, probs, device, seed, n=256):
    """copy-score on a fully-repeated probe whose tokens are drawn from `probs` (None = uniform = STRICT, unigram fails)."""
    rng = np.random.default_rng(seed); half = T // 2
    R = sample_tokens((n, half), vocab, probs, rng); x = np.concatenate([R, R], axis=1)
    xt = torch.from_numpy(x).to(device)
    pred = model(xt).logits[:, half:2 * half - 1].argmax(-1)
    tgt = torch.from_numpy(x[:, half + 1:2 * half]).to(device)
    return float((pred == tgt).float().mean())


def c_copy_heldout(model, k, device, sample=1500, steps=400, lr=5e-3):
    """held-out rank-k OV copy capacity of the current embeddings (geometry quality)."""
    WE = model.gpt_neox.embed_in.weight.detach(); WU = model.embed_out.weight.detach()
    V, d = WE.shape; rng = np.random.default_rng(7); n = min(sample, V // 2); perm = rng.permutation(V)
    tr = torch.from_numpy(perm[:n]).long().to(device); te = torch.from_numpy(perm[n:2 * n]).long().to(device)
    A = torch.zeros(d, k, device=device, requires_grad=True); B = torch.zeros(d, k, device=device, requires_grad=True)
    torch.nn.init.normal_(A, std=0.02); torch.nn.init.normal_(B, std=0.02)
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(((WE[tr] @ B) @ A.t()) @ WU.t(), tr); loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        return float(((((WE[te] @ B) @ A.t()) @ WU.t()).argmax(1) == te).float().mean())


@torch.no_grad()
def qk_lookback(model, T, vocab, probs, device, seed, n=128):
    rng = np.random.default_rng(seed); half = T // 2
    R = sample_tokens((n, half), vocab, probs, rng); x = np.concatenate([R, R], axis=1)
    xt = torch.from_numpy(x).to(device)
    att = model(xt, output_attentions=True).attentions
    idx = torch.arange(half + 1, 2 * half - 1, device=device); tgt = idx - half + 1
    best = 0.0
    for a in att:
        best = max(best, float(a[:, :, idx, tgt].mean(dim=(0, 2)).max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["zipf", "uniform"], required=True)
    ap.add_argument("--vocab", type=int, default=8192); ap.add_argument("--nu", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--max_steps", type=int, default=14000)
    ap.add_argument("--ee", type=int, default=100); ap.add_argument("--init_emb", default=None)
    ap.add_argument("--emb_alpha", type=float, default=1.0, help="blend: alpha*init_emb + (1-alpha)*random (sweep geometry quality)")
    ap.add_argument("--freeze_emb", action="store_true"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    probs = freq_probs(args.vocab, args.freq)
    cfg = GPTNeoXConfig(vocab_size=args.vocab, hidden_size=args.hidden, num_hidden_layers=args.layers,
                        num_attention_heads=args.heads, intermediate_size=4 * args.hidden,
                        max_position_embeddings=args.T + 2, use_parallel_residual=True, rotary_pct=0.25,
                        attn_implementation="eager")
    model = GPTNeoXForCausalLM(cfg).to(device)
    if args.init_emb:                                        # give it a (blended) real geometry to drive the COPY
        is_local = os.path.isdir(args.init_emb)
        pm = (AutoModelForCausalLM.from_pretrained(args.init_emb, dtype=torch.float32) if is_local
              else AutoModelForCausalLM.from_pretrained(args.init_emb, revision="step143000", dtype=torch.float32))
        a = args.emb_alpha
        with torch.no_grad():                                # blend: alpha*Pythia + (1-alpha)*random (sweep C_copy)
            model.gpt_neox.embed_in.weight.mul_(1 - a).add_(a * pm.gpt_neox.embed_in.weight.to(device))
            model.embed_out.weight.mul_(1 - a).add_(a * pm.embed_out.weight.to(device))
        if args.freeze_emb:
            model.gpt_neox.embed_in.weight.requires_grad_(False); model.embed_out.weight.requires_grad_(False)
        del pm; torch.cuda.empty_cache(); print(f"  blended {args.init_emb} alpha={a} (frozen={args.freeze_emb})", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    k_ov = args.hidden // args.heads; C0 = c_copy_heldout(model, k_ov, device)   # geometry quality at start (held-out)
    print(f"  C_copy(held-out, start) = {C0:.3f}", flush=True)
    # unigram-shortcut baseline copy-score: a model that ALWAYS predicts the argmax-frequency token, on each probe
    uni_argmax = 0 if probs is None else int(np.argmax(probs))
    print(f"device={device} freq={args.freq} V={args.vocab} (ind_same can cheat via unigram; ind_STRICT cannot)", flush=True)

    traj = []; t_same = t_strict = None
    print(f"\n  {'step':>6} {'ind_same':>9} {'ind_STRICT':>11} {'qk':>6}", flush=True)
    for s in range(0, args.max_steps + 1):
        if s % args.ee == 0:
            model.eval()
            i_same = copy_score(model, args.T, args.vocab, probs, device, 111)       # same-freq probe
            i_strict = copy_score(model, args.T, args.vocab, None, device, 222)       # UNIFORM probe (strict)
            qk = qk_lookback(model, args.T, args.vocab, None, device, 333)
            traj.append(dict(step=s, ind_same=i_same, ind_strict=i_strict, qk=qk))
            if t_same is None and i_same > 0.5: t_same = s
            if t_strict is None and i_strict > 0.5: t_strict = s
            print(f"  {s:>6} {i_same:>9.2f} {i_strict:>11.2f} {qk:>6.2f}"
                  + ("  <-- t*_strict" if s == t_strict else ""), flush=True)
            json.dump({"freq": args.freq, "vocab": args.vocab, "emb_alpha": args.emb_alpha, "c_copy0": C0,
                       "seed": args.seed, "t_same": t_same, "t_strict": t_strict, "traj": traj},
                      open(os.path.join(RESULTS, args.out), "w"), indent=2)
            model.train()
            if t_strict is not None and s >= t_strict + 300: break
        x = torch.from_numpy(make_seq(args.batch, args.T, args.vocab, args.nu, probs, rng)).to(device)
        out = model(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()

    print(f"\n  ===== freq={args.freq}: t*_same={t_same}  t*_STRICT={t_strict} =====", flush=True)
    print(f"  (if t*_STRICT is also small -> REAL induction, not a unigram shortcut; if STRICT never forms -> shortcut)", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
