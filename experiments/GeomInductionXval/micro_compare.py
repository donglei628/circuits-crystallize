"""
FIRST-PRINCIPLES microscope: nucleation in a CLEAN lab (toy) vs an INDUSTRIAL melt (Pythia-like) -- frame by frame,
until the induction circuit forms. We track the formation as TWO sub-skills developing, not one output:
  (1) C_copy   = can the OV copy through the current embeddings (the 'legible alphabet' / geometry)
  (2) qk_prev  = does a head attend back to the induction position (the 'look-back' routing skill)
  (3) induction = the full copy-score (both skills present)
By dialing in industrial 'impurities' one at a time -- a Zipfian token frequency (long-tail rare tokens, like natural
text) and a large vocabulary -- we find the FRAME where the trajectory diverges from the clean toy, i.e. which skill the
impurity stalls. Then we MASK the impurity (freeze a good embedding geometry) and check the system snaps back to the
theory's clean behavior.

  clean toy:    python micro_compare.py --vocab 1024              --out mc_clean.json
  industrial:   python micro_compare.py --vocab 8192 --zipf       --out mc_indus.json
  purified:     python micro_compare.py --vocab 8192 --zipf --init_emb <pythia> --freeze_emb --out mc_pure.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoModelForCausalLM

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def zipf_probs(vocab, s=1.07):
    p = 1.0 / np.power(np.arange(1, vocab + 1), s); return p / p.sum()


def train_batch(B, T, vocab, nu, rng, device, probs=None):
    half = T // 2
    if probs is None:
        R = rng.integers(0, vocab, (B, half), dtype=np.int64); rand2 = rng.integers(0, vocab, (B, half), dtype=np.int64)
    else:
        R = rng.choice(vocab, (B, half), p=probs).astype(np.int64); rand2 = rng.choice(vocab, (B, half), p=probs).astype(np.int64)
    copy = rng.random((B, half)) < nu
    second = np.where(copy, R, rand2)
    return torch.from_numpy(np.concatenate([R, second], axis=1)).to(device)


def probe(T, vocab, device, probs, seed=12345, n=128):
    rng = np.random.default_rng(seed); half = T // 2
    R = (rng.integers(0, vocab, (n, half), dtype=np.int64) if probs is None
         else rng.choice(vocab, (n, half), p=probs).astype(np.int64))
    return np.concatenate([R, R], axis=1)


@torch.no_grad()
def induction_score(model, x, device):
    half = x.shape[1] // 2; xt = torch.from_numpy(x).to(device)
    pred = model(xt).logits[:, half:2 * half - 1].argmax(-1)
    tgt = torch.from_numpy(x[:, half + 1:2 * half]).to(device)
    return float((pred == tgt).float().mean())


@torch.no_grad()
def qk_prev(model, x, device):
    """max over heads/layers of the induction prefix-match: attention from 2nd-block pos to (its 1st-block match +1)."""
    half = x.shape[1] // 2; xt = torch.from_numpy(x).to(device)
    att = model(xt, output_attentions=True).attentions      # tuple[L] of (B,H,T,T)
    idx = torch.arange(half + 1, 2 * half - 1, device=device)       # query positions in 2nd block
    tgt = idx - half + 1                                            # induction target = match position + 1
    best = 0.0
    for a in att:
        v = a[:, :, idx, tgt].mean(dim=(0, 2))              # (H,) avg prefix-match per head
        best = max(best, float(v.max()))
    return best


def c_copy_now(model, k, device, sample=1500, steps=400, lr=5e-3):
    """rank-k OV copy capacity of the live embeddings, measured on HELD-OUT tokens (fit on tr, eval on disjoint te) so a
    rank-k map cannot fake it by memorizing the fit set -- the true GENERALIZING geometry quality."""
    WE = model.gpt_neox.embed_in.weight.detach(); WU = model.embed_out.weight.detach()
    V, d = WE.shape; rng = np.random.default_rng(7)
    n = min(sample, V // 2); perm = rng.permutation(V)
    tr = torch.from_numpy(perm[:n]).to(device); te = torch.from_numpy(perm[n:2 * n]).to(device)  # disjoint held-out
    A = torch.zeros(d, k, device=device, requires_grad=True); B = torch.zeros(d, k, device=device, requires_grad=True)
    torch.nn.init.normal_(A, std=0.02); torch.nn.init.normal_(B, std=0.02)
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(((WE[tr] @ B) @ A.t()) @ WU.t(), tr); loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        return float(((((WE[te] @ B) @ A.t()) @ WU.t()).argmax(1) == te).float().mean())  # held-out copy-fidelity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024); ap.add_argument("--zipf", action="store_true")
    ap.add_argument("--nu", type=float, default=0.4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=128); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=12000); ap.add_argument("--ee", type=int, default=100)
    ap.add_argument("--init_emb", default=None); ap.add_argument("--freeze_emb", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); k = args.hidden // args.heads
    probs = zipf_probs(args.vocab) if args.zipf else None
    cfg = GPTNeoXConfig(vocab_size=args.vocab, hidden_size=args.hidden, num_hidden_layers=args.layers,
                        num_attention_heads=args.heads, intermediate_size=4 * args.hidden,
                        max_position_embeddings=args.T + 2, use_parallel_residual=True, rotary_pct=0.25,
                        attn_implementation="eager")
    model = GPTNeoXForCausalLM(cfg).to(device)
    if args.init_emb:
        pm = AutoModelForCausalLM.from_pretrained(args.init_emb, revision="step143000", dtype=torch.float32)
        with torch.no_grad():
            model.gpt_neox.embed_in.weight.copy_(pm.gpt_neox.embed_in.weight.to(device))
            model.embed_out.weight.copy_(pm.embed_out.weight.to(device))
        if args.freeze_emb:
            model.gpt_neox.embed_in.weight.requires_grad_(False); model.embed_out.weight.requires_grad_(False)
        del pm; torch.cuda.empty_cache()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    pb = probe(args.T, args.vocab, device, probs)
    print(f"device={device} V={args.vocab} zipf={args.zipf} init_emb={args.init_emb} freeze={args.freeze_emb}", flush=True)

    traj = []; tstar = None
    print(f"\n  {'step':>6} {'induc':>6} {'C_copy':>7} {'qk_prev':>8}", flush=True)
    for s in range(0, args.max_steps + 1):
        if s % args.ee == 0:
            model.eval()
            ind = induction_score(model, pb, device); qk = qk_prev(model, pb, device); cc = c_copy_now(model, k, device)
            traj.append(dict(step=s, induction=ind, c_copy=cc, qk_prev=qk))
            if tstar is None and ind > 0.5:
                tstar = s
            print(f"  {s:>6} {ind:>6.2f} {cc:>7.3f} {qk:>8.3f}" + ("  <-- t*" if s == tstar else ""), flush=True)
            json.dump({"vocab": args.vocab, "zipf": args.zipf, "init_emb": args.init_emb, "freeze": args.freeze_emb,
                       "tstar": tstar, "traj": traj}, open(os.path.join(RESULTS, args.out), "w"), indent=2)
            model.train()
            if tstar is not None and s >= tstar + 300:
                break
        x = train_batch(args.batch, args.T, args.vocab, args.nu, rng, device, probs)
        out = model(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()

    print(f"\n  ===== V={args.vocab} zipf={args.zipf}: t*={tstar} ; final C_copy/qk = "
          f"{traj[-1]['c_copy']:.3f}/{traj[-1]['qk_prev']:.3f} =====", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
