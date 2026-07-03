"""
DATA-DRIVEN WHEN (the paper's "central next step"): test the rate law's DATA-driven timing prediction t* proportional to
1/nu on a REAL GPTNeoX architecture trained FROM SCRATCH. The real-LM evidence so far only shows SIZE-invariant timing
(t*=912+-44 across 70M-1B); the other half of the rate law -- how t* SHIFTS when the data statistic nu changes -- has
not been tested on a real architecture. Here nu = the fraction of induction-predictable tokens in the training stream
(the same quantity measured as p_ind_margin=0.042 on the Pile). The nucleation rate law J proportional to N_sites*nu
predicts the induction-head formation time t* proportional to 1/nu. We sweep nu and fit.

  python data_driven_when.py --nu 0.25 --seed 0 --out ddw_nu025_s0.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def train_batch(B, T, vocab, nu, rng, device):
    """Induction-predictable stream: seq = [block R | second], where each second-half token copies the aligned R token
    with probability nu (induction-predictable), else random. nu controls the density of induction attempts."""
    half = T // 2
    R = rng.integers(0, vocab, size=(B, half), dtype=np.int64)
    rand2 = rng.integers(0, vocab, size=(B, half), dtype=np.int64)
    copy = rng.random((B, half)) < nu
    second = np.where(copy, R, rand2)
    x = np.concatenate([R, second], axis=1)
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def induction_score(model, T, vocab, device, seed=12345):
    """copy-score on a held-out FULLY-repeated probe (nu=1): fraction of 2nd-half positions predicted correctly."""
    rng = np.random.default_rng(seed); half = T // 2
    R = rng.integers(0, vocab, size=(256, half), dtype=np.int64)
    x = np.concatenate([R, R], axis=1)
    xt = torch.from_numpy(x).to(device)
    logits = model(xt).logits
    pred = logits[:, half:2 * half - 1].argmax(-1)            # predict 2nd-half tokens
    tgt = torch.from_numpy(x[:, half + 1:2 * half]).to(device)
    return float((pred == tgt).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, required=True, help="induction density (copy prob in 2nd half)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=512); ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--ee", type=int, default=50); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    cfg = GPTNeoXConfig(vocab_size=args.vocab, hidden_size=args.hidden, num_hidden_layers=args.layers,
                        num_attention_heads=args.heads, intermediate_size=4 * args.hidden,
                        max_position_embeddings=args.T + 2, use_parallel_residual=True, rotary_pct=0.25)
    model = GPTNeoXForCausalLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"device={device} nu={args.nu} seed={args.seed} {args.layers}L x{args.heads}H d{args.hidden} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    traj = []; tstar = None
    print(f"\n  {'step':>6} {'ind':>6}", flush=True)
    for s in range(0, args.max_steps + 1):
        if s % args.ee == 0:
            model.eval(); ind = induction_score(model, args.T, args.vocab, device)
            traj.append((s, ind))
            if tstar is None and ind > args.thresh:
                tstar = s
                print(f"  {s:>6} {ind:>6.2f}  <-- t* (induction formed)", flush=True)
            elif s % (args.ee * 4) == 0:
                print(f"  {s:>6} {ind:>6.2f}", flush=True)
            json.dump({"nu": args.nu, "seed": args.seed, "tstar": tstar, "traj": traj,
                       "config": {"layers": args.layers, "hidden": args.hidden, "heads": args.heads, "vocab": args.vocab}},
                      open(os.path.join(RESULTS, args.out), "w"), indent=2)
            model.train()
            if tstar is not None and s >= tstar + 200:        # stop a bit after formation
                break
        x = train_batch(args.batch, args.T, args.vocab, args.nu, rng, device)
        out = model(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()

    print(f"\n  ===== nu={args.nu} seed={args.seed}: t*={tstar} (censored if None) =====", flush=True)
    print(f"  rate-law prediction: t* proportional to 1/nu -> t**nu = {tstar*args.nu if tstar else None}", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
