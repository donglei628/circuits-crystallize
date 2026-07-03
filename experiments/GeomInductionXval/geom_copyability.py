"""
QUANTIFY THE DIFFICULTY: the barrier ladder localized the formation difficulty to the embedding geometry's
"copy-ability". Here we turn that into a COMPUTABLE NUMBER, from the embeddings alone (no full-model training):

  C_copy(W_E, W_U, k) = the best copy-fidelity achievable by a rank-k (= head_dim) OV operator L, i.e. the fraction of
  tokens X for which argmax_Y u_Y . (L e_X) = X, where L = A B^T is rank-k. We fit only this single low-rank linear map
  (frozen embeddings) -- a pure property of the token geometry. If C_copy predicts the ladder's formation time t*
  (random-50k unable to copy -> circuit can't form; Pythia-50k copies -> circuit forms fast), the difficulty is
  quantified by a quantity readable from the embeddings.

  python geom_copyability.py --vocab 50304 --hidden 768 --init_emb <local pythia path> --out geom_R3.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoModelForCausalLM

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def get_embeddings(args, device):
    cfg = GPTNeoXConfig(vocab_size=args.vocab, hidden_size=args.hidden, num_hidden_layers=1,
                        num_attention_heads=args.heads, intermediate_size=4 * args.hidden,
                        max_position_embeddings=8, use_parallel_residual=True, rotary_pct=0.25)
    if args.init_emb:                                        # real (e.g. Pythia) trained geometry
        pm = AutoModelForCausalLM.from_pretrained(args.init_emb, dtype=torch.float32)
        WE = pm.gpt_neox.embed_in.weight.detach().to(device)        # (V, d)
        WU = pm.embed_out.weight.detach().to(device)                # (V, d)
        del pm
    else:                                                   # fresh GPTNeoX random init (matches the ladder's R0/R1/R2)
        torch.manual_seed(args.seed); m = GPTNeoXForCausalLM(cfg)
        WE = m.gpt_neox.embed_in.weight.detach().to(device); WU = m.embed_out.weight.detach().to(device)
        del m
    return WE, WU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=512); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--init_emb", default=None)
    ap.add_argument("--k", type=int, default=64, help="OV rank = head_dim (single induction head)")
    ap.add_argument("--sample", type=int, default=2000); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WE, WU = get_embeddings(args, device); V, d = WE.shape
    rng = np.random.default_rng(args.seed)
    tr = torch.from_numpy(rng.choice(V, size=min(args.sample, V), replace=False)).to(device)
    te = torch.from_numpy(rng.choice(V, size=min(args.sample, V), replace=False)).to(device)

    # fit a single rank-k OV operator L = A B^T to copy: u_Y . (L e_X) should peak at Y=X
    A = torch.zeros(d, args.k, device=device, requires_grad=True)
    B = torch.zeros(d, args.k, device=device, requires_grad=True)
    torch.nn.init.normal_(A, std=0.02); torch.nn.init.normal_(B, std=0.02)
    opt = torch.optim.Adam([A, B], lr=args.lr)
    print(f"device={device} V={V} d={d} k={args.k} emb={'real:'+args.init_emb if args.init_emb else 'random'}", flush=True)

    @torch.no_grad()
    def copy_fid(idx):
        out = (WE[idx] @ B) @ A.t()                         # (n, d) low-rank OV applied to e_X
        logits = out @ WU.t()                               # (n, V)
        return float((logits.argmax(1) == idx).float().mean())

    for s in range(args.steps + 1):
        out = (WE[tr] @ B) @ A.t(); logits = out @ WU.t()   # (n, V)
        loss = F.cross_entropy(logits, tr)
        if s % 300 == 0:
            print(f"  step {s:>5} loss {loss.item():.3f}  train_copyfid {copy_fid(tr):.3f}", flush=True)
        loss.backward(); opt.step(); opt.zero_grad()

    c_train = copy_fid(tr); c_test = copy_fid(te)
    res = dict(vocab=V, hidden=d, k=args.k, emb=("real:" + args.init_emb if args.init_emb else "random"),
               C_copy_train=c_train, C_copy_test=c_test)
    json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
    print(f"\n  ===== C_copy (rank-{args.k} copy capacity) = {c_test:.3f} (held-out tokens) =====", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
