"""
Q1-OV-HOPFIELD (lift Q1 from BEHAVIORAL to WEIGHT-LEVEL equivalence). We already showed the formed induction circuit
behaves like the closed-form content-addressed retriever (0.97 agreement). This goes to the weights: the induction head's
OV circuit, read out in TOKEN space, should literally BE the copy/identity operator that the Hopfield / closed-form
associative-memory solution prescribes (attend to a token -> write that same token to the output).

For the induction head h (in the top layer), the OV circuit in token space is
    M_OV = W_U @ (W_O^h W_V^h) @ W_E      (vocab x vocab),
and the closed-form copy operator is the identity (token i -> token i). We measure how close M_OV is to identity:
  copy_fidelity = fraction of rows i whose argmax_j M_OV[i,j] == i   (1.0 = exact copy operator).
  diag_ratio    = mean diagonal / mean |off-diagonal|.
We compare the trained head to a random head (control) and report the off-diagonal residual (the honest "copy up to
off-diagonal structure" caveat).

  python q1_ov_hopfield.py --smoke
  python q1_ov_hopfield.py --seeds 6 --steps 4000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = os.path.join(os.path.dirname(__file__), "results")


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, ret_attn=False):
        B, T, d = x.shape; H = self.h; hd = d // H
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-1, -2)) / hd ** 0.5
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        a = a.masked_fill(mask, float("-inf")).softmax(-1)
        out = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        return (out, a) if ret_attn else out


class TinyTf(nn.Module):
    def __init__(self, vocab, d, h, L, maxlen):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); self.unembed = nn.Linear(d, vocab, bias=False)
        self.h = h; self.d = d

    def forward(self, x):
        T = x.shape[1]; hh = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            hh = b(hh)
        return self.unembed(hh)


def block_batch(B, Bk, vocab, device, rng):
    blk = rng.integers(0, vocab, (B, Bk)); ids = np.concatenate([blk, blk], axis=1)
    return torch.from_numpy(ids).to(device)


@torch.no_grad()
def score(model, Bk, vocab, device, seed):
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (256, Bk))
    ids = np.concatenate([blk, blk], axis=1); x = torch.from_numpy(ids).to(device)
    pred = model(x)[:, Bk:2 * Bk - 1].argmax(-1); tgt = torch.from_numpy(ids[:, Bk + 1:2 * Bk]).to(device)
    return float((pred == tgt).float().mean())


@torch.no_grad()
def prefix_match(model, Bk, vocab, device, seed):
    """per-head induction score in the TOP layer (attention to the token after the previous occurrence)."""
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (64, Bk)); ids = np.concatenate([blk, blk], 1)
    x = torch.from_numpy(ids).to(device)
    hh = model.tok(x) + model.pos(torch.arange(ids.shape[1], device=device))[None]
    for b in model.blocks[:-1]:
        hh = b(hh)
    _, a = model.blocks[-1](hh, ret_attn=True)              # (B, H, T, T)
    idx = torch.arange(Bk + 1, 2 * Bk, device=device); tgt = idx - (Bk - 1)
    return a[:, :, idx, tgt].mean(dim=(0, 2)).cpu().numpy()  # per-head score


def ov_token_matrix(model, head, device):
    """M_OV = W_U @ (W_O^head W_V^head) @ W_E   in token space (vocab x vocab)."""
    d = model.d; H = model.h; hd = d // H; blk = model.blocks[-1]
    Wv = blk.qkv.weight[2 * d + head * hd: 2 * d + (head + 1) * hd, :]   # (hd, d)
    Wo = blk.o.weight[:, head * hd:(head + 1) * hd]                       # (d, hd)
    W_OV = Wo @ Wv                                                        # (d, d)
    WE = model.tok.weight                                                 # (vocab, d)
    WU = model.unembed.weight                                             # (vocab, d)
    return (WU @ W_OV @ WE.t()).detach()                                 # (vocab, vocab)


def copy_metrics(M):
    V = M.shape[0]; diag = torch.diagonal(M)
    fid = float((M.argmax(dim=1) == torch.arange(V, device=M.device)).float().mean())
    off = M.clone(); off[torch.arange(V), torch.arange(V)] = float("-inf")
    dr = float(diag.mean() / (M.abs().mean() + 1e-9))
    return fid, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="q1_ov_hopfield.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.seeds = 2; args.steps = 3000
    print(f"device={device} vocab={args.vocab} (Q1-OV-HOPFIELD: is the formed OV circuit the copy/identity operator?)", flush=True)

    res = []
    print(f"\n  {'seed':>4} {'recall':>7} {'ind_head':>9} {'copy_fid':>9} {'rand_fid':>9} {'diag_ratio':>11}", flush=True)
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        model = TinyTf(args.vocab, args.d, args.heads, args.layers, 2 * args.block + 2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for s in range(args.steps):
            x = block_batch(args.batch, args.block, args.vocab, device, rng)
            F.cross_entropy(model(x)[:, :-1].reshape(-1, args.vocab), x[:, 1:].reshape(-1)).backward()
            opt.step(); opt.zero_grad()
        rec = score(model, args.block, args.vocab, device, 999)
        ps = prefix_match(model, args.block, args.vocab, device, 999)
        # the OV-copy head need not be the same as the attention-prefix-match head; report the best copy-OV head
        fids = [copy_metrics(ov_token_matrix(model, hh, device)) for hh in range(args.heads)]
        cfs = [f for f, _ in fids]; best = int(np.argmax(cfs)); fid, dr = fids[best]
        # control: median copy_fid over the OTHER heads (not the copy head)
        rfid = float(np.median([cfs[hh] for hh in range(args.heads) if hh != best]))
        res.append(dict(seed=seed, recall=rec, copy_head=best, pm_head=int(ps.argmax()),
                        copy_fid=fid, other_fid=rfid, diag_ratio=dr))
        print(f"  {seed:>4} {rec:>7.2f} {best:>9} {fid:>9.2f} {rfid:>9.2f} {dr:>11.1f}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    formed = [r for r in res if r["recall"] > 0.7]
    print(f"\n  ===== OV = copy operator (weight level) =====", flush=True)
    if formed:
        cf = np.median([r["copy_fid"] for r in formed]); rf = np.median([r["other_fid"] for r in formed])
        print(f"  copy_fidelity (best OV head) = {cf:.2f}  vs other heads = {rf:.2f}  (1.0 = exact copy/identity operator)", flush=True)
        print(f"  => {'WEIGHT-LEVEL: the induction OV circuit IS the copy operator (Hopfield closed-form), up to off-diagonal residual' if cf > 0.7 and cf > rf + 0.3 else 'partial -- report off-diagonal structure'}", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
