"""
MORE-CIRCUITS (generality: does the barrier law M*~vocab^a hold for MORE circuit types, with their own constants?). We
have induction (vocab^0.53) and offset-copy (vocab^0.43). To claim the calculator is general we add more distinct circuit
computations and check the SAME power-law FORM holds (different exponent/prefactor = per-circuit constants):
  first_copy : target[t] = x[0]                 (fixed-position attention, K=1)
  induction  : repeated block, copy-after-match (K~2-3, content addressed)
  offset5    : target[t] = x[t-5]               (relative-position copy)
  maxtok     : target[t] = max(x[0..t])         (a comparison/aggregation circuit, not a copy)

  python more_circuits.py --circuit first_copy --smoke
  python more_circuits.py --circuit maxtok --vocabs 256 1024 4096 --seeds 5
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
        self.ln = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, d = x.shape; H = self.h; hd = d // H
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-1, -2)) / hd ** 0.5
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        a = a.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class Tf(nn.Module):
    def __init__(self, vocab, d, h, L, maxlen):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); self.unembed = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        T = x.shape[1]; hh = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            hh = b(hh)
        return self.unembed(hh)


def gen(circuit, B, T, V, device, rng):
    """return (x, target) with target = -100 where undefined."""
    if circuit == "induction":
        Bk = T // 2; blk = rng.integers(0, V, (B, Bk)); x = np.concatenate([blk, blk], 1)
        tgt = np.full((B, 2 * Bk), -100, np.int64); tgt[:, Bk:2 * Bk - 1] = x[:, 1:Bk]    # next within 2nd copy
        return torch.from_numpy(x.astype(np.int64)).to(device), torch.from_numpy(tgt).to(device)
    x = rng.integers(0, V, (B, T)); tgt = np.full((B, T), -100, np.int64)
    if circuit == "first_copy":
        tgt[:, 1:] = x[:, 0:1]
    elif circuit == "offset5":
        k = 5; tgt[:, k:] = x[:, :T - k]
    elif circuit == "maxtok":
        tgt = np.maximum.accumulate(x, axis=1)                              # running max
        tgt[:, 0] = -100
    return torch.from_numpy(x.astype(np.int64)).to(device), torch.from_numpy(tgt).to(device)


@torch.no_grad()
def acc(model, circuit, T, V, device, seed):
    rng = np.random.default_rng(seed); x, tgt = gen(circuit, 256, T, V, device, rng)
    pred = model(x).argmax(-1); m = tgt != -100
    return float((pred[m] == tgt[m]).float().mean())


def tstar(circuit, V, d, H, L, T, steps, batch, lr, seed, device, thr=0.85, ee=25):
    rng = np.random.default_rng(seed); model = Tf(V, d, H, L, T + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(1, steps + 1):
        x, tgt = gen(circuit, batch, T, V, device, rng)
        F.cross_entropy(model(x).reshape(-1, V), tgt.reshape(-1), ignore_index=-100).backward()
        opt.step(); opt.zero_grad()
        if s % ee == 0 and acc(model, circuit, T, V, device, 7000 + s) >= thr:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="first_copy", choices=["first_copy", "induction", "offset5", "maxtok"])
    ap.add_argument("--vocabs", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--T", type=int, default=24); ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=None); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.vocabs = [256, 4096]; args.seeds = 3; args.steps = 6000
    out = args.out or f"more_{args.circuit}.json"; d = args.d; tps = args.batch * args.T
    print(f"device={device} circuit={args.circuit} vocabs={args.vocabs} (MORE-CIRCUITS: M*~vocab^a?)", flush=True)

    res = {}
    print(f"\n  {'vocab':>6} {'t*':>6} {'M*':>10} {'n':>3}", flush=True)
    for V in args.vocabs:
        ts = [t for s in range(args.seeds)
              if (t := tstar(args.circuit, V, d, args.heads, args.layers, args.T, args.steps, args.batch, args.lr, s, device)) is not None]
        tm = float(np.median(ts)) if ts else None; Ms = tm * tps if tm else None
        res[f"{V}"] = dict(vocab=V, tstar=tm, Mstar=Ms, n=len(ts))
        print(f"  {V:>6} {str(int(tm)) if tm else 'None':>6} {str(int(Ms)) if Ms else 'None':>10} {len(ts):>3}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, out), "w"), indent=2, default=float)

    pts = [(r["vocab"], r["Mstar"]) for r in res.values() if r["Mstar"]]
    print(f"\n  ===== {args.circuit} barrier law (cf. induction 0.53, offset-copy 0.43) =====", flush=True)
    if len(pts) >= 2:
        X = np.log([p[0] for p in pts]); Y = np.log([p[1] for p in pts])
        a, b = np.polyfit(X, Y, 1); r2 = 1 - ((Y - (a * X + b)) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-9)
        print(f"  M*({args.circuit}) ~ vocab^{a:.2f}  (prefactor {np.exp(b):.2e}, R2={r2:.2f})  "
              f"{'<- same power-law FORM, own constants' if r2 > 0.7 else '(noisy)'}", flush=True)
    print(f"\n  saved results/{out}", flush=True)


if __name__ == "__main__":
    main()
