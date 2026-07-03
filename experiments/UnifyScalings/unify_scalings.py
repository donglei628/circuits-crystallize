"""
UNIFY-SCALINGS (unification: does our nucleation t* CONTAIN the saddle-to-saddle and scan-and-snap timing laws?). Two
prior theories predict when a circuit/attention forms, each via a different knob:
  * saddle-to-saddle (Pesme-Flammarion): with initialization scale alpha->0, escape time t ~ ln(1/alpha).
  * scan-and-snap (Tian 2023): attention concentration time t0 ~ ln(M)/eta  (M = context length, eta = learning rate).
Our nucleation rate law says t* = (barrier)/(rate). If our t* in the SAME toy follows ln(1/alpha) when we vary init,
1/eta when we vary lr, and ln(L) when we vary context length, then our t* literally contains both prior laws as special
parameter dependences -- a concrete unification, not just a verbal analogy.

We sweep three knobs independently on the standard repeated-block induction toy and fit:
  init scale alpha -> t* vs ln(1/alpha)   (saddle-to-saddle)
  lr eta           -> t* vs 1/eta         (scan-and-snap, lr)
  context L        -> t* vs ln(L)         (scan-and-snap, context)

  python unify_scalings.py --smoke
  python unify_scalings.py --seeds 5
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
    def __init__(self, vocab, d, h, L, maxlen, init_scale=1.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); self.unembed = nn.Linear(d, vocab, bias=False)
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(init_scale)

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


def tstar(vocab, d, H, L, Bk, alpha, lr, steps, batch, seed, device, thr=0.7, ee=10):
    rng = np.random.default_rng(seed)
    model = Tf(vocab, d, H, L, 2 * Bk + 2, init_scale=alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(1, steps + 1):
        x = block_batch(batch, Bk, vocab, device, rng)
        F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
        opt.step(); opt.zero_grad()
        if s % ee == 0 and score(model, Bk, vocab, device, 7000 + s) >= thr:
            return s
    return None


def med(fn, seeds):
    ts = [t for s in range(seeds) if (t := fn(s)) is not None]
    return float(np.median(ts)) if ts else None


def lin_r2(x, y):
    x = np.array(x, float); y = np.array(y, float)
    a, b = np.polyfit(x, y, 1); yh = a * x + b
    return a, b, 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.25, 0.1, 0.05, 0.02])
    ap.add_argument("--etas", type=float, nargs="+", default=[4e-3, 2e-3, 1e-3, 5e-4])
    ap.add_argument("--Ls", type=int, nargs="+", default=[6, 12, 24, 48])
    ap.add_argument("--vocab", type=int, default=1024); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="unify_scalings.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.alphas = [0.5, 0.1]; args.etas = [4e-3, 1e-3]; args.Ls = [6, 24]; args.seeds = 3; args.steps = 4000
    d, H, Lyr = args.d, args.heads, args.layers
    print(f"device={device} (UNIFY: t* vs init-alpha / lr-eta / context-L -> saddle + scan-snap)", flush=True)
    res = {"alpha": {}, "eta": {}, "L": {}}

    print(f"\n  -- saddle-to-saddle: vary init alpha (eta=2e-3, L=12) --\n  {'alpha':>6} {'t*':>6}", flush=True)
    for al in args.alphas:
        t = med(lambda s: tstar(args.vocab, d, H, Lyr, 12, al, 2e-3, args.steps, args.batch, s, device), args.seeds)
        res["alpha"][str(al)] = t; print(f"  {al:>6.2f} {str(int(t)) if t else 'None':>6}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)

    print(f"\n  -- scan-and-snap (lr): vary eta (alpha=0.1, L=12) --\n  {'eta':>7} {'t*':>6}", flush=True)
    for et in args.etas:
        t = med(lambda s: tstar(args.vocab, d, H, Lyr, 12, 0.1, et, args.steps, args.batch, s, device), args.seeds)
        res["eta"][str(et)] = t; print(f"  {et:>7.0e} {str(int(t)) if t else 'None':>6}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)

    print(f"\n  -- scan-and-snap (context): vary L=block (alpha=0.1, eta=2e-3) --\n  {'L':>5} {'t*':>6}", flush=True)
    for Lb in args.Ls:
        t = med(lambda s: tstar(args.vocab, d, H, Lyr, Lb, 0.1, 2e-3, args.steps, args.batch, s, device), args.seeds)
        res["L"][str(Lb)] = t; print(f"  {Lb:>5} {str(int(t)) if t else 'None':>6}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)

    print(f"\n  ===== unification fits =====", flush=True)
    a = [(al, t) for al, t in res["alpha"].items() if t]
    if len(a) >= 3:
        s, b, r2 = lin_r2([np.log(1 / float(al)) for al, _ in a], [t for _, t in a])
        print(f"  SADDLE   t* vs ln(1/alpha): slope={s:.0f} R2={r2:.2f}  {'<- ln(1/alpha) scaling (saddle-to-saddle)' if r2 > 0.8 else '(weak)'}", flush=True)
    e = [(et, t) for et, t in res["eta"].items() if t]
    if len(e) >= 3:
        s, b, r2 = lin_r2([1 / float(et) for et, _ in e], [t for _, t in e])
        print(f"  SCANSNAP t* vs 1/eta:       slope={s:.2e} R2={r2:.2f}  {'<- 1/eta scaling (scan-and-snap lr)' if r2 > 0.8 else '(weak)'}", flush=True)
    l = [(Lb, t) for Lb, t in res["L"].items() if t]
    if len(l) >= 3:
        s, b, r2 = lin_r2([np.log(float(Lb)) for Lb, _ in l], [t for _, t in l])
        print(f"  SCANSNAP t* vs ln(L):       slope={s:.0f} R2={r2:.2f}  {'<- ln(L) scaling (scan-and-snap context)' if r2 > 0.8 else '(weak)'}", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
