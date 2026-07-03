"""
SEED-COMPONENTS (deepen ④: per-component seed threshold s*). Transplanting a fraction s of a trained circuit's weights
accelerates re-nucleation above a joint threshold s*~0.18. Which COMPONENT is rate-limiting? We decompose the seed into
the induction circuit's parts and find s* for each separately:
  EMBED   : token + position embeddings
  QK_prev : layer-0 attention (the previous-token head's query/key)
  QK_ind  : layer-1 attention query/key (the induction match)
  OV_ind  : layer-1 attention value/output (the copy)
For each component C and seed strength s, we build a FRESH model with only component C interpolated toward a trained
donor (W = (1-s) W_rand + s W_donor) and measure the formation time t*(C, s). The component whose s* is LOWEST (seeding
it alone accelerates formation most / at smallest s) is the rate-limiting nucleus.

  python seed_components.py --smoke
  python seed_components.py --ss 0.0 0.1 0.2 0.35 0.6 1.0 --seeds 5
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

    def forward(self, x):
        B, T, d = x.shape; H = self.h; hd = d // H
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-1, -2)) / hd ** 0.5
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        a = a.masked_fill(mask, float("-inf")).softmax(-1)
        return x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))


class TinyTf(nn.Module):
    def __init__(self, vocab, d, h, L, maxlen):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); self.unembed = nn.Linear(d, vocab, bias=False)
        self.d = d

    def forward(self, x):
        T = x.shape[1]; hh = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            hh = b(hh)
        return self.unembed(hh)


def block_batch(B, Bk, vocab, device, rng):
    blk = rng.integers(0, vocab, (B, Bk)); ids = np.concatenate([blk, blk], 1)
    return torch.from_numpy(ids).to(device)


@torch.no_grad()
def score(model, Bk, vocab, device, seed):
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (256, Bk)); ids = np.concatenate([blk, blk], 1)
    x = torch.from_numpy(ids).to(device)
    pred = model(x)[:, Bk:2 * Bk - 1].argmax(-1); tgt = torch.from_numpy(ids[:, Bk + 1:2 * Bk]).to(device)
    return float((pred == tgt).float().mean())


def train_donor(vocab, d, H, L, Bk, steps, batch, lr, device):
    rng = np.random.default_rng(0); model = TinyTf(vocab, d, H, L, 2 * Bk + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(steps):
        x = block_batch(batch, Bk, vocab, device, rng)
        F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
        opt.step(); opt.zero_grad()
    return model


COMPONENTS = {
    "ALL":     lambda m: list(m.parameters()),               # joint seed (positive control: should accelerate ~s*0.18)
    "EMBED":   lambda m: [m.tok.weight, m.pos.weight],
    "QK_prev": lambda m: [m.blocks[0].qkv.weight],
    "QK_ind":  lambda m: [m.blocks[-1].qkv.weight],
    "OV_ind":  lambda m: [m.blocks[-1].o.weight],
}


def seed_and_train(donor, comp, s, vocab, d, H, L, Bk, steps, batch, lr, seed, device, thr=0.7, ee=25):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = TinyTf(vocab, d, H, L, 2 * Bk + 2).to(device)
    with torch.no_grad():                                   # interpolate ONE component toward the donor
        for pf, pd in zip(COMPONENTS[comp](model), COMPONENTS[comp](donor)):
            pf.mul_(1 - s).add_(s * pd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for st in range(1, steps + 1):
        x = block_batch(batch, Bk, vocab, device, rng)
        F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
        opt.step(); opt.zero_grad()
        if st % ee == 0 and score(model, Bk, vocab, device, 7000 + st) >= thr:
            return st
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.6, 1.0])
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="seed_components.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.ss = [0.0, 0.3, 1.0]; args.seeds = 3
    print(f"device={device} ss={args.ss} comps={list(COMPONENTS)} (SEED-COMPONENTS: per-component s*)", flush=True)
    donor = train_donor(args.vocab, args.d, args.heads, args.layers, args.block, 4000, args.batch, args.lr, device)
    print(f"  donor recall = {score(donor, args.block, args.vocab, device, 999):.2f}", flush=True)

    res = {}
    print(f"\n  {'comp':>8} " + " ".join(f"{'s='+str(s):>8}" for s in args.ss), flush=True)
    for comp in COMPONENTS:
        row = []
        for s in args.ss:
            ts = [t for sd in range(args.seeds)
                  if (t := seed_and_train(donor, comp, s, args.vocab, args.d, args.heads, args.layers, args.block,
                                          args.steps, args.batch, args.lr, sd, device)) is not None]
            row.append(float(np.median(ts)) if ts else None)
        res[comp] = {str(s): t for s, t in zip(args.ss, row)}
        print(f"  {comp:>8} " + " ".join(f"{(str(int(t)) if t else '-'):>8}" for t in row), flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    print(f"\n  ===== per-component seed threshold s* (s where t* drops to ~half of s=0) =====", flush=True)
    base = {c: res[c][str(args.ss[0])] for c in COMPONENTS}
    for comp in COMPONENTS:
        sstar = None
        for s in args.ss:
            t = res[comp][str(s)]
            if t and base[comp] and t <= 0.6 * base[comp]:
                sstar = s; break
        print(f"  {comp:>8}: s* = {sstar}  (t*@s=0 -> {base[comp]}, accelerates at s={sstar})", flush=True)
    print(f"  => the component with the LOWEST s* is the rate-limiting nucleus", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
