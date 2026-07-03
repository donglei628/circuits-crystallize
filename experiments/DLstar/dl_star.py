"""
DL-STAR (strengthen ②: the ΔL* driving-force law with >=4 points). The rate law says formation time depends on the
driving force ΔL* (the loss the circuit removes) as t* ~ exp(b/ΔL*) (a small drive => exponentially longer wait). We only
had 2 ΔL* points. Here we vary ΔL* by the repeat RELIABILITY rho (second copy matches the first w.p. rho, else uniform),
which monotonically sets how much loss induction can remove: ΔL* = log(V) - H(target | match), a precomputable DATA
statistic. We measure t* (clean-probe formation) at 5 rho values and fit t* vs ΔL* to pin the functional form.

  python dl_star.py --smoke
  python dl_star.py --rhos 0.35 0.5 0.65 0.8 1.0 --seeds 6
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


def delta_L_star(rho, V):
    """precomputable driving force: loss removed by perfect induction = log V - H(target|match)."""
    p_copy = rho + (1 - rho) / V; p_other = (1 - rho) / V
    H = -(p_copy * np.log(p_copy) + (V - 1) * p_other * np.log(p_other + 1e-12))
    return float(np.log(V) - H)


def noisy_batch(B, Bk, vocab, rho, device, rng):
    b1 = rng.integers(0, vocab, (B, Bk)); keep = rng.random((B, Bk)) < rho
    b2 = np.where(keep, b1, rng.integers(0, vocab, (B, Bk)))
    return torch.from_numpy(np.concatenate([b1, b2], 1).astype(np.int64)).to(device)


@torch.no_grad()
def clean_score(model, Bk, vocab, device, seed):
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (256, Bk)); ids = np.concatenate([blk, blk], 1)
    x = torch.from_numpy(ids).to(device)
    pred = model(x)[:, Bk:2 * Bk - 1].argmax(-1); tgt = torch.from_numpy(ids[:, Bk + 1:2 * Bk]).to(device)
    return float((pred == tgt).float().mean())


def tstar(rho, vocab, d, H, L, Bk, steps, batch, lr, seed, device, thr=0.6, ee=25):
    rng = np.random.default_rng(seed); model = Tf(vocab, d, H, L, 2 * Bk + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(1, steps + 1):
        x = noisy_batch(batch, Bk, vocab, rho, device, rng)
        F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
        opt.step(); opt.zero_grad()
        if s % ee == 0 and clean_score(model, Bk, vocab, device, 7000 + s) >= thr:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.35, 0.5, 0.65, 0.8, 1.0])
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3); ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=9000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="dl_star.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.rhos = [0.5, 1.0]; args.seeds = 3; args.steps = 6000
    print(f"device={device} rhos={args.rhos} V={args.vocab} (DL-STAR: t* vs driving force ΔL*)", flush=True)

    res = {}
    print(f"\n  {'rho':>5} {'dL*':>6} {'t*':>6} {'n':>3}", flush=True)
    for rho in args.rhos:
        dL = delta_L_star(rho, args.vocab)
        ts = [t for s in range(args.seeds)
              if (t := tstar(rho, args.vocab, args.d, args.heads, args.layers, args.block, args.steps, args.batch, args.lr, s, device)) is not None]
        tm = float(np.median(ts)) if ts else None
        res[f"{rho}"] = dict(rho=rho, dLstar=dL, tstar=tm, n=len(ts))
        print(f"  {rho:>5.2f} {dL:>6.2f} {str(int(tm)) if tm else 'None':>6} {len(ts):>3}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    pts = [(r["dLstar"], r["tstar"]) for r in res.values() if r["tstar"]]
    print(f"\n  ===== driving-force law: t* vs ΔL* =====", flush=True)
    if len(pts) >= 3:
        dl = np.array([p[0] for p in pts]); t = np.array([p[1] for p in pts])
        # form A: ln t* = a + b/ΔL*  (exponential gating)
        a1, b1 = np.polyfit(1 / dl, np.log(t), 1)
        r2a = 1 - ((np.log(t) - (a1 / dl + b1)) ** 2).sum() / max(((np.log(t) - np.log(t).mean()) ** 2).sum(), 1e-9)
        # form B: ln t* = a + b*ln(ΔL*)  (power law)
        c1, c0 = np.polyfit(np.log(dl), np.log(t), 1)
        r2b = 1 - ((np.log(t) - (c1 * np.log(dl) + c0)) ** 2).sum() / max(((np.log(t) - np.log(t).mean()) ** 2).sum(), 1e-9)
        print(f"  exp-gating  ln t* = {b1:.2f} + {a1:.2f}/ΔL*    R2={r2a:.2f}", flush=True)
        print(f"  power-law   ln t* = {c0:.2f} + {c1:.2f}ln(ΔL*)  R2={r2b:.2f}", flush=True)
        print(f"  => {'exp(b/ΔL*) multiplicative gating wins' if r2a > r2b else 'power-law wins'} "
              f"(t* grows as ΔL* shrinks)", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
