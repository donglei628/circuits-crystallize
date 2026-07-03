"""
K-CONJUNCTION (split C_circuit: measure how the barrier M* grows with conjunction depth K). The 288x C_circuit residual
is suspected to be dominated by the conjunction depth (induction = a 3-way AND). The earlier k-gram MQAR knob failed to
train. Here we use a TRAINABLE conjunction task: the target token is a fixed function of K specific earlier positions,

    target[t] = ( x[t-d_1] + x[t-d_2] + ... + x[t-d_K] )  mod V

so the circuit must attend to AND combine K positions at once -- conjunction depth = K -- but it is POSITIONAL (fixed
offsets), which trains far better than content retrieval. K=1 is plain offset-copy. We sweep K, measure the formation
step t* and the barrier M* = t* * tokens/step, and test the combinatorial prediction ln(M*) ~ linear in K (i.e. each
extra conjunct multiplies the barrier by a constant 1/p). If so, we have isolated K's contribution to C_circuit.

  python k_conjunction.py --smoke
  python k_conjunction.py --Ks 1 2 3 4 --seeds 5
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OFFSETS = [2, 4, 6, 8, 10, 12]


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


def make_batch(B, T, V, offs, device, rng):
    """x random; target[t] = sum_{d in offs} x[t-d] mod V (only valid for t >= max(offs))."""
    x = rng.integers(0, V, (B, T))
    tgt = np.full((B, T), -100, dtype=np.int64)        # -100 = ignore_index
    mo = max(offs)
    acc = np.zeros((B, T), dtype=np.int64)
    for d in offs:
        acc[:, mo:] += x[:, mo - d:T - d]
    tgt[:, mo:] = acc[:, mo:] % V
    return torch.from_numpy(x.astype(np.int64)).to(device), torch.from_numpy(tgt).to(device)


@torch.no_grad()
def acc_score(model, T, V, offs, device, seed):
    rng = np.random.default_rng(seed); x, tgt = make_batch(256, T, V, offs, device, rng)
    pred = model(x).argmax(-1)
    m = tgt != -100
    return float((pred[m] == tgt[m]).float().mean())


def train_to_form(K, V, d, H, L, T, steps, batch, lr, seed, device, thr=0.9, ee=25):
    offs = OFFSETS[:K]; rng = np.random.default_rng(seed)
    model = Tf(V, d, H, L, T + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(1, steps + 1):
        x, tgt = make_batch(batch, T, V, offs, device, rng)
        F.cross_entropy(model(x).reshape(-1, V), tgt.reshape(-1), ignore_index=-100).backward()
        opt.step(); opt.zero_grad()
        if s % ee == 0 and acc_score(model, T, V, offs, device, 7000 + s) >= thr:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--V", type=int, default=16); ap.add_argument("--T", type=int, default=28)
    ap.add_argument("--hd", type=int, default=64); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="k_conjunction.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.Ks = [1, 2]; args.seeds = 3; args.steps = 5000
    d = args.hd * args.heads; tps = args.batch * args.T
    print(f"device={device} Ks={args.Ks} V={args.V} d={d} (K-CONJUNCTION: does ln(M*) grow linearly in K?)", flush=True)

    res = {}
    print(f"\n  {'K':>3} {'offs':>14} {'t*':>6} {'M*=t*·tok':>11} {'n':>3}", flush=True)
    for K in args.Ks:
        ts = [t for s in range(args.seeds)
              if (t := train_to_form(K, args.V, d, args.heads, args.layers, args.T, args.steps, args.batch, args.lr, s, device)) is not None]
        tm = float(np.median(ts)) if ts else None
        Ms = tm * tps if tm else None
        res[f"{K}"] = dict(K=K, offs=OFFSETS[:K], tstar=tm, Mstar=Ms, n=len(ts), seeds=args.seeds)
        print(f"  {K:>3} {str(OFFSETS[:K]):>14} {str(int(tm)) if tm else 'None':>6} {str(int(Ms)) if Ms else 'None':>11} {len(ts):>3}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    pts = [(r["K"], r["Mstar"]) for r in res.values() if r["Mstar"]]
    print(f"\n  ===== conjunction barrier law =====", flush=True)
    if len(pts) >= 2:
        Kx = np.array([p[0] for p in pts]); lnM = np.log([p[1] for p in pts])
        a, b = np.polyfit(Kx, lnM, 1); r2 = 1 - ((lnM - (a * Kx + b)) ** 2).sum() / max(((lnM - lnM.mean()) ** 2).sum(), 1e-9)
        print(f"  ln(M*) = {b:.2f} + {a:.2f}*K   => each extra conjunct multiplies barrier by e^{a:.2f} = {np.exp(a):.1f}x  (R2={r2:.2f})", flush=True)
        print(f"  => extrapolate K=1->3: barrier grows {np.exp(a*2):.0f}x  (target: explain part of the 80x residual)", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
