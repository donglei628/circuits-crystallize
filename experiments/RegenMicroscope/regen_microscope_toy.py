"""
REGEN-MICROSCOPE (toy): record the PROCESS of regeneration frame-by-frame, to test the whirlpool/basin picture. After
lesioning a formed induction circuit, we continue training and every few steps record, PER HEAD: the QK prefix-match
score (the "where to look" component) and the OV copy-fidelity (the "copy it" component), plus which head is the top-QK
and top-OV, and the full recall. The basin hypothesis predicts: in the toy (one basin) the QK and OV grow TOGETHER in
the SAME, STABLE head and recall recovers (the ball falls back into the one whirlpool). The contrast with Pythia (many
basins) is the point.

  python regen_microscope_toy.py
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
    blk = rng.integers(0, vocab, (B, Bk)); ids = np.concatenate([blk, blk], 1)
    return torch.from_numpy(ids).to(device)


@torch.no_grad()
def recall(model, Bk, vocab, device, seed):
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (256, Bk)); ids = np.concatenate([blk, blk], 1)
    x = torch.from_numpy(ids).to(device)
    pred = model(x)[:, Bk:2 * Bk - 1].argmax(-1); tgt = torch.from_numpy(ids[:, Bk + 1:2 * Bk]).to(device)
    return float((pred == tgt).float().mean())


@torch.no_grad()
def per_head_qk(model, Bk, vocab, device, seed):
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (64, Bk)); ids = np.concatenate([blk, blk], 1)
    x = torch.from_numpy(ids).to(device); hh = model.tok(x) + model.pos(torch.arange(2 * Bk, device=device))[None]
    for b in model.blocks[:-1]:
        hh = b(hh)
    _, a = model.blocks[-1](hh, ret_attn=True)
    idx = torch.arange(Bk + 1, 2 * Bk, device=device); tgt = idx - (Bk - 1)
    return a[:, :, idx, tgt].mean(dim=(0, 2)).cpu().numpy()      # per-head prefix-match (QK)


@torch.no_grad()
def per_head_ov(model, vocab, device):
    """per-head OV copy-fidelity in token space: frac of rows where argmax(M_OV)=diagonal."""
    d = model.d; H = model.h; hd = d // H; blk = model.blocks[-1]
    WE = model.tok.weight; WU = model.unembed.weight; fids = []
    for hh in range(H):
        Wv = blk.qkv.weight[2 * d + hh * hd:2 * d + (hh + 1) * hd, :]; Wo = blk.o.weight[:, hh * hd:(hh + 1) * hd]
        M = WU @ (Wo @ Wv) @ WE.t()
        fids.append(float((M.argmax(1) == torch.arange(vocab, device=device)).float().mean()))
    return np.array(fids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--donor_steps", type=int, default=3000)
    ap.add_argument("--regen_steps", type=int, default=800); ap.add_argument("--ee", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="regen_microscope_toy.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0); torch.manual_seed(0)
    model = TinyTf(args.vocab, args.d, args.heads, args.layers, 2 * args.block + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for s in range(args.donor_steps):                            # form the donor circuit
        x = block_batch(args.batch, args.block, args.vocab, device, rng)
        F.cross_entropy(model(x)[:, :-1].reshape(-1, args.vocab), x[:, 1:].reshape(-1)).backward(); opt.step(); opt.zero_grad()
    print(f"donor recall={recall(model, args.block, args.vocab, device, 9):.2f}", flush=True)

    with torch.no_grad():                                        # LESION: re-randomize all attention
        for b in model.blocks:
            b.qkv.weight.normal_(0, 0.02); b.o.weight.normal_(0, 0.02)
    print(f"after lesion recall={recall(model, args.block, args.vocab, device, 9):.2f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    traj = []
    print(f"\n  {'step':>5} {'recall':>7} {'topQK_h':>7} {'QK':>5} {'topOV_h':>7} {'OV':>5} {'same?':>5}", flush=True)
    for s in range(0, args.regen_steps + 1):
        if s % args.ee == 0:
            qk = per_head_qk(model, args.block, args.vocab, device, 9); ov = per_head_ov(model, args.vocab, device)
            tq = int(qk.argmax()); to = int(ov.argmax()); rec = recall(model, args.block, args.vocab, device, 9)
            traj.append(dict(step=s, recall=rec, topQK=tq, QK=float(qk[tq]), topOV=to, OV=float(ov[to]),
                             qk_all=qk.tolist(), ov_all=ov.tolist(), same=int(tq == to)))
            print(f"  {s:>5} {rec:>7.2f} {tq:>7} {qk[tq]:>5.2f} {to:>7} {ov[to]:>5.2f} {('Y' if tq==to else 'n'):>5}", flush=True)
            json.dump(traj, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        if s < args.regen_steps:
            x = block_batch(args.batch, args.block, args.vocab, device, rng)
            F.cross_entropy(model(x)[:, :-1].reshape(-1, args.vocab), x[:, 1:].reshape(-1)).backward(); opt.step(); opt.zero_grad()

    # summary: head stability + QK-OV co-location
    forms = [t for t in traj if t["recall"] > 0.7]
    if forms:
        first = forms[0]["step"]; heads_after = [t["topQK"] for t in traj if t["step"] >= first]
        from collections import Counter
        stab = Counter(heads_after).most_common(1)[0][1] / len(heads_after)
        coloc = np.mean([t["same"] for t in traj if t["recall"] > 0.4])
        print(f"\n  ===== microscope summary (toy) =====", flush=True)
        print(f"  recovered recall at step {first}; top-QK-head stability after = {stab:.2f} (1=one basin)", flush=True)
        print(f"  QK & OV in SAME head (when forming) = {coloc:.2f} (1=one complete whirlpool)", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
