"""
BRIDGE-CALIBRATE (use the realistic-architecture BRIDGE as the instrument that calibrates the absolute formation law).
The toy proved the FORM t* = C/(N_sites * nu); the bridge (MLP + depth, which already reproduces the Pythia laws) lets us
calibrate the two real-unit pieces so we can PREDICT Pythia:

  q(width)  -- does the per-attempt success saturate with head width? Pythia shows 31m(head_dim 32)->70m(head_dim 64)
               halves the onset at fixed #heads/#layers, so the size effect is WIDTH-driven, not site-count-driven. We
               sweep head_dim at fixed heads/layers and check t* saturates with width (and locate the knee).
  M* (barrier in EVENT units) -- the clock-invariant quantity. Steps differ across models, but the number of induction
               EVENTS seen by formation is N_events = t* * (tokens/step) * nu, which should be ~constant for the same
               circuit. We sweep nu and report N_events at formation = the barrier in universal units.

With q(width) and M* from the bridge, plus the measured Pythia nu (~0.042) and Pythia's batch tokens/step, we can predict
Pythia's onset in absolute steps -- the clock-sync is then just the (computable) tokens-per-step factor.

  python bridge_calibrate.py --smoke
  python bridge_calibrate.py --hds 4 8 16 32 64 --nus 0.15 0.4 1.0 --seeds 4
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = os.path.join(os.path.dirname(__file__), "results")


class Block(nn.Module):
    def __init__(self, d, h, mlp=True):
        super().__init__(); self.h = h; self.use_mlp = mlp
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        if mlp:
            self.ln = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, d = x.shape; H = self.h; hd = d // H
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-1, -2)) / hd ** 0.5
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        a = a.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        if self.use_mlp:
            x = x + self.fc2(F.gelu(self.fc1(self.ln(x))))
        return x


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


def make_batch(B, Bk, vocab, nu_seq, device, rng):
    seqs = np.empty((B, 2 * Bk), dtype=np.int64)
    for i in range(B):
        if rng.random() < nu_seq:
            blk = rng.integers(0, vocab, Bk); seqs[i] = np.concatenate([blk, blk])
        else:
            seqs[i] = rng.integers(0, vocab, 2 * Bk)
    return torch.from_numpy(seqs).to(device)


@torch.no_grad()
def copy_score(model, Bk, vocab, device, seed):
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (256, Bk))
    ids = np.concatenate([blk, blk], axis=1); x = torch.from_numpy(ids).to(device)
    pred = model(x)[:, Bk:2 * Bk - 1].argmax(-1); tgt = torch.from_numpy(ids[:, Bk + 1:2 * Bk]).to(device)
    return float((pred == tgt).float().mean())


def train_to_form(hd, H, L, nu_seq, vocab, Bk, steps, batch, lr, seed, device, thr=0.7, ee=25):
    rng = np.random.default_rng(seed); d = hd * H
    model = Tf(vocab, d, H, L, 2 * Bk + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(1, steps + 1):
        x = make_batch(batch, Bk, vocab, nu_seq, device, rng)
        F.cross_entropy(model(x)[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
        opt.step(); opt.zero_grad()
        if s % ee == 0 and copy_score(model, Bk, vocab, device, 7000 + s) >= thr:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hds", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.15, 0.4, 1.0])
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--fix_d", type=int, default=0, help="if >0, fix model width d and set heads=d/hd (isolates head_dim from width)")
    ap.add_argument("--vocab", type=int, default=512); ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="bridge_calibrate.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.hds = [4, 16, 64]; args.nus = [0.3, 1.0]; args.seeds = 2; args.steps = 2500
    tok_per_step = args.batch * 2 * args.block
    print(f"device={device} hds={args.hds} nus={args.nus} heads={args.heads} layers={args.layers} "
          f"tok/step={tok_per_step} (BRIDGE-CALIBRATE: q(width) + barrier M*)", flush=True)

    grid = {}
    print(f"\n  {'hd':>4} {'d':>4} {'nu':>5} {'t*':>6} {'N_events=t*·tok·nu':>18}", flush=True)
    for hd in args.hds:
        H = (args.fix_d // hd) if args.fix_d else args.heads     # fix_d: hold model width, vary how it splits into heads
        for nu in args.nus:
            ts = [t for seed in range(args.seeds)
                  if (t := train_to_form(hd, H, args.layers, nu, args.vocab, args.block,
                                         args.steps, args.batch, args.lr, seed, device)) is not None]
            tmed = float(np.median(ts)) if ts else None
            nev = tmed * tok_per_step * nu if tmed else None      # induction events seen by formation (clock-invariant)
            grid[f"{hd},{nu}"] = dict(hd=hd, d=hd * H, heads=H, nu=nu, tstar=tmed, n_events=nev, n=len(ts))
            print(f"  {hd:>4} {hd*H:>4} {nu:>5.2f} {str(int(tmed)) if tmed else 'None':>6} "
                  f"{str(int(nev)) if nev else 'None':>18}", flush=True)
            json.dump(grid, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    rows = [v for v in grid.values() if v["tstar"]]
    numax = max(args.nus)
    print(f"\n  ===== q(width): t*(head_dim) at nu={numax} (does it saturate?) =====", flush=True)
    for hd in args.hds:
        v = grid.get(f"{hd},{numax}")
        if v and v["tstar"]:
            print(f"    head_dim={hd:>3}: t*={int(v['tstar'])}", flush=True)
    print(f"  ===== barrier M* (induction events at formation, clock-invariant) =====", flush=True)
    evs = [v["n_events"] for v in rows if v["hd"] == max(args.hds)]
    if evs:
        print(f"    at head_dim={max(args.hds)} (saturated q): N_events = {[int(e) for e in evs]}  "
              f"median={int(np.median(evs))}", flush=True)
    print(f"\n  PREDICT PYTHIA: t*_pred = M* / (tok_per_step_pythia * nu_pythia * q)   "
          f"[pythia tok/step~2.1e6, nu~0.042]", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
