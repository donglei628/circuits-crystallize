"""
GROK-NUCLEATION (unification: is grokking a full nucleation cycle?). Modular addition (a+b) mod p with weight decay
groks: train accuracy saturates early, test accuracy stays ~0 on a plateau, then SUDDENLY jumps. We test three
nucleation signatures, going beyond the usual accuracy-sigmoid:
  (1) PLATEAU -> ABRUPT JUMP in test accuracy (the metastable-state-then-ignition picture).
  (2) LOSS KINK: the test LOSS has a sharp derivative spike at the transition (a first-order-like jump, not a gradual
      slide -- the signature that distinguishes nucleation from smooth convergence).
  (3) REGENERATION: after grokking, lesion the circuit (re-randomize a weight group) and continue training -- does it
      RE-grok, and faster than the first time (a second nucleation on the existing scaffold)?

  python grok_nucleation.py --smoke
  python grok_nucleation.py --p 97 --train_frac 0.4 --steps 40000 --seeds 3
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
        a = a.softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class GrokTf(nn.Module):
    def __init__(self, p, d=128, h=4, L=1):
        super().__init__()
        self.tok = nn.Embedding(p + 1, d); self.pos = nn.Embedding(3, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); self.unembed = nn.Linear(d, p, bias=False)

    def forward(self, x):
        T = x.shape[1]; hh = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            hh = b(hh)
        return self.unembed(hh[:, -1])                      # predict at the '=' position


def make_data(p, device):
    a, b = np.meshgrid(np.arange(p), np.arange(p)); a = a.flatten(); b = b.flatten()
    eq = np.full_like(a, p)                                 # '=' token id = p
    X = np.stack([a, b, eq], 1); Y = (a + b) % p
    return torch.tensor(X, device=device), torch.tensor(Y, device=device)


@torch.no_grad()
def evaluate(model, X, Y):
    logit = model(X); loss = F.cross_entropy(logit, Y); acc = (logit.argmax(-1) == Y).float().mean()
    return float(loss), float(acc)


def lesion(model, scale=0.1):
    with torch.no_grad():                                   # re-randomize the attention readout (the circuit's output)
        for b in model.blocks:
            b.o.weight.normal_(0, scale)


def run_grok(p, train_frac, d, steps, lr, wd, seed, device, lesion_at=None):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    X, Y = make_data(p, device); n = X.shape[0]
    perm = rng.permutation(n); ntr = int(train_frac * n)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    model = GrokTf(p, d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    curve = []; grok_step = None; regrok_step = None; lesioned = False
    for s in range(1, steps + 1):
        model.train(); logit = model(Xtr); loss = F.cross_entropy(logit, Ytr)
        loss.backward(); opt.step(); opt.zero_grad()
        if s % 100 == 0:
            trl, tra = evaluate(model, Xtr, Ytr); tel, tea = evaluate(model, Xte, Yte)
            curve.append(dict(step=s, tr_acc=tra, te_acc=tea, tr_loss=trl, te_loss=tel))
            if grok_step is None and tea >= 0.9:
                grok_step = s
            if lesion_at and not lesioned and grok_step and s >= grok_step + lesion_at:
                lesion(model); lesioned = True; lesion_step = s
            if lesioned and regrok_step is None and tea >= 0.9 and s > lesion_step + 200:
                regrok_step = s
    return dict(seed=seed, grok_step=grok_step, regrok_step=regrok_step,
                lesion_step=lesion_step if lesioned else None, curve=curve)


def loss_kink(curve, grok_step):
    """max |d te_loss / d step| near the grok transition = the sharpness of the loss jump."""
    if grok_step is None:
        return None
    steps = np.array([c["step"] for c in curve]); tel = np.array([c["te_loss"] for c in curve])
    d = np.abs(np.diff(tel) / np.diff(steps))
    mid = (steps[:-1] + steps[1:]) / 2
    win = (mid > grok_step * 0.5) & (mid < grok_step * 1.5)
    base = np.median(d[mid < grok_step * 0.5]) if (mid < grok_step * 0.5).any() else 1e-9
    peak = d[win].max() if win.any() else 0
    return float(peak / max(base, 1e-9))                    # kink sharpness = peak slope / baseline slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97); ap.add_argument("--train_frac", type=float, default=0.4)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--lesion_at", type=int, default=3000)
    ap.add_argument("--out", default="grok_nucleation.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.p = 53; args.steps = 20000; args.seeds = 1
    print(f"device={device} p={args.p} train_frac={args.train_frac} wd={args.wd} steps={args.steps} "
          f"(GROK-NUCLEATION: plateau->jump + loss-kink + regen)", flush=True)

    runs = []
    for seed in range(args.seeds):
        r = run_grok(args.p, args.train_frac, args.d, args.steps, args.lr, args.wd, seed, device, args.lesion_at)
        r["kink"] = loss_kink(r["curve"], r["grok_step"])
        runs.append(r)
        rg = r["regrok_step"]; ls = r["lesion_step"]
        regrow = (rg - ls) if (rg and ls) else None
        print(f"  seed {seed}: grok@{r['grok_step']}  loss-kink={r['kink']:.0f}x baseline  "
              f"lesion@{ls} regrok@{rg} (regrow={regrow} vs first-grok={r['grok_step']})", flush=True)
        json.dump([{k: v for k, v in rr.items() if k != "curve"} for rr in runs] +
                  [{"_last_curve": runs[-1]["curve"]}], open(os.path.join(RESULTS, args.out), "w"), indent=2)

    print(f"\n  ===== grokking-as-nucleation =====", flush=True)
    gs = [r["grok_step"] for r in runs if r["grok_step"]]
    ks = [r["kink"] for r in runs if r["kink"]]
    regrows = [(r["regrok_step"] - r["lesion_step"]) for r in runs if r["regrok_step"] and r["lesion_step"]]
    print(f"  (1) plateau->jump: grok at {np.median(gs) if gs else None} steps (test acc 0->1)", flush=True)
    print(f"  (2) loss-kink: test-loss slope peaks {np.median(ks):.0f}x baseline at transition "
          f"{'(SHARP = nucleation signature)' if ks and np.median(ks) > 3 else '(gradual)'}" if ks else "  (2) no grok", flush=True)
    if regrows:
        print(f"  (3) regen: re-grok in {np.median(regrows):.0f} steps vs first-grok {np.median(gs):.0f} "
              f"{'(FASTER = second nucleation on scaffold)' if np.median(regrows) < np.median(gs) else ''}", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
