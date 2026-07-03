"""
END-TO-END CONSTANT TEST (is the gap "just a constant" or a structure?).

The rate law is  t* = t0 + 1/(nu * p^K).  From the nucleation product law J = N_sites * nu * p^K we get
    t* * nu  ~=  a  =  1/(N_sites * p^K)            (a = slope of t* vs 1/nu, t0 cancels)
so the "gap" between the independently-measured per-component barrier p^K and the measured slope a is
    C  =  a * p^K  =  1 / N_sites .
=> "is the gap just a constant?" is LITERALLY "is N_sites architecture-invariant?". We measure C at several widths d
on the SAME 2-layer induction toy, each time getting the THREE inputs INDEPENDENTLY:
   - a, t0   : fit t* = t0 + a/nu over a nu-sweep (timing; never touches the barrier)
   - |ln p|  : lesion all 3 induction components, RESTORE subsets, ln(rate) ~ (#restored)*|ln p|  (barrier; never touches nu)
   - K = 3   : the conjunction depth (restore crater: only the full set recovers)
Then C = a * exp(-K*|ln p|).  If C is the same across widths -> ONE constant (Arrhenius prefactor, measure once,
end-to-end closes).  If C scales with d -> N_sites is structured (a function, not a constant).

2-layer attention+MLP toy, clean component slices (layer0 = prev-token head, layer1 = induction head):
  PREV   = blocks[0].qkv + blocks[0].o          (the previous-token head)
  IND_QK = blocks[1].qkv[0:2d]   (q,k)          (the induction match)
  IND_OV = blocks[1].qkv[2d:3d] (v) + blocks[1].o (the copy)

  python end2end_const.py --ds 128 256 --seeds 3 --out e2e_const.json
  python end2end_const.py --smoke
"""
from __future__ import annotations
import argparse, copy, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = os.path.join(os.path.dirname(__file__), "results")


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d); self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, d = x.shape; H = self.h; hd = d // H
        z = self.ln1(x)
        qkv = self.qkv(z).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4); q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-1, -2)) / hd ** 0.5
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        a = a.masked_fill(mask, float("-inf")).softmax(-1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Tf(nn.Module):
    def __init__(self, vocab, d, h, maxlen):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(2)]); self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():                                  # default Embedding init (~N(0,1)) is too large
            self.tok.weight.mul_(0.1); self.pos.weight.mul_(0.1)

    def forward(self, x):
        T = x.shape[1]; hh = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            hh = b(hh)
        return self.unembed(hh)


def batch(B, T, V, nu, rng, device):
    """seq = [R | second]; second[j] = R[j] with prob nu (induction-predictable), else random. nu = induction density."""
    half = T // 2
    R = rng.integers(0, V, (B, half)); r2 = rng.integers(0, V, (B, half))
    second = np.where(rng.random((B, half)) < nu, R, r2)
    x = np.concatenate([R, second], 1).astype(np.int64)
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def ind_score(model, T, V, device, seed=12345):
    """strict copy-score on a held-out FULLY-repeated probe."""
    rng = np.random.default_rng(seed); half = T // 2
    R = rng.integers(0, V, (256, half)); x = np.concatenate([R, R], 1).astype(np.int64)
    xt = torch.from_numpy(x).to(device)
    pred = model(xt)[:, half:2 * half - 1].argmax(-1)
    tgt = torch.from_numpy(x[:, half + 1:2 * half].astype(np.int64)).to(device)
    return float((pred == tgt).float().mean())


def train_tstar(model, T, V, nu, lr, max_steps, ee, thr, seed, device):
    """train at density nu; return the step where strict induction first exceeds thr (or None)."""
    rng = np.random.default_rng(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for s in range(0, max_steps + 1):
        if s % ee == 0:
            model.eval()
            if ind_score(model, T, V, device) > thr:
                model.train(); return s
            model.train()
        x = batch(64, T, V, nu, rng, device)
        logits = model(x)                                       # NEXT-token prediction (shift by 1)
        F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None


# ---------- component slices (for the restore-barrier) ----------
def comp_state(model, d):
    """return the donor weight tensors for each of the 3 induction components."""
    b0, b1 = model.blocks[0], model.blocks[1]
    return {
        "PREV":   [("blocks.0.qkv.weight", b0.qkv.weight.detach().clone()),
                   ("blocks.0.o.weight",   b0.o.weight.detach().clone())],
        "IND_QK": [("blocks.1.qkv.weight.qk", b1.qkv.weight.detach()[:2 * d].clone())],
        "IND_OV": [("blocks.1.qkv.weight.v",  b1.qkv.weight.detach()[2 * d:].clone()),
                   ("blocks.1.o.weight",       b1.o.weight.detach().clone())],
    }


def set_components(model, src_donor, which, d):
    """set model's components in `which` to the weights stored in src_donor (a comp_state dict)."""
    b0, b1 = model.blocks[0], model.blocks[1]
    with torch.no_grad():
        if "PREV" in which:
            b0.qkv.weight.copy_(src_donor["PREV"][0][1]); b0.o.weight.copy_(src_donor["PREV"][1][1])
        if "IND_QK" in which:
            b1.qkv.weight[:2 * d].copy_(src_donor["IND_QK"][0][1])
        if "IND_OV" in which:
            b1.qkv.weight[2 * d:].copy_(src_donor["IND_OV"][0][1]); b1.o.weight.copy_(src_donor["IND_OV"][1][1])


def measure_width(d, args, device):
    V, T, H = args.vocab, args.T, args.heads
    # ---- (1) nu-sweep: t* = t0 + a/nu ----
    nus = args.nus; tstars = {}
    for nu in nus:
        ts = []
        for sd in range(args.seeds):
            torch.manual_seed(1000 + sd)
            m = Tf(V, d, H, T + 2).to(device)
            t = train_tstar(m, T, V, nu, args.lr, args.max_steps, args.ee, args.thresh, sd, device)
            if t is not None: ts.append(t)
        tstars[nu] = float(np.median(ts)) if ts else None
        print(f"    [d{d}] nu={nu}: t*={tstars[nu]}  (n={len(ts)}/{args.seeds})", flush=True)
    pts = [(1.0 / nu, tstars[nu]) for nu in nus if tstars[nu] is not None]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        a_slope, t0 = np.polyfit(xs, ys, 1)
        r2 = 1 - ((ys - (a_slope * xs + t0)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9)
    else:
        a_slope = t0 = r2 = None

    # ---- (2) restore-barrier (k_scale style): start from a TRAINED donor, LESION all 3 induction components, RESTORE
    #          a subset (rest of the model -- embeddings, MLP, unembed -- KEPT at donor), measure re-formation rate ----
    torch.manual_seed(7); donor_m = Tf(V, d, H, T + 2).to(device)
    train_tstar(donor_m, T, V, 1.0, args.lr, args.max_steps, args.ee, 0.85, 0, device)
    donor = comp_state(donor_m, d)
    comps = ["PREV", "IND_QK", "IND_OV"]
    from itertools import combinations
    subsets = [()] + [c for r in (1, 2, 3) for c in combinations(comps, r)]
    rate = {}
    for sub in subsets:
        rs = []
        for sd in range(args.seeds):
            m = copy.deepcopy(donor_m)                                   # keep embeddings/MLP/unembed at donor
            torch.manual_seed(3000 + sd); rand_m = Tf(V, d, H, T + 2).to(device)
            set_components(m, comp_state(rand_m, d), set(comps), d)      # LESION all 3 (fresh random)
            if sub: set_components(m, donor, set(sub), d)                # RESTORE the subset from donor
            t = train_tstar(m, T, V, 1.0, args.lr, args.restore_steps, args.restore_ee, args.thresh, sd, device)
            if t is None: rs.append(1.0 / (args.restore_steps * 2))      # censored -> floor rate
            else:         rs.append(1.0 / max(t, args.restore_ee))       # t=0 (instant donor) capped at one eval
        rate[sub] = float(np.median(rs))
        print(f"    [d{d}] restore {('+'.join(sub) or 'none'):>20}: rate={rate[sub]:.5f}", flush=True)
    # fit ln(rate) vs #restored
    nx = np.array([len(s) for s in subsets]); ly = np.log([rate[s] for s in subsets])
    lnp, b = np.polyfit(nx, ly, 1); r2b = 1 - ((ly - (lnp * nx + b)) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-9)
    # crater check: full-3 rate vs best 2-subset
    full = rate[tuple(comps)]; best2 = max(rate[s] for s in subsets if len(s) == 2)

    K = 3
    pK = float(np.exp(-K * lnp)) if lnp else None
    C = float(a_slope * pK) if (a_slope and pK) else None        # C = a * p^K = 1/N_sites
    res = dict(d=d, tstars=tstars, a_slope=a_slope, t0=t0, nu_r2=r2,
               restore_rate={('+'.join(s) or 'none'): rate[s] for s in subsets},
               lnp=float(lnp), barrier_r2=float(r2b), full3_rate=full, best2_rate=best2,
               pK=pK, C=C, N_sites=float(1.0 / C) if C else None)
    print(f"  ==[d{d}]== a={a_slope:.1f}(t0={t0:.0f},R2={r2:.2f}) | |ln p|={lnp:.3f}(R2={r2b:.2f}) "
          f"crater full/best2={full:.4f}/{best2:.4f} | p^K={pK:.4f} | C=a*p^K={C:.2f} (N_sites~{1/C:.1f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[128, 192, 256])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.25, 0.4, 0.6, 1.0])
    ap.add_argument("--vocab", type=int, default=256); ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--restore_steps", type=int, default=2500); ap.add_argument("--ee", type=int, default=25)
    ap.add_argument("--restore_ee", type=int, default=10)
    ap.add_argument("--thresh", type=float, default=0.5); ap.add_argument("--out", default="e2e_const.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.ds = [128]; args.nus = [0.5, 1.0]; args.seeds = 1; args.max_steps = 2000; args.restore_steps = 1500
    print(f"device={device} ds={args.ds} nus={args.nus} seeds={args.seeds} "
          f"(C=a*p^K=1/N_sites; constant across d  <=>  gap is ONE constant)", flush=True)

    out = []
    for d in args.ds:
        out.append(measure_width(d, args, device))
        json.dump({"args": vars(args), "results": out}, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    Cs = [r["C"] for r in out if r["C"]]
    print(f"\n  ===== VERDICT =====", flush=True)
    for r in out:
        print(f"   d={r['d']:>4}: C={r['C']:.2f}  N_sites~{r['N_sites']:.1f}  (a={r['a_slope']:.1f}, |ln p|={r['lnp']:.3f})", flush=True)
    if len(Cs) >= 2:
        cv = float(np.std(Cs) / (np.mean(Cs) + 1e-9))
        print(f"   C across widths: {[round(c,2) for c in Cs]}  mean={np.mean(Cs):.2f} CV={cv:.2f}", flush=True)
        print(f"   => {'CONSTANT (gap is ONE number; end-to-end closes)' if cv < 0.25 else 'STRUCTURED (C scales with d; N_sites is a function)'}", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
