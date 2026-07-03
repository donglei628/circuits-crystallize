"""
END-TO-END CONSTANT on the REAL Pythia ARCHITECTURE (GPTNeoX, 2-layer, from scratch). Same procedure as end2end_const
but with GPTNeoX (rotary, parallel-residual, real MLP) so the measured prefactor C = a*p^K = 1/N_sites is directly
comparable across the TOY (small GPTNeoX) and PYTHIA-scale widths (d up to 768 = pythia-160m width). Answers:
  (i)  is C constant across widths for the REAL architecture? (within-family invariance)
  (ii) is GPTNeoX's C the same number as the custom-toy's C~9, or does each architecture/model have its OWN constant?

We can't sweep nu on the released pretrained Pythia (data is fixed), so the faithful "Pythia constant, measured the same
way as the toy" is from-scratch GPTNeoX at Pythia configs. Components for the restore-barrier (2-layer induction):
  PREV   = layer0 attention (query_key_value + dense)   -- previous-token head
  IND_QK = layer1 q,k  slices of query_key_value         -- induction match
  IND_OV = layer1 v slice of query_key_value + dense     -- the copy
GPTNeoX packs qkv per-head interleaved: weight.view(num_heads, 3*head_dim, hidden) -> [:, :hd]=q, [hd:2hd]=k, [2hd:3hd]=v.

  python end2end_gptneox.py --smoke
  python end2end_gptneox.py --ds 256 512 768 --seeds 3 --out e2e_gptneox.json
"""
from __future__ import annotations
import argparse, copy, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def build(d, heads, vocab, T, device):
    cfg = GPTNeoXConfig(vocab_size=vocab, hidden_size=d, num_hidden_layers=2, num_attention_heads=heads,
                        intermediate_size=4 * d, max_position_embeddings=T + 2, use_parallel_residual=True,
                        rotary_pct=0.25, hidden_act="gelu")
    return GPTNeoXForCausalLM(cfg).to(device)


def batch(B, T, V, nu, rng, device):
    half = T // 2
    R = rng.integers(0, V, (B, half)); r2 = rng.integers(0, V, (B, half))
    second = np.where(rng.random((B, half)) < nu, R, r2)
    x = np.concatenate([R, second], 1).astype(np.int64)
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def ind_score(model, T, V, device, seed=12345):
    rng = np.random.default_rng(seed); half = T // 2
    R = rng.integers(0, V, (256, half)); x = np.concatenate([R, R], 1).astype(np.int64)
    xt = torch.from_numpy(x).to(device)
    logits = model(xt).logits
    pred = logits[:, half:2 * half - 1].argmax(-1)
    tgt = torch.from_numpy(x[:, half + 1:2 * half].astype(np.int64)).to(device)
    return float((pred == tgt).float().mean())


def train_tstar(model, T, V, nu, lr, max_steps, ee, thr, seed, device):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for s in range(0, max_steps + 1):
        if s % ee == 0:
            model.eval()
            if ind_score(model, T, V, device) > thr:
                model.train(); return s
            model.train()
        x = batch(64, T, V, nu, rng, device)
        out = model(x, labels=x); out.loss.backward()       # HF shifts labels internally (next-token)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None


# ---------- GPTNeoX component access (per-head interleaved qkv) ----------
def _qkv_view(layer, heads, hd):
    return layer.attention.query_key_value.weight.data.view(heads, 3 * hd, -1)   # [heads, 3*hd, hidden]


def comp_state(model, heads, hd):
    L = model.gpt_neox.layers
    qkv1 = _qkv_view(L[1], heads, hd)
    return {
        "PREV":   [("l0.qkv", L[0].attention.query_key_value.weight.data.clone()),
                   ("l0.dense", L[0].attention.dense.weight.data.clone())],
        "IND_QK": [("l1.qk", qkv1[:, :2 * hd, :].clone())],                       # q,k
        "IND_OV": [("l1.v",  qkv1[:, 2 * hd:, :].clone()),                        # v
                   ("l1.dense", L[1].attention.dense.weight.data.clone())],
    }


def set_components(model, src, which, heads, hd):
    L = model.gpt_neox.layers
    with torch.no_grad():
        if "PREV" in which:
            L[0].attention.query_key_value.weight.data.copy_(src["PREV"][0][1])
            L[0].attention.dense.weight.data.copy_(src["PREV"][1][1])
        if "IND_QK" in which:
            _qkv_view(L[1], heads, hd)[:, :2 * hd, :].copy_(src["IND_QK"][0][1])
        if "IND_OV" in which:
            _qkv_view(L[1], heads, hd)[:, 2 * hd:, :].copy_(src["IND_OV"][0][1])
            L[1].attention.dense.weight.data.copy_(src["IND_OV"][1][1])


def measure_width(d, args, device):
    V, T, H = args.vocab, args.T, args.heads; hd = d // H
    # ---- (1) nu-sweep: t* = t0 + a/nu ----
    tstars = {}
    for nu in args.nus:
        ts = []
        for sd in range(args.seeds):
            torch.manual_seed(1000 + sd); m = build(d, H, V, T, device)
            t = train_tstar(m, T, V, nu, args.lr, args.max_steps, args.ee, args.thresh, sd, device)
            if t is not None: ts.append(t)
        tstars[nu] = float(np.median(ts)) if ts else None
        print(f"    [d{d}] nu={nu}: t*={tstars[nu]}  (n={len(ts)}/{args.seeds})", flush=True)
    pts = [(1.0 / nu, tstars[nu]) for nu in args.nus if tstars[nu] is not None]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        a_slope, t0 = np.polyfit(xs, ys, 1)
        r2 = 1 - ((ys - (a_slope * xs + t0)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-9)
    else:
        a_slope = t0 = r2 = None

    # ---- (2) restore-barrier: donor -> lesion 3 -> restore subset (keep embeddings/MLP/unembed) ----
    torch.manual_seed(7); donor_m = build(d, H, V, T, device)
    train_tstar(donor_m, T, V, 1.0, args.lr, args.max_steps, args.ee, 0.85, 0, device)
    donor = comp_state(donor_m, H, hd)
    comps = ["IND_QK", "IND_OV"]          # the empirically rate-limiting components ([R|R]+rotary: PREV not needed, K=2)
    from itertools import combinations
    subsets = [()] + [c for r in (1, 2) for c in combinations(comps, r)]
    rate = {}
    for sub in subsets:
        rs = []
        for sd in range(args.seeds):
            m = copy.deepcopy(donor_m)
            torch.manual_seed(3000 + sd); rand_m = build(d, H, V, T, device)
            set_components(m, comp_state(rand_m, H, hd), set(comps), H, hd)        # LESION all 3
            if sub: set_components(m, donor, set(sub), H, hd)                       # RESTORE subset
            t = train_tstar(m, T, V, 1.0, args.lr, args.restore_steps, args.restore_ee, args.thresh, sd, device)
            rs.append(1.0 / (args.restore_steps * 2) if t is None else 1.0 / max(t, args.restore_ee))
        rate[sub] = float(np.median(rs))
        print(f"    [d{d}] restore {('+'.join(sub) or 'none'):>20}: rate={rate[sub]:.5f}", flush=True)
    nx = np.array([len(s) for s in subsets]); ly = np.log([rate[s] for s in subsets])
    lnp, b = np.polyfit(nx, ly, 1); r2b = 1 - ((ly - (lnp * nx + b)) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-9)
    full = rate[tuple(comps)]; best2 = max(rate[s] for s in subsets if len(s) == 1)

    K = 2; pK = float(np.exp(-K * lnp)) if lnp else None
    C = float(a_slope * pK) if (a_slope and pK) else None
    res = dict(d=d, tstars=tstars, a_slope=a_slope, t0=t0, nu_r2=r2,
               restore_rate={('+'.join(s) or 'none'): rate[s] for s in subsets},
               lnp=float(lnp), barrier_r2=float(r2b), full3_rate=full, best2_rate=best2, pK=pK, C=C)
    print(f"  ==[d{d} GPTNeoX]== a={a_slope:.1f}(t0={t0:.0f},R2={r2:.2f}) | |ln p|={lnp:.3f}(R2={r2b:.2f}) "
          f"| p^K={pK:.4f} | C=a*p^K={C:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[256, 512, 768])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.25, 0.4, 0.6, 1.0])
    ap.add_argument("--vocab", type=int, default=256); ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--restore_steps", type=int, default=2500); ap.add_argument("--ee", type=int, default=25)
    ap.add_argument("--restore_ee", type=int, default=10); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="e2e_gptneox.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.ds = [256]; args.nus = [0.5, 1.0]; args.seeds = 1; args.max_steps = 3000; args.restore_steps = 1500
    print(f"device={device} ds={args.ds} (GPTNeoX real arch; C const across d & =toy?  -> universal vs per-model)", flush=True)

    out = []
    for d in args.ds:
        out.append(measure_width(d, args, device))
        json.dump({"args": vars(args), "results": out}, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    Cs = [r["C"] for r in out if r["C"]]
    print(f"\n  ===== VERDICT (GPTNeoX) =====", flush=True)
    for r in out:
        print(f"   d={r['d']:>4}: C={r['C']:.2f}  (a={r['a_slope']:.1f}, |ln p|={r['lnp']:.3f})", flush=True)
    if len(Cs) >= 2:
        cv = float(np.std(Cs) / (np.mean(Cs) + 1e-9))
        print(f"   C across widths: {[round(c,2) for c in Cs]}  mean={np.mean(Cs):.2f} CV={cv:.2f}", flush=True)
        print(f"   custom-toy C was ~9.1 -> {'SAME ballpark (more universal)' if 6 < np.mean(Cs) < 13 else 'DIFFERENT (each arch its own constant)'}", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
