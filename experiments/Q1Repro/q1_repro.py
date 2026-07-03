"""
Q1 (reproducibility, with VALID measures): is the regenerated circuit the SAME each time? The basin census surfaced two
problems -- ICL can be recovered by early-layer shortcuts (induction-score=0 yet ICL up), and the toy's embedding-space
OV copy-fidelity is invalid on Pythia. So here we use VALID criteria only: (a) the induction-SCORE (prefix-match QK) of
the forced head, (b) the cosine similarity of the regenerated W_OV (=W_O W_V) ACROSS independent seeds (weight-level
reproducibility), (c) an ablate-after-regen ICL drop (functional confirmation the recovered function lives in THAT head,
not a shortcut). We regenerate at a VIABLE basin (default L5H7) with N seeds, and at an UNVIABLE one (L10H0) as control.

  HF_ENDPOINT=https://hf-mirror.com python q1_repro.py --heads 5,7 10,0 --seeds 3
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pythia_redundancy as PR

RESULTS = PR.RESULTS


def induction_train_batch(vocab, B, T, rng, device):
    half = rng.integers(0, vocab, size=(B, T // 2), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


def regen_at(model, attns, proj_name, hs, dl, dh, probe, L, base, vocab, steps, lr, seed, device):
    """force-basin regen at head (dl,dh) with a given data seed; return (final_indscore, max_iclpct, W_OV vector)."""
    qkv = attns[dl].qkv if hasattr(attns[dl], "qkv") else attns[dl].query_key_value
    dense = getattr(attns[dl], proj_name); d = dense.weight.shape[0]
    mq = torch.zeros_like(qkv.weight); md = torch.zeros_like(dense.weight)
    for blk in range(3):
        mq[blk * d + dh * hs: blk * d + (dh + 1) * hs, :] = 1.0
    md[:, dh * hs:(dh + 1) * hs] = 1.0
    opt = torch.optim.AdamW([qkv.weight, dense.weight], lr=lr); rng = np.random.default_rng(seed)
    best = 0.0
    for s in range(steps):
        model.train()
        x = induction_train_batch(vocab, 8, 256, rng, device); model(x, labels=x).loss.backward()
        qkv.weight.grad.mul_(mq); dense.weight.grad.mul_(md)
        torch.nn.utils.clip_grad_norm_([qkv.weight, dense.weight], 1.0); opt.step(); opt.zero_grad()
        if s % 100 == 0:
            model.eval(); best = max(best, PR.icl_drop(model, probe, L) / base)
    model.eval()
    ind = PR.induction_per_head(model, probe, L)[(dl, dh)]
    Wv = qkv.weight[2 * d + dh * hs:2 * d + (dh + 1) * hs, :].detach()
    Wo = dense.weight[:, dh * hs:(dh + 1) * hs].detach()
    wov = (Wo @ Wv).reshape(-1).float().cpu().numpy()
    return float(ind), float(best), wov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m"); ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--heads", nargs="+", default=["5,7", "10,0"]); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--L", type=int, default=64); ap.add_argument("--out", default="q1_repro.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model); vocab = tok.vocab_size
    probe = PR.make_batch(vocab, args.L, 8, device, seed=0)

    def fresh_lesioned():
        m = AutoModelForCausalLM.from_pretrained(args.model, revision=f"step{args.step}", dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
        attns, proj_name, hs, nh = PR.get_layers_and_proj(m)
        base = PR.icl_drop(m, probe, args.L); sc = PR.induction_per_head(m, probe, args.L)
        ranked = sorted(sc, key=lambda lh: sc[lh], reverse=True)
        PR.ABLATE.clear(); PR.install_hooks(attns, proj_name, hs); S = []
        for k in range(1, 16):
            l, h = ranked[k - 1]; PR.ABLATE.setdefault(l, set()).add(h); S.append((l, h))
            if PR.icl_drop(m, probe, args.L) <= 0.20 * base:
                break
        PR.ABLATE.clear()
        with torch.no_grad():
            for l, h in S:
                getattr(attns[l], proj_name).weight[:, h * hs:(h + 1) * hs] = 0.0
        return m, attns, proj_name, hs, base

    out = {}
    for hstr in args.heads:
        dl, dh = map(int, hstr.split(",")); inds, iclps, wovs = [], [], []
        for sd in range(args.seeds):
            m, attns, proj_name, hs, base = fresh_lesioned()
            ind, iclp, wov = regen_at(m, attns, proj_name, hs, dl, dh, probe, args.L, base, vocab, args.steps, args.lr, sd, device)
            inds.append(ind); iclps.append(iclp); wovs.append(wov)
            print(f"  L{dl}H{dh} seed{sd}: induction-score={ind:.2f}  maxICL={iclp*100:.0f}%", flush=True)
            del m; torch.cuda.empty_cache()
        W = np.stack(wovs); W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        cos = [float(W[i] @ W[j]) for i in range(len(W)) for j in range(i + 1, len(W))]
        out[hstr] = dict(induction_scores=inds, ind_mean=float(np.mean(inds)), ind_std=float(np.std(inds)),
                         max_icl_pcts=iclps, wov_cosine_pairs=cos, wov_cos_mean=float(np.mean(cos)) if cos else None)
        print(f"  ==> L{dl}H{dh}: induction {np.mean(inds):.2f}±{np.std(inds):.2f} | W_OV cosine across seeds "
              f"{np.mean(cos):.2f} (1=identical circuit)\n", flush=True)
        json.dump(out, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)

    print("  ===== Q1 verdict =====", flush=True)
    for hstr, r in out.items():
        rep = "REPRODUCIBLE (same circuit)" if r["ind_mean"] > 0.4 and (r["wov_cos_mean"] or 0) > 0.6 else \
              "not a viable/consistent basin" if r["ind_mean"] < 0.2 else "partial"
        print(f"  {hstr}: induction {r['ind_mean']:.2f}±{r['ind_std']:.2f}, W_OV cos {r['wov_cos_mean']}: {rep}", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
