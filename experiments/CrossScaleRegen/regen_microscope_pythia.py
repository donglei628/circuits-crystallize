"""
REGEN-MICROSCOPE (Pythia): same frame-by-frame measurement as the toy, on a real LM. After lesioning the induction SET,
continue training under induction pressure and every few steps record the per-head induction (prefix-match) scores ->
the TOP head + its score, the concentration (does the QK energy sit in ONE head or spread/shift?), and ICL. The basin
picture predicts the toy SETTLES (top-head stability ~0.94, function recovers) while Pythia WANDERS (top head keeps
changing, function doesn't recover) -- the ball can't fall back into the same whirlpool among many.

  HF_ENDPOINT=https://hf-mirror.com python regen_microscope_pythia.py --model EleutherAI/pythia-160m
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pythia_redundancy as PR
from collections import Counter

RESULTS = PR.RESULTS


def induction_train_batch(vocab, B, T, rng, device):
    half = rng.integers(0, vocab, size=(B, T // 2), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m"); ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--lesion_mode", choices=["clamp", "oneshot"], default="oneshot")
    ap.add_argument("--train_steps", type=int, default=600); ap.add_argument("--ee", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--train_batch", type=int, default=8); ap.add_argument("--train_seq", type=int, default=256)
    ap.add_argument("--probe_batch", type=int, default=8); ap.add_argument("--max_clamp", type=int, default=15)
    ap.add_argument("--out", default="regen_microscope_pythia.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    probe = PR.make_batch(tok.vocab_size, args.L, args.probe_batch, device, seed=0)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=f"step{args.step}", dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    attns, proj_name, head_size, nh = PR.get_layers_and_proj(model); nl = len(attns)
    print(f"device={device} {args.model} {nl}L x {nh}H lesion={args.lesion_mode}", flush=True)

    base = PR.icl_drop(model, probe, args.L)
    scores = PR.induction_per_head(model, probe, args.L)
    ranked = sorted(scores, key=lambda lh: scores[lh], reverse=True)
    PR.ABLATE.clear(); PR.install_hooks(attns, proj_name, head_size); S = []
    for k in range(1, args.max_clamp + 1):
        l, h = ranked[k - 1]; PR.ABLATE.setdefault(l, set()).add(h); S.append((l, h))
        if PR.icl_drop(model, probe, args.L) <= 0.20 * base:
            break
    Sset = set(S)
    if args.lesion_mode == "oneshot":
        PR.ABLATE.clear()
        with torch.no_grad():
            for l, h in S:
                getattr(attns[l], proj_name).weight[:, h * head_size:(h + 1) * head_size] = 0.0
    print(f"  baseline ICL={base:.2f}; |S|={len(S)}; crater ICL={PR.icl_drop(model, probe, args.L):.2f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); rng = np.random.default_rng(0)
    traj = []
    print(f"\n  {'step':>5} {'ICL%':>6} {'topFreeHead':>12} {'score':>6} {'top3heads':>22}", flush=True)
    for s in range(0, args.train_steps + 1):
        if s % args.ee == 0:
            model.eval()
            sc = PR.induction_per_head(model, probe, args.L)
            free = sorted([(lh, v) for lh, v in sc.items() if lh not in Sset], key=lambda kv: -kv[1])
            top = free[0]; icl = PR.icl_drop(model, probe, args.L)
            top3 = [(f"L{l}H{h}", round(v, 2)) for (l, h), v in free[:3]]
            traj.append(dict(step=s, icl=icl, icl_pct=icl / base, top_head=list(top[0]), top_score=top[1], top3=top3))
            print(f"  {s:>5} {icl/base*100:>5.0f}% {f'L{top[0][0]}H{top[0][1]}':>12} {top[1]:>6.2f} {str(top3):>22}", flush=True)
            json.dump(traj, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
            model.train()
        if s < args.train_steps:
            x = induction_train_batch(tok.vocab_size, args.train_batch, args.train_seq, rng, device)
            out = model(x, labels=x); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()

    # summary: top-free-head stability (vs toy's 0.94)
    formed = [t for t in traj if t["top_score"] > 0.5]
    if formed:
        first = formed[0]["step"]; heads_after = [tuple(t["top_head"]) for t in traj if t["step"] >= first]
        stab = Counter(heads_after).most_common(1)[0][1] / len(heads_after)
        n_distinct = len(set(heads_after))
        print(f"\n  ===== microscope summary (Pythia) =====", flush=True)
        print(f"  top-QK head relocated by step {first} (score>0.5); AFTER that: stability={stab:.2f}, "
              f"{n_distinct} distinct heads visited (toy stability was 0.94, 1 head)", flush=True)
        print(f"  max ICL recovered = {max(t['icl_pct'] for t in traj)*100:.0f}% of base", flush=True)
        print(f"  => {'WANDERS among many basins (low stability) -> function does not return' if stab < 0.6 else 'settles'}", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
