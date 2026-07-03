"""
DIRECT-PYTHIA test of the geometry law: on real Pythia checkpoints (trained on real text, which genuinely rewards real
induction), does the embedding geometry's copy-capacity C_copy rise and PREDICT the REAL-induction onset? We measure, per
checkpoint:
  - C_copy   : held-out rank-(head_dim) OV copy capacity of the checkpoint's embeddings (the geometry, fixed metric)
  - induction: STRICT copy-score on a repeated-RANDOM-token probe (uniform tokens -> no frequency shortcut -> real copy)
  - qk       : max prefix-match attention (the look-back routing)
If C_copy crosses up right as `induction` forms, the toy's geometry->speed law governs the REAL model.

  HF_ENDPOINT=https://hf-mirror.com python ckpt_geom_induction.py --model EleutherAI/pythia-160m
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pythia_redundancy as PR

RESULTS = PR.RESULTS


def c_copy_heldout(WE, WU, k, device, sample=1500, steps=400, lr=5e-3):
    V, d = WE.shape; rng = np.random.default_rng(7); n = min(sample, V // 2); perm = rng.permutation(V)
    tr = torch.from_numpy(perm[:n]).long().to(device); te = torch.from_numpy(perm[n:2 * n]).long().to(device)
    A = torch.zeros(d, k, device=device, requires_grad=True); B = torch.zeros(d, k, device=device, requires_grad=True)
    torch.nn.init.normal_(A, std=0.02); torch.nn.init.normal_(B, std=0.02)
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(((WE[tr] @ B) @ A.t()) @ WU.t(), tr); loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        return float(((((WE[te] @ B) @ A.t()) @ WU.t()).argmax(1) == te).float().mean())


@torch.no_grad()
def induction_copyscore(model, vocab, L, device, n=128, seed=0):
    """STRICT: repeated RANDOM (uniform) tokens; copy-score = fraction of 2nd-half tokens predicted correctly."""
    rng = np.random.default_rng(seed); blk = rng.integers(0, vocab, (n, L), dtype=np.int64)
    x = np.concatenate([blk, blk], axis=1); xt = torch.from_numpy(x).long().to(device)
    pred = model(xt).logits[:, L:2 * L - 1].argmax(-1)
    tgt = torch.from_numpy(x[:, L + 1:2 * L]).long().to(device)
    return float((pred == tgt).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--steps_list", default="0,16,64,128,256,512,1000,2000,4000,8000,16000,64000,143000")
    ap.add_argument("--L", type=int, default=64); ap.add_argument("--out", default="ckpt_geomind_160m.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model); vocab = tok.vocab_size
    probe = PR.make_batch(vocab, args.L, 8, device, seed=0)
    steps = [int(x) for x in args.steps_list.split(",")]
    traj = []
    print(f"  {'step':>7} {'C_copy':>8} {'induction':>10} {'qk':>6}", flush=True)
    for stp in steps:
        try:
            m = AutoModelForCausalLM.from_pretrained(args.model, revision=f"step{stp}", dtype=torch.float32,
                                                     attn_implementation="eager").to(device)
        except Exception as e:
            print(f"  step{stp}: load failed {str(e)[:50]}", flush=True); continue
        nh = m.config.num_attention_heads; k = m.config.hidden_size // nh
        WE = m.gpt_neox.embed_in.weight.detach(); WU = m.embed_out.weight.detach()
        cc = c_copy_heldout(WE, WU, k, device)
        ind = induction_copyscore(m, vocab, args.L, device)
        qk = max(PR.induction_per_head(m, probe, args.L).values())
        traj.append(dict(step=stp, c_copy=cc, induction=ind, qk=qk))
        print(f"  {stp:>7} {cc:>8.3f} {ind:>10.3f} {qk:>6.3f}", flush=True)
        json.dump({"model": args.model, "traj": traj}, open(os.path.join(RESULTS, args.out), "w"), indent=2)
        del m; torch.cuda.empty_cache()

    # does C_copy lead the real-induction onset?
    ind_onset = next((t["step"] for t in traj if t["induction"] > 0.4), None)
    cc_at = next((t["c_copy"] for t in traj if t["induction"] > 0.4), None)
    cc_before = [t["c_copy"] for t in traj if ind_onset is not None and t["step"] < ind_onset]
    print(f"\n  ===== real-induction onset (copy>0.4) at step {ind_onset}, C_copy there = {cc_at} =====", flush=True)
    print(f"  C_copy before onset: {[round(c,3) for c in cc_before]}  (did it climb ahead of induction?)", flush=True)
    print(f"  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
