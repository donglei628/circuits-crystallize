"""
REGEN-FORCE-BASIN (the causal test of the whirlpool/basin hypothesis on Pythia). The microscope showed Pythia's
regeneration WANDERS across ~12 heads (stability 0.26) and the function never returns, while the toy SETTLES into one
basin (stability 0.94) and recovers. Hypothesis: the failure is the MANY-BASIN wandering, not an inability to
regenerate. Decisive test: FORCE the gradient into ONE basin -- after lesioning, designate a single free head D and
let ONLY head D's QK (query/key/value) and OV (output) update; freeze everything else. Now there is exactly ONE basin
available. If the induction FUNCTION (ICL) recovers, the hypothesis is confirmed: Pythia CAN regenerate when constrained
to one basin; the earlier failure was the wandering.

  HF_ENDPOINT=https://hf-mirror.com python regen_force_basin.py --model EleutherAI/pythia-160m
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pythia_redundancy as PR

RESULTS = PR.RESULTS


def induction_train_batch(vocab, B, T, rng, device):
    half = rng.integers(0, vocab, size=(B, T // 2), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m"); ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--train_steps", type=int, default=800); ap.add_argument("--ee", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--train_batch", type=int, default=8); ap.add_argument("--train_seq", type=int, default=256)
    ap.add_argument("--probe_batch", type=int, default=8); ap.add_argument("--max_clamp", type=int, default=15)
    ap.add_argument("--nfree", type=int, default=1, help="how many free heads to unfreeze (1 = single basin)")
    ap.add_argument("--force_lh", default=None, help="force a SPECIFIC head 'l,h' as the basin (else auto best free head)")
    ap.add_argument("--out", default="regen_force_basin.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    probe = PR.make_batch(tok.vocab_size, args.L, args.probe_batch, device, seed=0)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=f"step{args.step}", dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    attns, proj_name, hs, nh = PR.get_layers_and_proj(model); nl = len(attns)
    print(f"device={device} {args.model} {nl}L x {nh}H  force {args.nfree} basin(s)", flush=True)

    base = PR.icl_drop(model, probe, args.L)
    scores = PR.induction_per_head(model, probe, args.L)
    ranked = sorted(scores, key=lambda lh: scores[lh], reverse=True)
    PR.ABLATE.clear(); PR.install_hooks(attns, proj_name, hs); S = []
    for k in range(1, args.max_clamp + 1):
        l, h = ranked[k - 1]; PR.ABLATE.setdefault(l, set()).add(h); S.append((l, h))
        if PR.icl_drop(model, probe, args.L) <= 0.20 * base:
            break
    Sset = set(S)
    PR.ABLATE.clear()                                       # oneshot lesion (no maintained clamp)
    with torch.no_grad():
        for l, h in S:
            getattr(attns[l], proj_name).weight[:, h * hs:(h + 1) * hs] = 0.0
    print(f"  baseline ICL={base:.2f}; |S|={len(S)}; crater ICL={PR.icl_drop(model, probe, args.L):.2f}", flush=True)

    # designate the free head(s) D
    if args.force_lh:                                       # a SPECIFIC head (for the basin-landscape census)
        l, h = map(int, args.force_lh.split(",")); D = [(l, h)]
    else:                                                   # auto: best-positioned free head(s)
        free_ranked = [lh for lh in ranked if lh not in Sset]; D = free_ranked[:args.nfree]
    in_seed = all(d in Sset for d in D)                     # is the forced head one of the ORIGINAL lesioned heads?
    print(f"  designated basin head(s) D = {D}  (only these update; everything else frozen)", flush=True)

    # build gradient masks: only head-D rows of qkv + head-D cols of dense are allowed to update
    params, masks = [], []
    by_layer = {}
    for (dl, dh) in D:
        by_layer.setdefault(dl, []).append(dh)
    for dl, dhs in by_layer.items():
        qkv = attns[dl].qkv if hasattr(attns[dl], "qkv") else attns[dl].query_key_value
        dense = getattr(attns[dl], proj_name)
        d = dense.weight.shape[0]
        mq = torch.zeros_like(qkv.weight); md = torch.zeros_like(dense.weight)
        for dh in dhs:
            for blk in range(3):                           # q,k,v blocks each [d, d]
                mq[blk * d + dh * hs: blk * d + (dh + 1) * hs, :] = 1.0
            md[:, dh * hs:(dh + 1) * hs] = 1.0
        params += [qkv.weight, dense.weight]; masks += [mq, md]
    opt = torch.optim.AdamW(params, lr=args.lr)

    rng = np.random.default_rng(0); traj = []
    print(f"\n  {'step':>5} {'ICL%':>6} {'Dhead_indscore':>14}", flush=True)
    for s in range(0, args.train_steps + 1):
        if s % args.ee == 0:
            model.eval(); sc = PR.induction_per_head(model, probe, args.L); icl = PR.icl_drop(model, probe, args.L)
            dscore = float(np.mean([sc[d] for d in D]))
            traj.append(dict(step=s, icl=icl, icl_pct=icl / base, Dscore=dscore))
            print(f"  {s:>5} {icl/base*100:>5.0f}% {dscore:>14.2f}", flush=True)
            json.dump(traj, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
            model.train()
        if s < args.train_steps:
            x = induction_train_batch(tok.vocab_size, args.train_batch, args.train_seq, rng, device)
            out = model(x, labels=x); out.loss.backward()
            for p, m in zip(params, masks):                # freeze everything except head-D slices
                if p.grad is not None:
                    p.grad.mul_(m)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()

    # Q1: is the regenerated OV the COPY operator? (invariant test) -- sampled token-space copy-fidelity of head D
    @torch.no_grad()
    def ov_copy_fid(dl, dh, n=400):
        qkv = attns[dl].qkv if hasattr(attns[dl], "qkv") else attns[dl].query_key_value
        d = getattr(attns[dl], proj_name).weight.shape[0]
        Wv = qkv.weight[2 * d + dh * hs:2 * d + (dh + 1) * hs, :]            # (hs, d)
        Wo = getattr(attns[dl], proj_name).weight[:, dh * hs:(dh + 1) * hs]  # (d, hs)
        WE = model.get_input_embeddings().weight; WU = model.get_output_embeddings().weight
        idx = torch.randperm(WE.shape[0], device=device)[:n]
        M = WU @ (Wo @ Wv) @ WE[idx].t()                                     # (vocab, n)
        return float((M.argmax(0) == idx).float().mean())
    ov_fid = None
    try:
        ov_fid = float(np.mean([ov_copy_fid(d[0], d[1]) for d in D]))
    except Exception as e:
        print(f"  (ov_copy_fid failed: {str(e)[:60]})", flush=True)

    mx = max(t["icl_pct"] for t in traj); dmax = max(t["Dscore"] for t in traj)
    summary = dict(D=D, in_seed=in_seed, max_icl_pct=mx, max_Dscore=dmax, ov_copy_fid=ov_fid,
                   final_icl_pct=traj[-1]["icl_pct"])
    json.dump({"traj": traj, "summary": summary}, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
    print(f"\n  ===== forced-basin result (head D={D}, in_original_seed={in_seed}) =====", flush=True)
    print(f"  max ICL recovered = {mx*100:.0f}% of base; D-head induction reached {dmax:.2f}; "
          f"OV copy-fidelity (Q1 invariant) = {ov_fid}", flush=True)
    print(f"  => {'REGENERATION SUCCEEDS in this basin' if mx > 0.45 else 'this head is NOT a viable induction basin'}", flush=True)
    print(f"\n  saved results/{args.out}", flush=True)


if __name__ == "__main__":
    main()
