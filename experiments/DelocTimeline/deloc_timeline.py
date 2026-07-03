"""
N14 单头→离域时间序(消 App E/D 矛盾)。矛盾:App D 说再生/形成是"单点成核"(一个头先点火),App E 说
真 LM 的归纳是"冗余集合"(切一个头别的顶上=Hydra)。和解假设 = 时间序:先单头成核,冗余是形成后慢慢
摊开的(delocalization),两者是同一过程的不同时刻。测法:扫 Pythia 160m 的全部预训练 checkpoint(184 本地
models/pythia-160m/step*),每个 checkpoint 记录:每头的 prefix-match 归纳分、最强头、超阈值头数、functional
copy(fp32)。预言:超阈值头数 = 先 0 → 1(成核)→ 缓慢增多(离域),而非一步到位多头齐现。

  python deloc_timeline.py --local_models /path/to/workdir/models
  python deloc_timeline.py --local_models ... --smoke
"""
from __future__ import annotations
import argparse, gc, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

RESULTS = os.path.join(os.path.dirname(__file__), "results")
REVS = ["step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128", "step256",
        "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000",
        "step128000", "step143000"]


def make_rep_batch(vocab, L, B, device, seed=0):
    g = np.random.default_rng(seed)
    half = g.integers(0, vocab, size=(B, L), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


@torch.no_grad()
def all_head_induction(model, rep, L):
    """per-head prefix-match induction score -> (n_layers, n_heads) numpy."""
    out = model(rep, output_attentions=True); attns = out.attentions; dev = rep.device
    dest = torch.arange(L, 2 * L - 1, device=dev); src = dest - L + 1; D = dest.numel()
    sc = []
    for A in attns:
        a = A[:, :, dest, :]; pick = a[:, :, torch.arange(D, device=dev), src]
        sc.append(pick.mean(dim=(0, 2)).float().cpu().numpy())
    return np.stack(sc)                                                     # (L, H)


@torch.no_grad()
def copy_score(model, rep, L):
    logits = model(rep).logits
    pos = torch.arange(L, 2 * L - 1, device=rep.device)
    tgt = rep[:, pos + 1]
    lp = logits[:, pos].float().log_softmax(-1)                             # fp32 softmax (bf16 collapses it)
    return float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--local_models", required=True)                        # e.g. /path/to/workdir/models
    ap.add_argument("--L", type=int, default=128); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thr_lo", type=float, default=0.15); ap.add_argument("--thr_hi", type=float, default=0.5)
    ap.add_argument("--out", default="deloc_timeline.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    revs = ["step0", "step512", "step143000"] if args.smoke else REVS
    print(f"device={device} model={args.model} revs={len(revs)} (N14 单头成核→冗余离域 时间序)", flush=True)
    res = []
    rep = None
    for rev in revs:
        path = os.path.join(args.local_models, args.model, rev)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                     attn_implementation="eager").to(device)
        if rep is None:
            rep = make_rep_batch(model.config.vocab_size, args.L, args.batch, device, 0)
        sc = all_head_induction(model, rep, args.L)
        cs = copy_score(model, rep, args.L)
        top = float(sc.max()); tl, th = np.unravel_index(int(sc.argmax()), sc.shape)
        n_lo = int((sc > args.thr_lo).sum()); n_hi = int((sc > args.thr_hi).sum())
        tops = sorted([(float(sc[l, h]), int(l), int(h)) for l in range(sc.shape[0]) for h in range(sc.shape[1])],
                      reverse=True)[:5]
        res.append(dict(rev=rev, step=int(rev[4:]), top=top, top_head=[int(tl), int(th)],
                        n_above_015=n_lo, n_above_05=n_hi, copy=cs, top5=tops))
        print(f"  {rev:>11}: top={top:.3f}@L{tl}H{th}  头数>0.15={n_lo:>2} >0.5={n_hi:>2}  copy={cs:.3f}", flush=True)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
        del model; gc.collect(); torch.cuda.empty_cache()
    forming = [r for r in res if r["n_above_05"] >= 1]
    if forming:
        first = forming[0]
        later = res[-1]
        print(f"\n=== N14 时间序 ===", flush=True)
        print(f"  首次成核(>0.5): {first['rev']}(头数 {first['n_above_05']})", flush=True)
        print(f"  最终({later['rev']}): 头数>0.5={later['n_above_05']} >0.15={later['n_above_015']}", flush=True)
        print(f"  判定: {'✅ 先单头/少头成核 → 冗余随训练摊开(离域=后续过程,App E/D 和解)' if first['n_above_05'] <= 2 and later['n_above_05'] > first['n_above_05'] else '一步多头齐现或未增长 — 如实报'}", flush=True)
    print("N14_DONE", flush=True)


if __name__ == "__main__":
    main()
