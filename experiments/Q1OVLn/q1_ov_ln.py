"""
N4 Q1 OV 权重级(折 LayerNorm + 传播有效嵌入)。q1_identity 的 OV 原始权重对角分位只 ~0.54;smoke 证明只折
LN gamma 不够(0.545→0.542)——因为头读的是「层 0..li-1 加工过的残差」,不是裸词嵌入。三档探针递进:
  raw : M_OV = U @ Wo @ Wv @ E^T                        (q1_identity 原始)
  fold: 折两处 LN gamma(输入 ln1 行归一化×γ1,输出 ×γf)  (smoke 证明不够)
  eff : 传播有效嵌入 —— 单 token 前向捕获进入层 li 的真实残差 h_j(含层 0..li-1 的加工),
        M_OV[i,j] = (U_i) @ LN_f( Wo @ Wv @ LN1(h_j) )   (logit-lens 式;V/O 路径无位置码,单 token 合法)
对照:同层非归纳头照样算 —— 归纳头 eff 对角分位↑而对照不↑ ⇒ OV=复制算子在权重级成立(Q1 取值半边)。

  python q1_ov_ln.py --models EleutherAI/pythia-160m EleutherAI/pythia-410m
  python q1_ov_ln.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from k_scale import make_rep_batch, induction_score
from q1_identity import head_weights, diag_metrics

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def ln_apply(X, gamma, eps=1e-5):
    """proper LN row-wise: center/normalize then learned gamma. beta shifts rows uniformly (rank-invariant) -- omitted."""
    mu = X.mean(-1, keepdim=True); sd = X.var(-1, keepdim=True, unbiased=False).add(eps).sqrt()
    return (X - mu) / sd * gamma


@torch.no_grad()
def eff_hidden(model, sub, li, device, bs=256):
    """propagated effective embedding: the residual ENTERING layer li when the input is the bare token (pos 0).
    Captures layers 0..li-1 processing. V/O path is rotary-free so a single-token forward is legitimate."""
    gn = model.gpt_neox
    grabbed = []
    def hook(mod, inp, kw=None):
        grabbed.append(inp[0][:, 0, :].detach())
    h = gn.layers[li].register_forward_pre_hook(hook)
    try:
        for i in range(0, len(sub), bs):
            model(sub[i:i + bs].unsqueeze(1))
    finally:
        h.remove()
    return torch.cat(grabbed, 0)                                             # (N, d)


def probes(model, gn, li, hi, hd, Es, Us, Heff, g1, gf):
    Wq, Wk, Wv, Wo = head_weights(gn, li, hi, hd)
    with torch.no_grad():
        raw = diag_metrics(Us @ (Wo @ (Wv @ Es.T)))
        fold = diag_metrics((Us * gf) @ (Wo @ (Wv @ ln_apply(Es, g1).T)))
        out = (Wo @ (Wv @ ln_apply(Heff, g1).T)).T                           # (N, d) head output per source token
        eff = diag_metrics(Us @ ln_apply(out, gf).T)                         # logit-lens: proper LN_f on the component
    return raw, fold, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["EleutherAI/pythia-160m", "EleutherAI/pythia-410m"])
    ap.add_argument("--nsub", type=int, default=800); ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--out", default="q1_ov_ln.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.models = ["EleutherAI/pythia-160m"]; args.nsub = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = []
    for mn in args.models:
        model = AutoModelForCausalLM.from_pretrained(mn, revision="step143000", dtype=torch.float32,
                                                     attn_implementation="eager").to(device)
        gn = model.gpt_neox; cfg = model.config; H = cfg.num_attention_heads; hd = cfg.hidden_size // H; V = cfg.vocab_size
        WE = gn.embed_in.weight.detach(); WU = model.embed_out.weight.detach()
        rep = make_rep_batch(V, args.L, 4, device, 0)
        _, (li, hi) = induction_score(model, rep, args.L)
        rng = np.random.default_rng(0); sub = torch.from_numpy(rng.choice(V, args.nsub, replace=False)).to(device)
        Es = WE[sub]; Us = WU[sub]
        g1 = gn.layers[li].input_layernorm.weight.detach(); gf = gn.final_layer_norm.weight.detach()
        Heff = eff_hidden(model, sub, li, device)                            # (N,d) 传播有效嵌入
        hr = (hi + H // 2) % H                                               # 对照头(同层错开)
        r_i = probes(model, gn, li, hi, hd, Es, Us, Heff, g1, gf)
        r_c = probes(model, gn, li, hr, hd, Es, Us, Heff, g1, gf)
        names = ["raw", "fold", "eff"]
        rec = dict(model=mn, ind_head=[li, hi], ctrl_head=[li, hr], nsub=args.nsub)
        print(f"  {mn}: ind=({li},{hi}) ctrl=({li},{hr})", flush=True)
        for nm, mi, mc in zip(names, r_i, r_c):
            rec[f"ind_{nm}"] = dict(zip(["am", "pct", "pos"], mi)); rec[f"ctrl_{nm}"] = dict(zip(["am", "pct", "pos"], mc))
            print(f"    {nm:>4}: 归纳头 argmax={mi[0]:.3f} 分位={mi[1]:.3f} 正性={mi[2]:.3f}   | 对照 分位={mc[1]:.3f}", flush=True)
        res.append(rec)
        json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
        del model; torch.cuda.empty_cache()
    print("\n=== N4 判定:eff(传播嵌入)下归纳头对角分位显著>对照 ⇒ OV=复制算子在权重级成立;若仍 ~0.5 则如实报「复制身份只在功能级、权重级被残差流合成掩盖」===", flush=True)
    print("N4_DONE", flush=True)


if __name__ == "__main__":
    main()
