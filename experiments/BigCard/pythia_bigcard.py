"""
BIG-CARD tasks for ① (density floor) and ② (combinatorial barrier ∝ K) on LARGE pretrained Pythia (1b / 2.8b) that need
ONE clean 24G 3090. Built on the PROVEN pythia_barrier machinery (real_lm_f2_K: markov background + offset-sum conjunction,
continue-train a pretrained Pythia to learn a controllable-K skill). Two modes:
  --mode floor   : ①. For each K, sweep density nu DOWN -> floor nu* = lowest nu that still forms; slope over K = floor law.
  --mode barrier : ②. For each K (nu=1), measure t*; held-out: calibrate ln t* = c0+|ln p|*K on K=2,3 -> predict K=4,5.

24G memory (bitsandbytes is NOT installed on 184 -> use Adafactor, optimizer state ~0):
  1b   : --opt adamw                 (regular AdamW+bf16 ~14-16G; DIRECTLY comparable to the existing 160m/410m results)
  2.8b : --opt adafactor --grad_ckpt (bf16 weights+grad ~11G + Adafactor ~0 + checkpointed activations -> ~14-16G, fits 24G)
CAVEAT: 2.8b uses Adafactor (not AdamW) -> absolute t* may shift vs AdamW; the floor (① binary) and the slope (② barrier∝K)
are robust to optimizer choice. ALWAYS --smoke first on the real card to confirm (a) it forms and (b) peak VRAM < 24G.

VERIFIED lr (160m smoke, offset-sum task): adamw=1e-4 (3e-5 too slow, plateaus ~0.33), adafactor=3e-4 (1e-3 too high).
  # T1 ①1b   floor:  python pythia_bigcard.py --mode floor   --model EleutherAI/pythia-1b   --opt adamw     --lr 1e-4 --bf16 --out bc_floor_1b.json
  # T2 ①2.8b floor:  python pythia_bigcard.py --mode floor   --model EleutherAI/pythia-2.8b --opt adafactor --lr 3e-4 --grad_ckpt --bf16 --out bc_floor_28b.json
  # T3 ②2.8b barr:   python pythia_bigcard.py --mode barrier --model EleutherAI/pythia-2.8b --opt adafactor --lr 3e-4 --grad_ckpt --bf16 --Ks 2 3 4 5 --nus 1.0 --out bc_barr_28b.json
  #    smoke FIRST:  python pythia_bigcard.py --mode floor --model <m> --opt <o> --lr <lr> [--grad_ckpt] --bf16 --smoke   (confirms forms + peak VRAM<24G)
"""
from __future__ import annotations
import argparse, copy, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.optimization import Adafactor
import real_lm_f2_K as RK
from real_lm_f2_K import make_markov, gen_batch_K, conj_score

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def make_opt(model, name, lr):
    if name == "adafactor":                                              # state-free optimizer -> fits 2.8b in 24G
        return Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def train_tstar(model, base_state, M, T, V, nu, K, opt_name, lr, max_steps, ee, thr, batch, seed, device):
    model.load_state_dict(base_state); model.train()
    opt = make_opt(model, opt_name, lr); rng = np.random.default_rng(seed)
    for s in range(0, max_steps + 1):
        if s % ee == 0:
            model.eval()
            if conj_score(model, M, T, V, K, device) > thr:
                model.train(); return s
            model.train()
        x = gen_batch_K(M, T, V, batch, nu, K, rng, device)
        out = model(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["floor", "barrier"], required=True)
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--nus", type=float, nargs="+", default=[1.0, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08])
    ap.add_argument("--offsets", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--opt", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--ee", type=int, default=10); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--bf16", action="store_true"); ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--revision", default="step143000"); ap.add_argument("--out", default="bc.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    RK.OFFSETS = args.offsets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = args.vocab; M = make_markov(V, 0, peak=4.0)
    dt = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dt,
                                                 attn_implementation="eager").to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    base_state = copy.deepcopy(model.state_dict())
    print(f"device={device} model={args.model} mode={args.mode} opt={args.opt} grad_ckpt={args.grad_ckpt} bf16={args.bf16}", flush=True)

    def peak():
        return torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    if args.smoke:                                                        # verify (a) forms, (b) peak VRAM < 24G
        t = train_tstar(model, base_state, M, args.T, V, 1.0, 2, args.opt, args.lr, min(args.max_steps, 1500),
                        args.ee, args.thresh, args.batch, 100, device)
        print(f"  SMOKE K=2 nu=1: t*={t} ({'FORMS' if t else 'NO-FORM'})  峰值显存={peak():.1f}G / 24G", flush=True)
        return

    if args.mode == "floor":                                             # ① floor: per K, lowest nu that still forms
        res = {}
        for K in args.Ks:
            floor = None
            for nu in sorted(args.nus, reverse=True):                    # high -> low; stop once it fails to form
                ts = [t for sd in range(args.seeds)
                      if (t := train_tstar(model, base_state, M, args.T, V, nu, K, args.opt, args.lr,
                                           args.max_steps, args.ee, args.thresh, args.batch, 100 + sd, device)) is not None]
                ok = len(ts) >= max(1, (args.seeds + 1) // 2)            # majority of seeds form
                print(f"  K={K} nu={nu}: formed {len(ts)}/{args.seeds} t*~{np.median(ts) if ts else None}  显存{peak():.1f}G", flush=True)
                if ok:
                    floor = nu
                else:
                    break
            res[f"K{K}"] = floor
            print(f"  => K={K} 地板 nu*={floor}", flush=True)
            json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)
        Ks = [k for k in args.Ks if res.get(f"K{k}")]
        if len(Ks) >= 2:
            fl = [res[f"K{k}"] for k in Ks]
            A, B = np.polyfit([k - 1 for k in Ks], fl, 1)
            print(f"\n=== ① 地板律: 临界密度 nu* = {A:.4f}×(部件数-1) + {B:.4f} (斜率={A:.4f} 基底={B:.4f}) ===", flush=True)
    else:                                                                # ② barrier ∝ K + held-out (K=2,3 -> predict K=4,5)
        tK = {}
        for K in args.Ks:
            for nu in args.nus:
                ts = [t for sd in range(args.seeds)
                      if (t := train_tstar(model, base_state, M, args.T, V, nu, K, args.opt, args.lr,
                                           args.max_steps, args.ee, args.thresh, args.batch, 100 + sd, device)) is not None]
                v = float(np.median(ts)) if ts else None
                tK[(K, nu)] = v
                print(f"  K={K} nu={nu}: t*={v} (n={len(ts)}/{args.seeds})  显存{peak():.1f}G", flush=True)
                json.dump({str(k): v for k, v in tK.items()}, open(os.path.join(RESULTS, args.out), "w"), indent=2)
        nu1 = {K: tK.get((K, 1.0)) for K in args.Ks if tK.get((K, 1.0))}  # held-out at nu=1
        cal = [k for k in [2, 3] if k in nu1]
        if len(cal) >= 2:
            A, B = np.polyfit(cal, [np.log(nu1[k]) for k in cal], 1)      # ln t* = A*K + B
            print(f"\n=== ② 势垒∝部件数: |ln p|={A:.3f} (p={np.exp(-A):.3f}) ===", flush=True)
            for k in args.Ks:
                if k not in cal and k in nu1:
                    pred = float(np.exp(A * k + B)); err = (pred - nu1[k]) / nu1[k] * 100
                    print(f"  留一预测 K={k}: 预测 t*={pred:.0f} 实测 {nu1[k]:.0f} 误差 {err:+.0f}%", flush=True)
    print("BC_DONE", flush=True)


if __name__ == "__main__":
    main()
