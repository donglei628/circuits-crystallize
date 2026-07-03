"""
N2 再生两层分离(晶种"真降势垒" vs 只是"填浅伤口")。同位点(SAME SITE):把训练好的电路权重按强度 frac
线性混回一份**新的随机初始**(不重定位、不换头),测再生时间 T_regrow。分离两层:
  - 层1 平凡(驱动力/伤口深度 ΔL*):晶种越强、伤口越浅 → 初始 loss(减地板)随 frac 平滑下降。
  - 层2 势垒(结构阈值 s*):若 T_regrow 在某个 frac 处 SHARP 骤降、且比 wound 平滑曲线更陡 → 晶种是靠跨过
    结构阈值/降势垒起效,而非只把伤口填浅。呼应 seed_tool 的 s*≈0.18(结构非范数)。
交叉一个驱动力轴 nu:每个 frac 上量 a_r(frac)=T_regrow 对 1/nu 的斜率(=势垒×f(frac));若斜率随 frac 下降
=晶种降势垒。这是 M2(Pythia 整数 s 两轴)的 toy 连续版对照。

  python regen_two_layer.py --fracs 0 0.1 0.18 0.25 0.4 0.6 0.8 --nus 0.5 1.0 --seeds 4
  python regen_two_layer.py --smoke
"""
from __future__ import annotations
import argparse, json, os, copy
import numpy as np
import torch
from real_lm_f2 import make_markov, build
from real_lm_f2_K import gen_batch_K, conj_score
import real_lm_f2_K as RK

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def train_reference(d, H, V, T, M, K, nu, lr, max_steps, ee, thr, device):
    """train a reference model until the K-conjunction forms (+polish); return its trained state_dict and formed-step."""
    torch.manual_seed(0); m = build(d, H, V, T, device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(0); formed = None
    for s in range(max_steps + 1):
        if s % ee == 0:
            m.eval()
            if conj_score(m, M, T, V, K, device) > thr: formed = s; break
            m.train()
        x = gen_batch_K(M, T, V, 64, nu, K, rng, device)
        out = m(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    for _ in range(400):                                                 # polish to fully consolidate the circuit
        x = gen_batch_K(M, T, V, 64, nu, K, rng, device)
        out = m(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    return copy.deepcopy(m.state_dict()), formed


def blend_state(theta_star, theta_rand, frac):
    """SAME-SITE seed: interpolate a fraction of the trained weights into a fresh random init (no relocation).
    Only blend floating-point weights; copy non-float buffers (rotary inv_freq, causal masks) verbatim."""
    out = {}
    for k in theta_star:
        a = theta_star[k]
        out[k] = frac * a + (1.0 - frac) * theta_rand[k] if a.is_floating_point() else theta_rand[k].clone()
    return out


def regrow(d, H, V, T, M, K, nu, lr, theta_star, frac, max_steps, ee, thr, floor, seed, device):
    """seed at strength frac (same site) on a fresh random init; measure wound(ΔL*_init) and steps to re-form."""
    torch.manual_seed(2000 + seed); theta_rand = build(d, H, V, T, device).state_dict()
    m = build(d, H, V, T, device); m.load_state_dict(blend_state(theta_star, theta_rand, frac))
    rng = np.random.default_rng(seed)
    m.eval()
    with torch.no_grad():
        x = gen_batch_K(M, T, V, 256, nu, K, rng, device)
        wound = float(m(x, labels=x).loss) - floor                       # ΔL*_init proxy = initial loss above floor
        init_cs = float(conj_score(m, M, T, V, K, device))
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for s in range(max_steps + 1):
        if s % ee == 0:
            m.eval()
            if conj_score(m, M, T, V, K, device) > thr: m.train(); return s, wound, init_cs
            m.train()
        x = gen_batch_K(M, T, V, 64, nu, K, rng, device)
        out = m(x, labels=x); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    return None, wound, init_cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.1, 0.18, 0.25, 0.4, 0.6, 0.8])
    ap.add_argument("--nus", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=64); ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--ee", type=int, default=25); ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--ref_steps", type=int, default=8000)
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out", default="regen_two_layer.json"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.fracs = [0.0, 0.5]; args.nus = [1.0]; args.seeds = 2; args.max_steps = 3000; args.ref_steps = 3000
    RK.OFFSETS = args.offsets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = make_markov(args.vocab, 0)
    print(f"device={device} K={args.K} fracs={args.fracs} nus={args.nus} (N2 再生两层分离: 同位点晶种 vs 驱动力)", flush=True)
    theta_star, formed = train_reference(args.d, args.heads, args.vocab, args.T, M, args.K, 1.0, args.lr,
                                         args.ref_steps, args.ee, args.thresh, device)
    print(f"  参考电路已形成 @ step {formed}", flush=True)
    ref = build(args.d, args.heads, args.vocab, args.T, device); ref.load_state_dict(theta_star); ref.eval()
    rng0 = np.random.default_rng(7); floors = {}
    with torch.no_grad():
        for nu in args.nus:
            x = gen_batch_K(M, args.T, args.vocab, 256, nu, args.K, rng0, device)
            floors[nu] = float(ref(x, labels=x).loss)
    print(f"  floor_loss per nu = { {k: round(v,4) for k,v in floors.items()} }", flush=True)

    res = []
    for frac in args.fracs:
        for nu in args.nus:
            trip = [regrow(args.d, args.heads, args.vocab, args.T, M, args.K, nu, args.lr, theta_star, frac,
                           args.max_steps, args.ee, args.thresh, floors[nu], sd, device) for sd in range(args.seeds)]
            ts = [t for t, _, _ in trip if t is not None]
            med = float(np.median(ts)) if ts else None
            wound = float(np.mean([w for _, w, _ in trip])); init_cs = float(np.mean([c for _, _, c in trip]))
            res.append(dict(frac=frac, nu=nu, med_tregrow=med, n=len(ts), wound=wound, init_cs=init_cs,
                            all=[t for t, _, _ in trip]))
            print(f"  frac={frac} nu={nu}: T_regrow={med} (n={len(ts)}/{args.seeds}) "
                  f"wound(ΔL*init)={wound:.3f} init_cs={init_cs:.3f}", flush=True)
            json.dump(res, open(os.path.join(RESULTS, args.out), "w"), indent=2)

    print("\n=== N2 两层分离 ===", flush=True)
    for frac in args.fracs:                                              # 层2: barrier slope a_r(frac) = T_regrow vs 1/nu
        pts = [(1.0 / r["nu"], r["med_tregrow"]) for r in res if r["frac"] == frac and r["med_tregrow"]]
        if len(pts) >= 2:
            xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
            print(f"  frac={frac}: a_r(势垒斜率 T_regrow~1/nu)={float(np.polyfit(xs, ys, 1)[0]):.0f}", flush=True)
    numax = max(args.nus)
    seq = sorted([(r["frac"], r["med_tregrow"]) for r in res if r["nu"] == numax and r["med_tregrow"]])
    wq = sorted([(r["frac"], r["wound"]) for r in res if r["nu"] == numax])
    print(f"  T_regrow(frac)@nu={numax}: {[(f, round(t)) for f, t in seq]}", flush=True)
    print(f"  wound ΔL*init(frac):     {[(f, round(w, 3)) for f, w in wq]}", flush=True)
    print(f"  判据: 若 T_regrow 在某 frac 处骤降而 wound 仍平滑 → 晶种真降势垒(结构非深度,=s*阈值)", flush=True)
    print("N2_DONE", flush=True)


if __name__ == "__main__":
    main()
