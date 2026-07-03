"""
A1 周期表组装(强版)分析:用冻结的 160m 表(② 势垒律 + ① 地板律)联合预言 held-out (K,nu) 配置,对实测。
- ② 何时:t* = C / (nu × p^K) -> ln(1/t*) = ln(1/C) + ln(nu) - K×|ln p|;在校准子集拟合(|ln p|,C),预言 held-out t*。
- ① 会不会:forms? <=> nu ≥ 地板(K),地板 = 斜率×(K-1) + 基底(160m: 斜率0.05, 基底0.11);实测 forms = t* 非 None。
冻结表 = 校准子集(K∈{2,3} × nu∈{0.6,1.0});held-out = 其余全部(含 K=4 的外推 + 低 nu 的 ① 地板测试)。

  python assemble_analysis.py --grid results/bc_assemble_160m.json
"""
from __future__ import annotations
import argparse, json
import numpy as np

# 冻结的 160m ① 地板常数(来自 E3:floor K2/K3/K4 = 0.16/0.21/0.26)
FLOOR_SLOPE_160M = 0.05
FLOOR_BASE_160M = 0.11
CAL_KS = [2, 3]; CAL_NUS = [0.6, 1.0]                                         # 校准子集(易形成)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    args = ap.parse_args()
    raw = json.load(open(args.grid))
    grid = {}
    for k, v in raw.items():
        K, nu = eval(k); grid[(int(K), float(nu))] = v                       # str "(2, 1.0)" -> (2,1.0)

    cal = [(K, nu, grid[(K, nu)]) for K in CAL_KS for nu in CAL_NUS if grid.get((K, nu))]
    print(f"=== A1 周期表组装(强版):用冻结 160m 表联合预言 held-out ===")
    print(f"校准子集(冻结): {[(K, nu) for K, nu, _ in cal]}")
    # ② 拟合 ln(1/t*) = a×ln(nu) - b×K + c
    A = np.array([[np.log(nu), -K, 1.0] for K, nu, t in cal]); y = np.array([np.log(1.0 / t) for K, nu, t in cal])
    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"② 冻结律: ln(1/t*) = {a:.2f}×ln(nu) − {b:.2f}×K + {c:.2f}  (密度指数 a≈1?, |ln p|={b:.2f})")

    cal_set = {(K, nu) for K, nu, _ in cal}
    ho = [(K, nu, t) for (K, nu), t in sorted(grid.items()) if (K, nu) not in cal_set]
    print(f"\n--- held-out 配置({len(ho)} 个)---")
    t_errs = []; form_correct = 0; form_total = 0
    for K, nu, t in ho:
        floor = FLOOR_SLOPE_160M * (K - 1) + FLOOR_BASE_160M
        pred_form = nu >= floor                                              # ① 预言: 形成?
        act_form = t is not None
        ok = "✓" if pred_form == act_form else "✗"
        form_total += 1; form_correct += int(pred_form == act_form)
        line = f"  K={K} nu={nu}: ①预言{'形成' if pred_form else '不形成'}(地板{floor:.2f}) 实测{'形成' if act_form else '不形成'} {ok}"
        if act_form:                                                         # ② t* 预言(仅对形成的)
            pred_t = 1.0 / np.exp(a * np.log(nu) - b * K + c); err = abs(pred_t - t) / t * 100
            t_errs.append(err); line += f" | ②预言t*={pred_t:.0f} 实测{t:.0f} 误差{err:.0f}%"
        print(line)
    print(f"\n=== 组装结果 ===")
    print(f"① 形成预言准确率: {form_correct}/{form_total} = {form_correct/form_total*100:.0f}%")
    if t_errs:
        print(f"② t* 联合留一: 平均|误差|={np.mean(t_errs):.0f}% 中位={np.median(t_errs):.0f}% (n={len(t_errs)})")
    print(f"→ 冻结表能在没见过的 (K,nu) 配置上联合预言 ①(会不会)+②(何时) = 周期表能算")


if __name__ == "__main__":
    main()
