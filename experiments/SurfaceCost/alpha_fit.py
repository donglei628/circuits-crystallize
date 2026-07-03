"""
STEP-2 UNIFIED FIT — measure the model's surface-volume exponent alpha from the barrier shape, and run the
ONE-DEFINITION-MANY-LAWS anti-overfit checks (L1-L5).

Mapping:  nucleus size n  <-  transplant fraction s (amount of nucleus assembled)
          free energy ΔG(n) <- L(s) - L_plateau (loss above the shortcut plateau)
          driving force Δμ <- ΔL* = L_plateau - L(s=1)
CNT form: ΔG(n) = -Δμ·n + σ·n^α   (volume gain linear; surface cost ∝ n^α, α<1 -> a barrier hump)

Master parameter alpha is fit from the barrier SHAPE at each (fixed) config -> NO driving-force-sweep confound.
Then ONE alpha must satisfy:
  L1  the form fits (good R^2) and alpha is STABLE across the 4 driving-force configs
  L2  the predicted critical size n* = (alpha*sigma/Δμ)^(1/(1-alpha)) = the barrier PEAK location
  L3  n* matches the dynamic critical seed s* ~ 0.18 (seed_tool)
  L4  n*(Δμ) scaling exponent = -1/(1-alpha) (same alpha) across configs
  L5  barrier height ΔG* tracks the rate-law b (Arrhenius), Δμ scaling consistent

  python alpha_fit.py
"""
import glob, json, os
import numpy as np
from scipy.optimize import curve_fit

DATA = os.path.join(os.path.dirname(__file__), "data")
S_STAR_DYN = 0.18   # dynamic critical seed from seed_tool


def dG_model(s, dmu, sigma, alpha):
    return -dmu * s + sigma * np.power(np.clip(s, 1e-9, None), alpha)


def fit_one(path):
    d = json.load(open(path))
    Lp = d["L_plateau"]; rows = sorted(d["rows"], key=lambda r: r["s"])
    s = np.array([r["s"] for r in rows]); L = np.array([r["L_mean"] for r in rows]); n = np.array([r["n_mean"] for r in rows])
    dG = L - Lp                                  # free energy above the plateau
    dLstar = Lp - L[s == 1.0][0] if (s == 1.0).any() else Lp - L[-1]   # driving force
    try:
        p0 = [max(dLstar, 0.3), max(dG.max(), 0.5), 0.6]
        popt, _ = curve_fit(dG_model, s, dG, p0=p0, bounds=([0, 0, 0.1], [20, 50, 0.97]), maxfev=20000)
        dmu, sigma, alpha = popt
        pred = dG_model(s, *popt); r2 = 1 - ((dG - pred) ** 2).sum() / ((dG - dG.mean()) ** 2).sum()
    except Exception as e:
        return None
    nstar = (alpha * sigma / dmu) ** (1 / (1 - alpha)) if dmu > 0 else float("nan")
    speak = s[int(np.argmax(dG))]
    return dict(path=os.path.basename(path), dLstar=float(dLstar), alpha=float(alpha), sigma=float(sigma),
                dmu_fit=float(dmu), r2=float(r2), nstar=float(nstar), speak=float(speak), barrier=float(dG.max()))


def main():
    files = sorted(glob.glob(os.path.join(DATA, "surface_cost_vp*.json")))
    fits = [f for f in (fit_one(p) for p in files) if f]
    if not fits:
        print("no surface_cost_vp*.json found in", DATA); return
    print("  ===========  STEP-2: surface-volume exponent alpha + anti-overfit checks  ===========")
    print(f"  {'config':>22} {'ΔL*':>5} {'alpha':>6} {'R^2':>5} {'n*(fit)':>8} {'peak_s':>7} {'barrier':>8}")
    for f in fits:
        print(f"  {f['path']:>22} {f['dLstar']:>5.2f} {f['alpha']:>6.2f} {f['r2']:>5.2f} {f['nstar']:>8.3f} {f['speak']:>7.2f} {f['barrier']:>8.2f}")

    alphas = np.array([f["alpha"] for f in fits]); dL = np.array([f["dLstar"] for f in fits]); nst = np.array([f["nstar"] for f in fits])
    print("\n  ----- L1: form fits + alpha stable across driving force -----")
    print(f"    alpha = {alphas.mean():.2f} +- {alphas.std():.2f}  (R^2 range {min(f['r2'] for f in fits):.2f}-{max(f['r2'] for f in fits):.2f})")
    print(f"    {'STABLE -> L1 OK (one alpha across configs)' if alphas.std() < 0.12 else 'alpha drifts across configs -> L1 weak'}")
    a = alphas.mean()
    print("\n  ----- L2/L3: critical size n* vs dynamic critical seed s*~0.18 -----")
    nstar_med = float(np.median(nst))
    print(f"    median n*(from barrier fit) = {nstar_med:.3f}  vs dynamic s* = {S_STAR_DYN}")
    print(f"    {'MATCH -> L3 OK (barrier n* = transplant s*)' if abs(nstar_med - S_STAR_DYN) < 0.06 else 'mismatch -> L3 weak; the static and dynamic critical sizes differ'}")
    print("\n  ----- L4: does n* scale with ΔL* as the SAME alpha predicts? -----")
    if len(fits) >= 3 and (nst > 0).all():
        b_meas = np.polyfit(np.log(dL), np.log(nst), 1)[0]   # measured n* ~ ΔL*^b_meas
        b_pred = -1.0 / (1.0 - a)                            # CNT: n* ~ Δμ^(-1/(1-alpha))
        print(f"    measured n* ~ ΔL*^{b_meas:+.2f}   vs   alpha-predicted ^{b_pred:+.2f}")
        print(f"    {'CONSISTENT -> L4 OK (one alpha ties shape AND scaling)' if abs(b_meas - b_pred) < 0.8 else 'inconsistent -> L4 weak (shape-alpha != scaling-alpha)'}")
    else:
        print("    need >=3 configs with positive n*")
    print("\n  ----- VERDICT (how far we got) -----")
    print(f"    Tier A (form+alpha): {'YES' if alphas.std()<0.12 and min(f['r2'] for f in fits)>0.7 else 'partial'}  alpha={a:.2f}")
    print(f"    Tier B (+n*=s*):     {'YES' if abs(nstar_med-S_STAR_DYN)<0.06 else 'no'}")
    print(f"    (Tier C needs L4+L5: nucleation-theorem n* and rate-law b cross-checks)")
    print(f"  honest read: alpha={a:.2f} is the model's effective 'nucleation surface dimension' "
          f"({'~2/3: strikingly crystal-like' if abs(a-0.667)<0.12 else 'NOT 2/3: a non-geometric nucleation class -- itself a finding'})")


if __name__ == "__main__":
    main()
