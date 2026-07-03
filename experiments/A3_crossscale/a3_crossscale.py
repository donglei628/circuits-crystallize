"""
A3 — CROSS-SCALE FORMATION TIME: does the real-LM induction t* scale with model size in a rate-law-consistent way?

Pure analysis (no GPU): reads the A1 formation curves for pythia-70m/160m/410m, extracts each t* (the step where
induction snaps past 0.5*final, interpolated in log-step) and the snap interval, and reports t* vs model size. This
is the cross-scale spine of the "9th point" (P8): a real-LM formation-time series to confront with the rate law.

  python a3_crossscale.py
  (expects ../A1_formation_time/data/a1_pythia{70m,160m,410m}.json — A1's pulled-back outputs)
"""
import json, os, glob
import numpy as np

PARAMS = {"pythia-70m": 70e6, "pythia-160m": 160e6, "pythia-410m": 410e6, "pythia-1b": 1.0e9, "pythia-1.4b": 1.4e9}
A1_DIR = os.path.join(os.path.dirname(__file__), "..", "A1_formation_time", "data")


def tstar_of(rows):
    rows = sorted(rows, key=lambda r: r["step"])
    steps = np.array([r["step"] for r in rows]); ind = np.array([r["induction"] for r in rows])
    final = ind[-1]; thr = 0.5 * final
    tstar = None; snap = None
    for i in range(1, len(steps)):
        if ind[i - 1] < thr <= ind[i]:
            ls0, ls1 = np.log10(max(steps[i - 1], 1)), np.log10(steps[i])
            frac = (thr - ind[i - 1]) / (ind[i] - ind[i - 1] + 1e-9)
            tstar = 10 ** (ls0 + frac * (ls1 - ls0)); snap = (int(steps[i - 1]), int(steps[i])); break
    peak = float(ind.max()); peak_step = int(steps[int(np.argmax(ind))])
    return tstar, snap, final, peak, peak_step


def main():
    files = sorted(glob.glob(os.path.join(A1_DIR, "a1_pythia*.json")))
    print(f"reading {len(files)} A1 curves from {A1_DIR}")
    pts = []
    for f in files:
        d = json.load(open(f))
        name = d["model"].split("/")[-1]
        if "smoke" in os.path.basename(f):
            continue
        tstar, snap, final, peak, peak_step = tstar_of(d["rows"])
        P = PARAMS.get(name, float("nan"))
        pts.append((name, P, tstar, snap, final, peak, peak_step))
        print(f"  {name:>14} ({P/1e6:.0f}M params): t*={None if tstar is None else round(tstar)}  "
              f"snap{snap}  final={final:.3f}  peak={peak:.3f}@{peak_step}")
    pts = [p for p in pts if p[2] is not None and np.isfinite(p[1])]
    print("\n  ===========  A3: t* vs model size  ===========")
    if len(pts) >= 3:
        P = np.array([p[1] for p in pts]); T = np.array([p[2] for p in pts])
        order = np.argsort(P); P, T = P[order], T[order]
        # log-log slope: t* ~ P^a
        a, b = np.polyfit(np.log(P), np.log(T), 1)
        r2 = 1 - ((np.log(T) - (a * np.log(P) + b)) ** 2).sum() / ((np.log(T) - np.log(T).mean()) ** 2).sum()
        print(f"  t* vs params (log-log): t* ~ P^{a:+.2f}  (R^2={r2:.2f})")
        direction = ("RISES with size (bigger model forms induction LATER)" if a > 0.1 else
                     "FALLS with size (bigger model forms induction EARLIER)" if a < -0.1 else
                     "roughly CONSTANT across size")
        print(f"  ==> formation time {direction}")
        print(f"  (this is the real-LM t*(scale) series; pairing it with a per-model driving force / addressing")
        print(f"   complexity is the remaining step to confront the rate law quantitatively — P8 future work)")
    else:
        print(f"  only {len(pts)} usable points; need 70m+160m+410m A1 outputs present.")


if __name__ == "__main__":
    main()
