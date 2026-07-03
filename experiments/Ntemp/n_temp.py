"""
N-temp — IS THE RATE ARRHENIUS IN TEMPERATURE? (pins the "temperature T <-> noise" row; independent of M2.)

CNT rate: J ∝ exp(-ΔG*/kT). The SGD "temperature" T ∝ lr/batch (smaller batch = more gradient noise = hotter). If the
snap is THERMALLY activated, hotter (smaller batch) -> faster, and ln(rate) vs 1/T (∝ batch) is LINEAR with slope
-ΔG*/k -> a temperature-route measurement of the barrier. If instead formation is gradient/terrain-driven (our earlier
"no thermal nose"), colder (bigger batch) is faster -> ANTI-Arrhenius -> the temperature does NOT map thermally (the
rate's real Arrhenius axis is the DRIVING FORCE exp(-b/ΔL*), not T).

Sweep batch (temperature), measure the formation rate λ = 1/(median t* - t0) and the cross-seed CV (Poisson stochasticity
grows when hotter, per E1). Report the Arrhenius plot ln λ vs batch.

  python n_temp.py --batches 16 32 64 128 256 --seeds 10
  python n_temp.py --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from seed_tool import _data, _new_model, KEY_LEN, KEY_POOL, VAL_POOL
from run_F1 import acc_and_ce
from run_expA import RESULTS

T0 = 550.0


def one_run(seed, batch, device, max_steps, eval_every):
    pool, ev, vocab, rng = _data(seed, device)
    model = _new_model(vocab, seed, device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    formed = None
    for st in range(max_steps + 1):
        if st % eval_every == 0:
            a, _ = acc_and_ce(model, *ev)
            if formed is None and not np.isnan(a) and a >= 0.80:
                formed = st; break
        model.train()
        idx = torch.from_numpy(rng.integers(pool.shape[0], size=batch)).to(device); tok = pool[idx]
        loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, vocab.size), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return formed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--out", default="n_temp.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.batches, args.seeds = [32, 256], 3
    print(f"device={device} batches={args.batches} seeds={args.seeds}", flush=True)
    rows = []
    for batch in args.batches:
        for seed in range(args.seeds):
            t = one_run(seed, batch, device, args.max_steps, args.eval_every)
            rows.append(dict(batch=batch, seed=seed, tstar=t))
            json.dump(rows, open(os.path.join(RESULTS, args.out), "w"), indent=2, default=float)
        sub = [x["tstar"] for x in rows if x["batch"] == batch and x["tstar"] is not None]
        print(f"  batch{batch:>4}: formed {len(sub)}/{args.seeds}  t*med={np.median(sub) if sub else float('nan'):.0f}", flush=True)
    analyze(rows, args.batches)
    print(f"\n  saved results/{args.out}")


def analyze(rows, batches):
    print("\n  ===========  N-temp: Arrhenius rate vs temperature (T ∝ 1/batch)  ===========")
    print(f"  {'batch':>6} {'T~1/b':>7} {'t*med':>7} {'lambda':>9} {'lnλ':>7} {'CV':>6}")
    pts = []
    for b in batches:
        sub = [x["tstar"] for x in rows if x["batch"] == b and x["tstar"] is not None]
        if len(sub) < 2:
            continue
        tm = float(np.median(sub)); lam = 1.0 / max(tm - T0, 10); cv = float(np.std(sub) / np.mean(sub))
        pts.append((b, lam, cv, tm)); print(f"  {b:>6} {1.0/b:>7.4f} {tm:>7.0f} {lam:>9.5f} {np.log(lam):>7.2f} {cv:>6.2f}")
    if len(pts) >= 3:
        b = np.array([p[0] for p in pts]); lam = np.array([p[1] for p in pts]); cv = np.array([p[2] for p in pts])
        # Arrhenius: ln λ vs 1/T (∝ batch). slope<0 (hotter=faster) = thermal; slope>0 (colder=faster) = anti-thermal
        slope = np.polyfit(b, np.log(lam), 1)[0]
        print(f"\n  ln λ vs batch(∝1/T): slope = {slope:+.4f}")
        if slope < -0.001:
            print(f"  => THERMAL (hotter/smaller-batch = faster): Arrhenius in T holds; barrier crossed thermally.")
        elif slope > 0.001:
            print(f"  => ANTI-THERMAL (colder/bigger-batch = faster): NOT thermal activation -> the 'temperature' does")
            print(f"     NOT map thermally; the rate's Arrhenius axis is the DRIVING FORCE (exp(-b/ΔL*)), not T.")
        else:
            print(f"  => rate ~ insensitive to temperature.")
        print(f"  CV vs batch: CV={cv.min():.2f}-{cv.max():.2f}; rising CV at small batch = Poisson stochasticity (E1).")


if __name__ == "__main__":
    main()
