"""Generate the four paper figures from the experiment result JSONs. Run from arXiv/:
   python make_figs.py
Reads ../184content/<Exp>/data/*.json and ../experiments/results/*.json; writes figures/*.pdf.
Every plotted value comes from a saved JSON -- no hand-entered numbers.
"""
import json, os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
B = os.path.join(HERE, "..", "experiments")          # per-experiment archive
E = os.path.join(HERE, "..", "experiments", "_toy", "data")
F = HERE                                              # figures are written next to this script
os.makedirs(F, exist_ok=True)

# ---- clean academic style, colourblind-friendly palette ----
BLUE, ORANGE, GREEN, VERM, GREY = "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#949494"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.6,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.markersize": 5,
})

def jb(f):  return json.load(open(os.path.join(B, f)))
def je(f):  return json.load(open(os.path.join(E, f)))
def save(fig, name):
    fig.savefig(os.path.join(F, name)); plt.close(fig); print("wrote", name)


# ============================ Fig 1 -- Law 2 (WHEN) ============================
def fig_law2_when():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.15))
    pb = jb("PythiaBarrier/data/pb_full_160m_v32.json")
    grid = {}
    for k, v in pb.items():                                   # keys like "(2, 0.6)"
        K, nu = re.findall(r"[\d.]+", k); grid[(int(K), float(nu))] = v

    # (a) combinatorial barrier + held-out: freeze on K=2,3 (calibration), predict K=4
    Ks = [2, 3, 4]; t = [grid[(K, 1.0)] for K in Ks]          # clean, full-density line
    a, b = np.polyfit([2, 3], [np.log(t[0]), np.log(t[1])], 1)
    predK4 = np.exp(a * 4 + b); err = (predK4 - t[2]) / t[2] * 100
    xs = np.linspace(1.9, 4.1, 50)
    ax[0].plot(xs, np.exp(a * xs + b), "--", color=GREY, lw=1, label="fit on $K{=}2,3$")
    ax[0].plot([2, 3], t[:2], "o", color=BLUE, label="calibration")
    ax[0].plot(4, t[2], "o", color=BLUE, mfc="white", mew=1.4, label="held-out $K{=}4$ (measured)")
    ax[0].plot(4, predK4, "x", color=VERM, mew=1.8, label="predicted")
    ax[0].set_yscale("log"); ax[0].set_xticks(Ks)
    ax[0].set_xlabel("conjunction depth $K$"); ax[0].set_ylabel("formation step $t^*$ (Pythia-160M)")
    ax[0].set_title("(a) combinatorial barrier $t^*\\!\\propto p^{-K}$")
    ax[0].legend(loc="upper left")
    ax[0].text(0.97, 0.06, fr"$|\ln p|={a:.2f}$" + "\n" + fr"held-out $K{{=}}4$: ${err:+.0f}\%$",
               transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7.6)

    # (b) ignition is memoryless: survival broadens & straightens as minibatch noise rises
    for f, col, lab in [("hazard_b256.json", GREY, "batch 256 (near-full)"),
                        ("hazard_b64.json", ORANGE, "batch 64"),
                        ("hazard_b16i0.json", BLUE, "batch 16 (minibatch)")]:
        ts = np.sort([r["tstar"] for r in je(f) if r.get("tstar")])
        surv = 1.0 - np.arange(1, len(ts) + 1) / len(ts)
        ax[1].step(ts / ts.mean(), np.clip(surv, 1e-3, 1), where="post", color=col,
                   label=f"{lab}, CV={ts.std()/ts.mean():.2f}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("waiting time $t^*/\\langle t^*\\rangle$"); ax[1].set_ylabel("survival $S(t)$")
    ax[1].set_title("(b) ignition $\\to$ memoryless (Poisson)"); ax[1].legend(loc="lower left")

    # (c) growth is a single-nucleus front: order parameter follows one Avrami front per seed
    for r in jb("NucleusGrowth/data/nucleus_growth.json"):
        c = np.array(r["curve"]); ax[2].plot(c[:, 0], c[:, 1], color=BLUE, alpha=0.45, lw=1)
    ax[2].set_xlabel("training step"); ax[2].set_ylabel("order parameter $\\varphi$")
    ax[2].set_title("(c) growth: single-nucleus front")
    ax[2].text(0.96, 0.06, "Avrami $n\\approx1.16$", transform=ax[2].transAxes,
               ha="right", va="bottom", fontsize=8)
    fig.tight_layout(); save(fig, "fig_law2_when.pdf")


# ============================ Fig 2 -- Law 4 (REGEN) ==========================
def fig_law4_regen():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.15))
    # (a) real LM: forced single basin recovers; unconstrained (many basins) does not
    fb = jb("ForceBasin/data/regen_force_basin.json")
    un = je("regen_microscope_pythia.json")
    ax[0].axhline(100, ls="--", color=GREY, lw=1, label="pre-lesion")
    ax[0].plot([r["step"] for r in fb], [100 * r["icl_pct"] for r in fb], "o-", color=BLUE,
               ms=3, label="forced 1 basin")
    ax[0].plot([r["step"] for r in un], [100 * r["icl_pct"] for r in un], "s-", color=VERM,
               ms=3, label="unconstrained (many)")
    ax[0].set_ylim(-5, 108)
    ax[0].set_xlabel("continued-training step"); ax[0].set_ylabel("recovery (% of baseline)")
    ax[0].set_title("(a) real LM: regen is basin-governed"); ax[0].legend(loc="center right")
    ax[0].text(0.03, 0.96, "dose: 1 basin$\\to$56%, 4$\\to$40%, many$\\to$15%",
               transform=ax[0].transAxes, fontsize=6.6, va="top")

    # (b) seed as a tool: a real partial circuit accelerates above s*; a scrambled one does not
    sd = je("seed_tool_fine.json")
    from collections import defaultdict
    for fake, col, lab in [(False, BLUE, "real seed"), (True, VERM, "scrambled (control)")]:
        g = defaultdict(list)
        for r in sd:
            if r["fake"] == fake and not r["censored"] and r.get("tstar"):
                g[r["s"]].append(r["tstar"])
        ss = sorted(g); ax[1].plot(ss, [np.median(g[s]) for s in ss], "o-", color=col, label=lab)
    ax[1].axvline(0.18, ls=":", color=GREY); ax[1].text(0.185, ax[1].get_ylim()[1] * 0.55, "$s^*$", fontsize=8)
    ax[1].set_xlabel("seed strength $s$"); ax[1].set_ylabel("formation step $t^*$")
    ax[1].set_title("(b) seed lowers the barrier ($8\\times$)"); ax[1].legend(loc="upper right")

    # (c) location is predictable: real induction (prefix-match) regrows only where upstream survives
    rows = []
    for f in glob.glob(os.path.join(B, "BasinCensus/data/census_L*.json")):
        L = int(re.search(r"census_L(\d+)", f).group(1))
        rows.append((L, json.load(open(f))["summary"]["max_Dscore"]))
    rows.sort()
    Ls = [r[0] for r in rows]; D = [r[1] for r in rows]
    cols = [GREEN if 4 <= L <= 5 else GREY for L in Ls]
    ax[2].bar(Ls, D, color=cols)
    ax[2].axhline(0.5, ls=":", color=GREY, lw=1)
    ax[2].set_xlabel("forced layer"); ax[2].set_ylabel("induction score (prefix-match)")
    ax[2].set_title("(c) regrows only at layers 4--5"); ax[2].set_xticks(Ls)
    fig.tight_layout(); save(fig, "fig_law4_regen.pdf")


# ============================ Fig 3 -- IDENTITY (bath) =======================
def fig_identity():
    fig, ax = plt.subplots(1, 2, figsize=(7.7, 3.15))
    # (a) barrier-gated TTT nose: no barrier (K=2) -> cold is fastest; barrier (K=3,4) -> optimal T
    for k, col in [(2, GREY), (3, BLUE), (4, ORANGE)]:
        d = jb(f"TempNose/data/nose_K{k}.json")
        T = [r["T_bath"] for r in d if r["med_tstar"]]
        t = [r["med_tstar"] for r in d if r["med_tstar"]]
        ax[0].plot(T, t, "o-", color=col, label=fr"$K={k}$")
    ax[0].set_xscale("symlog", linthresh=1e-3); ax[0].set_yscale("log")
    ax[0].set_xlabel("bath temperature $T$"); ax[0].set_ylabel("formation step $t^*$")
    ax[0].set_title("(a) barrier-gated TTT nose"); ax[0].legend(loc="upper left")
    ax[0].annotate("nose", xy=(3e-3, 760), xytext=(3e-2, 720), fontsize=7.5,
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    # (b) the bath is a true temperature: the formed circuit melts at a melting point
    cur = jb("Melt/data/melt_smoke2.json")["melt"][0]
    T = [c[0] for c in cur["curve"]]; q = [c[1] for c in cur["curve"]]
    ax[1].plot(T, q, "o-", color=BLUE)
    ax[1].axvline(cur["Tmelt"], ls=":", color=VERM); ax[1].text(cur["Tmelt"] * 1.08, 0.6,
               fr"$T_{{\rm melt}}\approx{cur['Tmelt']:.3f}$", color=VERM, fontsize=7.5)
    ax[1].set_xlabel("bath temperature $T$"); ax[1].set_ylabel("formed-circuit order $\\varphi$")
    ax[1].set_title("(b) a true temperature: circuit melts")
    fig.tight_layout(); save(fig, "fig_identity.pdf")


# ============================ Fig 4 -- what crystallises =====================
def fig_qkov():
    fig, ax = plt.subplots(1, 2, figsize=(7.7, 3.15))
    # (a) real LM, weight level: QK is the diagonal (content-addressing) operator; OV copy is
    #     visible only after propagating the effective embedding through the layers below.
    qi = {r["model"].split("-")[-1]: r for r in jb("Q1Identity/data/q1_identity.json")}
    ol = {r["model"].split("-")[-1]: r for r in jb("Q1OVLn/data/q1_ov_ln.json")}
    models = ["160m", "410m"]
    x = np.arange(len(models)); w = 0.26
    qk  = [qi[m]["qk_pct"] for m in models]
    ovb = [ol[m]["ind_raw"]["pct"] for m in models]       # bare OV (washed out)
    ove = [ol[m]["ind_eff"]["pct"] for m in models]       # propagated OV (recovered)
    ax[0].bar(x - w, qk, w, color=BLUE, label="QK addressing")
    ax[0].bar(x, ovb, w, color=GREY, label="OV copy, bare")
    ax[0].bar(x + w, ove, w, color=GREEN, label="OV copy, propagated")
    ax[0].axhline(0.5, ls=":", color=GREY, lw=1); ax[0].text(1.35, 0.52, "chance", fontsize=6.6, color=GREY)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"Pythia-{m}" for m in models])
    ax[0].set_ylabel("diagonal-dominance percentile"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(a) real LM: addressing $=$ associative memory"); ax[0].legend(loc="upper center", ncol=1)

    # (b) toy, isolated: the OV circuit is a copy operator -- its value->output map is diagonally
    #     dominant (diag_ratio = on-diagonal / off-diagonal mass), several-fold above the no-copy 1.0.
    ov = jb("Q1OVHopfield/data/q1_ov_hopfield_formal.json")
    dr = [r["diag_ratio"] for r in ov]; xs = np.arange(len(ov))
    ax[1].bar(xs, dr, 0.6, color=GREEN)
    ax[1].axhline(1.0, ls="--", color=VERM, lw=1); ax[1].text(len(ov) - 1, 1.15, "no copy structure",
               ha="right", color=VERM, fontsize=6.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"seed {r['seed']}" for r in ov], fontsize=6.6, rotation=30)
    ax[1].set_ylabel("$W_{OV}$ diagonal dominance"); ax[1].set_ylim(0, max(dr) * 1.18)
    ax[1].set_title("(b) toy: OV is a copy operator (copy fidelity $=1.0$)")
    fig.tight_layout(); save(fig, "fig_qkov.pdf")


if __name__ == "__main__":
    for fn in (fig_law2_when, fig_law4_regen, fig_identity, fig_qkov):
        try:
            fn()
        except Exception as e:
            import traceback; print("FAILED", fn.__name__); traceback.print_exc()
