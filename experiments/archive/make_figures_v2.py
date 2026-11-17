"""Figures 2-4 and results/summary.txt (every number quoted in the paper)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _s; _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.6, "legend.fontsize": 7, "legend.frameon": False})
OUT = []
COL = {"no store": "C0", "store": "C1", "store+substrings": "C2"}
LAB = {"no store": "DFE without store", "store": "DFE with store", "store+substrings": "store with substrings"}


def note(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


MEDFAC = 0.6745 * np.sqrt(2 / np.pi) / np.sqrt(2 / np.pi)   # median of |N(0, s)| = 0.6745 s


def med_iqr(g):
    """median and the distances to the 25% and 75% quantiles of a grouped series."""
    m = g.median()
    return m, m - g.quantile(0.25), g.quantile(0.75) - m


def label(ax, k, x=-0.12):
    ax.text(x, 1.03, f"({k})", transform=ax.transAxes, fontsize=9)


# ---------------------------------------------------------------- Fig. 2 (simulation 1)
d = pd.read_csv(os.path.join(RESULTS, "sim1_methods.csv"))
tr = pd.read_csv(os.path.join(RESULTS, "sim1_trajectories.csv"))
ours = d[d.method.isin(COL)]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
ours = ours.assign(err=(ours.F_hat - 0.9).abs())
NF1 = int((ours.copies - ours.search_shots).iloc[0])
for v in COL:
    g = ours[ours.method == v].groupby("n")
    m, lo, hi = med_iqr(g.err)
    lim = g.hit_limit.mean() > 0.5
    ax[0].errorbar(m.index, m, [lo, hi], color=COL[v], lw=0.9, capsize=2, label=LAB[v], zorder=2)
    ax[0].scatter(m.index[~lim], m[~lim], color=COL[v], s=14, zorder=3)
    ax[0].scatter(m.index[lim], m[lim], facecolor="white", edgecolor=COL[v], s=14, zorder=3)
    note(f"S1 {v}: median |FSTAB-Fhat|", m.round(4).to_dict(), "q25", (m - lo).round(4).to_dict(),
         "q75", (m + hi).round(4).to_dict(), "mean", g.err.mean().round(4).to_dict(),
         "mean F/FSTAB", g.ratio.mean().round(3).to_dict(),
         "P(opt)", g.optimal.mean().round(2).to_dict(), "hit limit", g.hit_limit.mean().round(2).to_dict(),
         "median shots", g.search_shots.median().to_dict(), "median it", g.iterations.median().to_dict(),
         "certified", g.certified.mean().round(2).to_dict())
floor1 = 0.6745 * np.sqrt((1 - 0.9**2) / NF1)
ax[0].axhline(floor1, color="0.4", ls="--", lw=0.8, label="statistical error of final DFE")
note("S1 statistical floor (median |error| of final DFE):", round(floor1, 5), "N_f", NF1)
ax[0].set_yscale("log")
ax[0].set_xlabel("number of qubits $n$")
ax[0].set_ylabel(r"$|F_{\rm STAB}-\hat F|$")
ax[0].set_ylim(1e-3, 2)
ax[0].legend(loc="upper left", fontsize=6.5)
label(ax[0], "a")
est = pd.read_csv(os.path.join(RESULTS, "sim1b_estimates.csv"))
cmap = plt.get_cmap("viridis")
for k, n in enumerate(sorted(est.n.unique())):
    g = est[est.n == n]
    itmax = int(g.it.max())
    curves = []
    for _, run in g.groupby("state"):
        e = (run.set_index("it").f_hat - 0.9).abs()
        curves.append(e.reindex(range(1, itmax + 1)).ffill().values)
    mean = np.median(curves, axis=0)
    ax[1].plot(np.arange(1, itmax + 1), mean, color=cmap(k / 5.5), lw=1.0, label=f"$n={n}$")
    note(f"S1b n={n}: median |FSTAB - f_hat| after it 1, 5, 10, 20, end:",
         [round(float(mean[min(j, itmax) - 1]), 4) for j in (1, 5, 10, 20, itmax)])
ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[1].set_xlim(1, 100)
ax[1].set_xlabel("iteration")
ax[1].set_ylabel(r"median $|F_{\rm STAB}-\hat f(z)|$")
ax[1].legend(loc="lower right", ncol=2)
label(ax[1], "b")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig2_methods.pdf"))
fig.savefig(os.path.join(RESULTS, "fig2_methods.png"), dpi=200)
p = ours.pivot_table(index=["n", "state"], columns="method", values="search_shots")
for v in ["store", "store+substrings"]:
    r = (p["no store"] / p[v]).groupby("n").agg(["mean", "sem"])
    note(f"S1 shot saving {v} (no store / {v}):", r.round(2).to_dict())
note("S1 false certificates:", int(((ours.certified) & (~ours.optimal.astype(bool))).sum()))
note("S1 final copies M_DFE:", int((ours.copies - ours.search_shots).iloc[0]))

# ---------------------------------------------------------------- Fig. 4 (simulation 1)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for v in COL:
    g = ours[ours.method == v].groupby("n")
    m = g.copies.median()
    lim = g.hit_limit.mean() > 0.5
    ax.errorbar(m.index, m.values, yerr=[m - g.copies.quantile(.25), g.copies.quantile(.75) - m],
                color=COL[v], lw=0.8, capsize=1.5, label=f"self-guided, {v}", zorder=2)
    ax.scatter(m.index[~lim], m[~lim], color=COL[v], s=12, zorder=3)
    ax.scatter(m.index[lim], m[lim], facecolor="white", edgecolor=COL[v], s=12, zorder=3)
    note(f"S1 copies {v}: median", m.to_dict())
th = d[d.method == "Grewal et al. (theory)"].set_index("n").copies
ax.plot(th.index, th.values, "k--", lw=0.8, marker="d", ms=3, label="Grewal et al., Thm. 6.7")
emp = d[d.method == "Grewal et al. (empirical)"]
bst = emp[emp.success >= 0.95].sort_values("copies").groupby("n").head(1).set_index("n").sort_index()
ax.plot(bst.index, bst.copies, "k-", lw=0.8, marker="x", ms=4, label="Grewal et al., tuned")
note("S1 Grewal theory copies:", th.round(0).to_dict())
note("S1 Grewal tuned copies:", bst.copies.round(0).to_dict(), "scale", bst.scale.to_dict())
ax.set_yscale("log")
ax.set_ylim(8e2, 2e6)
ax.set_xlabel("number of qubits $n$")
ax.set_ylabel(r"copies of $\rho$ for $\varepsilon=0.05$")
ax.legend(loc="upper left", fontsize=5.8, ncol=2, columnspacing=0.8, handlelength=1.6)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig4_copies.pdf"))
fig.savefig(os.path.join(RESULTS, "fig4_copies.png"), dpi=200)

# ---------------------------------------------------------------- Fig. 3 (simulation 2)
path = os.path.join(RESULTS, "sim2_families.csv")
if os.path.exists(path):
    import glob
    d = pd.concat([pd.read_csv(path)] + [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RESULTS, "sim2_ext_*.csv")))],
                  ignore_index=True).drop_duplicates(["family", "par", "n", "state"])
    if os.environ.get("N12_STATES"):          # use only the first states for n = 12 (partial runs)
        d = d[(d.n != 12) | (d.state < int(os.environ["N12_STATES"]))]
    fams = [("perturbed", r"$\tau$", "(a) stabilizer state + perturbation"),
            ("T-doped", "$t$", "(b) T-doped states"),
            ("depolarized", "$p$", "(c) depolarized stabilizer states")]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    ax, axs = axes[0], axes[1]
    d = d.assign(err=(d.F_hat - d.F_STAB).abs(), total=d.search_shots + 10_000)
    for k, (fam, sym, title) in enumerate(fams):
        g0 = d[d.family == fam]
        for j, (par, g) in enumerate(g0.groupby("par")):
            gg = g.groupby("n")
            m, lo, hi = med_iqr(gg.err)
            e = gg.err.sem()
            lim = gg.hit_limit.mean() > 0.5
            c = plt.get_cmap("plasma")(0.15 + 0.3 * j)
            xs = m.index + (j - 1) * 0.08
            ax[k].errorbar(xs, m, [lo, hi], color=c, lw=0.9, capsize=1.5, label=f"{sym}$={par:g}$", zorder=2)
            ax[k].scatter(xs[~lim], m[~lim], color=c, s=12, zorder=3)
            ax[k].scatter(xs[lim], m[lim], facecolor="white", edgecolor=c, s=12, zorder=3)
            Fs = g.groupby("n").F_STAB.mean()
            fl = 0.6745 * np.sqrt((1 - Fs**2) / 1e4)
            ax[k].plot(Fs.index, fl, color=c, ls="--", lw=0.7)
            med = gg.total.median()
            axs[k].errorbar(xs, med, yerr=[med - gg.total.quantile(.25), gg.total.quantile(.75) - med],
                            color=c, lw=0.9, capsize=1.5, marker="o", ms=3, label=f"{sym}$={par:g}$")
            note(f"S2 {fam} {par}: median total shots", med.to_dict(), "states", gg.size().to_dict())
            note(f"S2 {fam} {par}: median |err|", m.round(4).to_dict(), "q25", (m - lo).round(4).to_dict(),
                 "q75", (m + hi).round(4).to_dict(), "mean", gg.err.mean().round(4).to_dict(), "floor", fl.round(4).to_dict(),
                 "mean ratio", gg.ratio.mean().round(3).to_dict(),
                 "P(opt)", gg.optimal.mean().round(2).to_dict(), "FSTAB", round(g.F_STAB.mean(), 4),
                 "certified", gg.certified.mean().round(2).to_dict(),
                 "median shots", gg.search_shots.median().to_dict(), "hit limit", gg.hit_limit.mean().round(2).to_dict())
        ax[k].set_title(title, fontsize=8, loc="left")
        axs[k].set_xlabel("number of qubits $n$")
        axs[k].set_yscale("log")
        axs[k].set_ylim(8e3, 2e5)
        axs[k].axhline(1e4, color="0.4", ls="--", lw=0.7)
        axs[k].set_title("(" + "def"[k] + ")", fontsize=8, loc="left")
        if k == 0:
            axs[k].text(3.0, 1.06e4, "final DFE, $N_f=10^4$", fontsize=6.5, color="0.35")
        if k:
            ax[k].set_yticklabels([])
            axs[k].set_yticklabels([])
        ax[k].legend(loc="upper left", fontsize=6.5)
        xt = sorted(d.n.unique()) if fam != "perturbed" else list(range(3, 9))
        ax[k].set_xticks(xt)
        axs[k].set_xticks(xt)
        ax[k].set_xlim(xt[0] - 0.4, xt[-1] + 0.4)
        axs[k].set_xlim(xt[0] - 0.4, xt[-1] + 0.4)
        ax[k].set_yscale("log")
        ax[k].set_ylim(1e-3, 2)
    opt = d[d.optimal.astype(bool)]
    note("S2 optimal runs: median |err| / median floor:",
         round(float(opt.err.median() / (0.6745 * np.sqrt((1 - opt.F_STAB**2) / 1e4)).median()), 3))
    ax[0].set_ylabel(r"$|F_{\rm STAB}-\hat F|$")
    axs[0].set_ylabel("total shots")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_families.pdf"))
    fig.savefig(os.path.join(RESULTS, "fig3_families.png"), dpi=200)
    note("S2 false certificates:", int(((d.certified) & (~d.optimal.astype(bool))).sum()),
         "| certified among optimal with FSTAB>0.884:",
         round(d[(d.optimal) & (d.F_STAB > 0.884)].certified.mean(), 3))
    note("S2 overall P(opt):", round(d.optimal.mean(), 3), "| min ratio:", round(d.ratio.min(), 3))


open(os.path.join(RESULTS, "summary.txt"), "w").write("\n".join(OUT) + "\n")
