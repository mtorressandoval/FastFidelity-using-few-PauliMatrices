"""Fig. 2 of the paper (four families of states) and results/summary.txt."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.6, "legend.fontsize": 6.5, "legend.frameon": False})
OUT = []


def note(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


d = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RESULTS, "sim6_*.csv"))], ignore_index=True)
NMAX = int(os.environ.get("NMAX", 14))           # partial results: n <= NMAX
d = d[d.n <= NMAX]
FAMS = [("perturbed", r"$\tau$", "(a) perturbed stabilizer"),
        ("T-doped deep", "$t$", "(b) T-doped, depth $O(n)$"),
        ("T-doped shallow", "$t$", r"(c) T-doped, depth $O(\log n)$"),
        ("depolarized", "$p$", r"(d) depolarized, $O(\log n)$")]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9))
LO_P, HI_S, LO_S = [1.0], [0.0], [1e12]
for k, (fam, sym, title) in enumerate(FAMS):
    g0 = d[d.family == fam]
    ax, axs = axes[0, k], axes[1, k]
    pars = sorted(g0.par.unique())
    for j, par in enumerate(pars):
        g = g0[g0.par == par].groupby("n")
        c = plt.get_cmap("plasma")(0.1 + 0.75 * j / max(1, len(pars) - 1))
        dx = (j - (len(pars) - 1) / 2) * 0.12
        m = g.err.median()
        ps = g.optimal.mean()
        se = np.sqrt(ps * (1 - ps) / g.size())
        ax.errorbar(ps.index + dx, ps, se, color=c, lw=0.8, marker="o", ms=2.5, capsize=1, label=f"{sym}$={par:g}$")
        LO_P.append(float((ps - se).min()))
        tot = g.search_shots.median() + 1e4
        HI_S.append(float((g.search_shots.quantile(.75) + 1e4).max()))
        LO_S.append(float((g.search_shots.quantile(.25) + 1e4).min()))
        axs.errorbar(tot.index + dx, tot, [tot - (g.search_shots.quantile(.25) + 1e4),
                                           (g.search_shots.quantile(.75) + 1e4) - tot],
                     color=c, lw=0.8, marker="o", ms=2.5, capsize=1)
        note(f"{fam} {par}: P(opt)", g.optimal.mean().round(2).to_dict(), "| median err", m.round(4).to_dict(),
             "| median search shots", g.search_shots.median().to_dict(), "| states", g.size().to_dict(),
             "| certified", g.certified.mean().round(2).to_dict(), "| jumps", g.local_jumps.mean().round(2).to_dict())
    nmax = g0.n.max() if len(g0) else 8
    axs.set_yscale("log")
    for a in (ax, axs):
        if nmax > 12:
            a.axvline(12, color="0.6", lw=0.6, ls="--")
        a.set_xlim(2.3, nmax + 0.8)

    for a in (ax, axs):
        a.set_xticks([v for v in sorted(g0.n.unique()) if v in (3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20)])
    ax.set_title(title, fontsize=7.5, loc="left")
    axs.set_title("(" + "efgh"[k] + ")", fontsize=7.5, loc="left")
    axs.set_xlabel("number of qubits $n$")
    ax.legend(loc="lower left", fontsize=6)
for k in range(4):
    axes[0, k].set_ylim(np.floor((min(LO_P) - 0.03) * 20) / 20, 1.01)
    axes[1, k].set_ylim(0.9 * min(LO_S), 1.15 * max(HI_S))
    if k:
        for r in range(2):
            axes[r, k].tick_params(axis="y", which="both", labelleft=False)
axes[0, 0].set_ylabel("success probability")
axes[1, 0].set_ylabel("total shots")
fig.tight_layout(w_pad=0.4, h_pad=0.6)
fig.savefig(os.path.join(RESULTS, "fig2_families.pdf"))
fig.savefig(os.path.join(RESULTS, "fig2_families.png"), dpi=200)
opt = d[d.optimal]
note("all: P(opt)", round(d.optimal.mean(), 3), "| false certificates", int((d.certified & ~d.optimal).sum()),
     "| median err/floor over optimal runs",
     round(float(opt.err.median() / (0.6745 * np.sqrt((1 - opt.F_STAB**2) / 1e4)).median()), 3))
f = d[~d.optimal]
if len(f):
    note("failed runs:", len(f), "| max F/F_STAB", round(f.ratio.max(), 3), "| hit limit", int(f.hit_limit.sum()))
for fam in ["T-doped shallow", "depolarized"]:
    g = d[(d.family == fam) & (d.n >= 10)]
    if g.n.nunique() > 2:
        x, y = np.log(g.n.values), np.log(g.search_shots.values)
        A = np.vstack([np.ones_like(x), x]).T
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ c
        cov = (r @ r / (len(x) - 2)) * np.linalg.inv(A.T @ A)
        note(f"{fam}: search shots ~ n^({c[1]:.1f} +- {np.sqrt(cov[1, 1]):.1f}) for n >= 10")
open(os.path.join(RESULTS, "summary.txt"), "w").write("\n".join(OUT) + "\n")
