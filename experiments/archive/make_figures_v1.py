"""Figures 2-4 (main text), Figs. 5-6 (appendix) and results/summary.txt
with every number quoted in the paper.  Reads the CSV files in results/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.6, "legend.fontsize": 6.5, "legend.frameon": False})
OUT = []


def note(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def fit_line(x, y, w=None):
    """Weighted least squares y = a + b x with standard errors."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    W = np.ones_like(x) if w is None else np.asarray(w, float)
    A = np.vstack([np.ones_like(x), x]).T
    Aw = A * np.sqrt(W)[:, None]
    cov = np.linalg.inv(Aw.T @ Aw)
    coef = cov @ (Aw.T @ (y * np.sqrt(W)))
    res = y - A @ coef
    s2 = np.sum(W * res**2) / max(1, len(x) - 2)
    err = np.sqrt(np.diag(cov) * s2) if w is None else np.sqrt(np.diag(cov) * max(1.0, s2))
    return coef, err


COL = {"no store": "C0", "store": "C1", "store+substrings": "C2"}
MK = {"pure tau=0.9": "o", "depolarized p=0.05": "s", "stabilizer mixture": "^"}

# ---------------------------------------------------------------- E1 / Fig 2
d = pd.read_csv(os.path.join(RESULTS, "e1_store_savings.csv"))
note("E1: optimal fraction", d.optimal.mean(), " runs", len(d))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
for fam, g in d.groupby("family"):
    for v, gv in g.groupby("variant"):
        m = gv.groupby("n").search_shots.agg(["median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
        ax[0].errorbar(m.index, m["median"], yerr=[m["median"] - m.iloc[:, 1], m.iloc[:, 2] - m["median"]],
                       color=COL[v], marker=MK[fam], ms=3, lw=0.8, capsize=1.5,
                       label=f"{v}, {fam}" if fam == "pure tau=0.9" else None, ls="-" if fam == "pure tau=0.9" else ":")
ax[0].set_yscale("log")
ax[0].set_xlabel("number of qubits $n$")
ax[0].set_ylabel("measured Pauli strings (shots)")
ax[0].legend(handles=[plt.Line2D([], [], color=COL[v], label=v) for v in COL] +
             [plt.Line2D([], [], color="0.3", marker=MK[f], ls="", ms=3, label=f.replace("tau", r"$\tau$")) for f in MK],
             loc="center", bbox_to_anchor=(0.5, 0.6), ncol=2)
p = d.pivot_table(index=["family", "n", "state"], columns="variant", values="search_shots")
for v in ["store", "store+substrings"]:
    r = (p["no store"] / p[v]).rename("s").reset_index()
    for fam, g in r.groupby("family"):
        m = g.groupby("n").s.agg(["mean", "sem"])
        ax[1].errorbar(m.index + (0.1 if v == "store+substrings" else -0.1), m["mean"], m["sem"],
                       color=COL[v], marker=MK[fam], ms=3, lw=0.8, capsize=1.5)
    agg = r.groupby("n").s.agg(["mean", "sem"])
    coef, err = fit_line(agg.index, agg["mean"], 1 / agg["sem"]**2)
    note(f"E1 saving ({v}) vs n: a = {coef[0]:.2f} +- {err[0]:.2f}, b = {coef[1]:.3f} +- {err[1]:.3f} per qubit;",
         "mean over all", round(r.s.mean(), 2), "+-", round(r.s.sem(), 2),
         "| n=8:", round(r[r.n == 8].s.mean(), 2), "+-", round(r[r.n == 8].s.sem(), 2))
ax[1].set_xlabel("number of qubits $n$")
ax[1].set_ylabel("shot saving factor")
ax[1].set_ylim(0, None)
for a, lab in zip(ax, "ab"):
    a.text(-0.17, 1.02, f"({lab})", transform=a.transAxes, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig2_shots_vs_n.pdf"))
fig.savefig(os.path.join(RESULTS, "fig2_shots_vs_n.png"), dpi=200)
for v in COL:
    note(f"E1 median shots n=8 {v}:", d[(d.n == 8) & (d.variant == v)].search_shots.median())
note("E1 certified fraction by family:", d.groupby("family").certified.mean().round(3).to_dict())
note("E1 reuse rate (store, store+sub):", round(d[d.variant == "store"].reuse.mean(), 3),
     round(d[d.variant == "store+substrings"].reuse.mean(), 3))

# ---------------------------------------------------------------- E2 / Fig 3
if os.path.exists(os.path.join(RESULTS, "e2_success.csv")):
    d = pd.read_csv(os.path.join(RESULTS, "e2_success.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for k, (sweep, xcol) in enumerate([("tau", "tau"), ("p", "p")]):
        g = d[d.sweep == sweep]
        for n, gn in g.groupby("n"):
            for v, gv in gn.groupby("variant"):
                m = gv.groupby(xcol).optimal.mean()
                ax[k].plot(m.index, m.values, marker="os^"[n - 3], ms=3, lw=0.8, color=COL[v],
                           ls="-" if v == "store" else "--", label=f"$n={n}$, {v}")
        ax[k].set_xlabel(r"overlap $\tau$" if sweep == "tau" else r"depolarizing probability $p$")
    ax[0].axvline(np.cos(np.pi / 8)**2, color="0.6", lw=0.6, ls=":")
    ax[0].set_ylabel("probability of reaching the optimum")
    ax[0].legend(ncol=2, loc="lower right")
    for a, lab in zip(ax, "ab"):
        a.text(-0.16 if lab == "a" else -0.06, 1.03, f"({lab})", transform=a.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_success.pdf"))
    fig.savefig(os.path.join(RESULTS, "fig3_success.png"), dpi=200)
    t = d[d.sweep == "tau"]
    note("E2 P(opt) by tau (store):", t[t.variant == "store"].groupby("tau").optimal.mean().round(3).to_dict())
    note("E2 P(opt) by tau (no store):", t[t.variant == "no store"].groupby("tau").optimal.mean().round(3).to_dict())
    note("E2 mean F/FSTAB by tau (store):", t[t.variant == "store"].groupby("tau").ratio.mean().round(4).to_dict())
    note("E2 min F/FSTAB tau sweep:", round(t.ratio.min(), 4))
    q = d[d.sweep == "p"]
    note("E2 P(opt) by p (store):", q[q.variant == "store"].groupby("p").optimal.mean().round(3).to_dict())
    note("E2 P(opt) by p (no store):", q[q.variant == "no store"].groupby("p").optimal.mean().round(3).to_dict())
    note("E2 mean F/FSTAB by p (store):", q[q.variant == "store"].groupby("p").ratio.mean().round(4).to_dict())
    note("E2 certified by tau (store):", t[t.variant == "store"].groupby("tau").certified.mean().round(3).to_dict())
    note("E2 false certificates (certified but not optimal):", int(((d.certified) & (~d.optimal)).sum()))
    inside = ((d.F_out >= d.ci_lo) & (d.F_out <= d.ci_hi)).mean()
    note("E2 coverage of 99% CI for F(z_out):", round(inside, 3))

# ---------------------------------------------------------------- E3 / Fig 5
if os.path.exists(os.path.join(RESULTS, "e3_dense.csv")):
    d = pd.read_csv(os.path.join(RESULTS, "e3_dense.csv"))
    fams = ["Clifford+1T", "Clifford+3T", "Haar", "rank 2", "rank 4"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for k, n in enumerate([3, 4, 5]):
        data, pos, cols = [], [], []
        for i, fam in enumerate(fams):
            for j, v in enumerate(COL):
                s = d[(d.n == n) & (d.family == fam) & (d.variant == v)].ratio
                data.append(s.values)
                pos.append(i * 4 + j)
                cols.append(COL[v])
        bp = ax[k].boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=True,
                           flierprops=dict(ms=2), medianprops=dict(color="k", lw=0.8))
        for b, c in zip(bp["boxes"], cols):
            b.set_facecolor(c)
            b.set_alpha(0.6)
        ax[k].set_xticks([i * 4 + 1 for i in range(len(fams))])
        ax[k].set_xticklabels(fams, rotation=35, fontsize=6.5)
        ax[k].set_title(f"$n={n}$", fontsize=8)
    ax[0].set_ylabel(r"$F(z_{\rm out})/F_{\rm STAB}$")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COL[v], alpha=0.6, label=v) for v in COL], loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig5_dense.pdf"))
    fig.savefig(os.path.join(RESULTS, "fig5_dense.png"), dpi=200)
    note("E3 P(opt):", d.groupby(["family", "variant"]).optimal.mean().unstack().round(2).to_dict())
    note("E3 mean ratio:", d.groupby(["family", "variant"]).ratio.mean().unstack().round(3).to_dict())
    note("E3 min ratio:", d.groupby(["variant"]).ratio.min().round(3).to_dict())
    note("E3 shots:", d.groupby(["variant"]).search_shots.median().to_dict())

# ---------------------------------------------------------------- E4 / Fig 4
if os.path.exists(os.path.join(RESULTS, "e4_copies.csv")):
    d = pd.read_csv(os.path.join(RESULTS, "e4_copies.csv"))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ours = d[d.method.str.startswith("self-guided")]
    for v in COL:
        g = ours[ours.method == f"self-guided ({v})"].groupby("n")
        m = g.copies.median()
        ax.errorbar(m.index, m.values, yerr=[m - g.copies.quantile(.25), g.copies.quantile(.75) - m],
                    color=COL[v], marker="o", ms=3, lw=0.8, capsize=1.5, label=f"self-guided, {v}")
        note(f"E4 self-guided {v}: P(opt) by n", g.optimal.mean().round(2).to_dict(), "median copies", m.to_dict())
    th = d[d.method == "Grewal et al. (theory)"].set_index("n").copies
    ax.plot(th.index, th.values, "k--", lw=0.8, marker="d", ms=3, label="Grewal et al., Thm. 6.7")
    emp = d[d.method == "Grewal et al. (empirical)"]
    best = emp[emp.success >= 0.95].sort_values("copies").groupby("n").head(1).set_index("n").sort_index()
    ax.plot(best.index, best.copies, "k-", lw=0.8, marker="x", ms=4, label="Grewal et al., tuned")
    note("E4 Grewal theory copies:", th.round(0).to_dict())
    note("E4 Grewal tuned copies:", best.copies.round(0).to_dict(), "scale", best.scale.to_dict())
    ax.set_yscale("log")
    ax.set_xlabel("number of qubits $n$")
    ax.set_ylabel(r"copies of $\rho$ for $\varepsilon=0.05$")
    ax.set_ylim(8e2, 2e6)
    ax.legend(loc="upper left", fontsize=5.8, ncol=2, columnspacing=0.8, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig4_copies.pdf"))
    fig.savefig(os.path.join(RESULTS, "fig4_copies.png"), dpi=200)

# ---------------------------------------------------------------- E5 / Fig 6
if os.path.exists(os.path.join(RESULTS, "e5_ablation.csv")):
    d = pd.read_csv(os.path.join(RESULTS, "e5_ablation.csv"))
    order = list(dict.fromkeys(d.config))
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True)
    x = np.arange(len(order))
    for j, v in enumerate(["no store", "store"]):
        g = d[d.variant == v].groupby("config")
        ax[0].bar(x + (j - 0.5) * 0.38, g.search_shots.median()[order], 0.38, color=COL[v], label=v)
        ax[1].bar(x + (j - 0.5) * 0.38, g.optimal.mean()[order], 0.38, color=COL[v])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("median shots")
    ax[1].set_ylabel("P(optimum)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(order, rotation=30, fontsize=6.5)
    ax[0].legend()
    for a, lab in zip(ax, "ab"):
        a.text(-0.07, 1.0, f"({lab})", transform=a.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig6_ablation.pdf"))
    fig.savefig(os.path.join(RESULTS, "fig6_ablation.png"), dpi=200)
    note("E5:", d.groupby(["config", "variant"]).agg(opt=("optimal", "mean"), shots=("search_shots", "median")).unstack().round(2).to_dict())

if os.path.exists(os.path.join(RESULTS, "e4b_random_start_N.csv")):
    d = pd.read_csv(os.path.join(RESULTS, "e4b_random_start_N.csv"))
    note("E4b (n=8, random start):", d.groupby(["N", "variant"]).agg(opt=("optimal", "mean"),
         shots=("search_shots", "median")).round(2).to_dict())

open(os.path.join(RESULTS, "summary.txt"), "w").write("\n".join(OUT) + "\n")
