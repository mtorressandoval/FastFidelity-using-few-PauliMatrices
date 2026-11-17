"""Scaling laws of the number of shots of the search with n (working figure,
not in the paper).  Fits of the median search shots per n (pooled over the
parameters of each family) to S = A n^alpha and to S = B exp(beta n)."""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
                     "legend.fontsize": 6.5, "legend.frameon": False})
d = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RESULTS, "sim6_*.csv"))])
d = d[d.n <= int(os.environ.get("NMAX", 14))]


def fit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ c
    cov = (r @ r / (len(x) - 2)) * np.linalg.inv(A.T @ A)
    return c, np.sqrt(np.diag(cov)), float(r @ r)


FAMS = [("perturbed", "perturbed stabilizer", "C0"), ("T-doped deep", r"T-doped, depth $O(n)$", "C3"),
        ("T-doped shallow", r"T-doped, depth $O(\log n)$", "C2"), ("depolarized", r"depolarized, $O(\log n)$", "C1")]
lines = []
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for fam, lab, col in FAMS:
    m = d[d.family == fam].groupby("n").search_shots.median()
    n = m.index.values.astype(float)
    y = np.log(m.values)
    cp, ep, rp = fit(np.log(n), y)
    ce, ee, re = fit(n, y / np.log(2))      # base 2: log2 S = log2 B + gamma n
    A, dA = np.exp(cp[0]), np.exp(cp[0]) * ep[0]
    B, dB = 2.0 ** ce[0], 2.0 ** ce[0] * np.log(2) * ee[0]
    lines.append(f"{fam:16s}: S = ({A:.0f} +- {dA:.0f}) n^({cp[1]:.2f} +- {ep[1]:.2f})   [RSS {rp:.4f}]   |   "
                 f"S = ({B:.0f} +- {dB:.0f}) 2^[({ce[1]:.3f} +- {ee[1]:.3f}) n]   [RSS(ln) {re * np.log(2)**2:.4f}]   1/gamma = {1 / ce[1]:.2f}")
    nn = np.linspace(3, n.max(), 50)
    for k, a in enumerate(ax):
        a.plot(n, m.values, "o", color=col, ms=3.5, label=lab)
    ax[0].plot(nn, A * nn ** cp[1], "-", color=col, lw=0.8)
    ax[1].plot(nn, B * 2.0 ** (ce[1] * nn), "-", color=col, lw=0.8)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[0].set_title(r"(a) power law $S=A\,n^{\alpha}$ (log-log)", fontsize=8, loc="left")
ax[1].set_title(r"(b) exponential $S=B\,2^{\gamma n}$ (log-linear)", fontsize=8, loc="left")
for a in ax:
    a.set_xlabel("number of qubits $n$")
    a.set_xticks(sorted(d.n.unique()))
    a.set_xticklabels([str(k) for k in sorted(d.n.unique())])
    a.minorticks_off()
ax[0].set_ylabel("median search shots")
ax[0].legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_scaling_laws.png"), dpi=200)
open(os.path.join(RESULTS, "scaling_laws.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
