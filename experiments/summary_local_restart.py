"""Working figure and table for simulation 5 (local restart).  Not in the paper."""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
                     "legend.fontsize": 6.5, "legend.frameon": False})
d = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RESULTS, "sim5_*.csv"))])
d = d.drop_duplicates(["n", "kind", "state", "method"], keep="last")
tab = d.groupby(["kind", "method", "n"]).agg(states=("optimal", "size"), P_opt=("optimal", "mean"),
                                             median_shots=("search_shots", "median"),
                                             jumps=("local_jumps", "mean"),
                                             median_err=("err", "median")).round(3)
print(tab.to_string())
open(os.path.join(RESULTS, "sim5_summary.txt"), "w").write(tab.to_string() + "\n")
from matplotlib.lines import Line2D
col = {"neighbors": "C0", "random pairs": "C1", "deep random": "C3"}
off = {"neighbors": -0.15, "random pairs": 0.0, "deep random": 0.15}
name = {"neighbors": "neighbors, depth 2", "random pairs": "random pairs, depth 2", "deep random": "random (deep)"}
sty = {"local": dict(ls="-", marker="o", mfc=None), "plain": dict(ls="--", marker="s", mfc="white")}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
for (kind, meth), g in d.groupby(["kind", "method"]):
    gg = g.groupby("n")
    st = sty[meth]
    kw = dict(ls=st["ls"], marker=st["marker"], ms=4, color=col[kind], lw=1.0)
    if st["mfc"]:
        kw["mfc"] = st["mfc"]
    ax[0].plot(gg.optimal.mean().index + off[kind], gg.optimal.mean(), **kw)
    med = gg.search_shots.median()
    ax[1].errorbar(med.index + off[kind], med, [med - gg.search_shots.quantile(.25), gg.search_shots.quantile(.75) - med],
                   capsize=1.5, **kw)
ax[0].set_ylabel("fraction of runs at the optimum")
ax[0].set_ylim(-0.05, 1.08)
ax[1].set_ylabel("search shots (median)")
ax[1].set_yscale("log")
for a, k in zip(ax, "ab"):
    a.set_xlabel("number of qubits $n$")
    a.set_xticks([8, 10, 12, 14, 16, 20])
    a.text(-0.14, 1.03, f"({k})", transform=a.transAxes, fontsize=9)
h1 = [Line2D([], [], color=col[k], lw=2, label=name[k]) for k in col]
h2 = [Line2D([], [], color="0.2", ls="-", marker="o", ms=4, label="with local restart"),
      Line2D([], [], color="0.2", ls="--", marker="s", ms=4, mfc="white", label="without local restart")]
l1 = ax[0].legend(handles=h1, title="Clifford circuit", title_fontsize=6.5, loc="center right",
                  bbox_to_anchor=(1.0, 0.62))
ax[0].add_artist(l1)
ax[0].legend(handles=h2, title="protocol", title_fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.25))
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_local_restart.png"), dpi=200)
fig.savefig(os.path.join(RESULTS, "fig_local_restart.pdf"))
