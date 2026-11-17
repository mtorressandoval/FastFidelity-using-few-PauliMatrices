"""Figure of the worked examples (supplementary material draft)."""
import os, io, sys, math, runpy, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    g1 = runpy.run_path(os.path.join(HERE, "worked_example_1.py"))
    g2 = runpy.run_path(os.path.join(HERE, "worked_example_2.py"))
pl = g2["pl"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.6, "legend.fontsize": 6.5, "legend.frameon": False})
per, p, fhat, new, store1, F1 = g1["run"](3)
fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1, 1, 1.4, 1.25]})
C = {"est": "#1f77b4", "ex": "0.2", "acc": "#2ca02c", "rej": "#d62728", "low": "0.7"}
# (a) Walsh step
xs = ["00", "01", "10", "11"]
ax[0].bar(range(4), [p[x] for x in range(4)], color=C["est"], width=0.6, label=r"$\hat p_x$")
ax[0].plot(range(4), [0.6, 0, 0, 0.4], "_", color=C["ex"], ms=14, mew=1.4, label="exact")
ax[0].axhline((3 - 2 * math.sqrt(2)) * p[0], color="0.4", ls=":", lw=0.8)
ax[0].text(1.0, (3 - 2 * math.sqrt(2)) * p[0] + 0.02, r"$0.17\,\hat p_0$", ha="center", fontsize=6.5)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(xs); ax[0].set_xlabel("$x$")
ax[0].set_ylabel("fidelity"); ax[0].set_ylim(0, 1.05)
ax[0].set_title(r"(a) basis of $|00\rangle$", fontsize=7.5, loc="left")
ax[0].legend(loc="upper center")
# (b) neighbours in direction 11
exact_nb = [0.5 + math.sqrt(0.24) * math.cos(math.pi / 8 - m * math.pi / 2) for m in range(4)]
ax[1].bar(range(4), [fhat[m] for m in range(4)], color=C["est"], width=0.6)
ax[1].plot(range(4), exact_nb, "_", color=C["ex"], ms=14, mew=1.4)
ax[1].axhline(p[0], color="0.4", ls="--", lw=0.8)
ax[1].text(3.45, p[0] + 0.03, r"$\hat p_0$", ha="right", fontsize=6.5)
ax[1].set_xticks(range(4)); ax[1].set_xlabel(r"$m$ of $|\nu_{11,m}\rangle$"); ax[1].set_ylim(0, 1.05)
ax[1].set_title(r"(b) neighbors, $x=11$", fontsize=7.5, loc="left")
# (c) store scores
T2, f02, p02, store2, cand2, fc2 = g2["run"](1)          # store right after the first DFE
rows = sorted(((abs(s / m) - 3 / math.sqrt(m), int(k)) for k, (m, s) in store2.items()), reverse=True)[:16]
acc = {pl.from_label(L)[0] for L in ["IIZZ", "IIYY", "ZZXX", "YYII"]}
cols = [C["acc"] if k in acc else (C["rej"] if sc >= 0.2 else C["low"]) for sc, k in rows]
ax[2].bar(range(len(rows)), [sc for sc, _ in rows], color=cols, width=0.7)
ax[2].axhline(0.2, color="0.4", ls=":", lw=0.8)
ax[2].set_xticks(range(len(rows)))
ax[2].set_xticklabels([pl.to_label(k, 4) for _, k in rows], rotation=90, fontsize=5.5, family="monospace")
ax[2].set_ylabel(r"$|\bar P|-3/\sqrt{m_P}$")
ax[2].set_title("(c) stored strings", fontsize=7.5, loc="left")
from matplotlib.patches import Patch
ax[2].legend(handles=[Patch(color=C["acc"], label="accepted"), Patch(color=C["rej"], label="dependent"),
                      Patch(color=C["low"], label="below threshold")], loc="upper right", ncol=1)
ax[2].set_ylim(-0.15, 0.85)
# (d) fidelities, example 2
T = T2; E = g2["B"].E
flip = max(T.flip(x).fidelity_from_spectrum(E) for x in range(1, 16))
nb = max(T.neighbor(x, m).fidelity_from_spectrum(E) for x in range(1, 16) for m in range(4))
vals = [g2["f0"], flip, nb, g2["fc"]]
ax[3].bar(range(4), vals, color=[C["low"], C["low"], C["low"], C["acc"]], width=0.6)
ax[3].axhline(0.8125, color=C["ex"], ls="--", lw=0.8)
ax[3].text(0.0, 0.84, r"$F_{\rm STAB}$", fontsize=6.5)
ax[3].set_xticks(range(4)); ax[3].set_xticklabels([r"$|\phi\rangle$", "best\nflip", "best\nneighbor", r"$|\tilde\phi\rangle$"], fontsize=6.5)
ax[3].set_ylim(0, 1.05); ax[3].set_ylabel("exact fidelity")
ax[3].set_title("(d) one step", fontsize=7.5, loc="left")
fig.tight_layout(w_pad=0.6)
fig.savefig("results/figS1_worked_examples.pdf"); fig.savefig("results/figS1_worked_examples.png", dpi=200)
print("flip", round(flip, 4), "neighbor", round(nb, 4))
