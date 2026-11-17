"""Draft of Fig. 1(a): Bloch octahedron (1 qubit) and Q-sphere with the neighbors of |000> (3 qubits)."""
import os, math, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
RED, BLUE, GREY = "#c0392b", "#1f4e79", "0.45"
PHASE = {0: ("#1f77b4", r"$+1$"), 1: ("#2ca02c", r"$+i$"), 2: ("#d62728", r"$-1$"), 3: ("#ff7f0e", r"$-i$")}


def sphere(ax, nu=16, nv=9):
    u, v = np.mgrid[0:2 * np.pi:complex(nu), 0:np.pi:complex(nv)]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.9", lw=0.3, zorder=1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.patch.set_alpha(0)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)


fig = plt.figure(figsize=(4.8, 2.3))
# ---------------- 1 qubit: Bloch octahedron
ax = fig.add_axes([-0.02, 0.02, 0.44, 0.96], projection="3d", computed_zorder=False)
sphere(ax); ax.view_init(elev=18, azim=30)
pts = {r"$|0\rangle$": (0, 0, 1), r"$|1\rangle$": (0, 0, -1), r"$|+\rangle$": (1, 0, 0),
       r"$|-\rangle$": (-1, 0, 0), r"$|{+i}\rangle$": (0, 1, 0), r"$|{-i}\rangle$": (0, -1, 0)}
eq = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
for a, b in zip(eq, eq[1:] + eq[:1]):
    ax.plot(*zip(a, b), color=GREY, lw=0.6, ls="--", zorder=2)
    ax.plot(*zip(b, (0, 0, -1)), color=GREY, lw=0.6, ls="--", zorder=2)
for p in eq:
    ax.plot(*zip((0, 0, 1), p), color=RED, lw=2.0, zorder=3)
for lab, p in pts.items():
    c = BLUE if p == (0, 0, 1) else (RED if p[2] == 0 else GREY)
    ax.scatter(*p, color=c, s=20, depthshade=False, zorder=8)
    q = np.array(p) * 1.3
    ax.text(*q, lab, ha="center", va="center", fontsize=7.5, zorder=10)
ax.text2D(0.12, 0.93, "1 qubit", transform=ax.transAxes, fontsize=7.5)

# ---------------- 3 qubits: Q-sphere
n = 3
ax2 = fig.add_axes([0.40, 0.02, 0.50, 0.96], projection="3d", computed_zorder=False)
sphere(ax2); ax2.view_init(elev=15, azim=20)
pos = {}
for w in range(n + 1):
    xs = [x for x in range(2 ** n) if bin(x).count("1") == w]
    th = math.pi * w / n
    for k, x in enumerate(xs):
        ph = 2 * math.pi * k / len(xs) + (math.pi / 3 if w == 2 else 0)
        pos[x] = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
north = pos[0]
for x in range(1, 2 ** n):
    ax2.plot(*zip(north, pos[x]), color=RED, lw=1.2, zorder=3, alpha=0.85)
# four phase markers around each |x>, in the tangent plane
for x in range(1, 2 ** n):
    p = pos[x]
    t1 = np.cross(p, [0, 0, 1]); t1 = t1 / np.linalg.norm(t1) if np.linalg.norm(t1) > 1e-9 else np.array([1.0, 0, 0])
    t2 = np.cross(p, t1)
    for m in range(4):
        a = math.pi / 4 + m * math.pi / 2
        q = p + 0.11 * (math.cos(a) * t1 + math.sin(a) * t2)
        ax2.scatter(*q, color=PHASE[m][0], s=13, depthshade=False, zorder=8, edgecolors="white", linewidths=0.3)
ax2.scatter(*north, color=BLUE, s=26, depthshade=False, zorder=9)
for x, p in pos.items():
    lab = r"$|" + format(x, "03b") + r"\rangle$"
    q = p * (1.18 if x == 0 else 1.45)
    if x == 7:
        q = p * 1.22
    ax2.text(*q, lab, ha="center", va="center", fontsize=6.8, zorder=10)
ax2.text2D(0.02, 0.93, "3 qubits", transform=ax2.transAxes, fontsize=7.5)
handles = [plt.Line2D([], [], ls="", marker="o", ms=4, color=c, label=l) for c, l in PHASE.values()]
fig.legend(handles=handles, title=r"phase $i^m$", title_fontsize=6.5, fontsize=6.5, frameon=False,
           loc="center right", bbox_to_anchor=(1.0, 0.5), ncol=1, handletextpad=0.2)
os.makedirs("../results", exist_ok=True)
fig.savefig("../results/fig1a_qsphere_draft.png", dpi=220)
fig.savefig("../results/fig1a_qsphere_draft.pdf")
print("ok")
