"""Draft: correlation tree for two qubits (Fano form).  Root: correlation matrix T_ij = <s_i s_j>;
leaves: Bloch vectors of qubit 1 and qubit 2.  |00> and its neighbors in the directions 01, 10, 11."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
PH = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
BLUE = "#1f4e79"
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
S = [X, Y, Z]


def fano(v):
    rho = np.outer(v, v.conj())
    rA = np.array([np.trace(rho @ np.kron(s, I2)).real for s in S])
    rB = np.array([np.trace(rho @ np.kron(I2, s)).real for s in S])
    T = np.array([[np.trace(rho @ np.kron(a, b)).real for b in S] for a in S])
    return rA, rB, T


def sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:14j, 0:np.pi:8j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.88", lw=0.3, zorder=1)
    for a, b in [((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (-1, 0, 0)), ((-1, 0, 0), (0, -1, 0)), ((0, -1, 0), (1, 0, 0))]:
        ax.plot(*zip(a, b), color="0.75", lw=0.4, ls="--", zorder=2)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=30)
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(-0.9, 0.9)


cols = {r"$|00\rangle$": [np.array([1, 0, 0, 0], complex)]}
for x, lab in [(1, "01"), (2, "10"), (3, "11")]:
    L = []
    for m in range(4):
        v = np.zeros(4, complex); v[0] = 1 / math.sqrt(2); v[x] = (1j) ** m / math.sqrt(2); L.append(v)
    cols[r"$x=" + lab + r"$"] = L
fig = plt.figure(figsize=(7.0, 3.1))
W = 0.25
sub = [(-0.22, 0.22), (0.22, 0.22), (-0.22, -0.22), (0.22, -0.22)]     # positions of the four phases in a cell
for c, (title, L) in enumerate(cols.items()):
    data = [fano(v) for v in L]
    # ---- root: correlation matrix
    ax = fig.add_axes([c * W + 0.07, 0.53, 0.11, 0.34])
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((j - 0.5, 2 - i - 0.5), 1, 1, fill=False, ec="0.75", lw=0.5))
            for m, (rA, rB, T) in enumerate(data):
                val = round(T[i, j])
                if val == 0:
                    continue
                dx, dy = (0, 0) if len(L) == 1 else sub[m]
                col = BLUE if len(L) == 1 else PH[m]
                ax.scatter(j + dx, 2 - i + dy, s=16 if len(L) > 1 else 26, zorder=3,
                           facecolor=col if val > 0 else "white", edgecolor=col, linewidths=1.0)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5); ax.set_aspect("equal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["$X$", "$Y$", "$Z$"], fontsize=6.5); ax.set_yticklabels(["$Z$", "$Y$", "$X$"], fontsize=6.5)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(r"$T_{ij}$", fontsize=7, pad=2)
    fig.text(c * W + 0.125, 0.95, title, ha="center", fontsize=8)
    # ---- leaves: Bloch vectors of each qubit
    for q in range(2):
        bx = fig.add_axes([c * W + 0.0 + 0.115 * q, 0.16, 0.13, 0.33], projection="3d", computed_zorder=False)
        sphere(bx)
        groups = {}
        for m, d in enumerate(data):
            groups.setdefault(tuple(np.round(d[q], 6)), []).append(m)
        for p, ms in groups.items():
            col = BLUE if len(L) == 1 else (PH[ms[0]] if len(ms) == 1 else "0.25")
            bx.scatter(*p, color=col, s=22 if len(ms) > 1 or len(L) == 1 else 16, depthshade=False, zorder=8)
        fig.text(c * W + 0.065 + 0.115 * q, 0.15, ["qubit 1", "qubit 2"][q], fontsize=6, ha="center")
    fig.add_artist(plt.Line2D([c * W + 0.125, c * W + 0.065], [0.53, 0.48], color="0.5", lw=0.6))
    fig.add_artist(plt.Line2D([c * W + 0.125, c * W + 0.18], [0.53, 0.48], color="0.5", lw=0.6))
h = [plt.Line2D([], [], ls="", marker="o", ms=4, color=c, label=l) for c, l in zip(PH, [r"$+1$", r"$+i$", r"$-1$", r"$-i$"])]
h.append(plt.Line2D([], [], ls="", marker="o", ms=4.5, color="0.25", label="all phases"))
h.append(plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="0.3", label=r"$T_{ij}=-1$"))
fig.legend(handles=h, title=r"neighbor phase $i^m$", title_fontsize=6, fontsize=6, frameon=False, loc="lower center",
           bbox_to_anchor=(0.5, -0.02), ncol=6, handletextpad=0.1, columnspacing=0.8)
fig.savefig("../results/fig1a_corrtree_draft.png", dpi=220)
for title, L in cols.items():
    for v in L:
        rA, rB, T = fano(v)
        assert np.allclose(np.round(T), T) and np.allclose(np.round(rA), rA)
print("all entries in {0, +-1}")
