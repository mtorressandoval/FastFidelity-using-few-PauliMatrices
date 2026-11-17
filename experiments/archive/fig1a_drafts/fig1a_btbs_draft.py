"""Draft: Bloch sphere binary tree (Barthe et al., arXiv:2302.02957) of |00> and its 12 neighbors."""
import math, cmath
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
PH = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
BLUE = "#1f4e79"


def node(c0, c1):
    """Bloch vector, norm and propagated phase of r0 e^{i p0}|0> + r1 e^{i p1}|1>."""
    n = math.sqrt(abs(c0) ** 2 + abs(c1) ** 2)
    if n < 1e-12:
        return None, 0.0, 0.0
    th = 2 * math.acos(min(1, abs(c0) / n))
    pg = cmath.phase(c0) if abs(c0) > 1e-12 else cmath.phase(c1)
    pl_ = cmath.phase(c1) - cmath.phase(c0) if abs(c0) > 1e-12 and abs(c1) > 1e-12 else 0.0
    return np.array([math.sin(th) * math.cos(pl_), math.sin(th) * math.sin(pl_), math.cos(th)]), n, pg


def tree(psi):
    """Root (qubit 1) and children (qubit 2 given qubit 1 = 0 or 1) for two qubits; index = 2*q1 + q2."""
    left, nl, gl = node(psi[0], psi[1])
    right, nr, gr = node(psi[2], psi[3])
    root, _, _ = node(nl * cmath.exp(1j * gl), nr * cmath.exp(1j * gr))
    return root, left, right


def sphere(ax, empty=False):
    u, v = np.mgrid[0:2 * np.pi:14j, 0:np.pi:8j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.88" if not empty else "0.93",
                      lw=0.3, zorder=1)
    for a, b in [((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (-1, 0, 0)), ((-1, 0, 0), (0, -1, 0)), ((0, -1, 0), (1, 0, 0))]:
        ax.plot(*zip(a, b), color="0.75", lw=0.4, ls="--", zorder=2)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=30)
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(-0.9, 0.9)


states = {r"$|00\rangle$": [np.array([1, 0, 0, 0], complex)]}
for x, lab in [(1, "01"), (2, "10"), (3, "11")]:
    L = []
    for m in range(4):
        v = np.zeros(4, complex); v[0] = 1 / math.sqrt(2); v[x] = (1j) ** m / math.sqrt(2); L.append(v)
    states[r"$x=" + lab + r"$"] = L
octa = [np.array(p, float) for p in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]]
fig = plt.figure(figsize=(7.0, 2.8))
W = 0.25
for col, (title, L) in enumerate(states.items()):
    trees = [tree(v) for v in L]
    pos = {0: [col * W + 0.055, 0.53, 0.14, 0.38], 1: [col * W + 0.0, 0.13, 0.13, 0.38], 2: [col * W + 0.115, 0.13, 0.13, 0.38]}
    for k in range(3):
        ax = fig.add_axes(pos[k], projection="3d", computed_zorder=False)
        pts = [t[k] for t in trees]
        empty = all(p is None for p in pts)
        sphere(ax, empty)
        groups = {}
        for m, p in enumerate(pts):
            if p is None:
                continue
            assert min(np.linalg.norm(p - o) for o in octa) < 1e-9, "not an octahedron vertex"
            groups.setdefault(tuple(np.round(p, 6)), []).append(m)
        for p, ms in groups.items():
            if len(L) == 1:
                c = BLUE
            elif len(ms) == 1:
                c = PH[ms[0]]
            else:
                c = "0.25"                       # several neighbors at the same vertex
            ax.scatter(*p, color=c, s=24 if len(L) == 1 or len(ms) > 1 else 18, depthshade=False, zorder=8)
        if empty:
            ax.text2D(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center", fontsize=6, color="0.5")
    fig.text(col * W + 0.125, 0.94, title, ha="center", fontsize=8)
    # tree edges
    fig.add_artist(plt.Line2D([col * W + 0.125, col * W + 0.065], [0.575, 0.485], color="0.5", lw=0.6))
    fig.add_artist(plt.Line2D([col * W + 0.125, col * W + 0.18], [0.575, 0.485], color="0.5", lw=0.6))
    fig.text(col * W + 0.065, 0.13, r"$q_1=0$", fontsize=6, ha="center")
    fig.text(col * W + 0.18, 0.13, r"$q_1=1$", fontsize=6, ha="center")
h = [plt.Line2D([], [], ls="", marker="o", ms=4, color=c, label=l) for c, l in zip(PH, [r"$+1$", r"$+i$", r"$-1$", r"$-i$"])]
h.append(plt.Line2D([], [], ls="", marker="o", ms=4.5, color="0.25", label="all phases"))
fig.legend(handles=h, title=r"neighbor phase $i^m$", title_fontsize=6, fontsize=6, frameon=False, loc="lower center",
           bbox_to_anchor=(0.5, -0.01), ncol=5, handletextpad=0.1, columnspacing=0.8)
fig.savefig("../results/fig1a_btbs_draft.png", dpi=220)
print("all points are octahedron vertices")
