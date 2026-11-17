"""Draft: |00> and its 12 neighbors (2 qubits) in five representations."""
import os, sys, math, cmath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from fastfidelity import SpectrumBackend
from fastfidelity import paulis as pl

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
PH = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
BLUE, RED, GREY = "#1f4e79", "#c0392b", "0.25"
MK = {None: "o", 1: "o", 2: "s", 3: "^"}
octa = [np.array(p, float) for p in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]]

states = [(None, 0, np.array([1, 0, 0, 0], complex))]
for x in (1, 2, 3):
    for m in range(4):
        v = np.zeros(4, complex); v[0] = 1 / math.sqrt(2); v[x] = (1j) ** m / math.sqrt(2)
        states.append((x, m, v))


def gauge(v):
    k = int(np.argmax(np.abs(v) > 1e-9)); ph = cmath.phase(v[k]); return v * cmath.exp(-1j * ph), ph


def bloch(v):
    a, b = v; return np.array([2 * (a.conjugate() * b).real, 2 * (a.conjugate() * b).imag, abs(a) ** 2 - abs(b) ** 2])


def schmidt(psi):
    U, s, Vh = np.linalg.svd(psi.reshape(2, 2))
    a, pa = gauge(U[:, 0]); b, pb = gauge(Vh[0, :]); phi = 0.0
    if s[1] > 1e-9:
        _, qa = gauge(U[:, 1]); _, qb = gauge(Vh[1, :]); phi = (qa + qb) - (pa + pb)
    th = 2 * math.atan2(s[1], s[0])
    return [np.array([math.sin(th) * math.cos(phi), math.sin(th) * math.sin(phi), math.cos(th)]), bloch(a), bloch(b)], s[1] > 1e-9


def node(c0, c1):
    n = math.hypot(abs(c0), abs(c1))
    if n < 1e-12:
        return None, 0.0, 0.0
    th = 2 * math.acos(min(1, abs(c0) / n)); pg = cmath.phase(c0) if abs(c0) > 1e-12 else cmath.phase(c1)
    pl_ = cmath.phase(c1) - cmath.phase(c0) if abs(c0) > 1e-12 and abs(c1) > 1e-12 else 0.0
    return np.array([math.sin(th) * math.cos(pl_), math.sin(th) * math.sin(pl_), math.cos(th)]), n, pg


def btbs(psi):
    left, nl, gl = node(psi[0], psi[1]); right, nr, gr = node(psi[2], psi[3])
    root, _, _ = node(nl * cmath.exp(1j * gl), nr * cmath.exp(1j * gr))
    return [root, left, right]


def sphere(ax, elev=18, azim=30):
    u, v = np.mgrid[0:2 * np.pi:14j, 0:np.pi:8j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.88", lw=0.3, zorder=1)
    for p, q in [((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (-1, 0, 0)), ((-1, 0, 0), (0, -1, 0)), ((0, -1, 0), (1, 0, 0))]:
        ax.plot(*zip(p, q), color="0.75", lw=0.4, ls="--", zorder=2)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85); ax.set_zlim(-0.85, 0.85)


def draw_points(ax, pts, axis=None):
    """pts: list of (x, m, p) with p a Bloch vector or None.  Points of the same direction at the same
    vertex are merged (grey if several phases); different directions at the same vertex are spread."""
    groups = {}
    for x, m, p in pts:
        if p is None:
            continue
        assert min(np.linalg.norm(p - o) for o in octa) < 1e-9
        groups.setdefault((tuple(np.round(p, 6)), x), []).append(m)
    byvertex = {}
    for (p, x), ms in groups.items():
        byvertex.setdefault(p, []).append((x, ms))
    for p, lst in byvertex.items():
        p = np.array(p)
        t1 = np.cross(p, [0, 0, 1.0]) if abs(p[2]) < 0.9 else np.array([1.0, 0, 0])
        t1 = t1 / np.linalg.norm(t1); t2 = np.cross(p, t1)
        lst = sorted(lst, key=lambda t: -1 if t[0] is None else t[0])
        for j, (x, ms) in enumerate(lst):
            q = p if len(lst) == 1 else p + 0.17 * (math.cos(2 * math.pi * j / len(lst)) * t1 + math.sin(2 * math.pi * j / len(lst)) * t2)
            col = BLUE if x is None else (PH[ms[0]] if len(ms) == 1 else GREY)
            ax.scatter(*q, color=col, marker=MK[x], s=30 if x is None else 16, depthshade=False, zorder=8)


def tree(fig, x0, y0, w, h, getter, labels, title):
    """root on top, two leaves below; getter(psi) -> ([root, leaf1, leaf2], flags)."""
    pos = [[x0 + w * 0.27, y0 + h * 0.5, w * 0.46, h * 0.46], [x0, y0 + h * 0.04, w * 0.5, h * 0.46],
           [x0 + w * 0.5, y0 + h * 0.04, w * 0.5, h * 0.46]]
    res = [(x, m, getter(v)) for x, m, v in states]
    for k in range(3):
        ax = fig.add_axes(pos[k], projection="3d", computed_zorder=False); sphere(ax)
        draw_points(ax, [(x, m, r[0][k]) for x, m, r in res])
        if all(r[0][k] is None for _, _, r in res):
            ax.text2D(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", fontsize=6, color="0.5")
    fig.add_artist(plt.Line2D([x0 + w * 0.5, x0 + w * 0.27], [y0 + h * 0.53, y0 + h * 0.47], color="0.5", lw=0.6))
    fig.add_artist(plt.Line2D([x0 + w * 0.5, x0 + w * 0.73], [y0 + h * 0.53, y0 + h * 0.47], color="0.5", lw=0.6))
    for k, lab in enumerate(labels):
        fig.text([x0 + w * 0.5, x0 + w * 0.25, x0 + w * 0.75][k], [y0 + h * 0.97, y0 + h * 0.02, y0 + h * 0.02][k],
                 lab, ha="center", fontsize=6.3)
    fig.text(x0, y0 + h * 1.02, title, fontsize=7.5)


fig = plt.figure(figsize=(7.0, 5.4))
# (a) Schmidt tree, (b) Bloch sphere binary tree
tree(fig, 0.0, 0.53, 0.33, 0.42, lambda v: schmidt(v), ["Schmidt sphere", r"qubit 1: $|a\rangle$", r"qubit 2: $|b\rangle$"],
     "(a) Schmidt tree")
tree(fig, 0.34, 0.53, 0.33, 0.42, lambda v: (btbs(v), None), ["qubit 1", r"qubit 2 if $q_1=0$", r"qubit 2 if $q_1=1$"],
     "(b) Bloch sphere binary tree")
# (c) Q-sphere
ax = fig.add_axes([0.68, 0.55, 0.32, 0.38], projection="3d", computed_zorder=False); sphere(ax, elev=12, azim=25)
qpos = {0: np.array([0, 0, 1.0]), 1: np.array([1.0, 0, 0]), 2: np.array([-1.0, 0, 0]), 3: np.array([0, 0, -1.0])}
for x in (1, 2, 3):
    ax.plot(*zip(qpos[0], qpos[x]), color=RED, lw=1.0, zorder=3, alpha=0.85)
    p = qpos[x]; t1 = np.array([0, 1.0, 0]); t2 = np.cross(p, t1)
    for m in range(4):
        a = math.pi / 4 + m * math.pi / 2
        ax.scatter(*(p + 0.14 * (math.cos(a) * t1 + math.sin(a) * t2)), color=PH[m], marker=MK[x], s=14,
                   depthshade=False, zorder=8)
ax.scatter(*qpos[0], color=BLUE, s=30, depthshade=False, zorder=9)
for x, p in qpos.items():
    ax.text(*(p * (1.25 if x in (0, 3) else 1.45)), r"$|" + format(x, "02b") + r"\rangle$", ha="center", va="center", fontsize=6.5)
fig.text(0.68, 0.95, "(c) Q-sphere", fontsize=7.5)
# (d) dimensional circle notation (square)
bx = fig.add_axes([0.02, 0.08, 0.27, 0.36]); bx.set_aspect("equal"); bx.axis("off")
P = {0: np.array([0.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([1.0, 0.0]), 3: np.array([1.0, 1.0])}  # q1 horizontal, q2 vertical
for a_, b_ in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    bx.plot(*zip(P[a_], P[b_]), color="0.8", lw=0.6, zorder=1)
for x in (1, 2, 3):
    bx.plot(*zip(P[0], P[x]), color=RED, lw=1.0, zorder=2, alpha=0.85)
r = 0.16
for x, c in P.items():
    if x == 0:
        bx.add_patch(Circle(c, r, fc=BLUE, ec="k", lw=0.5, zorder=4))
    else:
        bx.add_patch(Circle(c, r, fc="white", ec="k", lw=0.5, zorder=4))
        for m in range(4):
            a = m * math.pi / 2
            bx.plot([c[0], c[0] + r * math.cos(a)], [c[1], c[1] + r * math.sin(a)], color=PH[m], lw=1.4, zorder=5)
    off = {0: (-0.28, -0.12), 1: (-0.28, 0.12), 2: (0.28, -0.12), 3: (0.28, 0.12)}[x]
    bx.text(c[0] + off[0], c[1] + off[1], r"$|" + format(x, "02b") + r"\rangle$", ha="center", va="center", fontsize=6.5)
bx.annotate("", xy=(1.35, -0.38), xytext=(-0.1, -0.38), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
bx.text(0.62, -0.5, "qubit 1", ha="center", fontsize=6)
bx.annotate("", xy=(-0.42, 1.25), xytext=(-0.42, -0.1), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
bx.text(-0.52, 0.55, "qubit 2", rotation=90, va="center", fontsize=6)
bx.set_xlim(-0.6, 1.5); bx.set_ylim(-0.6, 1.4)
fig.text(0.0, 0.47, "(d) dimensional circle notation", fontsize=7.5)


# (e) discrete phase space 4x4
def chi(psi):
    B = SpectrumBackend(psi); M = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            M[b, a] = B.E[pl.join(a, b, 2)] if (a or b) else 1.0
    return M


z = states[0][2]; prod = states[1][2]; bell = states[9][2]
Ms = [chi(z), chi(prod), chi(bell)]
titles = [r"$|00\rangle$", r"$|0\rangle|{+}\rangle$", r"$|\Phi^+\rangle$"]
cmap = matplotlib.colors.ListedColormap(["#d62728", "white", "#1f77b4"])
for k, (M, t) in enumerate(zip(Ms, titles)):
    cx = fig.add_axes([0.36 + 0.21 * k, 0.12, 0.17, 0.27])
    cx.imshow(np.round(M), cmap=cmap, vmin=-1, vmax=1, origin="lower")
    if k:
        sh = (np.abs(Ms[0]) > .5) & (np.abs(M) > .5)
        for b in range(4):
            for a in range(4):
                if sh[b, a]:
                    cx.add_patch(Rectangle((a - .5, b - .5), 1, 1, fill=False, ec="k", lw=1.1))
    lab = ["00", "01", "10", "11"]
    cx.set_xticks(range(4)); cx.set_yticks(range(4)); cx.set_xticklabels(lab, fontsize=5.5)
    cx.set_yticklabels(lab if k == 0 else [], fontsize=5.5); cx.tick_params(length=1.5)
    cx.set_xlabel(r"$a$", fontsize=6.5)
    if k == 0:
        cx.set_ylabel(r"$b$", fontsize=6.5)
    cx.set_title(t, fontsize=7)
fig.text(0.34, 0.47, r"(e) discrete phase space $\langle X^aZ^b\rangle$", fontsize=7.5)
h = [plt.Line2D([], [], ls="", marker="o", ms=5, color=BLUE, label=r"$|00\rangle$")]
h += [plt.Line2D([], [], ls="", marker=MK[x], ms=4.5, color="0.35", mfc="white", label=f"$x={x:02b}$") for x in (1, 2, 3)]
h += [plt.Line2D([], [], ls="", marker="o", ms=4.5, color=c, label=l) for c, l in zip(PH, [r"$i^m=+1$", r"$+i$", r"$-1$", r"$-i$"])]
h.append(plt.Line2D([], [], ls="", marker="o", ms=4.5, color=GREY, label="all phases"))
fig.legend(handles=h, fontsize=6.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=9,
           handletextpad=0.1, columnspacing=0.6)
fig.savefig("../results/fig1a_representations_2q.png", dpi=220)
print("ok")
