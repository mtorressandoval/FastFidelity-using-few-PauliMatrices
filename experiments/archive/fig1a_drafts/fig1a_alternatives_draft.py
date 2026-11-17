"""Draft: three representations of the neighbors of |000> (3 qubits).
(a) Q-sphere, (b) dimensional circle notation (cube), (c) discrete phase space (Pauli characteristic function)."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from fastfidelity import SpectrumBackend
from fastfidelity import paulis as pl

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
RED, BLUE = "#c0392b", "#1f4e79"
PH = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]          # phases +1, +i, -1, -i
n = 3
fig = plt.figure(figsize=(7.0, 2.55))

# ---------------- (a) Q-sphere
ax = fig.add_axes([0.0, 0.05, 0.30, 0.84], projection="3d", computed_zorder=False)
u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:9j]
ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.9", lw=0.3, zorder=1)
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=15, azim=20)
pos = {}
for w in range(n + 1):
    xs = [x for x in range(2 ** n) if bin(x).count("1") == w]
    th = math.pi * w / n
    for k, x in enumerate(xs):
        ph = 2 * math.pi * k / len(xs) + (math.pi / 3 if w == 2 else 0)
        pos[x] = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
for x in range(1, 8):
    ax.plot(*zip(pos[0], pos[x]), color=RED, lw=1.0, zorder=3, alpha=0.85)
    p = pos[x]; t1 = np.cross(p, [0, 0, 1])
    t1 = t1 / np.linalg.norm(t1) if np.linalg.norm(t1) > 1e-9 else np.array([1.0, 0, 0]); t2 = np.cross(p, t1)
    for m in range(4):
        a = math.pi / 4 + m * math.pi / 2
        ax.scatter(*(p + 0.11 * (math.cos(a) * t1 + math.sin(a) * t2)), color=PH[m], s=9, depthshade=False, zorder=8)
ax.scatter(*pos[0], color=BLUE, s=22, depthshade=False, zorder=9)
for x, p in pos.items():
    q = p * (1.18 if x == 0 else (1.22 if x == 7 else 1.45))
    ax.text(*q, r"$|" + format(x, "03b") + r"\rangle$", ha="center", va="center", fontsize=6.3, zorder=10)
fig.text(0.01, 0.95, "(a) Q-sphere", fontsize=7.5)

# ---------------- (b) dimensional circle notation (cube)
bx = fig.add_axes([0.31, 0.10, 0.29, 0.80]); bx.set_aspect("equal"); bx.axis("off")
e = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.33])]   # qubit axes
P = {x: sum(((x >> (n - 1 - i)) & 1) * e[i] for i in range(n)) for x in range(8)}
for x in range(8):
    for i in range(n):
        y = x ^ (1 << (n - 1 - i))
        if y > x:
            bx.plot(*zip(P[x], P[y]), color="0.8", lw=0.6, zorder=1)
for x in range(1, 8):
    bx.plot(*zip(P[0], P[x]), color=RED, lw=1.0, zorder=2, alpha=0.85)
r = 0.11
for x in range(8):
    c = P[x]
    if x == 0:
        bx.add_patch(Circle(c, r, fc=BLUE, ec="k", lw=0.5, zorder=4))
    else:
        bx.add_patch(Circle(c, r, fc="white", ec="k", lw=0.5, zorder=4))
        for m in range(4):                       # one pointer per neighbor, angle = phase m*pi/2
            a = m * math.pi / 2
            bx.plot([c[0], c[0] + r * math.cos(a)], [c[1], c[1] + r * math.sin(a)], color=PH[m], lw=1.3, zorder=5)
    off = {0: (-0.2, -0.13), 7: (0.2, 0.13)}.get(x, (0.0, -0.2) if x in (4, 6) else (0.0, 0.19))
    bx.text(c[0] + off[0], c[1] + off[1], r"$|" + format(x, "03b") + r"\rangle$", ha="center", va="center", fontsize=6.3)
for i, lab in enumerate(["qubit 1", "qubit 2", "qubit 3"]):
    pass
bx.set_xlim(-0.35, 1.75); bx.set_ylim(-0.35, 1.6)
fig.text(0.32, 0.95, "(b) dimensional circle notation", fontsize=7.5)

# ---------------- (c) discrete phase space: <X^a Z^b> for |000> and |GHZ>
def chi(psi):
    B = SpectrumBackend(psi); M = np.zeros((8, 8))
    for a in range(8):
        for b in range(8):
            M[b, a] = B.E[pl.join(a, b, n)] if (a or b) else 1.0
    return M
z = np.zeros(8, complex); z[0] = 1
g = np.zeros(8, complex); g[0] = g[7] = 1 / math.sqrt(2)
Ms = [chi(z), chi(g)]
cmap = matplotlib.colors.ListedColormap(["#d62728", "white", "#1f77b4"])
for k, (M, title) in enumerate(zip(Ms, [r"$|000\rangle$", r"$(|000\rangle+|111\rangle)/\sqrt{2}$"])):
    cx = fig.add_axes([0.655 + 0.175 * k, 0.2, 0.16, 0.6])
    cx.imshow(np.round(M), cmap=cmap, vmin=-1, vmax=1, origin="lower")
    shared = (np.abs(Ms[0]) > 0.5) & (np.abs(Ms[1]) > 0.5)
    for b in range(8):
        for a in range(8):
            if shared[b, a]:
                cx.add_patch(Rectangle((a - 0.5, b - 0.5), 1, 1, fill=False, ec="k", lw=1.0))
    cx.set_xticks(range(8)); cx.set_yticks(range(8))
    cx.set_xticklabels([format(a, "03b") for a in range(8)], rotation=90, fontsize=5)
    cx.set_yticklabels([format(b, "03b") for b in range(8)] if k == 0 else [], fontsize=5)
    cx.set_xlabel(r"$a$ ($X$ part)", fontsize=6.5)
    if k == 0:
        cx.set_ylabel(r"$b$ ($Z$ part)", fontsize=6.5)
    cx.set_title(title, fontsize=6.8)
    cx.tick_params(length=1.5)
fig.text(0.625, 0.95, r"(c) discrete phase space $\langle X^aZ^b\rangle$", fontsize=7.5)
h = [plt.Line2D([], [], ls="", marker="o", ms=4, color=c, label=l) for c, l in zip(PH, [r"$+1$", r"$+i$", r"$-1$", r"$-i$"])]
fig.legend(handles=h, title=r"phase $i^m$", title_fontsize=6, fontsize=6, frameon=False, loc="lower left",
           bbox_to_anchor=(0.26, 0.0), ncol=4, handletextpad=0.1, columnspacing=0.5)
fig.savefig("../results/fig1a_alternatives_draft.png", dpi=220)
print("shared cells:", int(((np.abs(Ms[0]) > .5) & (np.abs(Ms[1]) > .5)).sum()), "of", int((np.abs(Ms[1]) > .5).sum()))
