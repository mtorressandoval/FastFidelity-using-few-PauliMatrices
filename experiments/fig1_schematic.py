"""Fig. 1: (a) single-qubit stabilizer states: the four neighbours of |0>;
(b) flowchart of the self-guided estimation protocol."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from common import RESULTS

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
fig = plt.figure(figsize=(7.0, 3.2))

# ------------------------------------------------------------------ (a)
ax = fig.add_axes([-0.03, 0.0, 0.30, 1.0], projection="3d", computed_zorder=False)
pts = {r"$|0\rangle$": (0, 0, 1), r"$|1\rangle$": (0, 0, -1), r"$|+\rangle$": (1, 0, 0),
       r"$|-\rangle$": (-1, 0, 0), r"$|{+i}\rangle$": (0, 1, 0), r"$|{-i}\rangle$": (0, -1, 0)}
u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:9j]
ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.9", lw=0.3, zorder=1)
P = np.array(list(pts.values()))
for i in range(6):
    for j in range(i + 1, 6):
        if abs(np.dot(P[i], P[j])) < 1e-9:
            hot = i == 0 and P[j][2] == 0
            ax.plot(*zip(P[i], P[j]), color="#c0392b" if hot else "0.55", lw=1.8 if hot else 0.6,
                    ls="-" if hot else "--")
for lab, p in pts.items():
    c = "#1f4e79" if lab == r"$|0\rangle$" else ("#c0392b" if p[2] == 0 else "0.4")
    ax.scatter(*p, color=c, s=22, depthshade=False, zorder=8)
    off = {r"$|+\rangle$": (1.55, 0, -0.25), r"$|-\rangle$": (-1.45, 0, 0.3)}.get(lab)
    q = (p[0] * 1.42, p[1] * 1.42, p[2] * 1.25) if off is None else off
    ax.text(*q, lab, ha="center", va="center", fontsize=8, zorder=10)
ax.set_box_aspect((1, 1, 1))
ax.view_init(16, 22)
ax.set_axis_off()
ax.patch.set_alpha(0)
fig.text(0.01, 0.93, "(a)", fontsize=9)

# ------------------------------------------------------------------ (b)
ax2 = fig.add_axes([0.28, 0.0, 0.72, 1.0])
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.5, 4.75)
ax2.axis("off")
fig.text(0.262, 0.83, "(b)", fontsize=9)
BLUE, ORANGE, GREEN, YELLOW, GREY = "#dce9f5", "#fbe3cc", "#dcefd5", "#fff3c4", "0.3"


def num(x, y, k):
    ax2.add_patch(plt.Circle((x, y), 0.17, fc="white", ec=GREY, lw=0.7, zorder=5))
    ax2.text(x, y, str(k), ha="center", va="center", fontsize=6.5, zorder=6, fontweight="bold")


def box(cx, cy, w, h, txt, fc, k=None, fs=7):
    ax2.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                 fc=fc, ec=GREY, lw=0.7))
    ax2.text(cx, cy, txt, ha="center", va="center", fontsize=fs, linespacing=1.35)
    if k:
        num(cx - w / 2, cy + h / 2, k)


def diamond(cx, cy, w, h, txt, k=None):
    ax2.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
                          closed=True, fc=ORANGE, ec=GREY, lw=0.7))
    ax2.text(cx, cy, txt, ha="center", va="center", fontsize=7)
    if k:
        num(cx - w / 4 - 0.05, cy + h / 4 + 0.05, k)


def arrow(pts, txt=None, tpos=None, ls="-", color="0.15", both=False):
    xs, ys = zip(*pts)
    ax2.plot(xs, ys, color=color, lw=0.8, ls=ls, clip_on=False, zorder=4)
    ax2.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                  color=color, shrinkA=0, shrinkB=0))
    if both:
        ax2.add_patch(FancyArrowPatch(pts[1], pts[0], arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                      color=color, shrinkA=0, shrinkB=0))
    if txt:
        ax2.text(*tpos, txt, ha="center", va="center", fontsize=6.6, color="0.1")


yt, yb = 3.25, 0.5                               # top and bottom rows
box(0.75, yt, 1.3, 0.85, "initial\nstabilizer\nstate $|\\phi\\rangle$", "white", fs=6.8)
box(3.0, yt, 2.6, 0.95, "DFE on the group of $|\\phi\\rangle$:\n$\\hat p_x=\\hat f(\\phi_x)$ for the stabilizer\nbasis $\\{|\\phi_x\\rangle\\}$ with $|\\phi_0\\rangle=|\\phi\\rangle$", BLUE, 1, fs=6.3)
diamond(5.4, yt, 1.75, 1.05, "$\\hat p_x>\\hat p_0$ ?", 2)
box(8.2, yt, 3.0, 0.95, "DFE of the four neighbours\n$|\\nu_{x,m}\\rangle$ for the $K$ largest $\\hat p_x$\nwith $\\hat p_x>(3-2\\sqrt{2})\\,\\hat p_0$", BLUE, 3, fs=6.8)
diamond(8.25, yb, 1.75, 1.05, "$\\hat f(\\nu_{x,m})>\\hat p_0$ ?", 4)
diamond(6.0, yb, 1.85, 1.05, "$\\hat f(\\tilde\\phi)>\\hat p_0$ ?", 5)
diamond(3.85, yb, 1.75, 1.05, "retries left ?", 6)
box(1.55, yb, 2.3, 1.0, "stop: fresh DFE of $|\\phi\\rangle$,\n$\\hat F$ and its confidence\ninterval; optimum if\n$\\hat F_{\\rm low}>\\cos^2(\\pi/8)$", GREEN, 7, fs=6.5)

arrow([(1.42, yt), (1.68, yt)])
arrow([(4.32, yt), (4.51, yt)])
arrow([(6.29, yt), (6.68, yt)], "no", (6.48, yt + 0.17))
arrow([(8.25, yt - 0.5), (8.25, yb + 0.54)])
arrow([(7.36, yb), (6.94, yb)])
ax2.text(7.15, yb + 0.17, "no", ha="center", fontsize=6.6)
arrow([(5.06, yb), (4.74, yb)])
ax2.text(4.9, yb + 0.17, "no", ha="center", fontsize=6.6)
arrow([(2.96, yb), (2.72, yb)])
ax2.text(2.84, yb + 0.17, "no", ha="center", fontsize=6.6)
arrow([(5.4, yt + 0.54), (5.4, 4.05), (1.0, 4.05), (1.0, yt + 0.45)],
      "yes: $|\\phi\\rangle\\leftarrow|\\phi_x\\rangle$", (3.2, 4.2))
arrow([(9.14, yb), (9.85, yb), (9.85, 4.45), (0.5, 4.45), (0.5, yt + 0.45)],
      "yes: $|\\phi\\rangle\\leftarrow|\\nu_{x,m}\\rangle$", (7.6, 4.6))
arrow([(3.85, yb + 0.54), (3.85, 2.15), (2.6, 2.15), (2.6, yt - 0.5)],
      "yes: $N\\leftarrow 2N$", (3.2, 2.0))
arrow([(6.0, yb - 0.54), (6.0, -0.3), (-0.25, -0.3), (-0.25, yt), (0.08, yt)],
      "yes: $|\\phi\\rangle\\leftarrow|\\tilde\\phi\\rangle$", (3.0, -0.15))
box(6.0, 2.42, 2.3, 0.46, "Pauli store: outcomes\nof all substrings", YELLOW, fs=6.6)
box(6.0, 1.58, 3.25, 0.52, "local candidate $|\\tilde\\phi\\rangle$, generated by\nthe stored strings $P$ of weight $\\leq k$\nwith the largest $|\\bar P|$", YELLOW, fs=6.3)
for a, b in [((4.85, 2.5), (3.95, yt - 0.5)), ((7.15, 2.55), (7.9, yt - 0.5)), ((6.0, 2.19), (6.0, 1.83)), ((6.0, 1.33), (6.0, yb + 0.56))]:
    ax2.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=6, lw=0.6, ls=(0, (2, 2)),
                                  color="0.45", shrinkA=1, shrinkB=1))
fig.savefig(os.path.join(RESULTS, "fig1_schematic.pdf"))
fig.savefig(os.path.join(RESULTS, "fig1_schematic.png"), dpi=200)
print("ok")
