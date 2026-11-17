"""Draft: Schmidt tree for two qubits.  Root: Schmidt sphere (theta from the Schmidt coefficients,
phi = relative phase); leaves: Schmidt basis of each qubit.  |00> and its 12 neighbors in one tree."""
import math, cmath
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
PH = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
BLUE = "#1f4e79"
MK = {None: "o", 1: "o", 2: "s", 3: "^"}       # marker by direction x


def gauge(v):
    """Remove the phase of the first nonzero component; returns (vector, removed phase)."""
    k = int(np.argmax(np.abs(v) > 1e-9)); ph = cmath.phase(v[k])
    return v * cmath.exp(-1j * ph), ph


def bloch(v):
    a, b = v
    return np.array([2 * (a.conjugate() * b).real, 2 * (a.conjugate() * b).imag, abs(a) ** 2 - abs(b) ** 2])


def schmidt(psi):
    M = psi.reshape(2, 2)                       # M[q1, q2]
    U, s, Vh = np.linalg.svd(M)
    a, pa = gauge(U[:, 0]); b, pb = gauge(Vh[0, :])
    c1 = s[0] * cmath.exp(1j * (pa + pb))
    if s[1] > 1e-9:
        a2, qa = gauge(U[:, 1]); b2, qb = gauge(Vh[1, :])
        c2 = s[1] * cmath.exp(1j * (qa + qb))
        phi = cmath.phase(c2) - cmath.phase(c1)
    else:
        phi = 0.0
    th = 2 * math.atan2(s[1], s[0])
    root = np.array([math.sin(th) * math.cos(phi), math.sin(th) * math.sin(phi), math.cos(th)])
    return root, bloch(a), bloch(b), s[1] > 1e-9


def sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:14j, 0:np.pi:8j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.88", lw=0.3, zorder=1)
    for p, q in [((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (-1, 0, 0)), ((-1, 0, 0), (0, -1, 0)), ((0, -1, 0), (1, 0, 0))]:
        ax.plot(*zip(p, q), color="0.75", lw=0.4, ls="--", zorder=2)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=30)
    ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85); ax.set_zlim(-0.85, 0.85)


states = [(None, 0, np.array([1, 0, 0, 0], complex))]
for x in (1, 2, 3):
    for m in range(4):
        v = np.zeros(4, complex); v[0] = 1 / math.sqrt(2); v[x] = (1j) ** m / math.sqrt(2)
        states.append((x, m, v))
octa = [np.array(p, float) for p in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]]
data = [(x, m) + schmidt(v) for x, m, v in states]
fig = plt.figure(figsize=(3.4, 3.0))
axes = {0: fig.add_axes([0.25, 0.50, 0.5, 0.46], projection="3d", computed_zorder=False),
        1: fig.add_axes([0.0, 0.08, 0.5, 0.44], projection="3d", computed_zorder=False),
        2: fig.add_axes([0.5, 0.08, 0.5, 0.44], projection="3d", computed_zorder=False)}
for k, ax in axes.items():
    sphere(ax)
    groups = {}
    for x, m, root, ba, bb, ent in data:
        p = [root, ba, bb][k]
        assert min(np.linalg.norm(p - o) for o in octa) < 1e-9, (x, m, k, p)
        groups.setdefault((tuple(np.round(p, 6)), x), []).append((m, ent))
    for (p, x), ms in groups.items():
        p = np.array(p)
        col = BLUE if x is None else (PH[ms[0][0]] if len(ms) == 1 else "0.25")
        ax.scatter(*p, color=col, marker=MK[x], s=26 if x is None else 18, depthshade=False, zorder=8 + (x or 0))
        if k > 0 and any(e for _, e in ms):             # Schmidt basis axis for entangled states
            ax.plot(*zip(p, -p), color="0.35", lw=0.8, ls=":", zorder=3)
            ax.scatter(*(-p), facecolor="white", edgecolor="0.25", marker=MK[x], s=18, depthshade=False, zorder=8)
fig.text(0.5, 0.97, "Schmidt sphere", ha="center", fontsize=7.5)
fig.text(0.25, 0.08, r"qubit 1: $|a\rangle$", ha="center", fontsize=7)
fig.text(0.75, 0.08, r"qubit 2: $|b\rangle$", ha="center", fontsize=7)
fig.add_artist(plt.Line2D([0.5, 0.27], [0.53, 0.47], color="0.5", lw=0.6))
fig.add_artist(plt.Line2D([0.5, 0.73], [0.53, 0.47], color="0.5", lw=0.6))
h = [plt.Line2D([], [], ls="", marker="o", ms=4.5, color=BLUE, label=r"$|00\rangle$")]
h += [plt.Line2D([], [], ls="", marker=MK[x], ms=4, color="0.35", mfc="white", label=f"$x={x:02b}$") for x in (1, 2, 3)]
h += [plt.Line2D([], [], ls="", marker="o", ms=4, color=c, label=l) for c, l in zip(PH, [r"$+1$", r"$+i$", r"$-1$", r"$-i$"])]
h.append(plt.Line2D([], [], ls="", marker="o", ms=4, color="0.25", label="all phases"))
fig.legend(handles=h, fontsize=5.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5,
           handletextpad=0.1, columnspacing=0.5)
fig.text(0.02, 0.70, r"$\cos\frac{\theta}{2}|a\rangle|b\rangle$" "\n" r"$+e^{i\phi}\sin\frac{\theta}{2}|a^\perp\rangle|b^\perp\rangle$",
         fontsize=6.2)
fig.savefig("../results/fig1a_schmidt_tree_draft.png", dpi=250)
for x, m, root, ba, bb, ent in data:
    print(x, m, "root", np.round(root, 3), "a", np.round(ba, 3), "b", np.round(bb, 3), "entangled", ent)
