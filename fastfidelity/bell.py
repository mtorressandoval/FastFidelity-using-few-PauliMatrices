"""Simulation of Algorithm 3 of Grewal, Iyer, Kretschmer and Liang
(arXiv:2304.13915, Thm. 6.7): bounded-distance stabilizer approximation for a
pure state with fidelity > cos^2(pi/8) with some stabilizer state.

Measurements are entangled Bell measurements on two copies:
  1. m Bell difference samples (4 copies each) -> candidate Weyl operators;
  2. squared expectations <W_x>^2 of the candidates from k_B copies measured
     pairwise in the Bell basis (Huang, Kueng and Preskill estimator);
  3. keep x with estimate > 1/2, check that they span a Lagrangian subspace;
  4. measure k_M copies in the stabilizer basis of that subspace (Clifford
     circuit) and take the majority outcome to fix the signs.
Only small n (dense simulation of the two-copy measurement distribution).
"""
from __future__ import annotations

import math

import numpy as np

from . import paulis as pl
from .walsh import fwht_axis
from .f2 import Basis
from .tableau import StabTableau

__all__ = ["bell_distribution", "grewal_theory_copies", "run_grewal", "COS2_PI8"]

COS2_PI8 = math.cos(math.pi / 8) ** 2


def bell_distribution(psi):
    """Pr[y] of a Bell measurement on psi (x) psi, y = (a, b) as a Pauli key.

    Pr[y] = 2^-n |psi^T W_y psi|^2  (Bell states (W_y (x) I)|Phi+>).
    """
    psi = np.asarray(psi, dtype=complex)
    N = psi.shape[0]
    n = int(np.log2(N))
    xs = np.arange(N)
    V = psi[xs[None, :] ^ xs[:, None]] * psi[None, :]          # V[a, j] = psi_{j^a} psi_j
    W = fwht_axis(V, axis=1)                                    # sum_j V (-1)^{b.j}
    ph = 1j ** pl.popcount(xs[:, None] & xs[None, :])
    P = np.abs(ph * W) ** 2 / N
    return P.ravel() / P.sum()


def grewal_theory_copies(n, tau, delta=0.01):
    """Copy count of Algorithm 3 with the constants of Thm. 6.7 (gamma = tau - cos^2(pi/8))."""
    gamma = tau - COS2_PI8
    m = (8 + 4 * math.sqrt(3)) / math.cos(math.pi / 8) ** 8 * (n + math.log(3 / delta))
    kB = math.log(6 * m / delta) / (2 * gamma ** 2)
    kM = 4 * math.log(3 / delta)
    return dict(m=m, copies_bds=4 * m, copies_bell=kB, copies_majority=kM,
                total=4 * m + kB + kM)


def run_grewal(psi, m, kB, kM, rng, E=None):
    """One run of Algorithm 3. Returns (StabTableau or None, copies used)."""
    psi = np.asarray(psi, dtype=complex)
    n = int(np.log2(psi.shape[0]))
    pB = bell_distribution(psi)
    keys = np.arange(pB.shape[0])
    # (1) Bell difference sampling: y1 ^ y2
    y = rng.choice(keys, size=(m, 2), p=pB)
    cand = np.unique(y[:, 0] ^ y[:, 1])
    cand = cand[cand != 0]
    # (2) HKP: kB copies -> kB // 2 Bell samples shared by all candidates
    nb = max(1, kB // 2)
    yb = rng.choice(keys, size=nb, p=pB)
    est = []
    for x in cand:
        x = int(x)
        sgn = (-1) ** pl.popcount(pl.split(x, n)[0] & pl.split(x, n)[1])   # W_x^* = (-1)^{a.b} W_x
        par = pl.symp(np.full(nb, x, dtype=np.int64), yb, n)
        est.append(sgn * np.mean(1 - 2 * par))
    keep = [int(x) for x, e in zip(cand, est) if e > 0.5]
    # (3) Lagrangian check
    B = Basis()
    gens = []
    for x in keep:
        if all(pl.symp(x, g, n) == 0 for g in gens) and B.add(x):
            gens.append(x)
    copies = 4 * m + 2 * nb + kM
    if len(gens) != n or any(pl.symp(a, b, n) for a in keep for b in keep):
        return None, copies
    # (4) majority vote in the stabilizer basis of span(gens)
    T = StabTableau(n, gens, [1] * n)
    if E is None:
        from .backends import pauli_spectrum
        E = pauli_spectrum(psi)
    px = np.clip(T.walsh_from_spectrum(E), 0, None)
    px /= px.sum()
    outs = rng.choice(px.shape[0], size=kM, p=px)
    x = int(np.bincount(outs, minlength=px.shape[0]).argmax())
    return T.flip(x), copies
