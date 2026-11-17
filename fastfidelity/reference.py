"""Exact references and test states.

``fstab_exact`` enumerates all stabilizer states (quadratic-form representation
of Dehaene and De Moor) and is exact for pure or mixed states; it is practical
up to n = 5 (2 423 520 stabilizer states).  For larger n we use planted optima
with <phi|rho|phi> > cos^2(pi/8), which are unique (Appendix A of the paper).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .tableau import StabTableau
from .f2 import Basis

__all__ = ["num_stabilizer_states", "fstab_exact", "all_stabilizer_states",
           "near_stabilizer", "clifford_t_state", "haar_state", "depolarize",
           "random_rank_k", "stab_overlap", "mixture_fidelity", "planted_mixture"]


def num_stabilizer_states(n):
    return 2**n * math.prod(2**k + 1 for k in range(1, n + 1))


def _rref_subspaces(n, k):
    """All k-dim subspaces of F_2^n as (rows as ints, pivot list)."""
    for piv in itertools.combinations(range(n), k):
        free = []
        for i, p in enumerate(piv):
            free.append([c for c in range(p + 1, n) if c not in piv])
        nfree = sum(len(f) for f in free)
        for fill in range(1 << nfree):
            rows, t = [], 0
            for i, p in enumerate(piv):
                r = 1 << p
                for c in free[i]:
                    if (fill >> t) & 1:
                        r |= 1 << c
                    t += 1
                rows.append(r)
            yield rows, piv


def _phase_vectors(k):
    """All phase vectors i^{l.u} (-1)^{h.u + q(u)} on F_2^k, shape (2^{k(k+3)/2}, 2^k)."""
    U = np.arange(1 << k)
    bits = (U[:, None] >> np.arange(k)[None, :]) & 1                  # (2^k, k)
    # linear part: a_j = l_j + 2 h_j in Z_4 -> prod_j (i^{a_j})^{u_j}
    A = np.array(list(itertools.product(range(4), repeat=k)), dtype=int).reshape(4**k, k)
    lin = (1j) ** ((bits @ A.T) % 4)                                  # (2^k, 4^k)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if pairs:
        Q = np.array(list(itertools.product((0, 1), repeat=len(pairs))), dtype=int)
        quad = np.zeros((Q.shape[0], 1 << k), dtype=int)
        for t, (i, j) in enumerate(pairs):
            quad += np.outer(Q[:, t], bits[:, i] * bits[:, j])
        qs = (-1.0) ** (quad % 2)                                     # (2^{#pairs}, 2^k)
    else:
        qs = np.ones((1, 1 << k))
    return lin.T, qs                                                  # (4^k, 2^k), (nq, 2^k)


def _support(rows, c, k):
    U = np.arange(1 << k)
    x = np.full(1 << k, c, dtype=np.int64)
    for t, r in enumerate(rows):
        x ^= ((U >> t) & 1) * r
    return x


def fstab_exact(state, return_state=False, chunk=1 << 16):
    """Exact F_STAB of a state vector or density matrix (n <= 5 in practice)."""
    rho = np.asarray(getattr(state, "data", state))
    if rho.ndim == 1:
        vecs, lam = rho[None, :] / np.linalg.norm(rho), np.array([1.0])
    else:
        ev, V = np.linalg.eigh((rho + rho.conj().T) / 2)
        keep = ev > 1e-12
        vecs, lam = V[:, keep].T, ev[keep]
    n = int(np.log2(vecs.shape[1]))
    best, arg = -1.0, None
    for k in range(n + 1):
        lin, qs = _phase_vectors(k)
        for rows, piv in _rref_subspaces(n, k):
            for c in range(1 << n):
                if any((c >> p) & 1 for p in piv):
                    continue
                xs = _support(rows, c, k)
                sub = vecs[:, xs]                                     # (r, 2^k)
                for q0 in range(0, qs.shape[0], max(1, chunk >> (2 * k))):
                    W = (lin[None, :, :] * qs[q0:q0 + max(1, chunk >> (2 * k)), None, :]).reshape(-1, 1 << k)
                    amp = W.conj() @ sub.T                            # (#states, r)
                    F = (np.abs(amp) ** 2 @ lam) / (1 << k)
                    j = int(np.argmax(F))
                    if F[j] > best:
                        best = float(F[j])
                        if return_state:
                            phi = np.zeros(1 << n, dtype=complex)
                            phi[xs] = W[j] / np.sqrt(1 << k)
                            arg = phi
    return (best, arg) if return_state else best


def all_stabilizer_states(n):
    """Dense array with every n-qubit stabilizer state (small n, used in tests)."""
    out = []
    for k in range(n + 1):
        lin, qs = _phase_vectors(k)
        W = (lin[None, :, :] * qs[:, None, :]).reshape(-1, 1 << k) / np.sqrt(1 << k)
        for rows, piv in _rref_subspaces(n, k):
            for c in range(1 << n):
                if any((c >> p) & 1 for p in piv):
                    continue
                xs = _support(rows, c, k)
                block = np.zeros((W.shape[0], 1 << n), dtype=complex)
                block[:, xs] = W
                out.append(block)
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# test states
# ----------------------------------------------------------------------------
def near_stabilizer(n, tau, rng):
    """psi = sqrt(tau) phi + sqrt(1-tau) chi, chi Haar and orthogonal to phi."""
    T = StabTableau.random(n, rng)
    phi = T.statevector()
    chi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    chi -= np.vdot(phi, chi) * phi
    chi /= np.linalg.norm(chi)
    return np.sqrt(tau) * phi + np.sqrt(1 - tau) * chi, T


def clifford_t_state(n, t, rng):
    """C_t T C_{t-1} ... T C_0 |0>, with random Cliffords and T on random qubits."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_clifford, Statevector
    qc = QuantumCircuit(n)
    qc.compose(random_clifford(n, seed=int(rng.integers(2**31))).to_circuit(), inplace=True)
    for _ in range(t):
        qc.t(int(rng.integers(n)))
        qc.compose(random_clifford(n, seed=int(rng.integers(2**31))).to_circuit(), inplace=True)
    return Statevector(qc).data, qc


def haar_state(n, rng):
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return v / np.linalg.norm(v)


def depolarize(psi, p):
    """(1-p)|psi><psi| + p I/2^n ;  F_STAB = (1-p) F_STAB(psi) + p/2^n."""
    N = psi.shape[0]
    return (1 - p) * np.outer(psi, psi.conj()) + p * np.eye(N) / N


def random_rank_k(n, k, rng):
    from qiskit.quantum_info import random_density_matrix
    return random_density_matrix(1 << n, rank=k, seed=int(rng.integers(2**31))).data


def stab_overlap(A: StabTableau, B: StabTableau):
    """|<A|B>|^2 for two stabilizer states (any n)."""
    n = A.n
    I = Basis(A.g).intersect(Basis(B.g), 2 * n)
    for k in I.vectors:
        if A.sign_of(k) != B.sign_of(k):
            return 0.0
    return 2.0 ** (I.dim - n)


def mixture_fidelity(T, tableaus, weights):
    """<T| sum_i w_i |phi_i><phi_i| |T>."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    return float(sum(wi * stab_overlap(T, Ti) for Ti, wi in zip(tableaus, w)))


def planted_mixture(n, w1, n_comp, rng, near=None):
    """rho = w1 |phi_1><phi_1| + (1 - w1) sum_{i>1} |phi_i><phi_i| / (n_comp - 1).

    If ``near`` is an int, the other components are random walks of that length
    from phi_1 (harder instances); otherwise they are uniformly random.
    """
    T1 = StabTableau.random(n, rng)
    others = []
    for _ in range(n_comp - 1):
        others.append(T1.random_walk(near, rng) if near is not None else StabTableau.random(n, rng))
    Ts = [T1] + others
    w = [w1] + [(1 - w1) / max(1, n_comp - 1)] * (n_comp - 1)
    return Ts, w
