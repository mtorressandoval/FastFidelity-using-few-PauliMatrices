"""Stabilizer states as signed generator tableaux, and their neighbours.

A stabilizer state |z> is stored by n signed generators s_i W_{g_i} and n
destabilizers W_{d_i} (unsigned), with symp(d_i, g_j) = delta_ij and
symp(d_i, d_j) = 0.  The group element labelled by u in F_2^n is

    s_u W_{g_u} = prod_{i: u_i = 1} s_i W_{g_i} ,

and every Pauli of the group is reached exactly once.  The eigenbasis of the
group is phi_x = Q_x z with Q_x = prod_i W_{d_i}^{x_i}; p_x = <phi_x|rho|phi_x>
is the Walsh transform of u -> s_u tr(rho W_{g_u}).
"""
from __future__ import annotations

import numpy as np

from . import paulis as pl
from .f2 import Basis, Affine, random_bits

__all__ = ["StabTableau"]


def _signed_product(k1, s1, k2, s2, n):
    """(s1 W_k1)(s2 W_k2) for commuting Paulis -> (k, s)."""
    k, e = pl.mul(k1, k2, n)
    if isinstance(e, np.ndarray):
        if np.any(e % 2):
            raise ValueError("anticommuting product")
        return k, s1 * s2 * (1 - e)          # e in {0, 2}
    if e % 2:
        raise ValueError("anticommuting product")
    return k, s1 * s2 * (1 if e == 0 else -1)


def destabilizers_for(gens, n):
    """Destabilizers for n independent commuting generators (symplectic Gram-Schmidt)."""
    m = len(gens)
    rows = []
    for b in range(2 * n):                     # unit vectors and their images
        v = 1 << b
        img = 0
        for j, g in enumerate(gens):
            img |= pl.symp(v, g, n) << j
        rows.append([img, v])
    piv_rows = []
    for j in range(m):                         # Gauss-Jordan on the images
        r = next(i for i in range(len(rows)) if (rows[i][0] >> j) & 1)
        pr = rows.pop(r)
        for row in rows + piv_rows:
            if (row[0] >> j) & 1:
                row[0] ^= pr[0]
                row[1] ^= pr[1]
        piv_rows.append(pr)
    e = []
    for j in range(m):
        assert piv_rows[j][0] == 1 << j
        e.append(piv_rows[j][1])
    d = []                                     # make destabilizers commute
    for i in range(m):
        v = e[i]
        for j in range(i + 1, m):
            if pl.symp(e[i], e[j], n):
                v ^= gens[j]
        d.append(v)
    return d


class StabTableau:
    """Signed stabilizer tableau of an n-qubit stabilizer state."""

    __slots__ = ("n", "g", "s", "d", "_fast")

    def __init__(self, n, gens, signs, destabs=None, check=False):
        self.n = int(n)
        self.g = [int(k) for k in gens]
        self.s = [int(t) for t in signs]
        self.d = [int(k) for k in destabs] if destabs is not None else destabilizers_for(self.g, n)
        self._fast = self.n <= 31
        if check:
            self.check()

    # ------------------------------------------------------------------ basics
    def check(self):
        n = self.n
        assert len(self.g) == n and len(self.d) == n
        for i in range(n):
            for j in range(n):
                assert pl.symp(self.g[i], self.g[j], n) == 0
                assert pl.symp(self.d[i], self.d[j], n) == 0
                assert pl.symp(self.d[i], self.g[j], n) == (i == j)
        assert Basis(self.g).dim == n
        return True

    def copy(self):
        return StabTableau(self.n, list(self.g), list(self.s), list(self.d))

    @classmethod
    def from_generators(cls, n, gens, signs):
        return cls(n, gens, signs)

    @classmethod
    def zero_state(cls, n):
        """|0...0>: generators Z_j, destabilizers X_j."""
        return cls(n, [1 << j for j in range(n)], [1] * n, [1 << (j + n) for j in range(n)])

    @classmethod
    def from_qiskit(cls, state):
        """From ``StabilizerState`` or ``Clifford`` (stabilizer rows = generators)."""
        from qiskit.quantum_info import StabilizerState, Clifford
        cliff = state.clifford if isinstance(state, StabilizerState) else state
        n = cliff.num_qubits
        def keys(X, Z):
            return [pl.join(sum(1 << j for j in range(n) if X[r, j]),
                            sum(1 << j for j in range(n) if Z[r, j]), n) for r in range(n)]
        gens = keys(cliff.stab_x, cliff.stab_z)
        signs = [(-1 if p else 1) for p in cliff.stab_phase]
        des = keys(cliff.destab_x, cliff.destab_z)
        return cls(n, gens, signs, des)

    def to_qiskit(self):
        """Return a ``qiskit.quantum_info.StabilizerState``."""
        from qiskit.quantum_info import StabilizerState, Clifford
        n = self.n
        tab = np.zeros((2 * n, 2 * n + 1), dtype=bool)
        for r, (k, s) in enumerate([(k, 1) for k in self.d] + list(zip(self.g, self.s))):
            x, z = pl.split(k, n)
            for j in range(n):
                tab[r, j] = (x >> j) & 1
                tab[r, n + j] = (z >> j) & 1
            tab[r, 2 * n] = s < 0
        return StabilizerState(Clifford(tab, validate=False))

    @classmethod
    def random(cls, n, rng):
        from qiskit.quantum_info import random_clifford
        seed = int(rng.integers(0, 2**31 - 1))
        return cls.from_qiskit(random_clifford(n, seed=seed))

    def statevector(self):
        """Dense state vector (small n), built by projecting a random vector."""
        n = self.n
        N = 1 << n
        rng = np.random.default_rng(12345)
        phi = rng.normal(size=N) + 1j * rng.normal(size=N)
        xs = np.arange(N)
        for k, s in zip(self.g, self.s):
            x, z = pl.split(k, n)
            w = phi * (-1.0) ** pl.popcount(z & xs)
            w = (1j ** pl.popcount(x & z)) * w[xs ^ x]
            phi = (phi + s * w) / 2
        return phi / np.linalg.norm(phi)

    def canonical(self):
        """Hashable canonical form (reduced row echelon of signed generators)."""
        n = self.n
        rows = list(zip(self.g, self.s))
        out = []
        L = 2 * n
        for bit in range(L - 1, -1, -1):
            idx = next((i for i, (k, _) in enumerate(rows) if (k >> bit) & 1), None)
            if idx is None:
                continue
            pk, ps = rows.pop(idx)
            new = []
            for k, s in rows:
                if (k >> bit) & 1:
                    k, s = _signed_product(k, s, pk, ps, n)
                new.append((k, s))
            rows = new
            out = [(_signed_product(k, s, pk, ps, n) if (k >> bit) & 1 else (k, s)) for k, s in out]
            out.append((pk, ps))
        return tuple(sorted(out))

    def __eq__(self, other):
        return isinstance(other, StabTableau) and self.canonical() == other.canonical()

    def __hash__(self):
        return hash(self.canonical())

    # --------------------------------------------------------- group elements
    def element(self, u):
        """Keys and signs of the group elements labelled by u (int or array)."""
        n = self.n
        scalar = not isinstance(u, np.ndarray)
        if scalar:
            k, s = 0, 1
            for i in range(n):
                if (u >> i) & 1:
                    k, s = _signed_product(k, s, self.g[i], self.s[i], n)
            return k, s
        if self._fast and u.dtype != object:
            u = u.astype(np.int64)
            k = np.zeros(u.shape, dtype=np.int64)
            s = np.ones(u.shape, dtype=np.int64)
            for i in range(n):
                sel = ((u >> i) & 1).astype(bool)
                if not sel.any():
                    continue
                gi = np.int64(self.g[i])
                e = pl.mul_phase(k, np.full_like(k, gi), n)
                k = np.where(sel, k ^ gi, k)
                s = np.where(sel, s * self.s[i] * (1 - e), s)
            return k, s
        ks, ss = zip(*[self.element(int(t)) for t in u]) if len(u) else ((), ())
        return np.array(ks, dtype=object), np.array(ss, dtype=np.int64)

    def coords(self, k):
        """u such that W_k = +-W_{g_u} (assumes k is in the unsigned group)."""
        n = self.n
        if isinstance(k, np.ndarray):
            u = np.zeros(k.shape, dtype=np.int64 if k.dtype != object else object)
            for i, d in enumerate(self.d):
                dd = np.int64(d) if k.dtype != object else d
                u = u | (pl.symp(k, dd, n) << i)
            return u
        u = 0
        for i, d in enumerate(self.d):
            u |= pl.symp(k, d, n) << i
        return u

    def in_group(self, k):
        """True if W_k commutes with every generator (unsigned membership)."""
        n = self.n
        if isinstance(k, np.ndarray):
            ok = np.ones(k.shape, dtype=bool)
            for g in self.g:
                gg = np.int64(g) if k.dtype != object else g
                ok &= pl.symp(k, gg, n) == 0
            return ok
        return all(pl.symp(k, g, n) == 0 for g in self.g)

    def sign_of(self, k):
        """Sign s_k of W_k in the group (k must belong to the unsigned group)."""
        return self.element(self.coords(k))[1]

    def enumerate(self):
        """All 2^n elements: (keys, signs, u) as int64 arrays (small n)."""
        u = np.arange(1 << self.n, dtype=np.int64)
        k, s = self.element(u)
        return k, s, u

    def sample(self, rng, size, m=None):
        """Uniform group elements over the subgroup of the first m generators."""
        m = self.n if m is None else m
        u = random_bits(rng, size, m)
        k, s = self.element(u)
        return k, s, u

    def affine(self, m=None):
        """The (unsigned) group or the subgroup of the first m generators."""
        m = self.n if m is None else m
        return Affine(0, Basis(self.g[:m]), 2 * self.n)

    # -------------------------------------------------------------- dynamics
    def q_of(self, x):
        q = 0
        for i in range(self.n):
            if (x >> i) & 1:
                q ^= self.d[i]
        return q

    def flip(self, x):
        """phi_x = Q_x z: flip the signs of generators with x_i = 1."""
        s = [(-t if (x >> i) & 1 else t) for i, t in enumerate(self.s)]
        return StabTableau(self.n, list(self.g), s, list(self.d))

    def neighbor(self, x, m):
        """nu_{x,m} = (z + i^m phi_x)/sqrt(2) for x != 0, m in {0,1,2,3}."""
        n = self.n
        if x == 0:
            raise ValueError("x must be non-zero")
        i0 = next(i for i in range(n) if (x >> i) & 1)
        q = self.q_of(x)
        g0, s0 = self.g[i0], self.s[i0]
        g, s = list(self.g), list(self.s)
        for i in range(n):
            if i != i0 and (x >> i) & 1:
                g[i], s[i] = _signed_product(g[i], s[i], g0, s0, n)
        if m % 2 == 0:
            g[i0], s[i0] = q, (1 if m == 0 else -1)
        else:
            kh, e = pl.mul(q, g0, n)                  # W_q W_g0 = i^e W_kh, e odd
            sh = s0 * (1 if (1 + e) % 4 == 0 else -1)  # i * s0 * i^e
            g[i0], s[i0] = kh, (sh if m == 1 else -sh)
        d = list(self.d)
        d[i0] = g0
        return StabTableau(n, g, s, d)

    def neighbors(self):
        return [(x, m, self.neighbor(x, m)) for x in range(1, 1 << self.n) for m in range(4)]

    def random_walk(self, steps, rng):
        """Random walk of the given length over the neighbour graph."""
        z = self
        for _ in range(steps):
            x = int(rng.integers(1, 1 << z.n)) if z.n <= 62 else int(random_bits(rng, 1, z.n)[0]) or 1
            z = z.neighbor(x, int(rng.integers(4)))
        return z

    # ----------------------------------------------------------- references
    def fidelity_from_spectrum(self, E):
        """f(z) = 2^-n sum_P s_P tr(rho P) from a full Pauli spectrum E[key]."""
        k, s, _ = self.enumerate()
        return float(np.dot(s, E[k]) / (1 << self.n))

    def walsh_from_spectrum(self, E):
        """Exact p_x for all x from a full Pauli spectrum."""
        from .walsh import fwht
        k, s, u = self.enumerate()
        h = np.zeros(1 << self.n)
        h[u] = s * E[k]
        return fwht(h) / (1 << self.n)
