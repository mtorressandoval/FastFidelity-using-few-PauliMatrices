"""Hierarchical warm start by blocks of qubits.

The n qubits are split into 2^k contiguous blocks of similar size, with the
smallest k such that no block has more than ``max_block`` qubits.  The
protocol is first run on the reduced state of every block, measuring only Pauli
strings supported on the block.  Then pairs of neighbouring blocks are merged,
and the protocol runs on the merged block starting from the product of the two
results, until the whole system is reached.  Only single-copy Pauli
measurements are used at every level.
"""
from __future__ import annotations

import math

import numpy as np

from . import paulis as pl
from .tableau import StabTableau
from .search import self_guided_estimation

__all__ = ["SubsystemBackend", "embed_tableau", "product_tableau", "block_partition",
           "hierarchical_search"]


class SubsystemBackend:
    """Measurements of Pauli strings supported on ``qubits`` of a larger state."""

    def __init__(self, base, qubits):
        self.base = base
        self.q = list(qubits)
        self.n = len(self.q)
        self.N = base.n

    def embed(self, keys):
        keys = np.asarray(keys)
        n, N = self.n, self.N
        big = N > 31
        x, z = pl.split(keys, n)
        X = np.zeros(keys.shape, dtype=object if big else np.int64)
        Z = np.zeros(keys.shape, dtype=object if big else np.int64)
        for i, q in enumerate(self.q):
            X = X | (((x >> i) & 1) << q)
            Z = Z | (((z >> i) & 1) << q)
        return (X << N) | Z

    def measure(self, keys, rng):
        bits = self.base.measure(self.embed(keys), rng)
        out = np.zeros(np.asarray(keys).shape, dtype=np.int64)
        for i, q in enumerate(self.q):
            out |= ((bits >> q) & 1).astype(np.int64) << i
        return out


def embed_tableau(T, positions, m):
    """Tableau on a block, written on m qubits at the given positions."""
    def emb(k):
        x, z = pl.split(k, T.n)
        X = sum(((x >> i) & 1) << p for i, p in enumerate(positions))
        Z = sum(((z >> i) & 1) << p for i, p in enumerate(positions))
        return pl.join(X, Z, m)
    return [emb(k) for k in T.g], list(T.s), [emb(k) for k in T.d]


def product_tableau(TA, qa, TB, qb):
    """Tableau of |A> (x) |B> on the sorted union of the qubit lists qa and qb."""
    union = sorted(qa + qb)
    pos = {q: i for i, q in enumerate(union)}
    m = len(union)
    ga, sa, da = embed_tableau(TA, [pos[q] for q in qa], m)
    gb, sb, db = embed_tableau(TB, [pos[q] for q in qb], m)
    return StabTableau(m, ga + gb, sa + sb, da + db), union


def block_partition(n, max_block=8):
    """2^k contiguous blocks of similar size, k minimal with blocks <= max_block."""
    k = max(0, math.ceil(math.log2(n / max_block))) if n > max_block else 0
    return [list(map(int, b)) for b in np.array_split(np.arange(n), 2 ** k)], k


SCALE_N = {"on": False}


def _kw_for(m, kw, max_sub=12):
    """Substrings only for blocks of at most max_sub qubits (memory and benefit);
    with SCALE_N, N = 125 m shots per DFE and refresh of the store before retries."""
    out = dict(kw)
    if m > max_sub:
        out["substrings"] = False
    if SCALE_N["on"]:
        out.setdefault("N", max(1000, 125 * m))
        out.setdefault("refresh_on_retry", True)
    return out


def hierarchical_search(backend, max_block=8, rng=None, seed=None, final_shots=10_000,
                        **kw):
    """Block warm start followed by the global search.

    Returns (SearchResult of the global search, total shots of all levels, log),
    where the log lists (level, qubits, shots) for every search.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    n = backend.n
    blocks, k = block_partition(n, max_block)
    log, shots = [], 0
    states = []
    for q in blocks:                                   # leaves: random start
        sub = SubsystemBackend(backend, q)
        r = self_guided_estimation(sub, StabTableau.random(len(q), rng), rng=rng, final=False,
                                   keep_path=False, **_kw_for(len(q), kw))
        states.append((r.state, q))
        shots += r.search_shots
        log.append((k, q, r.search_shots))
    level = k
    while len(states) > 1:                             # merge pairs of blocks
        level -= 1
        merged = []
        for i in range(0, len(states), 2):
            (TA, qa), (TB, qb) = states[i], states[i + 1]
            T0, q = product_tableau(TA, qa, TB, qb)
            if len(q) == n:
                merged.append((T0, q))
                continue
            sub = SubsystemBackend(backend, q)
            r = self_guided_estimation(sub, T0, rng=rng, final=False, keep_path=False, **_kw_for(len(q), kw))
            merged.append((r.state, q))
            shots += r.search_shots
            log.append((level, q, r.search_shots))
        states = merged
    T0, q = states[0]
    assert q == list(range(n))
    r = self_guided_estimation(backend, T0, rng=rng, final_shots=final_shots, keep_path=False,
                               **_kw_for(n, kw))
    shots += r.search_shots
    log.append((0, q, r.search_shots))
    return r, shots, log, T0


class FrameBackend:
    """Measurements in the frame of a known Clifford C.

    A logical Pauli string P is measured as the physical string C P C^dag, which
    is again a Pauli string, so that only single-copy Pauli measurements are
    used.  Each shot gives one +-1 outcome for the whole string; it is returned
    as the bit of the lowest logical qubit of the support, so that substrings
    must not be used with this backend (substrings=False).
    """

    def __init__(self, base, clifford):
        from qiskit.quantum_info import Pauli
        self.base, self.C = base, clifford
        self.n = base.n
        n = self.n
        self.img = {}
        for j in range(n):
            for lab, (x, z) in (("X", (1, 0)), ("Z", (0, 1))):
                P = Pauli((np.eye(n, dtype=bool)[j] * bool(z), np.eye(n, dtype=bool)[j] * bool(x)))
                self.img[(j, lab)] = pl.from_qiskit(P.evolve(clifford, frame="s"))   # C P C^dag
            (ka, sa), (kb, sb) = self.img[(j, "X")], self.img[(j, "Z")]
            k, e = pl.mul(ka, kb, n)
            self.img[(j, "Y")] = (k, sa * sb * (1 if (1 + e) % 4 == 0 else -1))
        self._cache = {}

    def physical(self, key):
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = self.n
        k, s = 0, 1
        for j, c in pl.letters_on(key, n):
            kk, ss = self.img[(j, c)]
            k2, e = pl.mul(k, kk, n)
            k, s = k2, s * ss * (1 if e == 0 else -1)
        self._cache[key] = (k, s)
        return k, s

    def measure(self, keys, rng):
        keys = np.asarray(keys)
        n = self.n
        phys = [self.physical(int(k)) for k in keys.tolist()]
        pk = np.array([p[0] for p in phys], dtype=np.int64 if n <= 31 else object)
        ps = np.array([p[1] for p in phys])
        bits = self.base.measure(pk, rng)
        par = pl.popcount(bits & pl.support(pk, n)) & 1
        out = par ^ (ps < 0).astype(np.int64)                  # 1 means outcome -1
        sup = pl.support(keys, n)
        low = sup & -sup if keys.dtype != object else np.array([s & -s for s in sup], dtype=object)
        return np.where(out.astype(bool), low, 0).astype(np.int64 if n <= 31 else object)

    def to_physical(self, T):
        """The physical stabilizer state C|z_L> of a logical tableau."""
        from qiskit.quantum_info import StabilizerState
        return StabTableau.from_qiskit(StabilizerState(T.to_qiskit().clifford).evolve(self.C))
