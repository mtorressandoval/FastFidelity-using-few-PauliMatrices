"""Linear algebra over F_2 with vectors stored as Python ints.

Used for the scalable Pauli store (atoms are affine subspaces of F_2^{2n}),
for destabilizer computations and for the stabilizer-state samplers.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Basis", "Affine", "random_bits", "rank"]


class Basis:
    """Echelon basis of a subspace of F_2^L (highest set bit is the pivot)."""

    __slots__ = ("piv", "vectors")

    def __init__(self, vectors=()):
        self.piv = {}          # pivot bit -> reduced vector
        self.vectors = []      # independent input vectors, in insertion order
        for v in vectors:
            self.add(v)

    @property
    def dim(self):
        return len(self.vectors)

    def reduce(self, v):
        v = int(v)
        while v:
            p = v.bit_length() - 1
            w = self.piv.get(p)
            if w is None:
                return v
            v ^= w
        return 0

    def add(self, v):
        r = self.reduce(v)
        if r:
            self.piv[r.bit_length() - 1] = r
            self.vectors.append(int(v))
            return True
        return False

    def contains(self, v):
        return self.reduce(v) == 0

    def copy(self):
        b = Basis()
        b.piv = dict(self.piv)
        b.vectors = list(self.vectors)
        return b

    def is_subspace_of(self, other: "Basis"):
        return all(other.contains(v) for v in self.vectors)

    def intersect(self, other: "Basis", L: int) -> "Basis":
        """Zassenhaus algorithm: basis of self ∩ other (vectors of length L)."""
        rows = [(v << L) | v for v in self.vectors] + [(v << L) for v in other.vectors]
        ech = Basis()
        for r in rows:
            ech.add(r)
        low = (1 << L) - 1
        return Basis([w & low for w in ech.piv.values() if (w >> L) == 0])


def rank(vectors):
    return Basis(vectors).dim


def random_bits(rng, size, nbits):
    """Array (object dtype for nbits > 62) of uniformly random nbits-bit ints."""
    if nbits == 0:
        return np.zeros(size, dtype=np.int64)
    if nbits <= 62:
        return rng.integers(0, 1 << nbits, size=size, dtype=np.int64)
    out = np.zeros(size, dtype=object)
    done = 0
    while done < nbits:
        b = min(62, nbits - done)
        part = rng.integers(0, 1 << b, size=size, dtype=np.int64)
        out = out + np.array([int(p) << done for p in part], dtype=object)
        done += b
    return out


class Affine:
    """Affine subspace  offset + span(direction)  of F_2^L."""

    __slots__ = ("offset", "dirn", "L", "_canon")

    def __init__(self, offset, direction: Basis, L: int):
        self.dirn = direction
        self.L = L
        self.offset = direction.reduce(offset)   # canonical representative
        self._canon = None

    @property
    def dim(self):
        return self.dirn.dim

    @property
    def size(self):
        return 1 << self.dim

    def contains(self, v):
        return self.dirn.reduce(int(v) ^ self.offset) == 0

    def contains_many(self, keys):
        return np.array([self.contains(k) for k in keys], dtype=bool)

    def is_subset_of(self, other: "Affine"):
        return other.contains(self.offset) and self.dirn.is_subspace_of(other.dirn)

    def canon(self):
        """Hashable canonical form."""
        if self._canon is None:
            self._canon = (self.offset, tuple(sorted(self.dirn.piv.values())))
        return self._canon

    def __eq__(self, other):
        return isinstance(other, Affine) and self.canon() == other.canon()

    def __hash__(self):
        return hash(self.canon())

    def intersect(self, other: "Affine"):
        """Return self ∩ other as an Affine, or None if empty."""
        L = self.L
        rows = Basis()
        ech_rows = []
        for v in self.dirn.vectors:
            ech_rows.append((v << L) | v)
        for v in other.dirn.vectors:
            ech_rows.append(v << L)
        for r in ech_rows:
            rows.add(r)
        t = (self.offset ^ other.offset) << L
        t = rows.reduce(t)
        if (t >> L) != 0:
            return None
        v1 = t & ((1 << L) - 1)
        point = self.offset ^ v1
        low = (1 << L) - 1
        inter = Basis([w & low for w in rows.piv.values() if (w >> L) == 0])
        return Affine(point, inter, L)

    def split_by(self, sub: "Affine"):
        """Partition self into the 2^r cosets of sub.dirn, where sub ⊆ self.

        The first coset returned is ``sub`` itself.
        """
        comp = []
        b = sub.dirn.copy()
        for v in self.dirn.vectors:
            if b.add(v):
                comp.append(v)
        r = len(comp)
        out = []
        for eps in range(1 << r):
            off = sub.offset
            for t in range(r):
                if (eps >> t) & 1:
                    off ^= comp[t]
            out.append(Affine(off, sub.dirn, self.L))
        return out

    def sample(self, rng, size):
        """Uniform samples (Python ints in an object array, or int64 if L <= 62)."""
        d = self.dim
        vecs = self.dirn.vectors
        coef = random_bits(rng, size, d)
        use_int = self.L <= 62
        out = np.full(size, self.offset, dtype=np.int64 if use_int else object)
        for t, v in enumerate(vecs):
            bit = (coef >> t) & 1
            if use_int:
                out ^= np.where(bit.astype(bool), np.int64(v), np.int64(0))
            else:
                out = np.array([o ^ (v if b else 0) for o, b in zip(out, bit)], dtype=object)
        return out

    def __repr__(self):
        return f"Affine(dim={self.dim}, offset={self.offset:#x})"
