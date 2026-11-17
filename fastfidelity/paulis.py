"""Symplectic representation of n-qubit Pauli strings.

Convention (shared by the whole package)
----------------------------------------
A Hermitian Pauli string without sign is stored as an integer key

    k = (x << n) | z ,   W_k = i^{x.z} X^x Z^z ,

where bit j of ``x`` and ``z`` refers to qubit j (little endian, as in Qiskit).
``W_k`` is the tensor product of the letters I, X, Y, Z (Y = iXZ), so it is
Hermitian and has eigenvalues +-1.  Keys are Python ints for any n; for n <= 31
the vectorised helpers also accept ``numpy.int64`` arrays, which is the fast
path used in the simulations.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "popcount", "split", "join", "mul", "mul_phase", "symp", "support",
    "weight", "to_label", "from_label", "to_qiskit", "from_qiskit",
    "is_substring", "letters_on",
]


def popcount(v):
    """Number of set bits; works for ints, int64 arrays and object arrays."""
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return np.fromiter((int(t).bit_count() for t in v.ravel()), dtype=np.int64,
                               count=v.size).reshape(v.shape)
        return np.bitwise_count(v).astype(np.int64)
    return int(v).bit_count()


def _mask(n):
    return (1 << n) - 1


def split(k, n):
    """Return (x, z) parts of a key (works element-wise on arrays)."""
    m = _mask(n)
    if isinstance(k, np.ndarray) and k.dtype != object:
        return k >> n, k & m
    return k >> n, k & m


def join(x, z, n):
    return (x << n) | z


def mul_phase(k1, k2, n):
    """Exponent e (mod 4) such that W_k1 W_k2 = i^e W_{k1 ^ k2}."""
    x1, z1 = split(k1, n)
    x2, z2 = split(k2, n)
    x, z = x1 ^ x2, z1 ^ z2
    e = popcount(x1 & z1) + popcount(x2 & z2) + 2 * popcount(z1 & x2) - popcount(x & z)
    return e % 4


def mul(k1, k2, n):
    """W_k1 W_k2 = i^e W_k.  Returns (k, e)."""
    return k1 ^ k2, mul_phase(k1, k2, n)


def symp(k1, k2, n):
    """Symplectic product: 1 if W_k1 and W_k2 anticommute, 0 otherwise."""
    x1, z1 = split(k1, n)
    x2, z2 = split(k2, n)
    return (popcount(x1 & z2) + popcount(z1 & x2)) & 1


def support(k, n):
    """Bit mask of the qubits on which W_k acts non-trivially."""
    x, z = split(k, n)
    return x | z


def weight(k, n):
    return popcount(support(k, n))


def is_substring(q, p, n):
    """True if Q is obtained from P by replacing some letters with I (Q <= P)."""
    sq = support(q, n)
    xq, zq = split(q, n)
    xp, zp = split(p, n)
    return ((xp & sq) == xq) & ((zp & sq) == zq)


def letters_on(k, n):
    """List of (qubit, letter) for the non-identity letters of W_k."""
    x, z = split(int(k), n)
    out = []
    for j in range(n):
        a, b = (x >> j) & 1, (z >> j) & 1
        if a or b:
            out.append((j, "Y" if a and b else ("X" if a else "Z")))
    return out


def to_label(k, n):
    """Qiskit-ordered label (qubit 0 is the rightmost character)."""
    x, z = split(int(k), n)
    chars = []
    for j in range(n):
        a, b = (x >> j) & 1, (z >> j) & 1
        chars.append("I" if not (a or b) else ("Y" if a and b else ("X" if a else "Z")))
    return "".join(reversed(chars))


def from_label(label):
    """Inverse of :func:`to_label`; returns (key, n)."""
    n = len(label)
    x = z = 0
    for j, c in enumerate(reversed(label)):
        if c in "XY":
            x |= 1 << j
        if c in "ZY":
            z |= 1 << j
    return join(x, z, n), n


def to_qiskit(k, n, sign=1):
    """Return ``sign * W_k`` as a ``qiskit.quantum_info.Pauli``."""
    from qiskit.quantum_info import Pauli
    lab = to_label(k, n)
    return Pauli(("-" if sign < 0 else "") + lab)


def from_qiskit(P):
    """Return (key, sign) for a Hermitian ``qiskit.quantum_info.Pauli``."""
    n = P.num_qubits
    x = sum(1 << j for j in range(n) if P.x[j])
    z = sum(1 << j for j in range(n) if P.z[j])
    lab = P.to_label()
    if lab.startswith("-i") or lab.startswith("i"):
        raise ValueError("non-Hermitian Pauli")
    sign = -1 if lab.startswith("-") else 1
    return join(x, z, n), sign
