"""Walsh-Hadamard transforms used to obtain all p_x from one DFE run."""
from __future__ import annotations

import numpy as np

__all__ = ["fwht", "fwht_axis"]


def fwht(v):
    """Unnormalised Walsh-Hadamard transform  w[b] = sum_x v[x] (-1)^{b.x}."""
    v = np.asarray(v, dtype=float)
    n = int(np.log2(v.shape[0]))
    t = v.reshape([2] * n + list(v.shape[1:])) if n else v.copy()
    for ax in range(n):
        a = np.take(t, 0, axis=ax)
        b = np.take(t, 1, axis=ax)
        t = np.stack([a + b, a - b], axis=ax)
    return t.reshape(v.shape)


def fwht_axis(V, axis=-1):
    """Walsh-Hadamard transform along one axis of a 2-D array (in place copy)."""
    V = np.array(V, dtype=complex if np.iscomplexobj(V) else float)
    V = np.moveaxis(V, axis, -1)
    N = V.shape[-1]
    h = 1
    while h < N:
        V = V.reshape(V.shape[:-1] + (N // (2 * h), 2, h))
        a = V[..., 0, :].copy()
        b = V[..., 1, :]
        V[..., 0, :] = a + b
        V[..., 1, :] = a - b
        V = V.reshape(V.shape[:-3] + (N,))
        h *= 2
    return np.moveaxis(V, -1, axis)
