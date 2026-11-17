"""Candidate stabilizer states from k-local data (local restart).

The Pauli store with substrings keeps, for every string of weight <= k, the
number of outcomes m_Q and their mean o_Q.  If the optimal stabilizer state has
generators of weight <= k, they have large |<Q>|.  We order the strings by the
lower bound |o_Q| - 3/sqrt(m_Q) (accepted if > 0.2) and accept them greedily when they commute with,
and are independent of, the accepted ones, with the sign of o_Q.  This
maximizes greedily the local cost C_L = 1/2 + sum_i s_i <g_i> / (2n) over
stabilizer states with generators of weight <= k.
"""
from __future__ import annotations

import numpy as np

from . import paulis as pl
from .f2 import Basis
from .tableau import StabTableau

__all__ = ["local_candidate"]


def local_candidate(keys, cnt, sm, n, score_min=0.2, nsigma=3.0):
    """Return a StabTableau, or None if fewer than n generators are found."""
    keys = np.asarray(keys, dtype=np.int64)
    if keys.size == 0:
        return None
    m = np.asarray(cnt, dtype=float)
    o = np.asarray(sm, dtype=float) / np.maximum(m, 1)
    score = np.abs(o) - nsigma / np.sqrt(np.maximum(m, 1))
    order = np.argsort(-score)
    gens, signs, basis = [], [], Basis()
    for i in order:
        if score[i] < score_min:
            break
        k = int(keys[i])
        if all(pl.symp(k, g, n) == 0 for g in gens) and basis.add(k):
            gens.append(k)
            signs.append(1 if o[i] > 0 else -1)
            if len(gens) == n:
                return StabTableau(n, gens, signs)
    return None
