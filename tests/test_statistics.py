"""Statistical tests 5-7: unbiasedness, variance and exact-vs-affine store.

A fixed (non-adaptive) sequence of overlapping queries is repeated R times
with fresh randomness.  Tolerance: 3 standard errors.
"""
import os
import numpy as np
import pytest

from fastfidelity.tableau import StabTableau
from fastfidelity.backends import SpectrumBackend
from fastfidelity.dfe import FreshDFE, StoreDFE
from fastfidelity.reference import random_rank_k

R = int(os.environ.get("FF_STAT_REPS", 10_000))
N = 8


def _setup(n=3, seed=7):
    rng = np.random.default_rng(seed)
    rho = random_rank_k(n, 2, rng)
    B = SpectrumBackend(rho)
    z = StabTableau.random(n, rng)
    seq = [(z, None), (z.neighbor(3, 0), None), (z.neighbor(3, 1), None), (z.flip(1), None),
           (z.neighbor(5, 2), None), (z, 2), (z.neighbor(3, 0).neighbor(6, 3), None), (z, None)]
    exact = []
    for T, m in seq:
        k, s, u = T.enumerate()
        if m is not None:
            sel = (u >> m) == 0
            k, s = k[sel], s[sel]
        exact.append(float(np.mean(s * B.E[k])))
    return B, seq, np.array(exact)


def _run(kind, sub, B, seq, reps, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((reps, len(seq)))
    for r in range(reps):
        if kind == "fresh":
            est = FreshDFE(B, rng)
        elif kind == "store":
            est = StoreDFE(B, rng, substrings=sub)
        else:                                    # scalable storage (dicts, no dense arrays)
            est = StoreDFE(B, rng, substrings=sub, dense_max=0)
        for j, (T, m) in enumerate(seq):
            out[r, j] = est.estimate(T, N, m).value
    return out


@pytest.fixture(scope="module")
def runs():
    B, seq, exact = _setup()
    res = {}
    for i, (kind, sub) in enumerate([("fresh", False), ("store", False), ("scalable", False),
                                     ("store", True), ("scalable", True)]):
        res[(kind, sub)] = _run(kind, sub, B, seq, R, 100 + i)
    return exact, res


def test_5_unbiased(runs):
    exact, res = runs
    for key, v in res.items():
        se = v.std(axis=0, ddof=1) / np.sqrt(v.shape[0])
        z = np.abs(v.mean(axis=0) - exact) / se
        assert np.all(z < 3), (key, z)


def test_6_variance_not_above_fresh(runs):
    _, res = runs
    vf = res[("fresh", False)].var(axis=0, ddof=1)
    for key, v in res.items():
        ratio = v.var(axis=0, ddof=1) / vf
        assert np.all(ratio <= 1.1), (key, ratio.round(3))


def test_7_dense_and_scalable_storage_agree(runs):
    """Same seed -> identical results; independent seeds -> same statistics."""
    B, seq, _ = _setup()
    for sub in (False, True):
        a = _run("store", sub, B, seq, 50, 999)
        b = _run("scalable", sub, B, seq, 50, 999)
        assert np.allclose(a, b)
    _, res = runs
    for sub in (False, True):
        a, b = res[("store", sub)], res[("scalable", sub)]
        se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
        assert np.all(np.abs(a.mean(axis=0) - b.mean(axis=0)) < 3 * se)


def test_empty_store_reproduces_fresh_dfe():
    """Rule 2: the path without store is unchanged (same seed, same results)."""
    B, seq, _ = _setup()
    for T, m in seq:
        e1 = FreshDFE(B, np.random.default_rng(3)).estimate(T, 64, m)
        e2 = StoreDFE(B, np.random.default_rng(3)).estimate(T, 64, m)
        assert e1.value == e2.value and np.array_equal(e1.u, e2.u)
