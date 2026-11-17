"""Consistency of the three measurement backends."""
import numpy as np
from qiskit.quantum_info import Statevector, random_clifford

from fastfidelity import paulis as pl
from fastfidelity.tableau import StabTableau
from fastfidelity.backends import SpectrumBackend, StabilizerMixtureBackend, QiskitSamplerBackend


def _hist(b, w):
    return np.bincount(np.asarray(b, dtype=np.int64), minlength=2**w) / len(b)


def test_mixture_backend_matches_dense():
    rng = np.random.default_rng(0)
    n = 4
    Ts = [StabTableau.random(n, rng) for _ in range(3)]
    w = [0.5, 0.3, 0.2]
    M = StabilizerMixtureBackend(Ts, w)
    rho = sum(wi * np.outer(T.statevector(), T.statevector().conj()) for T, wi in zip(Ts, w))
    D = SpectrumBackend(rho)
    assert np.allclose(M.expectation(np.arange(4**n)), D.E)
    for lab in ["XZYX", "ZZII", "YYYY"]:
        P = pl.from_label(lab)[0]
        h1 = _hist(M.measure(np.full(30000, P), rng), n)
        h2 = _hist(D.measure(np.full(30000, P), rng), n)
        assert np.abs(h1 - h2).max() < 0.02


def test_qiskit_samplers_match_dense():
    rng = np.random.default_rng(1)
    qc = random_clifford(3, seed=5).to_circuit()
    qc.t(0)
    qc.h(1)
    qc.t(1)
    D = SpectrumBackend(Statevector(qc))
    for B in [QiskitSamplerBackend(qc, sampler="statevector"),
              QiskitSamplerBackend(qc, sampler="aer", method="matrix_product_state")]:
        for lab in ["XYZ", "ZIX"]:
            P = pl.from_label(lab)[0]
            h1 = _hist(B.measure(np.full(20000, P), rng), 3)
            h2 = _hist(D.measure(np.full(20000, P), rng), 3)
            assert np.abs(h1 - h2).max() < 0.02


def test_tableau_qiskit_roundtrip():
    rng = np.random.default_rng(2)
    for n in range(1, 6):
        c = random_clifford(n, seed=int(rng.integers(1e6)))
        T = StabTableau.from_qiskit(c)
        T.check()
        v = Statevector(c.to_circuit()).data
        assert abs(abs(np.vdot(v, T.statevector())) - 1) < 1e-10
        v2 = Statevector(T.to_qiskit().clifford.to_circuit()).data
        assert abs(abs(np.vdot(v2, v)) - 1) < 1e-10
        assert StabTableau(n, T.g, T.s) == T
