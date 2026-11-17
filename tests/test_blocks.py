"""Block warm start: product tableaux, block partition, subsystem and frame backends."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Statevector, partial_trace

from fastfidelity import paulis as pl
from fastfidelity.tableau import StabTableau
from fastfidelity.backends import SpectrumBackend
from fastfidelity.blocks import (product_tableau, block_partition, SubsystemBackend, FrameBackend)


def test_partition():
    for n in range(2, 40):
        blocks, k = block_partition(n)
        assert len(blocks) == 2**k and max(map(len, blocks)) <= 8
        assert sorted(q for b in blocks for q in b) == list(range(n))
        assert k == 0 or max(map(len, block_partition(n, 8)[0])) > 4


def test_product_tableau():
    rng = np.random.default_rng(0)
    A, B = StabTableau.random(3, rng), StabTableau.random(2, rng)
    T, q = product_tableau(A, [0, 2, 4], B, [1, 3])
    T.check()
    full = np.zeros(32, complex)
    va, vb = A.statevector(), B.statevector()
    for ia in range(8):
        for ib in range(4):
            x = sum(((ia >> i) & 1) << p for i, p in enumerate([0, 2, 4]))
            x |= sum(((ib >> i) & 1) << p for i, p in enumerate([1, 3]))
            full[x] = va[ia] * vb[ib]
    assert abs(abs(np.vdot(full, T.statevector())) - 1) < 1e-10


def test_subsystem_and_frame_backends():
    rng = np.random.default_rng(1)
    n = 5
    C = random_clifford(n, seed=3)
    qc = QuantumCircuit(n)
    qc.h(0)
    qc.t(0)
    qc.compose(C.to_circuit(), inplace=True)
    psi = Statevector(qc)
    B = SpectrumBackend(psi.data)
    S = SubsystemBackend(B, [1, 3, 4])
    red = SpectrumBackend(partial_trace(psi, [0, 2]).data)
    F = FrameBackend(B, C)
    qm = QuantumCircuit(n)
    qm.h(0)
    qm.t(0)
    L = SpectrumBackend(Statevector(qm).data)
    for _ in range(4):
        P = int(rng.integers(1, 4**3))
        b = S.measure(np.full(20000, P), rng)
        assert abs(np.mean(1 - 2 * (pl.popcount(b & pl.support(P, 3)) & 1)) - red.E[P]) < 0.035
        P = int(rng.integers(1, 4**n))
        b = F.measure(np.full(20000, P), rng)
        assert abs(np.mean(1 - 2 * (pl.popcount(b & pl.support(P, n)) & 1)) - L.E[P]) < 0.035
    zL = StabTableau.random(n, rng)
    assert abs(F.to_physical(zL).fidelity_from_spectrum(B.E) - zL.fidelity_from_spectrum(L.E)) < 1e-10
