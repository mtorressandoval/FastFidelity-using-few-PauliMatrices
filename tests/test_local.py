"""Local restart: k-local statistics of the store and candidate construction."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Statevector

from fastfidelity import paulis as pl
from fastfidelity.tableau import StabTableau
from fastfidelity.backends import SpectrumBackend
from fastfidelity.dfe import StoreDFE
from fastfidelity.local import local_candidate
from fastfidelity.search import self_guided_estimation


def test_local_stats_equal_dense_substrings():
    """The chunked k-local statistics equal the dense substring counts for the same shots."""
    from fastfidelity.dfe import _PauliData
    rng = np.random.default_rng(0)
    n, k = 6, 3
    B = SpectrumBackend(Statevector(random_clifford(n, seed=1).to_circuit()).data)
    dense = _PauliData(n, substrings=True)
    local = _PauliData(n, substrings=False, dense_max=0, local_k=k)
    local.local.chunk = 50
    for _ in range(5):
        keys = rng.integers(1, 4**n, size=400).astype(np.int64)
        bits = B.measure(keys, rng)
        dense.add(keys, bits)
        local.add(keys, bits)
    kd, cd, sd = dense.local_table(k)
    kl, cl, sl = local.local_table(k)
    assert np.array_equal(kd, kl) and np.array_equal(cd, cl) and np.array_equal(sd, sl)


def test_local_candidate_recovers_shallow_state_and_restart_runs():
    rng = np.random.default_rng(3)
    n = 8
    qc = QuantumCircuit(n)
    for layer in range(2):
        for q in range(layer % 2, n - 1, 2):
            qc.compose(random_clifford(2, seed=int(rng.integers(1e6))).to_circuit(), [q, q + 1], inplace=True)
    B = SpectrumBackend(Statevector(qc).data)            # a stabilizer state with generators of weight <= 4
    E = B.E
    keys = np.array([q for q in range(1, 4**n) if pl.weight(q, n) <= 4], dtype=np.int64)
    T = local_candidate(keys, np.full(keys.size, 10**6), np.round(E[keys] * 10**6).astype(np.int64), n)
    assert T is not None and abs(T.fidelity_from_spectrum(E) - 1) < 1e-9
    r = self_guided_estimation(B, StabTableau.random(n, rng), store="store", substrings=True, seed=1,
                               local_restart=4, final=False)
    assert abs(r.state.fidelity_from_spectrum(E) - 1) < 1e-9
