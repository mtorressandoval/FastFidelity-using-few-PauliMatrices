"""The legacy tomography code (tomography_sgqt/) still works and is unchanged:
Mean_Direct_Fidelity gives the same result with the same seeds."""
import os
import random
import sys

import numpy as np

LEGACY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tomography_sgqt")
sys.path.insert(0, LEGACY)


def test_mean_direct_fidelity_reproducible():
    from qiskit.quantum_info import Statevector, random_clifford
    from qiskit_aer.primitives import Estimator
    from tomography import Mean_Direct_Fidelity
    n = 3
    circ = random_clifford(n, seed=11).to_circuit()
    target = np.array(Statevector(circ))
    est = Estimator(backend_options={"shots": 1000, "method": "stabilizer", "seed_simulator": 5},
                    transpile_options={"optimization_level": 0})
    vals = []
    for _ in range(2):
        random.seed(123)
        M = Mean_Direct_Fidelity(n)
        vals.append(M.MeanFidelity(1, 30, target, circ, est, shots=1000))
    assert vals[0] == vals[1]
    assert abs(vals[0] - 1.0) < 1e-9          # stabilizer target = prepared state


def test_self_guided_tomography_runs():
    from tomography import SelfGuidedTomography
    rng = np.random.default_rng(0)
    target = rng.normal(size=4) + 1j * rng.normal(size=4)
    target /= np.linalg.norm(target)
    infid = lambda x: 1 - abs(np.vdot(target, x / np.linalg.norm(x)))**2
    guess = np.ones(4, complex) / 2
    SelfGuidedTomography(infid, guess, num_iter=50)
