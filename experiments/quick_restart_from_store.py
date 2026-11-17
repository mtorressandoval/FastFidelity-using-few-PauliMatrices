"""Quick test: restart the search from a candidate built with the k-local data
already in the Pauli store (substrings), without any additional measurement.
T-doped states (t = 2), n = 10, random start, N = 1000."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Statevector
from fastfidelity import StabTableau, SpectrumBackend, self_guided_estimation
from fastfidelity.f2 import Basis
from fastfidelity import paulis as pl

n, t, K, R = int(sys.argv[1]), 2, 4, int(sys.argv[2])
FST = np.cos(np.pi / 8) ** (2 * t)


def clifford(kind, rng):
    if kind == "deep random":
        return random_clifford(n, seed=int(rng.integers(2**31))).to_circuit()
    qc = QuantumCircuit(n)
    for layer in range(2):
        if kind == "neighbors":
            pairs = [(q, q + 1) for q in range(layer % 2, n - 1, 2)]
        else:
            p = rng.permutation(n)
            pairs = [(int(p[2 * i]), int(p[2 * i + 1])) for i in range(n // 2)]
        for a, b in pairs:
            qc.compose(random_clifford(2, seed=int(rng.integers(2**31))).to_circuit(), [a, b], inplace=True)
    return qc


def candidate_from_store(store, score_min=0.45):
    """Greedy commuting, independent k-local generators from the stored data."""
    cnt, sm = store.data.cnt, store.data.sm
    keys = np.nonzero(cnt)[0].astype(np.int64)
    keys = keys[(pl.weight(keys, n) <= K) & (keys != 0)]
    m = cnt[keys]
    o = sm[keys] / m
    score = np.abs(o) - 2 / np.sqrt(m)
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
                break
    return gens, signs


rng = np.random.default_rng(11)
for kind in ["neighbors", "random pairs", "deep random"]:
    for i in range(R):
        qc = QuantumCircuit(n)
        for j in range(t):
            qc.h(j); qc.t(j)
        qc.compose(clifford(kind, rng), inplace=True)
        B = SpectrumBackend(Statevector(qc).data)
        start = StabTableau.random(n, rng)
        r1 = self_guided_estimation(B, start, store="store", substrings=True, seed=int(rng.integers(2**31)),
                                    final=False, keep_path=False)
        F1 = r1.state.fidelity_from_spectrum(B.E) / FST
        gens, signs = candidate_from_store(r1.store)
        line = f"{kind:12s} {i}: first search F/F_STAB={F1:.3f} shots={r1.search_shots}; generators from store {len(gens)}/{n}"
        if len(gens) == n and F1 < 1 - 1e-6:
            z0 = StabTableau(n, gens, signs)
            r2 = self_guided_estimation(B, z0, store="store", substrings=True, seed=int(rng.integers(2**31)),
                                        final=False, keep_path=False)
            line += (f"; candidate F/F_STAB={z0.fidelity_from_spectrum(B.E) / FST:.3f}; after restart "
                     f"F/F_STAB={r2.state.fidelity_from_spectrum(B.E) / FST:.3f}, extra shots={r2.search_shots}")
        print(line, flush=True)
