"""Quick test: k-local objective from classical shadows of random Pauli measurements.
Every shot measures a random full-weight Pauli string; the outcome bits give
unbiased estimates of all Paulis of weight <= k (3^|P| reweighting).  The
k-local Paulis with large |<P>| are combined greedily into commuting, independent
generators, which maximizes the local cost C_L = 1/2 + sum_i s_i <g_i> / (2n)
over stabilizer states with k-local generators.  T-doped states (t = 2)."""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Statevector, Clifford
from fastfidelity import StabTableau, SpectrumBackend
from fastfidelity.backends import TDopedBackend
from fastfidelity.f2 import Basis
from fastfidelity import paulis as pl

t, K, M, THR = 2, 4, int(sys.argv[2]) if len(sys.argv) > 2 else 60000, 0.55
n = int(sys.argv[1])
FST = np.cos(np.pi / 8) ** (2 * t)


def clifford(kind, rng):
    if kind == "deep random":
        return random_clifford(n, seed=int(rng.integers(2**31))).to_circuit()
    qc = QuantumCircuit(n)
    for layer in range(2):
        if kind == "neighbors, depth 2":
            pairs = [(q, q + 1) for q in range(layer % 2, n - 1, 2)]
        else:
            p = rng.permutation(n)
            pairs = [(int(p[2 * i]), int(p[2 * i + 1])) for i in range(n // 2)]
        for a, b in pairs:
            qc.compose(random_clifford(2, seed=int(rng.integers(2**31))).to_circuit(), [a, b], inplace=True)
    return qc


def shadow_estimates(B, rng):
    letters = rng.integers(1, 4, size=(M, n))          # 1 = X, 2 = Z, 3 = Y
    x = ((letters == 1) | (letters == 3)).astype(np.int64)
    z = ((letters == 2) | (letters == 3)).astype(np.int64)
    w = 1 << np.arange(n, dtype=np.int64)
    keys = ((x @ w) << n) | (z @ w)
    bits = np.asarray(B.measure(keys, rng), dtype=np.int64)
    b = (bits[:, None] >> np.arange(n)) & 1
    est = {}
    for k in range(1, K + 1):
        for A in itertools.combinations(range(n), k):
            code = np.zeros(M, dtype=np.int64)
            for j, q in enumerate(A):
                code += letters[:, q] * 4**j
            sign = 1 - 2 * (b[:, list(A)].sum(axis=1) & 1)
            s = np.bincount(code, weights=sign, minlength=4**k)
            for c in np.nonzero(s)[0]:
                lets = [(c // 4**j) % 4 for j in range(k)]
                if 0 in lets:
                    continue
                X = sum(1 << q for q, l in zip(A, lets) if l in (1, 3))
                Z = sum(1 << q for q, l in zip(A, lets) if l in (2, 3))
                est[pl.join(X, Z, n)] = 3**k * s[c] / M
    return est


def local_generators(est):
    cand = sorted(((abs(v), k, v) for k, v in est.items() if abs(v) > THR), key=lambda t: (-t[0], pl.weight(t[1], n)))
    gens, signs, basis = [], [], Basis()
    for _, k, v in cand:
        if all(pl.symp(k, g, n) == 0 for g in gens) and basis.add(k):
            gens.append(k)
            signs.append(1 if v > 0 else -1)
            if len(gens) == n:
                break
    return gens, signs


rng = np.random.default_rng(7)
for kind in ["neighbors, depth 2", "random pairs, depth 2", "deep random"]:
    for i in range(3):
        Cq = clifford(kind, rng)
        if n <= 12:
            qc = QuantumCircuit(n)
            for j in range(t):
                qc.h(j); qc.t(j)
            qc.compose(Cq, inplace=True)
            B = SpectrumBackend(Statevector(qc).data)
            fid = lambda T: T.fidelity_from_spectrum(B.E)
        else:
            B = TDopedBackend(Clifford(Cq), t)
            fid = B.fidelity
        t0 = time.time()
        est = shadow_estimates(B, rng)
        gens, signs = local_generators(est)
        F = fid(StabTableau(n, gens, signs)) / FST if len(gens) == n else float("nan")
        print(f"n={n} {kind:22s} state {i}: generators found {len(gens)}/{n}, F/F_STAB = {F:.3f}, "
              f"shots = {M}, {time.time() - t0:.0f}s", flush=True)
