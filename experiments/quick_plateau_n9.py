"""Quick test (n = 9): is the failure for many qubits a barren plateau?
Noise-free greedy searches over the neighbour graph, with the global fidelity and
with a local cost that uses only group elements of weight <= k, compared with
the protocol with shots.  T-doped states (t = 2) with a deep random Clifford or a
depth-2 brickwork of neighbouring two-qubit Cliffords."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Statevector
from fastfidelity import StabTableau, SpectrumBackend, self_guided_estimation
from fastfidelity import paulis as pl

n, t, R = 9, 2, 10
FST = np.cos(np.pi / 8) ** (2 * t)
xs = np.arange(1 << n)


def apply(k, v):
    x, z = pl.split(k, n)
    w = v * (-1.0) ** pl.popcount(z & xs)
    return (1j ** pl.popcount(x & z)) * w[xs ^ x]


def overlaps(T, psi):
    """a_x = <phi_x|psi> for the whole stabilizer basis of T (Gray code)."""
    v = T.statevector()
    a = np.zeros(1 << n, complex)
    cur, x = v.copy(), 0
    a[0] = np.vdot(cur, psi)
    for g in range(1, 1 << n):
        i = (g & -g).bit_length() - 1              # flip bit i of the Gray code
        cur = apply(T.d[i], cur)
        x ^= 1 << i
        a[x] = np.vdot(cur, psi)
    return a


def greedy_global(T, psi, max_it=500):
    for it in range(max_it):
        a = overlaps(T, psi)
        p = np.abs(a) ** 2
        best = int(np.argmax(p))
        if best != 0 and p[best] > p[0] + 1e-12:
            T = T.flip(best)
            continue
        # f(nu_{x,m}) = |a_0 + i^{-m} a_x|^2 / 2 for every x and m
        f = np.abs(a[0] + np.outer([1, -1j, -1, 1j], a)) ** 2 / 2
        f[:, 0] = -1
        m, x = np.unravel_index(np.argmax(f), f.shape)
        if f[m, x] > p[0] + 1e-12:
            T = T.neighbor(int(x), int(m))
        else:
            return T, it
    return T, max_it


def local_cost(T, E, k):
    keys, signs, _ = T.enumerate()
    w = pl.weight(keys, n)
    sel = (w <= k) & (keys != 0)
    return float(np.mean(signs[sel] * E[keys[sel]])) if sel.any() else 0.0


def greedy_local(T, E, k, max_it=300):
    c = local_cost(T, E, k)
    for it in range(max_it):
        best, bc = None, c
        for x in range(1, 1 << n):
            for cand in [T.flip(x)] + [T.neighbor(x, m) for m in range(4)]:
                v = local_cost(cand, E, k)
                if v > bc + 1e-12:
                    best, bc = cand, v
        if best is None:
            return T, it
        T, c = best, bc
    return T, max_it


def circuit(kind, rng):
    qc = QuantumCircuit(n)
    for j in range(t):
        qc.h(j); qc.t(j)
    if kind == "deep random":
        qc.compose(random_clifford(n, seed=int(rng.integers(2**31))).to_circuit(), inplace=True)
    else:
        for layer in range(2):
            for q in range(layer % 2, n - 1, 2):
                qc.compose(random_clifford(2, seed=int(rng.integers(2**31))).to_circuit(), [q, q + 1], inplace=True)
    return Statevector(qc).data


rows = []
for kind in ["deep random", "local depth 2"]:
    rng = np.random.default_rng(42)
    for i in range(R):
        psi = circuit(kind, rng)
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng)
        F = lambda T: abs(np.vdot(T.statevector(), psi)) ** 2
        res = dict(kind=kind, state=i, F_start=F(start))
        t0 = time.time(); Tg, itg = greedy_global(start, psi); res["exact global"] = F(Tg) / FST
        for k in (2, 3):
            Tl, _ = greedy_local(start, B.E, k)
            Tl2, _ = greedy_global(Tl, psi)
            res[f"local k={k}"] = F(Tl) / FST
            res[f"local k={k} + global"] = F(Tl2) / FST
        r = self_guided_estimation(B, start, store="store", substrings=True, seed=int(rng.integers(2**31)),
                                   final=False, keep_path=False)
        res["protocol (shots)"] = F(r.state) / FST
        rows.append(res)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)

import pandas as pd
d = pd.DataFrame(rows)
d.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "quick_plateau_n9.csv"), index=False)
cols = ["exact global", "local k=2", "local k=2 + global", "local k=3", "local k=3 + global", "protocol (shots)"]
print("\nfraction that reached the optimum (F/F_STAB > 1 - 1e-6):")
print(d.groupby("kind")[cols].agg(lambda s: np.mean(s > 1 - 1e-6)).round(2).to_string())
