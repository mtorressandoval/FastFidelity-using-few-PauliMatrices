"""Quick test (n = 9): k-local mean of the stabilizer projector as a proxy cost.
f_k(z) = 2^-n sum_{P in S} s_P m_k(P), with m_k(P) = <P> if |P| <= k and the mean of
<Q> over the k-qubit substrings Q of P otherwise.  Noise-free values."""
import sys, os, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Statevector
from fastfidelity import StabTableau, SpectrumBackend, self_guided_estimation
from fastfidelity import paulis as pl

n, t, R = 9, 2, 10
FST = np.cos(np.pi / 8) ** (2 * t)
keys = np.arange(4 ** n, dtype=np.int64)
xk, zk = pl.split(keys, n)
supp = xk | zk
wt = pl.popcount(supp)


def mk_table(E, k):
    """m_k(P) for every Pauli string P."""
    m = E.copy()
    acc = np.zeros(4 ** n)
    for M in itertools.combinations(range(n), k):
        mask = sum(1 << j for j in M)
        inside = (supp & mask) == mask
        sub = ((xk & mask) << n) | (zk & mask)
        acc[inside] += E[sub[inside]]
    from math import comb
    big = wt > k
    m[big] = acc[big] / np.array([comb(int(w), k) for w in wt[big]])
    return m


def fk(T, m):
    kk, ss, _ = T.enumerate()
    return float(np.mean(ss * m[kk]))


def greedy(T, m, max_it=200):
    c = fk(T, m)
    for it in range(max_it):
        best, bc = None, c
        for x in range(1, 1 << n):
            for cand in [T.flip(x)] + [T.neighbor(x, mm) for mm in range(4)]:
                v = fk(cand, m)
                if v > bc + 1e-12:
                    best, bc = cand, v
        if best is None:
            break
        T, c = best, bc
    return T


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


rows, conc = [], []
for kind in ["deep random", "local depth 2"]:
    rng = np.random.default_rng(42)                      # same states as quick_plateau_n9.py
    for i in range(R):
        psi = circuit(kind, rng)
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng)
        seed = int(rng.integers(2**31))
        F = lambda T: abs(np.vdot(T.statevector(), psi)) ** 2
        tabs = {k: mk_table(B.E, k) for k in (2, 3)}
        if i < 3:                                        # concentration over random stabilizer states
            zs = [StabTableau.random(n, rng) for _ in range(100)]
            conc.append(dict(kind=kind, std_f=np.std([F(z) for z in zs]),
                             std_f2=np.std([fk(z, tabs[2]) for z in zs]),
                             std_f3=np.std([fk(z, tabs[3]) for z in zs]),
                             f2_opt_rel=None))
        res = dict(kind=kind, state=i, F_start=F(start))
        for k in (2, 3):
            Tk = greedy(start, tabs[k])
            res[f"F after f_{k}"] = F(Tk) / FST
            r = self_guided_estimation(B, Tk, store="store", substrings=True, seed=seed, final=False, keep_path=False)
            res[f"f_{k} + protocol"] = F(r.state) / FST
        rows.append(res)
        print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in res.items()}, flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "quick_klocal_n9.csv"), index=False)
print("\nstd over 100 random stabilizer states:")
print(pd.DataFrame(conc).groupby("kind")[["std_f", "std_f2", "std_f3"]].mean().to_string())
print("\nmedian F/F_STAB after maximizing f_k, and fraction reaching the optimum with the protocol (shots) from there:")
print(d.groupby("kind")[["F after f_2", "F after f_3"]].median().round(3).to_string())
print(d.groupby("kind")[["f_2 + protocol", "f_3 + protocol"]].agg(lambda s: np.mean(s > 1 - 1e-6)).round(2).to_string())
