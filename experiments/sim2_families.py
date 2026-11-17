"""Simulation 2: the best estimator (DFE with store and substrings) on three
families of states, starting from a random stabilizer state, n = 3..8.

(a) stabilizer state + random perturbation: sqrt(tau) phi + sqrt(1-tau) chi,
    tau in {0.86, 0.9, 0.95} > cos^2(pi/8), so that F_STAB = tau;
(b) T-doped states: C (T|+>)^{(x)t} (x) |0>^{(x)(n-t)} with a random Clifford C,
    t in {1, 2, 3}; F_STAB = F_STAB((T|+>)^{(x)t}) = cos^{2t}(pi/8) (Clifford
    invariance, a stabilizer factor does not change F_STAB, and exact
    enumeration for t <= 4);
(c) depolarized stabilizer states: (1-p)|phi><phi| + p I/2^n, p in {0.1, 0.3, 0.5};
    F_STAB = 1 - p + p/2^n.
"""
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Statevector
from common import save, seeds, SpectrumBackend, StabTableau, self_guided_estimation
from fastfidelity.reference import near_stabilizer, fstab_exact

NS, R, MAX_IT = [3, 4, 5, 6, 7, 8], 15, 200
Tplus = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
FT = {}
for t in (1, 2, 3):
    v = np.array([1.0 + 0j])
    for _ in range(t):
        v = np.kron(v, Tplus)
    FT[t] = fstab_exact(v)


def t_doped(n, t, rng):
    qc = QuantumCircuit(n)
    for j in range(t):
        qc.h(j)
        qc.t(j)
    qc.compose(random_clifford(n, seed=int(rng.integers(2**31))).to_circuit(), inplace=True)
    return Statevector(qc).data


rows = []
for fam, pars in [("perturbed", [0.86, 0.9, 0.95]), ("T-doped", [1, 2, 3]), ("depolarized", [0.1, 0.3, 0.5])]:
    for par in pars:
        for n in NS:
            for i, sd in enumerate(seeds(int(9000 + 1000 * ["perturbed", "T-doped", "depolarized"].index(fam)
                                             + 100 * pars.index(par) + n), R)):
                rng = np.random.default_rng(sd)
                if fam == "perturbed":
                    st, _ = near_stabilizer(n, par, rng)
                    Fst = par
                elif fam == "T-doped":
                    st, Fst = t_doped(n, par, rng), FT[par]
                else:
                    phi = StabTableau.random(n, rng).statevector()
                    st = (1 - par) * np.outer(phi, phi.conj()) + par * np.eye(2**n) / 2**n
                    Fst = 1 - par + par / 2**n
                B = SpectrumBackend(st)
                start = StabTableau.random(n, rng)
                r = self_guided_estimation(B, start, store="store", substrings=True,
                                           seed=int(rng.integers(2**31)), max_iter=MAX_IT, keep_path=False)
                F = r.state.fidelity_from_spectrum(B.E)
                rows.append(dict(family=fam, par=par, n=n, state=i, F_STAB=Fst, F_out=F, ratio=F / Fst,
                                 optimal=F >= Fst - 1e-9, F_hat=r.F_hat, lower=r.lower_bound,
                                 certified=r.certified, iterations=r.iterations,
                                 hit_limit=r.iterations >= MAX_IT and r.log[-1]["action"] != "stop",
                                 search_shots=r.search_shots))
            print(fam, par, n, flush=True)
        save(rows, "sim2_families.csv")
