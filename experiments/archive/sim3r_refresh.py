"""Simulation 3r: the runs of simulation 3 for n = 12 (t = 2 and p = 0.3) with the
store refreshed before each retry (refresh_on_retry=True); same seeds and starts."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, StabilizerState
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from common import save, seeds, StabTableau, self_guided_estimation
from fastfidelity.backends import TDopedBackend, DepolarizedStabilizerBackend

rows = []
n = 12
for fam, par, k in [("T-doped", 2, 1), ("depolarized", 0.3, 1)]:
    for i, sd in enumerate(seeds(int(12000 + 100 * n + (10 if fam == "T-doped" else 50) + 10 * k), 10)):
        rng = np.random.default_rng(sd)
        if fam == "T-doped":
            C = random_clifford(n, seed=int(rng.integers(2**31)))
            B = TDopedBackend(C, par)
            qc = QuantumCircuit(n)
            for j in range(par):
                qc.h(j)
            zopt = StabTableau.from_qiskit(StabilizerState(qc).evolve(C))
            Fst = np.cos(np.pi / 8) ** (2 * par)
        else:
            zopt = StabTableau.random(n, rng)
            B = DepolarizedStabilizerBackend(zopt, par)
            Fst = 1 - par + par / 2**n
        start = zopt.random_walk(3, rng)
        s = int(rng.integers(2**31))
        r = self_guided_estimation(B, start, store="store", seed=s, keep_path=False, refresh_on_retry=True)
        F = B.fidelity(r.state)
        rows.append(dict(n=n, family=fam, par=par, state=i, method="store, refresh", F_STAB=Fst, F_out=F,
                         optimal=F >= Fst - 1e-9, F_hat=r.F_hat, err=abs(r.F_hat - Fst),
                         search_shots=r.search_shots, iterations=r.iterations))
        print(fam, i, rows[-1]["optimal"], rows[-1]["search_shots"], flush=True)
save(rows, "sim3r_refresh_n12.csv")
