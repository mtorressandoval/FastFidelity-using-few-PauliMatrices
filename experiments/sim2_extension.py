"""Simulation 2, extension to more qubits for T-doped and depolarized stabilizer
states, with the same settings as sim2_families.py (store fed by substrings,
random start, N = 1000), using the exact backends instead of dense states.
The perturbed stabilizer states need a dense state vector and are not extended.

Usage: python sim2_extension.py n R fam:par[,par...] [--from i0] [--dense]
--dense uses the exact dense Pauli spectrum (n <= 12), which samples faster
than the exact T-doped / depolarized simulators; both are exact.
"""
import sys
import numpy as np
from qiskit.quantum_info import random_clifford
from common import save, seeds, StabTableau, self_guided_estimation
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from fastfidelity.backends import TDopedBackend, DepolarizedStabilizerBackend, SpectrumBackend

n, R = int(sys.argv[1]), int(sys.argv[2])
fam, pars = sys.argv[3].split(":")
pars = [float(p) if "." in p else int(p) for p in pars.split(",")]
I0 = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
DENSE = "--dense" in sys.argv
FAMS = ["perturbed", "T-doped", "depolarized"]
FULL = {"T-doped": [1, 2, 3], "depolarized": [0.1, 0.3, 0.5]}
rows = []
for par in pars:
    for i, sd in enumerate(seeds(int(9000 + 1000 * FAMS.index(fam) + 100 * FULL[fam].index(par) + n), R)):
        if i < I0:
            continue
        rng = np.random.default_rng(sd)
        if fam == "T-doped":
            C = random_clifford(n, seed=int(rng.integers(2**31)))
            if DENSE:
                qc = QuantumCircuit(n)
                for j in range(par):
                    qc.h(j)
                    qc.t(j)
                qc.compose(C.to_circuit(), inplace=True)
                B = SpectrumBackend(Statevector(qc).data)
            else:
                B = TDopedBackend(C, par)
            Fst = np.cos(np.pi / 8) ** (2 * par)
        else:
            phi = StabTableau.random(n, rng)
            if DENSE:
                B = SpectrumBackend(phi.statevector())
                B.E = (1 - par) * B.E
                B.E[0] = 1.0
            else:
                B = DepolarizedStabilizerBackend(phi, par)
            Fst = 1 - par + par / 2**n
        fid = (lambda z: z.fidelity_from_spectrum(B.E)) if DENSE else B.fidelity
        start = StabTableau.random(n, rng)
        r = self_guided_estimation(B, start, store="store", substrings=True,
                                   seed=int(rng.integers(2**31)), max_iter=200, keep_path=False)
        F = fid(r.state)
        rows.append(dict(family=fam, par=par, n=n, state=i, F_STAB=Fst, F_out=F, ratio=F / Fst,
                         optimal=F >= Fst - 1e-9, F_hat=r.F_hat, lower=r.lower_bound,
                         certified=r.certified, iterations=r.iterations,
                         hit_limit=r.iterations >= 200 and r.log[-1]["action"] != "stop",
                         search_shots=r.search_shots))
        print(fam, par, n, i, rows[-1]["optimal"], rows[-1]["search_shots"], flush=True)
        save(rows, f"sim2_ext_n{n}_{fam}_{'_'.join(map(str, pars))}{'_from' + str(I0) if I0 else ''}.csv")
