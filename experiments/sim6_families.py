"""Simulation 6 (paper, Fig. 2): the self-guided estimation protocol with Pauli
store, substrings and local restart (k = 4), from a random stabilizer state,
for four families of states:
  perturbed      deep random stabilizer + Haar perturbation, tau in {0.86, 0.9, 0.95}
  T-doped deep   C (T|+>)^t |0>^(n-t), C a random Clifford (depth O(n)), t in {0,1,2,3}
  T-doped shallow  the same with C = ceil(log2 n) layers of random two-qubit
                 Cliffords on neighbouring qubits (depth O(log n)), t in {0,1,2,3}
  depolarized    (1-p) C|0><0|C^dag + p I/2^n, C shallow as above, p in {0.1,0.3,0.5}
F_STAB is exact: tau, cos^{2t}(pi/8), and 1 - p + p/2^n.  For n <= 12 the whole
Pauli spectrum is used (and the store keeps every substring); for n > 12 the
exact T-doped and depolarized simulators are used and the store keeps only the
substrings of weight <= 4.  Resumable: finished runs in the CSV are skipped.

Usage: python sim6_families.py family n R [par ...]
"""
import math
import os
import sys
import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Clifford, Statevector, StabilizerState
from common import RESULTS, seeds, StabTableau, self_guided_estimation
from fastfidelity.backends import SpectrumBackend, TDopedBackend, DepolarizedStabilizerBackend
from fastfidelity.reference import near_stabilizer

fam, n, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
DEFAULT = {"perturbed": [0.86, 0.9, 0.95], "T-doped deep": [0, 1, 2, 3],
           "T-doped shallow": [0, 1, 2, 3], "depolarized": [0.1, 0.3, 0.5]}
pars = [float(p) if "." in p else int(p) for p in sys.argv[4:]] or DEFAULT[fam]
FAMS = list(DEFAULT)
KLOC = max(4, 2 * math.ceil(math.log2(n)))      # light-cone width of a circuit of depth log2 n
path = os.path.join(RESULTS, f"sim6_{fam.replace(' ', '_')}_n{n}.csv")
rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
done = {(r["par"], r["state"]) for r in rows}


def shallow(rng):
    qc = QuantumCircuit(n)
    for layer in range(math.ceil(math.log2(n))):
        for q in range(layer % 2, n - 1, 2):
            qc.compose(random_clifford(2, seed=int(rng.integers(2**31))).to_circuit(), [q, q + 1], inplace=True)
    return qc


for par in pars:
    for i, sd in enumerate(seeds(int(50000 + 1000 * FAMS.index(fam) + 100 * DEFAULT[fam].index(par) + n), R)):
        if (par, i) in done:
            continue
        rng = np.random.default_rng(sd)
        dense = n <= 12
        if fam == "perturbed":
            psi, _ = near_stabilizer(n, par, rng)
            B = SpectrumBackend(psi)
            Fst = par
        elif fam.startswith("T-doped"):
            Cq = random_clifford(n, seed=int(rng.integers(2**31))).to_circuit() if fam == "T-doped deep" else shallow(rng)
            Fst = np.cos(np.pi / 8) ** (2 * par)
            if dense:
                qc = QuantumCircuit(n)
                for j in range(par):
                    qc.h(j)
                    qc.t(j)
                qc.compose(Cq, inplace=True)
                B = SpectrumBackend(Statevector(qc).data)
            else:
                B = TDopedBackend(Clifford(Cq), par)
        else:
            phi = StabTableau.from_qiskit(StabilizerState(shallow(rng)))
            Fst = 1 - par + par / 2**n
            if dense:
                B = SpectrumBackend(phi.statevector())
                B.E = (1 - par) * B.E
                B.E[0] = 1.0
            else:
                B = DepolarizedStabilizerBackend(phi, par)
        fid = (lambda z: z.fidelity_from_spectrum(B.E)) if dense else B.fidelity
        start = StabTableau.random(n, rng)
        t0 = time.time()
        r = self_guided_estimation(B, start, store="store", substrings=True, local_restart=KLOC,
                                   seed=int(rng.integers(2**31)), max_iter=200, keep_path=False)
        F = fid(r.state)
        rows.append(dict(family=fam, par=par, n=n, state=i, F_STAB=Fst, F_out=F, ratio=F / Fst,
                         optimal=bool(F >= Fst - 1e-9), F_hat=r.F_hat, err=abs(r.F_hat - Fst),
                         lower=r.lower_bound, certified=r.certified, search_shots=r.search_shots,
                         iterations=r.iterations, local_jumps=sum(e["action"] == "local" for e in r.log),
                         hit_limit=bool(r.iterations >= 200 and r.log[-1]["action"] != "stop"),
                         dense=dense, k=KLOC, seconds=time.time() - t0))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(fam, par, n, i, round(F / Fst, 3), r.search_shots, f"{time.time() - t0:.0f}s", flush=True)
