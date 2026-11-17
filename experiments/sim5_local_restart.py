"""Simulation 5: protocol with local restart (k = 4) versus the protocol
without it, starting from a random stabilizer state.  T-doped states (t = 2)
whose Clifford circuit is two layers of random two-qubit Cliffords on
neighbouring qubits ("neighbors"), on random pairs ("random pairs"), or a random
n-qubit Clifford ("deep random").  Store fed by substrings, N = 1000.

Usage: python sim5_local_restart.py n R kind[,kind...] method[,method...] [--from i0]
methods: local (local_restart=4), plain (no local restart)."""
import sys
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Clifford, Statevector
from common import save, seeds, StabTableau, self_guided_estimation
from fastfidelity.backends import SpectrumBackend, TDopedBackend

n, R = int(sys.argv[1]), int(sys.argv[2])
kinds, methods = sys.argv[3].split(","), sys.argv[4].split(",")
I0 = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
t, KLOC = 2, 4
FST = np.cos(np.pi / 8) ** (2 * t)
KINDS = ["neighbors", "random pairs", "deep random"]


def clifford_circuit(kind, rng):
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


rows = []
for kind in kinds:
    for i, sd in enumerate(seeds(30000 + 100 * n + KINDS.index(kind), R)):
        if i < I0:
            continue
        rng = np.random.default_rng(sd)
        Cq = clifford_circuit(kind, rng)
        if n <= 12:
            qc = QuantumCircuit(n)
            for j in range(t):
                qc.h(j)
                qc.t(j)
            qc.compose(Cq, inplace=True)
            B = SpectrumBackend(Statevector(qc).data)
            fid = lambda z: z.fidelity_from_spectrum(B.E)
        else:
            B = TDopedBackend(Clifford(Cq), t)
            fid = B.fidelity
        start = StabTableau.random(n, rng)
        s = int(rng.integers(2**31))
        for meth in methods:
            t0 = time.time()
            r = self_guided_estimation(B, start, store="store", substrings=True, seed=s, keep_path=False,
                                       local_restart=KLOC if meth == "local" else None)
            F = fid(r.state)
            nloc = sum(e["action"] == "local" for e in r.log)
            rows.append(dict(n=n, kind=kind, state=i, method=meth, F_STAB=FST, F_out=F, ratio=F / FST,
                             optimal=F >= FST - 1e-9, F_hat=r.F_hat, err=abs(r.F_hat - FST),
                             certified=r.certified, search_shots=r.search_shots, iterations=r.iterations,
                             local_jumps=nloc, seconds=time.time() - t0))
            print(n, kind, i, meth, round(F / FST, 3), r.search_shots, "jumps", nloc,
                  f"{time.time() - t0:.0f}s", flush=True)
        save(rows, f"sim5_n{n}_{'_'.join(k.replace(' ', '') for k in kinds)}_{'_'.join(methods)}"
                   f"{'_from' + str(I0) if I0 else ''}.csv")
