"""Simulation 4: hierarchical block warm start versus random start for T-doped
states C (T|+>)^{(x)t} (x) |0>^{(x)(n-t)} with three kinds of Clifford circuits C:
  local2  : two brickwork layers of random 2-qubit Cliffords on neighbours,
  global2 : two layers of random 2-qubit Cliffords on random pairs of qubits,
  random  : a random n-qubit Clifford (deep, as in Fig. 3).
Best estimator (store fed by substrings), N = 1000.  F_STAB = cos^{2t}(pi/8).

Usage: python sim4_blocks.py n R kind[,kind...] [start ...] [--scaled]
start: random, blocks, frame (blocks in the frame of the known Clifford C, which
measures C P C^dag; no substrings).
"""
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Clifford, Statevector
from common import save, seeds, StabTableau, self_guided_estimation
from fastfidelity.backends import SpectrumBackend, TDopedBackend
from fastfidelity.blocks import hierarchical_search, SCALE_N, FrameBackend

n, R = int(sys.argv[1]), int(sys.argv[2])
kinds = sys.argv[3].split(",")
starts = [a for a in sys.argv[4:] if not a.startswith("--")] or ["random", "blocks"]
SCALE_N["on"] = "--scaled" in sys.argv
t = 2
Fst = np.cos(np.pi / 8) ** (2 * t)


def clifford_circuit(kind, rng):
    qc = QuantumCircuit(n)
    if kind == "random":
        return random_clifford(n, seed=int(rng.integers(2**31))).to_circuit()
    for layer in range(2):
        if kind == "local2":
            pairs = [(q, q + 1) for q in range(layer % 2, n - 1, 2)]
        else:
            p = rng.permutation(n)
            pairs = [(int(p[2 * i]), int(p[2 * i + 1])) for i in range(n // 2)]
        for a, b in pairs:
            qc.compose(random_clifford(2, seed=int(rng.integers(2**31))).to_circuit(), [a, b], inplace=True)
    return qc


rows = []
for kind in kinds:
    for i, sd in enumerate(seeds(20000 + 100 * n + ["local2", "global2", "random"].index(kind), R)):
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
        s = int(rng.integers(2**31))
        for st in starts:
            r_rng = np.random.default_rng(s)
            if st == "random":
                r = self_guided_estimation(B, StabTableau.random(n, r_rng), store="store", substrings=True,
                                           rng=r_rng, keep_path=False)
                shots, start_F = r.search_shots, None
            elif st == "blocks":
                r, shots, log, T0 = hierarchical_search(B, rng=r_rng, store="store", substrings=True)
                start_F = fid(T0)
            else:
                FB = FrameBackend(B, Clifford(Cq))
                r, shots, log, T0 = hierarchical_search(FB, rng=r_rng, store="store", substrings=False)
                start_F = fid(FB.to_physical(T0))
            F = fid(FB.to_physical(r.state)) if st == "frame" else fid(r.state)
            rows.append(dict(n=n, kind=kind, state=i, start=st, F_STAB=Fst, F_out=F, ratio=F / Fst,
                             optimal=F >= Fst - 1e-9, F_hat=r.F_hat, err=abs(r.F_hat - Fst),
                             search_shots=shots, F_start=start_F, iterations=r.iterations))
            print(n, kind, i, st, round(F / Fst, 3), shots, "F(start global)=", start_F, flush=True)
        save(rows, f"sim4_blocks_n{n}_{'_'.join(kinds)}_{'_'.join(starts)}{'_scaled' if SCALE_N['on'] else ''}.csv")
