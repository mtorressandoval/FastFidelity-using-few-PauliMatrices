"""E1 with circuit-prepared states on Qiskit Aer (prepared, not run in the paper).

psi = C (R_y(theta) (x) R_y(theta) (x) I ...) |0...0>, with C a shallow random
Clifford brickwork (low entanglement, so that MPS is efficient).  With
cos^4(theta/2) = 0.9 > cos^2(pi/8), C|0...0> is the unique optimum; success
means that the protocol returns exactly this stabilizer state.

Examples
  MPS, CPU :  python e1_circuits.py --method matrix_product_state --ns 20 40 60
  GPU      :  python e1_circuits.py --method statevector --device GPU --ns 20 26 30
(GPU requires the qiskit-aer-gpu package.)
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import save, seeds, VARIANTS  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Clifford, random_clifford  # noqa: E402
from fastfidelity import StabTableau, QiskitSamplerBackend, self_guided_estimation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=int, nargs="+", default=[20, 40])
ap.add_argument("--states", type=int, default=5)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--method", default="matrix_product_state")
ap.add_argument("--device", default="CPU")
ap.add_argument("--N", type=int, default=1000)
ap.add_argument("--block", type=int, default=4)
ap.add_argument("--out", default=None)
a = ap.parse_args()
theta = 2 * math.acos(0.9 ** 0.25)


def brickwork(n, depth, rng):
    qc = QuantumCircuit(n)
    for layer in range(depth):
        for q in range(layer % 2, n - 1, 2):
            qc.compose(random_clifford(2, seed=int(rng.integers(2**31))).to_circuit(), [q, q + 1], inplace=True)
    return qc


rows = []
for n in a.ns:
    for i, sd in enumerate(seeds(9000 + n, a.states)):
        rng = np.random.default_rng(sd)
        C = brickwork(n, a.depth, rng)
        prep = QuantumCircuit(n)
        prep.ry(theta, 0)
        prep.ry(theta, 1)
        prep.compose(C, inplace=True)
        target = StabTableau.from_qiskit(Clifford(C))
        B = QiskitSamplerBackend(prep, sampler="aer", method=a.method, device=a.device)
        start = target.random_walk(3, rng)
        s = int(rng.integers(2**31))
        for v, (store, sub) in VARIANTS.items():
            R = self_guided_estimation(B, start, store=store, substrings=sub, seed=s, N=a.N,
                                       walsh_block=None if n <= 20 else a.block, keep_path=False)
            rows.append(dict(n=n, state=i, variant=v, optimal=R.state == target, F_hat=R.F_hat,
                             certified=R.certified, search_shots=R.search_shots, reuse=R.reuse_rate,
                             method=a.method, device=a.device))
            print(rows[-1], flush=True)
        save(rows, a.out or f"large_e1_circuits_{a.method}_{a.device}.csv")
