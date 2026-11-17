"""E4b: random start at n = 8 (tau = 0.9).  Effect of the number N of shots per
DFE on the success probability and on the copies, with and without store.
From a random start the initial fidelities are of order 2^-n, below the DFE
noise 1/sqrt(N), so that N controls whether the search leaves the flat region."""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run, save, seeds, SpectrumBackend, StabTableau
from fastfidelity.reference import near_stabilizer

n, R = 8, 10
rows = []
for N in [1000, 4000]:
    for i, sd in enumerate(seeds(5000 + n, R)):          # same states as E4
        rng = np.random.default_rng(sd)
        psi, T = near_stabilizer(n, 0.9, rng)
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng)
        s = int(rng.integers(2**31))
        for v in ["no store", "store", "store+substrings"]:
            r = run(B, start, v, s, 0.9, N=N, final_shots=4239, max_iter=200)
            r.update(n=n, N=N, state=i)
            rows.append(r)
            print(N, i, v, r["optimal"], r["search_shots"], flush=True)
save(rows, "e4b_random_start_N.csv")
