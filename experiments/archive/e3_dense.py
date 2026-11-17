"""E3: dense states.  Haar-random pure states, random mixed states of rank k
(qiskit random_density_matrix) and Clifford+T states; n = 3, 4, 5; exact
F_STAB by enumeration.  Random start."""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run, save, seeds, SpectrumBackend, StabTableau
from fastfidelity.reference import haar_state, random_rank_k, clifford_t_state, fstab_exact

R = 25
rows = []
for n in [3, 4, 5]:
    fams = [("Haar", None), ("rank 2", 2), ("rank 4", 4), ("Clifford+1T", "t1"), ("Clifford+3T", "t3")]
    for fam, par in fams:
        for i, sd in enumerate(seeds(4000 + 10 * n + fams.index((fam, par)), R)):
            rng = np.random.default_rng(sd)
            if fam == "Haar":
                st = haar_state(n, rng)
            elif isinstance(par, int):
                st = random_rank_k(n, par, rng)
            else:
                st, _ = clifford_t_state(n, int(par[1:]), rng)
            Fst = fstab_exact(st)
            B = SpectrumBackend(st)
            start = StabTableau.random(n, rng)
            s = int(rng.integers(2**31))
            for v in ["no store", "store", "store+substrings"]:
                r = run(B, start, v, s, Fst)
                r.update(n=n, family=fam, state=i)
                rows.append(r)
        print(n, fam, flush=True)
save(rows, "e3_dense.csv")
