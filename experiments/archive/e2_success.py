"""E2: when is the optimum reached?  P(success) and F/F_STAB versus tau (pure)
and versus the depolarizing probability p (mixed).  n = 3, 4, 5 with exact
F_STAB by enumeration of all stabilizer states.  Random start."""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run, save, seeds, SpectrumBackend, StabTableau
from fastfidelity.reference import near_stabilizer, depolarize, fstab_exact

NS = [3, 4, 5]
TAUS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
R = 25
rows = []
for n in NS:
    for tau in TAUS:
        for i, sd in enumerate(seeds(int(2000 + 100 * n + 100 * tau), R)):
            rng = np.random.default_rng(sd)
            psi, T = near_stabilizer(n, tau, rng)
            Fst = tau if tau > 0.854 else fstab_exact(psi)
            B = SpectrumBackend(psi)
            start = StabTableau.random(n, rng)
            s = int(rng.integers(2**31))
            for v in ["no store", "store"]:
                r = run(B, start, v, s, Fst)
                r.update(n=n, sweep="tau", tau=tau, p=0.0, state=i)
                rows.append(r)
        print("n", n, "tau", tau, flush=True)
    for p in PS:
        for i, sd in enumerate(seeds(int(3000 + 100 * n + 100 * p), R)):
            rng = np.random.default_rng(sd)
            psi, T = near_stabilizer(n, 0.95, rng)
            Fst = (1 - p) * 0.95 + p / 2**n                  # exact: 0.95 > cos^2(pi/8)
            B = SpectrumBackend(depolarize(psi, p))
            start = StabTableau.random(n, rng)
            s = int(rng.integers(2**31))
            for v in ["no store", "store"]:
                r = run(B, start, v, s, Fst)
                r.update(n=n, sweep="p", tau=0.95, p=p, state=i)
                rows.append(r)
        print("n", n, "p", p, flush=True)
save(rows, "e2_success.csv")
