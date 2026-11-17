"""E1: how many shots does the store save, and how does the saving scale with n?

Pure states psi = sqrt(tau) phi + sqrt(1-tau) chi (tau = 0.9), their
depolarized versions (p = 0.05) and mixtures of four stabilizer states with a
planted optimum (w1 = 0.9).  In all cases the planted state has fidelity
above cos^2(pi/8), so it is the unique optimum and gives F_STAB exactly.
Start: 3 random neighbour steps away from the optimum (the same distance for
every n, so that only n changes).  n = 3..8, dense simulation.
"""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run, save, seeds, SpectrumBackend, StabTableau
from fastfidelity.reference import near_stabilizer, depolarize, planted_mixture, mixture_fidelity

NS = [3, 4, 5, 6, 7, 8]
R = 20
rows = []
for n in NS:
    for i, sd in enumerate(seeds(1000 + n, R)):
        rng = np.random.default_rng(sd)
        psi, T = near_stabilizer(n, 0.9, rng)
        Ts, w = planted_mixture(n, 0.9, 4, rng)
        rho_mix = sum(wi * np.outer(t.statevector(), t.statevector().conj()) for t, wi in zip(Ts, w))
        fams = [("pure tau=0.9", psi, T, 0.9),
                ("depolarized p=0.05", depolarize(psi, 0.05), T, 0.95 * 0.9 + 0.05 / 2**n),
                ("stabilizer mixture", rho_mix, Ts[0], mixture_fidelity(Ts[0], Ts, w))]
        for fam, state, Topt, Fref in fams:
            B = SpectrumBackend(state)
            assert abs(Topt.fidelity_from_spectrum(B.E) - Fref) < 1e-9
            start = Topt.random_walk(3, rng)
            run_seed = int(rng.integers(2**31))
            for v in ["no store", "store", "store+substrings"]:
                r = run(B, start, v, run_seed, Fref)
                r.update(n=n, family=fam, state=i)
                rows.append(r)
    print("n =", n, "done", flush=True)
save(rows, "e1_store_savings.csv")
