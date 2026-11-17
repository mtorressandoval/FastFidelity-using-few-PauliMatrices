"""Simulation 1b: the runs of simulation 1 with the store fed by substrings
(same seeds, so the same runs), saving the fidelity estimated by the protocol
after every iteration, for Fig. 2(b)."""
import math
import numpy as np
from common import save, seeds, SpectrumBackend, StabTableau, self_guided_estimation
from fastfidelity.reference import near_stabilizer

TAU, DELTA, EPS = 0.9, 0.01, 0.05
M_DFE = math.ceil(2 * math.log(2 / DELTA) / EPS**2)
rows = []
for n in [3, 4, 5, 6, 7, 8]:
    for i, sd in enumerate(seeds(5000 + n, 20)):
        rng = np.random.default_rng(sd)
        psi, T = near_stabilizer(n, TAU, rng)
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng)
        s = int(rng.integers(2**31))
        ex = {}
        cb = lambda it, z, sh: ex.__setitem__(it, z.fidelity_from_spectrum(B.E))
        r = self_guided_estimation(B, start, store="store", substrings=True, seed=s,
                                   final_shots=M_DFE, max_iter=200, callback=cb, keep_path=False)
        for e in r.log:
            rows.append(dict(n=n, state=i, it=e["it"], action=e["action"], f_hat=e["f_hat"],
                             F_exact=ex.get(e["it"], np.nan), shots=e["cum_shots"]))
        rows.append(dict(n=n, state=i, it=r.iterations + 1, action="final", f_hat=r.F_hat,
                         F_exact=r.state.fidelity_from_spectrum(B.E), shots=r.search_shots + M_DFE))
    print("n", n, flush=True)
save(rows, "sim1b_estimates.csv")
