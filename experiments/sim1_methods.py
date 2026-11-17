"""Simulation 1: comparison of the three estimators (DFE without store, with
store, with store and substrings) on stabilizer states with a random
perturbation, psi = sqrt(tau) phi + sqrt(1 - tau) chi with tau = 0.9, starting
from a random stabilizer state.  Since tau > cos^2(pi/8), F_STAB = tau exactly.
Records the exact fidelity of the current state after every iteration
(Fig. 2) and the copies needed for error eps = 0.05 (Fig. 4), and simulates
Algorithm 3 of Grewal et al. on the same states (Fig. 4).
"""
import math
import numpy as np
from common import save, seeds, SpectrumBackend, StabTableau, self_guided_estimation, VARIANTS
from fastfidelity.reference import near_stabilizer
from fastfidelity.bell import grewal_theory_copies, run_grewal

EPS, DELTA, TAU = 0.05, 0.01, 0.9
M_DFE = math.ceil(2 * math.log(2 / DELTA) / EPS**2)          # 4239 single-copy Pauli shots
M_BASIS = math.ceil(math.log(2 / DELTA) / (2 * EPS**2))       # 1060 copies in the stabilizer basis
NS, R, MAX_IT = [3, 4, 5, 6, 7, 8], 20, 200
runs, traj = [], []
for n in NS:
    states = []
    for i, sd in enumerate(seeds(5000 + n, R)):
        rng = np.random.default_rng(sd)
        psi, T = near_stabilizer(n, TAU, rng)
        states.append((psi, T))
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng)
        s = int(rng.integers(2**31))
        for v, (store, sub) in VARIANTS.items():
            rec = []
            cb = lambda it, z, sh: rec.append((it, z.fidelity_from_spectrum(B.E) / TAU, sh))
            r = self_guided_estimation(B, start, store=store, substrings=sub, seed=s,
                                       final_shots=M_DFE, max_iter=MAX_IT, callback=cb, keep_path=False)
            F = r.state.fidelity_from_spectrum(B.E)
            runs.append(dict(n=n, state=i, method=v, F_out=F, ratio=F / TAU, optimal=F >= TAU - 1e-9,
                             F_hat=r.F_hat, certified=r.certified, iterations=r.iterations,
                             hit_limit=r.iterations >= MAX_IT and r.log[-1]["action"] != "stop",
                             search_shots=r.search_shots, copies=r.search_shots + M_DFE, reuse=r.reuse_rate))
            traj += [dict(n=n, state=i, method=v, it=it, ratio=f, shots=sh) for it, f, sh in rec]
    th = grewal_theory_copies(n, TAU, DELTA)
    runs.append(dict(n=n, method="Grewal et al. (theory)", copies=th["total"] + M_BASIS, search_shots=th["total"]))
    rng = np.random.default_rng(6000 + n)
    for c in [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]:
        m, kB, kM = max(1, int(th["m"] * c)), max(2, int(th["copies_bell"] * c)), math.ceil(th["copies_majority"])
        ok, used = [], []
        for psi, T in states * 2:
            out, cop = run_grewal(psi, m, kB, kM, rng)
            ok.append(out is not None and out == T)
            used.append(cop)
        runs.append(dict(n=n, method="Grewal et al. (empirical)", scale=c, success=np.mean(ok),
                         copies=np.mean(used) + M_BASIS, search_shots=np.mean(used)))
    print("n", n, flush=True)
    save(runs, "sim1_methods.csv")
    save(traj, "sim1_trajectories.csv")
