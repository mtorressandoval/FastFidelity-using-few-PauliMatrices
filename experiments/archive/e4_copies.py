"""E4: copies needed to estimate F_STAB with error eps = 0.05 (confidence 99%)
versus n, for the self-guided protocol (Pauli measurements on one copy) and
for Algorithm 3 of Grewal et al. (Bell measurements on two copies), in the
regime tau = 0.9 > cos^2(pi/8) with pure states, where their guarantee holds.

Final estimate: our protocol uses a fresh DFE with M = 2 ln(2/delta)/eps^2
single-copy Pauli shots (Hoeffding, outcomes in [-1, 1]); for Grewal et al. we
count M = ln(2/delta)/(2 eps^2) copies measured in the stabilizer basis of the
output (Bernoulli outcomes), which favours their method.
Grewal et al.: (i) copies with the constants of their Thm. 6.7, (ii) empirical
copies, scaling m and k_B down by c in {1, 1/2, ..., 1/64} and keeping the
smallest c with success >= 95%.
"""
import math
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run, save, seeds, SpectrumBackend, StabTableau
from fastfidelity.reference import near_stabilizer
from fastfidelity.bell import grewal_theory_copies, run_grewal

EPS, DELTA, TAU = 0.05, 0.01, 0.9
M_DFE = math.ceil(2 * math.log(2 / DELTA) / EPS**2)
M_BASIS = math.ceil(math.log(2 / DELTA) / (2 * EPS**2))
NS = [3, 4, 5, 6, 7, 8]
R = 20
rows = []
for n in NS:
    states = []
    for i, sd in enumerate(seeds(5000 + n, R)):
        rng = np.random.default_rng(sd)
        psi, T = near_stabilizer(n, TAU, rng)
        states.append((psi, T))
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng)
        s = int(rng.integers(2**31))
        for v in ["no store", "store", "store+substrings"]:
            r = run(B, start, v, s, TAU, final_shots=M_DFE)
            r.update(n=n, method="self-guided (" + v + ")", state=i, copies=r["search_shots"] + M_DFE,
                     measurement="Pauli, one copy")
            rows.append(r)
    th = grewal_theory_copies(n, TAU, DELTA)
    rows.append(dict(n=n, method="Grewal et al. (theory)", copies=th["total"] + M_BASIS,
                     search_shots=th["total"], optimal=np.nan, measurement="Bell, two copies"))
    rng = np.random.default_rng(6000 + n)
    for c in [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]:
        m = max(1, int(th["m"] * c))
        kB = max(2, int(th["copies_bell"] * c))
        kM = math.ceil(th["copies_majority"])
        ok, used = [], []
        for psi, T in states * 2:
            out, cop = run_grewal(psi, m, kB, kM, rng)
            ok.append(out is not None and out == T)
            used.append(cop)
        rows.append(dict(n=n, method="Grewal et al. (empirical)", scale=c, success=np.mean(ok),
                         copies=np.mean(used) + M_BASIS, search_shots=np.mean(used),
                         measurement="Bell, two copies"))
    print("n", n, flush=True)
save(rows, "e4_copies.csv")
