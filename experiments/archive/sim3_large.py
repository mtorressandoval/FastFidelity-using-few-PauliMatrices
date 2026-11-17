"""Simulation 3: larger systems (n = 8..20) with the exact backends for T-doped
states C (T|+>)^{(x)t} (x) |0>^{(x)(n-t)} and depolarized stabilizer states.
The search starts three neighbour steps away from an optimal stabilizer state,
which models an experiment that knows its intended state approximately.
DFE with store for all parameters, and without store for t = 2 and p = 0.3; substrings are not used, since for large n
the substrings of a measured string almost never belong to the queried groups
(about 2^n / 4^n of them do).  Flips are confirmed with fresh DFE runs when
their advantage is below the multiple-comparison margin (confirm_flips).

Usage: python sim3_large.py [n ...] [--reduced] [--scaled] [--store-only] [--only fam:p1,p2]
--scaled uses N = 125 n shots per DFE and refreshes the store before each retry.
"""
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, StabilizerState
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from common import save, seeds, StabTableau, self_guided_estimation
from fastfidelity.backends import TDopedBackend, DepolarizedStabilizerBackend

REDUCED = "--reduced" in sys.argv          # one parameter per family (t = 2, p = 0.3)
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None   # e.g. depolarized:0.3,0.5
SCALED = "--scaled" in sys.argv            # N = 125 n and refresh_on_retry=True
STORE_ONLY = "--store-only" in sys.argv
NS = [int(a) for a in sys.argv[1:] if a.isdigit()] or [8, 12, 16, 20]
STEPS = 3
rows = []
SKIP = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0   # skip the first states of the first family
for n in NS:
    if ONLY:
        f0, p0 = ONLY.split(":")
        famlist = [(f0, [float(x) if "." in x else int(x) for x in p0.split(",")])]
    elif REDUCED:
        famlist = [("T-doped", [2]), ("depolarized", [0.3])]
    else:
        famlist = [("T-doped", [1, 2, 3]), ("depolarized", [0.1, 0.3, 0.5])]
    for fam, pars in famlist:
        for par in pars:
            R = 10 if n <= 12 else 6
            k = [1, 2, 3].index(par) if fam == "T-doped" else [0.1, 0.3, 0.5].index(par)
            for i, sd in enumerate(seeds(int(12000 + 100 * n + (10 if fam == "T-doped" else 50) + 10 * k), R)):
                if fam == famlist[0][0] and i < SKIP:
                    continue
                rng = np.random.default_rng(sd)
                if fam == "T-doped":
                    C = random_clifford(n, seed=int(rng.integers(2**31)))
                    B = TDopedBackend(C, par)
                    qc = QuantumCircuit(n)
                    for j in range(par):
                        qc.h(j)
                    zopt = StabTableau.from_qiskit(StabilizerState(qc).evolve(C))
                    Fst = np.cos(np.pi / 8) ** (2 * par)
                else:
                    zopt = StabTableau.random(n, rng)
                    B = DepolarizedStabilizerBackend(zopt, par)
                    Fst = 1 - par + par / 2**n
                start = zopt.random_walk(STEPS, rng)
                s = int(rng.integers(2**31))
                methods = [("store", "store")] + ([("no store", None)] if par in (2, 0.3) and not STORE_ONLY else [])
                for v, store in methods:
                    kw = dict(N=125 * n, refresh_on_retry=True) if SCALED else {}
                    r = self_guided_estimation(B, start, store=store, seed=s, keep_path=False,
                                               confirm_flips=True, max_iter=200, **kw)
                    F = B.fidelity(r.state)
                    rows.append(dict(n=n, family=fam, par=par, state=i, method=v, scaled=SCALED, F_STAB=Fst, F_out=F,
                                     optimal=F >= Fst - 1e-9, F_hat=r.F_hat, err=abs(r.F_hat - Fst),
                                     certified=r.certified, iterations=r.iterations,
                                     hit_limit=r.iterations >= 200 and r.log[-1]["action"] != "stop",
                                     search_shots=r.search_shots, reuse=r.reuse_rate))
                print(n, fam, par, i, [(r["method"], r["optimal"], r["search_shots"]) for r in rows[-len(methods):]], flush=True)
            save(rows, f"sim3_large_n{'_'.join(map(str, NS))}{'_reduced' if REDUCED else ''}{'_scaled' if SCALED else ''}{'_skip' + str(SKIP) if SKIP else ''}{'_' + ONLY.replace(':', '_').replace(',', '_') if ONLY else ''}.csv")
