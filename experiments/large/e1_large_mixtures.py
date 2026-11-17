"""E1 for large n (prepared, not run in the paper): mixtures of stabilizer
states with a planted optimum, simulated analytically with
StabilizerMixtureBackend.  The Walsh step uses the prefix search over
subgroups (walsh_block) because 2^n directions cannot be enumerated.

Usage:  python e1_large_mixtures.py --ns 20 40 60 80 100 --states 10
Runtime: tens of minutes per state at n = 100 on one CPU core.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import save, seeds, VARIANTS  # noqa: E402
from fastfidelity import StabilizerMixtureBackend, self_guided_estimation  # noqa: E402
from fastfidelity.reference import planted_mixture, mixture_fidelity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=int, nargs="+", default=[20, 40, 60, 80, 100])
ap.add_argument("--states", type=int, default=10)
ap.add_argument("--N", type=int, default=1000)
ap.add_argument("--block", type=int, default=4)
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--out", default="large_e1_mixtures.csv")
a = ap.parse_args()

rows = []
for n in a.ns:
    for i, sd in enumerate(seeds(8000 + n, a.states)):
        rng = np.random.default_rng(sd)
        Ts, w = planted_mixture(n, 0.9, 4, rng)
        Fref = mixture_fidelity(Ts[0], Ts, w)
        B = StabilizerMixtureBackend(Ts, w)
        start = Ts[0].random_walk(a.steps, rng)
        s = int(rng.integers(2**31))
        for v, (store, sub) in VARIANTS.items():
            R = self_guided_estimation(B, start, store=store, substrings=sub, seed=s, N=a.N,
                                       walsh_block=a.block, keep_path=False)
            F = mixture_fidelity(R.state, Ts, w)
            rows.append(dict(n=n, state=i, variant=v, F_out=F, F_ref=Fref, optimal=F >= Fref - 1e-9,
                             F_hat=R.F_hat, certified=R.certified, search_shots=R.search_shots,
                             reuse=R.reuse_rate))
            print(rows[-1], flush=True)
        save(rows, a.out)
