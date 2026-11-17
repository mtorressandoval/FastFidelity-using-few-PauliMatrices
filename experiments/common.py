"""Shared helpers for the experiments.  Every script writes a CSV in results/."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from fastfidelity import StabTableau, SpectrumBackend, self_guided_estimation  # noqa: E402

VARIANTS = {  # label -> (store, substrings)
    "no store": (None, False),
    "store": ("store", False),
    "store+substrings": ("store", True),
}


def seeds(master, k):
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(master).spawn(k)]


def run(backend, start, variant, seed, F_ref, F_exact_of=None, **kw):
    """One run of the protocol; returns a flat dict for the CSV."""
    store, sub = VARIANTS[variant]
    t = time.time()
    R = self_guided_estimation(backend, start, store=store, substrings=sub, seed=seed, **kw)
    F_out = F_exact_of(R.state) if F_exact_of else R.state.fidelity_from_spectrum(backend.E)
    return dict(variant=variant, F_out=F_out, F_ref=F_ref, ratio=F_out / F_ref,
                optimal=bool(F_out >= F_ref - 1e-9), F_hat=R.F_hat, ci_lo=R.ci[0], ci_hi=R.ci[1],
                lower=R.lower_bound, certified=R.certified, search_shots=R.search_shots,
                final_shots=R.final_shots, iterations=R.iterations, reuse=R.reuse_rate,
                requested=R.requested_shots, seconds=time.time() - t)


def save(rows, name):
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    print("wrote", path, len(df), "rows", flush=True)
    return df
