"""E5: which parameters matter?  Ablation of K, N, start distance, the
direction filter p_x > (3 - 2 sqrt 2) p_0 and the store refresh before retries.
n = 6, pure states with tau = 0.9 (unique optimum), DFE with store."""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run, save, seeds, SpectrumBackend, StabTableau
from fastfidelity.reference import near_stabilizer

n, R = 6, 20
base = dict(N=1000, K=3, theta=3 - 2 * np.sqrt(2), refresh_on_retry=False)
configs = [("default", {}, 3)]
configs += [(f"K={k}", dict(K=k), 3) for k in [1, 2, 5]]
configs += [(f"N={N}", dict(N=N), 3) for N in [250, 500, 2000]]
configs += [(f"start d={d}", {}, d) for d in [1, 5, 8]] + [("start random", {}, None)]
configs += [("no filter", dict(theta=0.0), 3), ("refresh on retry", dict(refresh_on_retry=True), 3)]
rows = []
states = []
for sd in seeds(7000, R):
    rng = np.random.default_rng(sd)
    psi, T = near_stabilizer(n, 0.9, rng)
    states.append((psi, T, rng))
for name, kw, d in configs:
    for i, (psi, T, _) in enumerate(states):
        rng = np.random.default_rng(10_000 * i + 7)
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng) if d is None else T.random_walk(d, rng)
        s = int(rng.integers(2**31))
        for v in ["no store", "store"]:
            p = dict(base)
            p.update(kw)
            r = run(B, start, v, s, 0.9, **p)
            r.update(config=name, state=i)
            rows.append(r)
    print(name, flush=True)
save(rows, "e5_ablation.csv")
