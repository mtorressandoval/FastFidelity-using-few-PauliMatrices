"""Control: reproduce the prototype configuration (vecinos_con_almacen.py):
N = 1000, K = 3, no direction filter, no retries.  n = 4 with tau = 0.9 and
random start, and n = 8 with tau = 0.9 and a 3-step start, must reach the
optimum in 100% of the cases with a shot saving of at least 4x; with
substrings the saving cannot be smaller."""
import numpy as np
import pytest

from fastfidelity import SpectrumBackend, StabTableau, self_guided_estimation
from fastfidelity.reference import near_stabilizer


@pytest.mark.parametrize("n,steps,R", [(4, None, 15), (8, 3, 8)])
def test_control_against_prototype(n, steps, R):
    rng = np.random.default_rng(0)
    res = {}
    for _ in range(R):
        psi, T = near_stabilizer(n, 0.9, rng)
        B = SpectrumBackend(psi)
        start = StabTableau.random(n, rng) if steps is None else T.random_walk(steps, rng)
        seed = int(rng.integers(1e9))
        for key, (store, sub) in {"fresh": (None, False), "store": ("store", False),
                                  "sub": ("store", True)}.items():
            r = self_guided_estimation(B, start, store=store, substrings=sub, seed=seed,
                                       theta=0.0, retries=0, final=False, confirm_flips=False)
            ok = r.state.fidelity_from_spectrum(B.E) > 0.9 - 1e-9
            res.setdefault(key, []).append((ok, r.search_shots))
    f = np.array(res["fresh"], float)
    for key in ["store", "sub"]:
        v = np.array(res[key], float)
        assert v[:, 0].all()
        saving = np.mean(f[:, 1] / v[:, 1])
        assert saving >= 4.0, (key, saving)
    assert np.mean(f[:, 1] / np.array(res["sub"], float)[:, 1]) >= np.mean(
        f[:, 1] / np.array(res["store"], float)[:, 1]) - 1e-9
