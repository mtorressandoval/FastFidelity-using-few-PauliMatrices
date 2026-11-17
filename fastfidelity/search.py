"""Self-guided estimation protocol for the stabilizer fidelity.

The iterate z is always a stabilizer state.  Each iteration estimates f(z)
and all p_x = <phi_x|rho|phi_x> from one DFE run on the group of z, moves to
phi_x if some p_x beats p_0, and otherwise compares z with its four
neighbours in the K most promising directions.  The protocol stops when no
neighbour improves after ``retries`` doublings of the number of shots.  The
reported fidelity is always estimated with fresh copies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .tableau import StabTableau
from .dfe import make_estimator, FreshDFE

__all__ = ["SearchResult", "self_guided_estimation", "walsh_step", "final_estimate",
           "THETA", "COS2_PI8"]

THETA = 3 - 2 * math.sqrt(2)          # only p_x > THETA p_0 can improve
COS2_PI8 = math.cos(math.pi / 8) ** 2  # uniqueness / certification threshold


@dataclass
class SearchResult:
    state: StabTableau
    F_hat: float
    stderr: float
    ci: tuple
    lower_bound: float
    certified: bool
    search_shots: int
    final_shots: int
    iterations: int
    path: list = field(default_factory=list)
    log: list = field(default_factory=list)
    reuse_rate: float = 0.0
    requested_shots: int = 0
    store: object = None

    @property
    def total_shots(self):
        return self.search_shots + self.final_shots


def walsh_step(est, z, N, block=None, beam=32):
    """Estimate p_x for x in F_2^n.

    block=None: exact Walsh transform of one DFE run (n <= ~22).
    block=b   : prefix search (Goldreich-Levin type) over subgroups of the first
                b, 2b, ..., n generators, keeping the ``beam`` heaviest prefixes;
                returns a dict {x: p_hat_x} for the surviving x.
    Returns (p, p0, estimate_of_f).
    """
    n = z.n
    if block is None or block >= n:
        e = est.estimate(z, N)
        p = e.walsh()
        return p, float(p[0]), e
    surv = [0]
    k_prev = 0
    e = None
    while k_prev < n:
        k = min(n, k_prev + block)
        e = est.estimate(z, N, m=k)
        cand = [a | (b << k_prev) for a in surv for b in range(1 << (k - k_prev))]
        val = e.character(cand)
        order = np.argsort(-np.asarray(val))
        keep = [cand[i] for i in order[:beam]]
        if 0 not in keep:
            keep.append(0)
        surv = keep
        k_prev = k
    vals = e.character(surv)
    p = dict(zip(surv, np.asarray(vals).tolist()))
    return p, float(p[0]), e


def _top(p, K, theta):
    if isinstance(p, dict):
        items = [(x, v) for x, v in p.items() if x != 0]
        items.sort(key=lambda t: -t[1])
        p0 = p[0]
    else:
        order = np.argsort(-p)
        items = [(int(x), float(p[x])) for x in order[:K + 1] if x != 0]
        p0 = float(p[0])
    best = items[0] if items else (None, -np.inf)
    cands = [x for x, v in items[:K] if v > theta * p0]
    return best, cands


def final_estimate(backend, z, shots, rng, confidence=0.99):
    """Fresh DFE of f(z) with a normal CI and a one-sided Hoeffding lower bound."""
    e = FreshDFE(backend, rng).estimate(z, shots)
    from scipy.stats import norm
    zq = norm.ppf(0.5 + confidence / 2)
    ci = (e.value - zq * e.stderr, e.value + zq * e.stderr)
    lb = e.value - math.sqrt(2 * math.log(1 / (1 - confidence)) / shots)
    return e.value, e.stderr, ci, lb


def self_guided_estimation(backend, start=None, *, N=1000, K=3, retries=2,
                           final_shots=10_000, confidence=0.99, store=None,
                           substrings=False, max_iter=200, seed=None, rng=None,
                           walsh_block=None, beam=32, theta=THETA, store_kw=None,
                           final=True, keep_path=True, refresh_on_retry=False,
                           callback=None, confirm_flips=True, local_restart=None,
                           local_every=5):
    """Run the self-guided estimation protocol.

    Parameters
    ----------
    backend : object with ``n`` and ``measure(keys, rng)`` (see backends.py).
    start   : StabTableau to start from (the target state of the experiment);
              a random stabilizer state if None.
    store   : None (DFE without store) or "store" (DFE with store).
    substrings : feed all substrings of every measured string to the store.
    refresh_on_retry : start an empty store before each retry, so that the
              decision to stop does not reuse the data that selected z.
    confirm_flips : a sign flip whose advantage hat p_x - hat p_0 is below the
              multiple-comparison margin sqrt(4 n ln 2) * stderr is accepted only
              after an independent fresh DFE of z and of phi_x confirms it.
    local_restart : k.  Every ``local_every`` iterations and whenever no move
              improves, a candidate is built from the stored strings of weight
              <= k (requires the store with substrings); the search jumps to it
              if its estimated fidelity is larger than hat p_0.
    callback : optional function callback(iteration, z, shots) called with the
              start (iteration 0) and after every iteration (simulations only).
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    rng_search = np.random.default_rng(rng.integers(2**62))
    rng_final = np.random.default_rng(rng.integers(2**62))
    n = backend.n
    z = StabTableau.random(n, rng) if start is None else start
    skw = dict(store_kw or {})
    if local_restart and store not in (None, False, "fresh", "none"):
        skw.setdefault("local_k", local_restart)
    est = make_estimator(store, backend, rng_search, substrings, **skw)
    tried = set()

    def try_local(z, p0, N):
        if not local_restart or not hasattr(est, "data"):
            return None
        from .local import local_candidate
        cand = local_candidate(*est.data.local_table(local_restart), n)
        if cand is None:
            return None
        key = cand.canonical()
        if key in tried or cand == z:
            return None
        tried.add(key)
        fc = est.estimate(cand, N).value
        return (cand, fc) if fc > p0 else None
    N0, tries, it = N, 0, 0
    log, path = [], [z] if keep_path else []
    if callback:
        callback(0, z, 0)
    for it in range(1, max_iter + 1):
        if callback and it > 1:
            callback(it - 1, z, est.shots)
        s0 = est.shots
        p, p0, e0 = walsh_step(est, z, N, walsh_block, beam)
        (bx, bv), cands = _top(p, K, theta)
        if local_restart and it % local_every == 0:
            jump = try_local(z, p0, N)
            if jump is not None:
                z = jump[0]
                log.append(dict(it=it, action="local", x=-1, m=-1, f_hat=jump[1], N=N,
                                shots=est.shots - s0, cum_shots=est.shots))
                N, tries = N0, 0
                continue
        flip = bx is not None and bv > p0
        if flip and confirm_flips and bv - p0 < math.sqrt(4 * n * math.log(2)) * e0.stderr:
            fresh = FreshDFE(backend, rng_search)
            a = fresh.estimate(z.flip(bx), N).value
            b = fresh.estimate(z, N).value
            est.shots += fresh.shots
            flip = a > b
        if flip:
            z = z.flip(bx)
            log.append(dict(it=it, action="flip", x=int(bx), m=-1, f_hat=bv, N=N,
                            shots=est.shots - s0, cum_shots=est.shots))
            if keep_path:
                path.append(z)
            N, tries = N0, 0
            continue
        moved = None
        for x in cands:
            V = [z.neighbor(x, m) for m in range(4)]
            f = [est.estimate(v, N).value for v in V]
            j = int(np.argmax(f))
            if f[j] > p0:
                moved = (x, j, f[j], V[j])
                break
        if moved is not None:
            x, j, fj, z = moved
            log.append(dict(it=it, action="move", x=int(x), m=j, f_hat=fj, N=N,
                            shots=est.shots - s0, cum_shots=est.shots))
            if keep_path:
                path.append(z)
            N, tries = N0, 0
        elif local_restart and (jump := try_local(z, p0, N)) is not None:
            z = jump[0]
            log.append(dict(it=it, action="local", x=-1, m=-1, f_hat=jump[1], N=N,
                            shots=est.shots - s0, cum_shots=est.shots))
            N, tries = N0, 0
        elif tries < retries:
            if refresh_on_retry and store not in (None, False, "fresh", "none"):
                # the decision to stop is taken with data that did not select z
                old = est
                est = make_estimator(store, backend, rng_search, substrings, **skw)
                est.shots, est.requested = old.shots, old.requested
            log.append(dict(it=it, action="retry", x=-1, m=-1, f_hat=p0, N=N,
                            shots=est.shots - s0, cum_shots=est.shots))
            N *= 2
            tries += 1
        else:
            log.append(dict(it=it, action="stop", x=-1, m=-1, f_hat=p0, N=N,
                            shots=est.shots - s0, cum_shots=est.shots))
            break
    if callback:
        callback(it, z, est.shots)
    if final:
        F, se, ci, lb = final_estimate(backend, z, final_shots, rng_final, confidence)
        fs = final_shots
    else:
        F, se, ci, lb, fs = float("nan"), float("nan"), (float("nan"),) * 2, float("nan"), 0
    return SearchResult(state=z, F_hat=F, stderr=se, ci=ci, lower_bound=lb,
                        certified=bool(lb > COS2_PI8), search_shots=est.shots,
                        final_shots=fs, iterations=it, path=path, log=log,
                        reuse_rate=est.reuse_rate, requested_shots=est.requested, store=est)
