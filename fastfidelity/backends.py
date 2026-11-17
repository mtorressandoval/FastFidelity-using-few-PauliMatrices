"""State oracles that simulate single-shot measurements of Pauli strings.

Every backend implements

    measure(keys, rng) -> bits

where ``keys`` is an array of unsigned Pauli keys (one shot each) and ``bits``
holds, for each shot, the outcome bit of every qubit in the product eigenbasis
of the measured string (bit j = 1 means eigenvalue -1 of the letter on qubit
j; bits outside the support are 0).  The outcome of the string itself, or of
any substring Q <= P, is the parity of these bits on the support of Q.

``expectation(keys)`` returns exact values tr(rho W_k) and is used only for
references and tests, never by the estimators.

Backends
--------
SpectrumBackend          dense rho (n <= ~11): exact Pauli spectrum + exact
                         joint outcome distribution of every measured string.
StabilizerMixtureBackend rho = sum_i w_i |phi_i><phi_i| with stabilizer phi_i;
                         analytic, scales to hundreds of qubits.
QiskitSamplerBackend     circuits run with a Qiskit sampler (StatevectorSampler
                         or Aer SamplerV2 with statevector, density_matrix,
                         matrix_product_state methods, CPU or GPU).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from . import paulis as pl
from .walsh import fwht, fwht_axis
from .f2 import random_bits

__all__ = ["pauli_spectrum", "SpectrumBackend", "StabilizerMixtureBackend", "TDopedBackend",
           "DepolarizedStabilizerBackend",
           "QiskitSamplerBackend", "outcomes_from_bits"]


def outcomes_from_bits(bits, keys, n):
    """+-1 outcome of each measured string from the stored bits."""
    sup = pl.support(keys, n)
    return 1 - 2 * (pl.popcount(bits & sup) & 1)


# ----------------------------------------------------------------------------
# dense backend
# ----------------------------------------------------------------------------
def pauli_spectrum(state, n=None):
    """E[(x << n) | z] = tr(rho W_{x,z}) for a state vector or density matrix."""
    rho = np.asarray(getattr(state, "data", state))
    if rho.ndim == 1:
        psi = rho / np.linalg.norm(rho)
        N = psi.shape[0]
        n = int(np.log2(N))
        xs = np.arange(N)
        V = np.conj(psi[xs[None, :] ^ xs[:, None]]) * psi[None, :]    # V[x, y] = rho[y, y^x]
    else:
        N = rho.shape[0]
        n = int(np.log2(N))
        xs = np.arange(N)
        V = rho[xs[None, :], xs[None, :] ^ xs[:, None]]                # V[x, y] = rho[y, y^x]
    W = fwht_axis(V, axis=1)                                           # sum_y V[x,y](-1)^{z.y}
    ph = 1j ** pl.popcount(xs[:, None] & xs[None, :])
    return np.real(ph * W).ravel()


class SpectrumBackend:
    """Exact simulation of Pauli-string measurements on a dense state."""

    def __init__(self, state, n=None, cache_size=4000):
        self.E = pauli_spectrum(state, n)
        self.n = int(round(np.log2(self.E.shape[0]) / 2))
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def expectation(self, keys):
        return self.E[np.asarray(keys, dtype=np.int64)]

    def _dist(self, P):
        hit = self._cache.get(P)
        if hit is not None:
            self._cache.move_to_end(P)
            return hit
        n = self.n
        x, z = pl.split(P, n)
        sup = x | z
        pos = [j for j in range(n) if (sup >> j) & 1]
        w = len(pos)
        A = np.arange(1 << w, dtype=np.int64)
        mask = np.zeros(1 << w, dtype=np.int64)
        for t, j in enumerate(pos):
            mask |= ((A >> t) & 1) << j
        sub = ((x & mask) << n) | (z & mask)
        prob = fwht(self.E[sub]) / (1 << w)
        prob = np.clip(prob, 0, None)
        prob /= prob.sum()
        out = (prob, mask)          # mask[b] deposits local bits b into qubit positions
        self._cache[P] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def measure(self, keys, rng):
        keys = np.asarray(keys, dtype=np.int64)
        bits = np.zeros(keys.shape, dtype=np.int64)
        uniq, inv = np.unique(keys, return_inverse=True)
        for t, P in enumerate(uniq):
            idx = np.nonzero(inv == t)[0]
            prob, dep = self._dist(int(P))
            b = rng.choice(prob.shape[0], size=idx.size, p=prob)
            bits[idx] = dep[b]
        return bits


# ----------------------------------------------------------------------------
# mixtures of stabilizer states (analytic, large n)
# ----------------------------------------------------------------------------
class StabilizerMixtureBackend:
    """rho = sum_i w_i |phi_i><phi_i| with stabilizer phi_i (StabTableau)."""

    def __init__(self, tableaus, weights):
        self.T = list(tableaus)
        self.w = np.asarray(weights, dtype=float)
        self.w = self.w / self.w.sum()
        self.n = self.T[0].n
        self._cache = {}

    def expectation(self, keys):
        keys = np.asarray(keys)
        out = np.zeros(keys.shape)
        for T, w in zip(self.T, self.w):
            inside = T.in_group(keys)
            if inside.any():
                s = np.zeros(keys.shape)
                s[inside] = T.sign_of(keys[inside])
                out += w * s
        return out

    def _constraints(self, i, P):
        """Parity constraints of the outcome bits of P on component i."""
        key = (i, P)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        T, n = self.T[i], self.n
        x, z = pl.split(P, n)
        pos = [j for j in range(n) if ((x | z) >> j) & 1]
        letters = [(((x >> j) & 1) << (j + n)) | (((z >> j) & 1) << j) for j in pos]
        # kernel of A -> (symp(sum_{j in A} e_j, g_l))_l
        ech, kernel = {}, []
        for t, e in enumerate(letters):
            img = 0
            for l, g in enumerate(T.g):
                img |= pl.symp(e, g, n) << l
            combo = 1 << t
            while img:
                p = img.bit_length() - 1
                if p in ech:
                    img ^= ech[p][0]
                    combo ^= ech[p][1]
                else:
                    ech[p] = (img, combo)
                    break
            if img == 0:
                kernel.append(combo)
        # reduced row echelon form (pivot = lowest set bit, unique per column)
        rows = []
        for v in kernel:
            for r in rows:
                if (v >> (r & -r).bit_length() - 1) & 1:
                    v ^= r
            if v:
                p = (v & -v)
                rows = [(r ^ v) if (r & p) else r for r in rows]
                rows.append(v)
        targets = []
        for r in rows:
            kA = 0
            for t in range(len(pos)):
                if (r >> t) & 1:
                    kA ^= letters[t]
            s = T.sign_of(kA)
            targets.append(0 if s > 0 else 1)
        out = (pos, rows, targets)
        self._cache[key] = out
        return out

    def measure(self, keys, rng):
        keys = np.asarray(keys)
        n = self.n
        comp = rng.choice(len(self.T), size=keys.shape[0], p=self.w)
        big = n > 62
        bits = np.zeros(keys.shape[0], dtype=object if big else np.int64)
        pairs = {}
        for idx, (c, P) in enumerate(zip(comp.tolist(), keys.tolist())):
            pairs.setdefault((c, int(P)), []).append(idx)
        for (c, P), idx in pairs.items():
            pos, rows, targets = self._constraints(c, P)
            w = len(pos)
            idx = np.asarray(idx)
            if w <= 62:                                   # vectorised path
                b = random_bits(rng, idx.size, w).astype(np.int64)
                for r, t in zip(rows, targets):
                    par = pl.popcount(b & np.int64(r)) & 1
                    b = np.where(par != t, b ^ np.int64(r & -r), b)
                if big:
                    for i_shot, bb in zip(idx, b.tolist()):
                        full = 0
                        for tt, j in enumerate(pos):
                            if (bb >> tt) & 1:
                                full |= 1 << j
                        bits[i_shot] = full
                else:
                    full = np.zeros(idx.size, dtype=np.int64)
                    for tt, j in enumerate(pos):
                        full |= ((b >> tt) & 1) << j
                    bits[idx] = full
            else:
                for i_shot in idx:
                    bb = int(random_bits(rng, 1, w)[0])
                    for r, t in zip(rows, targets):
                        if (int(bb & r).bit_count() & 1) != t:
                            bb ^= (r & -r)
                    full = 0
                    for tt, j in enumerate(pos):
                        if (bb >> tt) & 1:
                            full |= 1 << j
                    bits[i_shot] = full
        return bits


# ----------------------------------------------------------------------------
# Qiskit samplers (statevector, density matrix, MPS, GPU)
# ----------------------------------------------------------------------------
class QiskitSamplerBackend:
    """Measure Pauli strings by running circuits on a Qiskit sampler.

    Parameters
    ----------
    circuit : QuantumCircuit preparing the state (no measurements).
    sampler : "statevector" (qiskit.primitives.StatevectorSampler) or "aer".
    method  : Aer method, e.g. "statevector", "density_matrix",
              "matrix_product_state" (MPS) or "stabilizer".
    device  : "CPU" or "GPU" (requires qiskit-aer-gpu).
    noise_model : optional Aer noise model (mixed states).
    """

    def __init__(self, circuit, sampler="aer", method="automatic", device="CPU",
                 noise_model=None, seed=None, backend_options=None):
        self.circuit = circuit
        self.n = circuit.num_qubits
        self.kind = sampler
        opts = dict(method=method, device=device)
        if noise_model is not None:
            opts["noise_model"] = noise_model
        if backend_options:
            opts.update(backend_options)
        self.backend_options = opts
        self.seed = seed
        self._calls = 0

    def _sampler(self, rng):
        seed = int(rng.integers(0, 2**31 - 1))
        if self.kind == "statevector":
            from qiskit.primitives import StatevectorSampler
            return StatevectorSampler(seed=seed)
        from qiskit_aer.primitives import SamplerV2
        return SamplerV2(seed=seed, options=dict(backend_options=self.backend_options))

    def measurement_circuit(self, P):
        from qiskit import QuantumCircuit, ClassicalRegister
        n = self.n
        lets = pl.letters_on(P, n)
        qc = self.circuit.copy()
        if not lets:
            return qc, []
        creg = ClassicalRegister(len(lets), "c")
        qc.add_register(creg)
        for j, c in lets:
            if c == "X":
                qc.h(j)
            elif c == "Y":
                qc.sdg(j)
                qc.h(j)
        for t, (j, _) in enumerate(lets):
            qc.measure(j, creg[t])
        return qc, [j for j, _ in lets]

    def measure(self, keys, rng):
        keys = np.asarray(keys)
        uniq, inv = np.unique(keys, return_inverse=True)
        pubs, poss, idxs = [], [], []
        for t, P in enumerate(uniq):
            idx = np.nonzero(inv == t)[0]
            qc, pos = self.measurement_circuit(int(P))
            idxs.append(idx)
            poss.append(pos)
            pubs.append((qc, None, int(idx.size)) if pos else None)
        bits = np.zeros(keys.shape[0], dtype=object if self.n > 62 else np.int64)
        run = [p for p in pubs if p is not None]
        res = self._sampler(rng).run(run).result() if run else []
        r = 0
        for pub, pos, idx in zip(pubs, poss, idxs):
            if pub is None:
                continue
            strings = res[r].data.c.get_bitstrings()
            r += 1
            for i_shot, s in zip(idx, strings):
                full = 0
                for t, j in enumerate(pos):
                    if s[-1 - t] == "1":
                        full |= 1 << j
                bits[i_shot] = full
        return bits


# ----------------------------------------------------------------------------
# T-doped states and depolarized stabilizer states (exact, large n)
# ----------------------------------------------------------------------------
def _kernel(vecs):
    """Basis (as bit masks over the input indices) of {A : XOR_{j in A} vecs[j] = 0}."""
    ech, ker = {}, []
    for j, v in enumerate(vecs):
        combo = 1 << j
        while v:
            p = v.bit_length() - 1
            if p in ech:
                v ^= ech[p][0]
                combo ^= ech[p][1]
            else:
                ech[p] = (v, combo)
                break
        if v == 0:
            ker.append(combo)
    return ker


def _rref_rows(rows):
    """RREF over F_2 of bit-mask rows; returns (rows, combos) with combos over the inputs."""
    out = []
    for i, v in enumerate(rows):
        c = 1 << i
        for r, rc in out:
            if (v >> ((r & -r).bit_length() - 1)) & 1:
                v ^= r
                c ^= rc
        if v:
            p = v & -v
            out = [((r ^ v, rc ^ c) if (r & p) else (r, rc)) for r, rc in out]
            out.append((v, c))
    return [r for r, _ in out], [c for _, c in out]


class TDopedBackend:
    """psi = C (T|+>)^{(x)t} (x) |0>^{(x)(n-t)}, with a Clifford C (qiskit Clifford).

    A Pauli string P measured on psi is the string R = C^dag P C measured on the
    product state m = (T|+>)^{(x)t} |0...0>.  For the letters of P, the outcome
    bits obey (i) deterministic parities from the substrings whose image is a
    signed Z-string on the |0> qubits, and (ii) at most 2t further parities with
    an exact joint distribution given by <X> = <Y> = 1/sqrt(2), <Z> = 0 on T|+>.
    ``fidelity(T)`` returns the exact <z|psi|z>^2 for a stabilizer tableau.
    """

    E1 = {0: 1.0, 1: np.cos(np.pi / 4), 2: 0.0, 3: np.sin(np.pi / 4)}   # letter code x + 2z: I, X, Z, Y

    def __init__(self, clifford, t):
        from qiskit.quantum_info import Pauli
        self.C = clifford
        self.n = clifford.num_qubits
        self.t = t
        n = self.n
        self.img = {}
        for j in range(n):
            for lab, (x, z) in (("X", (1, 0)), ("Z", (0, 1))):
                P = Pauli((np.eye(n, dtype=bool)[j] * bool(z), np.eye(n, dtype=bool)[j] * bool(x)))
                Q = P.evolve(clifford, frame="h")                  # C^dag P C
                self.img[(j, lab)] = pl.from_qiskit(Q)
            (ka, sa), (kb, sb) = self.img[(j, "X")], self.img[(j, "Z")]
            k, e = pl.mul(ka, kb, n)                              # Y = i X Z
            self.img[(j, "Y")] = (k, sa * sb * (1 if (1 + e) % 4 == 0 else -1))
        self._cache = {}
        self.hmask = (1 << t) - 1

    def _signed_prod(self, items):
        n = self.n
        k, s = 0, 1
        for kk, ss in items:
            k2, e = pl.mul(k, kk, n)
            if e % 2:
                raise ValueError("anticommuting")
            k, s = k2, s * ss * (1 if e == 0 else -1)
        return k, s

    def _decompose(self, R):
        """R: list of commuting signed images.  Returns L0 rows+targets, coset reps, c-values."""
        n, t = self.n, self.t
        feats_tail, feats_all = [], []
        for k, _ in R:
            x, z = pl.split(k, n)
            tx = x >> t
            feats_tail.append(tx)
            feats_all.append(tx | ((x & self.hmask) << (n - t)) | ((z & self.hmask) << n))
        L = _kernel(feats_tail)
        L0 = _kernel(feats_all)
        from .f2 import Basis
        b0 = Basis(L0)
        reps = [v for v in L if b0.add(v)]
        def prod(A):
            return self._signed_prod([R[j] for j in range(len(R)) if (A >> j) & 1])
        sign0 = [prod(A)[1] for A in L0]
        cvals = np.zeros(1 << len(reps))
        for c in range(1 << len(reps)):
            A = 0
            for i, r in enumerate(reps):
                if (c >> i) & 1:
                    A ^= r
            k, s = prod(A)
            x, z = pl.split(k, n)
            val = float(s)
            for q in range(t):
                val *= self.E1[((x >> q) & 1) + 2 * ((z >> q) & 1)]
            cvals[c] = val
        return L0, sign0, reps, cvals

    def fidelity(self, T):
        """Exact <z|rho|z> for a StabTableau z."""
        R = [self._image_key(k, s) for k, s in zip(T.g, T.s)]
        L0, sign0, reps, cvals = self._decompose(R)
        if any(s < 0 for s in sign0):
            return 0.0
        return float(2.0 ** (len(L0) - T.n) * cvals.sum())

    def _image_key(self, k, s=1):
        items = [self.img[(j, c)] for j, c in pl.letters_on(k, self.n)]
        kk, ss = self._signed_prod(items)
        return kk, ss * s

    def _setup(self, P):
        hit = self._cache.get(P)
        if hit is not None:
            return hit
        lets = pl.letters_on(P, self.n)
        pos = [j for j, _ in lets]
        R = [self.img[(j, c)] for j, c in lets]
        L0, sign0, reps, cvals = self._decompose(R)
        k = len(reps)
        py = np.clip(fwht(cvals) / (1 << k), 0, None) if k else np.ones(1)
        py = py / py.sum()
        rows, combos = _rref_rows(list(L0) + list(reps))
        tgt0 = np.array([(0 if s > 0 else 1) for s in sign0], dtype=np.int64)
        out = (pos, rows, combos, len(L0), tgt0, py)
        if len(self._cache) < 100000:
            self._cache[P] = out
        return out

    def measure(self, keys, rng):
        keys = np.asarray(keys)
        big = self.n > 62
        bits = np.zeros(keys.shape[0], dtype=object if big else np.int64)
        uniq, inv = np.unique(keys, return_inverse=True)
        for u, P in enumerate(uniq.tolist()):
            idx = np.nonzero(inv == u)[0]
            pos, rows, combos, n0, tgt0, py = self._setup(int(P))
            w, m = len(pos), idx.size
            k = len(py).bit_length() - 1
            y = rng.choice(len(py), size=m, p=py)
            # per-shot targets of the original rows: L0 (fixed) then reps (y bits)
            orig = np.zeros((m, n0 + k), dtype=np.int64)
            if n0:
                orig[:, :n0] = tgt0[None, :]
            for i in range(k):
                orig[:, n0 + i] = (y >> i) & 1
            b = random_bits(rng, m, w).astype(np.int64) if w else np.zeros(m, dtype=np.int64)
            for r, c in zip(rows, combos):
                tg = np.zeros(m, dtype=np.int64)
                for i in range(n0 + k):
                    if (c >> i) & 1:
                        tg ^= orig[:, i]
                par = pl.popcount(b & np.int64(r)) & 1
                b = np.where(par != tg, b ^ np.int64(r & -r), b)
            full = np.zeros(m, dtype=np.int64)
            for tt, j in enumerate(pos):
                full |= ((b >> tt) & 1) << j
            bits[idx] = full
        return bits


class DepolarizedStabilizerBackend:
    """rho = (1-p)|phi><phi| + p I/2^n with a stabilizer phi (StabTableau)."""

    def __init__(self, T, p):
        self.T, self.p, self.n = T, float(p), T.n
        self._stab = StabilizerMixtureBackend([T], [1.0])

    def fidelity(self, z):
        from .reference import stab_overlap
        return (1 - self.p) * stab_overlap(z, self.T) + self.p / 2**self.n

    def measure(self, keys, rng):
        keys = np.asarray(keys)
        bits = self._stab.measure(keys, rng)
        noisy = rng.random(keys.shape[0]) < self.p
        if noisy.any():
            sup = pl.support(keys[noisy], self.n)
            bits[noisy] = random_bits(rng, int(noisy.sum()), self.n) & sup
        return bits
