"""
Estado estabilizador más cercano a un |psi> conocido, por umbralización del espectro de Pauli.

Garantía: si F_STAB(psi) > cos^2(pi/8) ≈ 0.854, el algoritmo recupera exactamente el
estabilizador más cercano: los Paulis de su grupo cumplen <P>^2 > 1/2 y el resto <P>^2 < 1/2
(Grewal, Iyer, Kretschmer, Liang, arXiv:2304.13915, Prop. 6.1, Prop. 6.3 y Cor. 6.4).
El umbral es óptimo. Por debajo es una heurística (buen punto de partida para búsqueda local).
Coste: O(n 4^n) para el espectro + O(n^2) por candidato en la eliminación gaussiana.
"""
import numpy as np


def popcounts(N):
    return np.array([bin(i).count("1") for i in range(N)])


def fwht(v, n):
    """Walsh–Hadamard sin normalizar: w[b] = sum_x v[x] (-1)^{b·x}."""
    t = v.reshape([2] * n)
    for ax in range(n):
        a, b = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
        t = np.stack([a + b, a - b], axis=ax)
    return t.reshape(-1)


def pauli_spectrum(psi, n):
    """E[a, b] = <psi| P_{a,b} |psi>, con P_{a,b} = i^{a·b} X^a Z^b (hermítico)."""
    N = 2 ** n
    xs, pc = np.arange(N), popcounts(N)
    E = np.empty((N, N))
    for a in range(N):
        w = fwht(np.conj(psi[xs ^ a]) * psi, n)      # <psi| X^a Z^b |psi> para todo b
        E[a] = np.real((1j ** pc[a & xs]) * w)
    return E


def apply_pauli(w, a, b, pc):
    xs = np.arange(len(w))
    w = w * (-1.0) ** pc[b & xs]                       # Z^b
    return (1j ** pc[a & b]) * w[xs ^ a]               # i^{a·b} X^a


def closest_stabilizer(psi, n):
    N, pc = 2 ** n, popcounts(2 ** n)
    E = pauli_spectrum(psi, n)
    gens, basis = [], {}
    for idx in np.argsort(-np.abs(E), axis=None):
        a, b = divmod(int(idx), N)
        if a == 0 and b == 0:
            continue
        if any((pc[a & b2] + pc[b & a2]) % 2 for a2, b2, _ in gens):
            continue                                   # no conmuta con lo ya elegido
        w = (a << n) | b                               # vector simpléctico en F_2^{2n}
        while w and (w.bit_length() - 1) in basis:
            w ^= basis[w.bit_length() - 1]
        if not w:
            continue                                   # dependiente
        basis[w.bit_length() - 1] = w
        gens.append((a, b, np.sign(E[a, b])))
        if len(gens) == n:
            break
    rng = np.random.default_rng(0)
    phi = rng.normal(size=N) + 1j * rng.normal(size=N)
    for a, b, s in gens:                               # proyector prod (I + s P)/2
        phi = (phi + s * apply_pauli(phi, a, b, pc)) / 2
    return phi / np.linalg.norm(phi), gens


def random_stabilizer(n, rng):
    """Estado estabilizador aleatorio vía forma cuadrática (soporte afín + fases)."""
    N = 2 ** n
    k = int(rng.integers(0, n + 1))
    cols, basis = [], {}
    while len(cols) < k:
        v = w = int(rng.integers(1, N))
        while w and (w.bit_length() - 1) in basis:
            w ^= basis[w.bit_length() - 1]
        if w:
            basis[w.bit_length() - 1] = w
            cols.append(v)
    c = int(rng.integers(0, N))
    Q = np.triu(rng.integers(0, 2, (k, k)), 1)
    h, l = rng.integers(0, 2, k), rng.integers(0, 2, k)
    phi = np.zeros(N, complex)
    for u in range(2 ** k):
        ub = np.array([(u >> j) & 1 for j in range(k)], dtype=int)
        x = c
        for j in range(k):
            x ^= cols[j] * int(ub[j])
        phi[x] = 1j ** int(l @ ub) * (-1) ** int(h @ ub + ub @ Q @ ub)
    return phi / np.linalg.norm(phi)


def near_stabilizer(n, tau, rng):
    phi = random_stabilizer(n, rng)
    chi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    chi -= np.vdot(phi, chi) * phi
    chi /= np.linalg.norm(chi)
    return np.sqrt(tau) * phi + np.sqrt(1 - tau) * chi, phi


if __name__ == "__main__":
    rng, n, trials = np.random.default_rng(1), 6, 200
    print(f"n = {n}, {trials} pruebas por tau (perturbación Haar ortogonal)")
    print(" tau   recuperado   F(phi_rec)/tau medio")
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        ok, ratio = 0, []
        for _ in range(trials):
            psi, phi = near_stabilizer(n, tau, rng)
            phi_rec, _ = closest_stabilizer(psi, n)
            ok += abs(np.vdot(phi_rec, phi)) ** 2 > 1 - 1e-9
            ratio.append(abs(np.vdot(phi_rec, psi)) ** 2 / tau)
        print(f" {tau:.2f}   {ok / trials:6.1%}      {np.mean(ratio):.3f}")
