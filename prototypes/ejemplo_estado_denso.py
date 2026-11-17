"""
Ejemplo: algoritmo de umbralización aplicado a un estado Haar-aleatorio (denso en la base de Pauli).
Se compara con el óptimo exacto (enumeración de los 36720 estados estabilizadores de 4 qubits)
y con la cota superior F_STAB <= 2^{-M_2/4}.
"""
import itertools
import numpy as np
from estabilizador_cercano import pauli_spectrum, closest_stabilizer, popcounts, apply_pauli


def label(a, b, n):
    return "".join("IZXY"[2 * ((a >> (n - 1 - q)) & 1) + ((b >> (n - 1 - q)) & 1)] for q in range(n))


def build_state(gens, n):
    pc, rng = popcounts(2 ** n), np.random.default_rng(0)
    phi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    for a, b, s in gens:
        phi = (phi + s * apply_pauli(phi, a, b, pc)) / 2
    return phi / np.linalg.norm(phi)


def best_signs(psi, gens, n):
    """Mismo grupo, mejor elección de signos (2^n estados de la base estabilizadora)."""
    cands = [[(a, b, s) for (a, b, _), s in zip(gens, sg)] for sg in itertools.product([1, -1], repeat=n)]
    return max((abs(np.vdot(build_state(g, n), psi)) ** 2, g) for g in cands)


def all_stabilizer_states(n):
    """Enumeración exacta vía forma cuadrática: soporte afín + fases i^{l·u}(-1)^{h·u+u^T Q u}."""
    N, states, layers = 2 ** n, [], [{frozenset([0]): []}]
    for _ in range(n):
        nxt = {}
        for V, basis in layers[-1].items():
            for v in range(1, N):
                if v not in V:
                    nxt.setdefault(frozenset(V | {x ^ v for x in V}), basis + [v])
        layers.append(nxt)
    for k, layer in enumerate(layers):
        us = np.array(list(itertools.product([0, 1], repeat=k)), dtype=int).reshape(2 ** k, k)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        for V, basis in layer.items():
            xb = [np.bitwise_xor.reduce([basis[j] for j in range(k) if u[j]] + [0]) for u in us]
            reps = {}
            for c in range(N):
                reps.setdefault(frozenset(x ^ c for x in V), c)
            for c in reps.values():
                idx = [x ^ c for x in xb]
                for qb in itertools.product([0, 1], repeat=len(pairs)):
                    quad = sum((q * us[:, i] * us[:, j] for q, (i, j) in zip(qb, pairs)), np.zeros(2 ** k, int))
                    for h in itertools.product([0, 1], repeat=k):
                        for l in itertools.product([0, 1], repeat=k):
                            phi = np.zeros(N, complex)
                            phi[idx] = 1j ** (us @ np.array(l, int)) * (-1.0) ** (us @ np.array(h, int) + quad)
                            states.append(phi / np.sqrt(2 ** k))
    return np.array(states)


if __name__ == "__main__":
    n, N = 4, 16
    STAB = all_stabilizer_states(n)
    print(f"Estados estabilizadores de {n} qubits enumerados: {len(STAB)}")

    rng = np.random.default_rng(7)
    psi = rng.normal(size=N) + 1j * rng.normal(size=N)
    psi /= np.linalg.norm(psi)

    E = pauli_spectrum(psi, n)
    flat = [(abs(E[a, b]), a, b) for a in range(N) for b in range(N) if (a, b) != (0, 0)]
    flat.sort(reverse=True)
    print(f"\nPaulis con |<P>| > 0.05: {sum(v > 0.05 for v, _, _ in flat)} de 255  (estabilizador: 15)")
    print("Los 8 mayores:", ", ".join(f"{label(a, b, n)}:{E[a, b]:+.3f}" for _, a, b in flat[:8]))

    phi_g, gens = closest_stabilizer(psi, n)
    F_g = abs(np.vdot(phi_g, psi)) ** 2
    print("\nGeneradores elegidos:", ", ".join(f"{'+' if s > 0 else '-'}{label(a, b, n)} (<P>={E[a, b]:+.3f})" for a, b, s in gens))
    F_s, gens_s = best_signs(psi, gens, n)

    overlaps = np.abs(STAB.conj() @ psi) ** 2
    phi_opt = STAB[np.argmax(overlaps)]
    _, gens_opt = closest_stabilizer(phi_opt, n)
    M2 = -np.log2(np.sum(E ** 4) / N)
    print(f"\nF voraz (signos de <P>):       {F_g:.4f}")
    print(f"F voraz + mejores signos:      {F_s:.4f}")
    print(f"F_STAB exacta (fuerza bruta):  {overlaps.max():.4f}")
    print(f"Cota superior 2^(-M2/4):       {2 ** (-M2 / 4):.4f}   (M2 = {M2:.3f})")
    print("Grupo óptimo:", ", ".join(f"{'+' if s > 0 else '-'}{label(a, b, n)}" for a, b, s in gens_opt))

    print("\nEstadística sobre 300 estados Haar-aleatorios:")
    r_g, r_s, hit = [], [], 0
    for _ in range(300):
        p = rng.normal(size=N) + 1j * rng.normal(size=N)
        p /= np.linalg.norm(p)
        Fopt = (np.abs(STAB.conj() @ p) ** 2).max()
        ph, gg = closest_stabilizer(p, n)
        Fs, _ = best_signs(p, gg, n)
        r_g.append(abs(np.vdot(ph, p)) ** 2 / Fopt)
        r_s.append(Fs / Fopt)
        hit += Fs > Fopt - 1e-9
    print(f"  F/F_opt medio: voraz {np.mean(r_g):.3f}, con signos {np.mean(r_s):.3f}; peor caso {min(r_s):.3f}")
    print(f"  encuentra el óptimo exacto en {hit / 300:.0%} de los casos")
