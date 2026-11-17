"""
Búsqueda entre vecinos estabilizadores con DFE y SOLO mediciones de Pauli (un Pauli por disparo),
comparando dos rutinas de estimación de fidelidad:
  - sin almacén: cada llamada a DFE usa N disparos nuevos, uniformes sobre el grupo del estado objetivo;
  - con almacén: los resultados se guardan por Pauli (sin signo) y se reutilizan con un estimador
    estratificado por "átomos" (conjuntos de Paulis con el mismo historial de muestreo). Es insesgado y
    garantiza al menos la asignación proporcional de N disparos, así que su precisión no es peor.
Todo se representa con grupos de Paulis (sin vectores de estado), así que escala a n moderado.
Requiere estabilizador_cercano.py y ejemplo_estado_denso.py en la misma carpeta.
"""
import math
import numpy as np
from estabilizador_cercano import pauli_spectrum, random_stabilizer, near_stabilizer

POP = np.array([bin(i).count("1") for i in range(1 << 16)])
pc = lambda x: int(POP[x & 0xFFFF] + POP[x >> 16])

class Paulis:
    """Clave k = (a << n) | b  ↔  W_k = i^{a·b} X^a Z^b (hermítico)."""
    def __init__(self, n): self.n, self.m = n, (1 << n) - 1
    def mul(self, k1, k2):                      # W_k1 W_k2 = i^e W_k
        n, m = self.n, self.m
        a1, b1, a2, b2 = k1 >> n, k1 & m, k2 >> n, k2 & m
        a, b = a1 ^ a2, b1 ^ b2
        return (a << n) | b, (pc(a1 & b1) + pc(a2 & b2) + 2 * pc(b1 & a2) - pc(a & b)) % 4
    def sp(self, k1, k2):                       # 1 si anticonmutan
        n, m = self.n, self.m
        return (pc((k1 >> n) & (k2 & m)) + pc((k1 & m) & (k2 >> n))) & 1

def group(P, gens):
    """gens: [(clave, signo)] que conmutan. Devuelve {clave: (signo, u)} con u = máscara de generadores."""
    G = {0: (1, 0)}
    for i, (k, s) in enumerate(gens):
        new = dict(G)
        for kk, (ss, u) in G.items():
            k2, e = P.mul(kk, k)
            new[k2] = (ss * s * (1 if e == 0 else -1), u | (1 << i))
        G = new
    return G

def gens_from_signed(U):
    """Generadores independientes de un grupo dado como {clave: signo}."""
    basis, gens = {}, []
    for k in sorted(U, key=lambda k: pc(k)):
        w = k
        while w and (w.bit_length() - 1) in basis: w ^= basis[w.bit_length() - 1]
        if w:
            basis[w.bit_length() - 1] = w; gens.append((k, U[k]))
    return gens

def gens_from_state(vec, n):
    E = pauli_spectrum(vec / np.linalg.norm(vec), n).ravel()
    return gens_from_signed({int(k): int(np.sign(E[k])) for k in np.nonzero(np.abs(E) > 0.999)[0] if k})

def neighbors(P, G, gens, x):
    """Los 4 vecinos en la dirección x (patrón de anticonmutación con los generadores)."""
    n = P.n
    allk = np.arange(1 << (2 * n))
    rows = sum(((POP[(allk >> n) & (g & P.m)] + POP[(allk & P.m) & (g >> n)]) & 1) << i for i, (g, _) in enumerate(gens))
    q = int(np.nonzero(rows == x)[0][0])                                   # un Q_x
    T = {k: s for k, (s, u) in G.items() if pc(u & x) % 2 == 0}
    i0 = next(i for i in range(len(gens)) if (x >> i) & 1)
    g0, s0 = gens[i0]
    kh, e = P.mul(q, g0); sh = s0 * (1 if (e + 1) % 4 == 0 else -1)        # h = i Q g0 (hermítico)
    out = []
    for base, sb in [(q, 1), (kh, sh)]:
        C = {}
        for t, st in T.items():
            k2, e2 = P.mul(base, t); C[k2] = sb * st * (1 if e2 == 0 else -1)
        for sign in (1, -1):
            V = dict(T); V.update({k: sign * s for k, s in C.items()}); out.append(V)
    return out                                                              # [+Q, -Q, +iQg0, -iQg0]

class Fresh:
    """Rutina sin almacén."""
    def __init__(self, E, rng, N): self.E, self.rng, self.N, self.shots = E, rng, N, 0
    def est(self, U, w=None):
        keys = np.array(list(U)); pick = keys[self.rng.integers(len(keys), size=self.N)]
        o = np.where(self.rng.random(self.N) < (1 + self.E[pick]) / 2, 1, -1)
        s = np.array([U[k] for k in pick]); self.shots += self.N
        v = s * o
        return v.mean() if w is None else (np.array([w(k) for k in pick]) * v[:, None]).mean(0)

class Store:
    """Rutina con almacén: muestras por Pauli sin signo + estimador estratificado por átomos."""
    def __init__(self, E, rng, N):
        self.E, self.rng, self.N, self.shots = E, rng, N, 0
        self.origins, self.samples = [], {}
    def est(self, U, w=None):
        atoms = {}
        for k in U:
            atoms.setdefault(tuple(i for i, O in enumerate(self.origins) if k in O), []).append(k)
        total, val = len(U), 0.0
        for atom in atoms.values():
            req = math.ceil(self.N * len(atom) / total)
            have = sum(len(self.samples.get(k, ())) for k in atom)
            if have < req:                                                  # completar con disparos nuevos
                self.origins.append(frozenset(atom))
                for k in np.array(atom)[self.rng.integers(len(atom), size=req - have)]:
                    o = 1 if self.rng.random() < (1 + self.E[k]) / 2 else -1
                    self.samples.setdefault(int(k), []).append(o)
                self.shots += req - have
            ks = [k for k in atom for _ in self.samples.get(k, ())]
            os_ = np.array([o for k in atom for o in self.samples.get(k, ())])
            v = np.array([U[k] for k in ks]) * os_
            m = v.mean() if w is None else (np.array([w(k) for k in ks]) * v[:, None]).mean(0)
            val = val + len(atom) / total * m
        return val

def search(P, gens, r, K=3, max_iter=80, retries=0):
    """r: rutina de estimación (Fresh o Store). Si no hay mejora, puede reintentar con 2N, 4N, ...
    disparos antes de declarar un máximo local (con almacén, solo se miden los disparos que faltan)."""
    n = P.n; xs = np.arange(1 << n); N0 = r.N; tries = 0
    for _ in range(max_iter):
        G = group(P, gens); U = {k: s for k, (s, u) in G.items()}
        p = r.est(U, lambda k: (-1.0) ** POP[xs & G[k][1]])                # histograma de Walsh
        order = [x for x in np.argsort(-p) if x != 0][:K]
        if p[order[0]] > p[0]:                                              # cambio de signos
            gens = [(k, s * (-1 if (order[0] >> i) & 1 else 1)) for i, (k, s) in enumerate(gens)]
            r.N, tries = N0, 0; continue
        moved = False
        for x in order:
            V = neighbors(P, G, gens, int(x)); f = [r.est(v) for v in V]
            j = int(np.argmax(f))
            if f[j] > p[0]:
                gens = gens_from_signed(V[j]); moved = True; break
        if moved:
            r.N, tries = N0, 0
        elif tries < retries:
            r.N *= 2; tries += 1                                            # más precisión antes de parar
        else:
            break
    r.N = N0
    return gens

def exact_fid(P, gens, E):
    G = group(P, gens); return sum(s * E[k] for k, (s, u) in G.items()) / 2 ** P.n

def warm_start(P, gens, rng, steps):
    for _ in range(steps):
        G = group(P, gens); x = int(rng.integers(1, 1 << P.n))
        gens = gens_from_signed(neighbors(P, G, gens, x)[rng.integers(4)])
    return gens

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # --- comprobaciones con vectores de estado (n = 4)
    n = 4; P = Paulis(n)
    for _ in range(20):
        z = random_stabilizer(n, rng); psi = rng.normal(size=16) + 1j * rng.normal(size=16); psi /= np.linalg.norm(psi)
        E = pauli_spectrum(psi, n).ravel(); gens = gens_from_state(z, n); G = group(P, gens)
        assert abs(exact_fid(P, gens, E) - abs(np.vdot(z, psi)) ** 2) < 1e-9
        x = int(rng.integers(1, 16)); V = neighbors(P, G, gens, x)
        Ez = pauli_spectrum(z, n).ravel()
        for v in V:
            assert len(v) == 16
            assert abs(sum(s * Ez[k] for k, s in v.items()) / 16 - 0.5) < 1e-9      # solapamiento 1/2 con z
            g2 = gens_from_signed(v)
            assert len(group(P, g2)) == 16 and all(group(P, g2)[k][0] == s for k, s in v.items())
    print("Comprobaciones: fidelidades por grupo = vectores de estado; vecinos con solapamiento 1/2. OK")

    from ejemplo_estado_denso import all_stabilizer_states
    STAB4 = all_stabilizer_states(4)
    N = 1000
    VARIANTS = [("sin almacén", Fresh, 0), ("con almacén", Store, 0),
                ("sin almacén + 2 reintentos", Fresh, 2), ("con almacén + 2 reintentos", Store, 2)]
    cases = [(4, "tau=0.9, arranque aleatorio", 0.9, None, 15),
             (4, "Haar, arranque aleatorio", None, None, 15),
             (6, "tau=0.9, arranque a 3 pasos", 0.9, 3, 10),
             (8, "tau=0.9, arranque a 3 pasos", 0.9, 3, 8)]
    for n, name, tau, steps, R in cases:
        P = Paulis(n); res = {}
        for _ in range(R):
            if tau is None:
                psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n); psi /= np.linalg.norm(psi)
                Fst = (np.abs(STAB4.conj() @ psi) ** 2).max(); start = gens_from_state(random_stabilizer(n, rng), n)
            else:
                psi, phi = near_stabilizer(n, tau, rng); Fst = abs(np.vdot(phi, psi)) ** 2
                start = gens_from_state(random_stabilizer(n, rng), n) if steps is None else warm_start(P, gens_from_state(phi, n), rng, steps)
            E = pauli_spectrum(psi, n).ravel()
            seed = int(rng.integers(1 << 30))
            for label, cls, rt in VARIANTS:
                r = cls(E, np.random.default_rng(seed), N)
                g = search(P, start, r, retries=rt)
                res.setdefault(label, []).append((exact_fid(P, g, E) / Fst, r.shots))
        print(f"\nn={n}, {name} ({R} estados, N={N} disparos por estimación)")
        for label, v in res.items():
            v = np.array(v)
            print(f"  {label}: F/F_STAB {v[:,0].mean():.3f}, óptimo {np.mean(v[:,0] > 1-1e-6):.0%}, disparos medios {v[:,1].mean():9.0f}")
        for a_, b_ in [("sin almacén", "con almacén"), ("sin almacén + 2 reintentos", "con almacén + 2 reintentos")]:
            ratio = np.mean([a[1] / b[1] for a, b in zip(res[a_], res[b_])])
            print(f"  ahorro de disparos ({b_} frente a {a_}): {ratio:.1f}x")
