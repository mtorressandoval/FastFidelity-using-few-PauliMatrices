"""Exact tests 1-4 of the plan (against Qiskit Pauli / DensityMatrix)."""
import numpy as np
import pytest
from qiskit.quantum_info import Pauli, DensityMatrix, random_density_matrix, random_statevector

from fastfidelity import paulis as pl
from fastfidelity.tableau import StabTableau
from fastfidelity.backends import pauli_spectrum, SpectrumBackend
from fastfidelity.walsh import fwht


def pmat(k, n):
    return pl.to_qiskit(k, n).to_matrix()


@pytest.mark.parametrize("n", [1, 2, 3])
def test_1_products_and_phases_match_qiskit(n):
    for k1 in range(4**n):
        P1 = pl.to_qiskit(k1, n)
        assert pl.from_qiskit(P1) == (k1, 1)
        for k2 in range(4**n):
            k, e = pl.mul(k1, k2, n)
            prod = P1.dot(pl.to_qiskit(k2, n))           # P1 @ P2
            assert np.allclose(prod.to_matrix(), (1j**e) * pmat(k, n))
            assert pl.symp(k1, k2, n) == (0 if P1.commutes(pl.to_qiskit(k2, n)) else 1)


@pytest.mark.parametrize("n,count", [(2, 12), (3, 28), (4, 60)])
def test_2_neighbors_distinct_overlap_half(n, count):
    rng = np.random.default_rng(n)
    for _ in range(5):
        z = StabTableau.random(n, rng)
        vz = z.statevector()
        seen = set()
        for x, m, v in z.neighbors():
            v.check()
            assert abs(abs(np.vdot(vz, v.statevector()))**2 - 0.5) < 1e-10
            seen.add(v.canonical())
        assert len(seen) == count


def _random_rho(n, rng, pure):
    if pure:
        return random_statevector(2**n, seed=int(rng.integers(1e9))).to_operator().data
    return random_density_matrix(2**n, rank=int(rng.integers(1, 2**n + 1)), seed=int(rng.integers(1e9))).data


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("pure", [True, False])
def test_3_neighbor_formula_cosets_and_walsh(n, pure):
    rng = np.random.default_rng(10 * n + pure)
    rho = _random_rho(n, rng, pure)
    E = pauli_spectrum(rho)
    for _ in range(2 if n > 4 else 4):
        z = StabTableau.random(n, rng)
        vz = z.statevector()
        p0 = np.vdot(vz, rho @ vz).real
        assert abs(z.fidelity_from_spectrum(E) - p0) < 1e-10
        pw = z.walsh_from_spectrum(E)
        xs = range(1, 2**n) if n <= 4 else rng.integers(1, 2**n, size=6)
        for x in xs:
            x = int(x)
            phix = pmat(z.q_of(x), n) @ vz
            px = np.vdot(phix, rho @ phix).real
            rx = np.vdot(vz, rho @ phix)
            assert abs(pw[x] - px) < 1e-10                       # Walsh formula for p_x
            assert abs(rx) <= np.sqrt(p0 * px) + 1e-12            # |r_x| <= sqrt(p0 px)
            # the four cosets of T_x
            kz, _, u = z.enumerate()
            Tx = set(int(k) for k, uu in zip(kz, u) if pl.popcount(int(uu) & x) % 2 == 0)
            i0 = next(i for i in range(n) if (x >> i) & 1)
            g0, q = z.g[i0], z.q_of(x)
            cosets = [Tx, {k ^ g0 for k in Tx}, {k ^ q for k in Tx}, {k ^ q ^ g0 for k in Tx}]
            assert set(int(k) for k in kz) == cosets[0] | cosets[1]
            for m in range(4):
                nu = z.neighbor(x, m)
                f = nu.fidelity_from_spectrum(E)
                assert abs(f - ((p0 + px) / 2 + (1j**m * rx).real)) < 1e-10
                kn = set(int(k) for k in nu.enumerate()[0])
                assert kn == (cosets[0] | cosets[2] if m % 2 == 0 else cosets[0] | cosets[3])


@pytest.mark.parametrize("pure", [True, False])
def test_4_substring_parity(pure):
    n = 3
    rng = np.random.default_rng(4 + pure)
    rho = _random_rho(n, rng, pure)
    B = SpectrumBackend(rho)
    D = DensityMatrix(rho)
    shots = 100_000
    for lab in ["XYZ", "ZZX", "YIY"]:
        P = pl.from_label(lab)[0]
        bits = B.measure(np.full(shots, P), rng)
        x, z = pl.split(P, n)
        sup = x | z
        for A in range(8):
            mask = A & sup
            Q = pl.join(x & mask, z & mask, n)
            est = np.mean(1 - 2 * (pl.popcount(bits & mask) & 1))
            exact = D.expectation_value(pl.to_qiskit(Q, n)).real
            assert abs(est - exact) < 4.5 / np.sqrt(shots)
