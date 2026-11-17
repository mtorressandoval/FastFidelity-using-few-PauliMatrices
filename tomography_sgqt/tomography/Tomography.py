import numpy as np
from scipy.optimize import minimize
from .LinearAlgebra import InnerProductMatrices
from .cspsa import CSPSA
from .FastFidelity import Mean_Direct_Fidelity


def NearSparseTomography(phi, MDF: Mean_Direct_Fidelity):
    """
    phi : initial condition for the search
    MDF : FastFidelity or RandomMeasurements class
    """

    phi = phi / np.linalg.norm(phi)
    phi = np.array([phi.real, phi.imag]).reshape(-1)

    def CostF(psi, MDF: Mean_Direct_Fidelity):
        psi = psi.reshape(2, -1)
        psi = psi[0] + 1j * psi[1]
        M_d = MDF.Measures
        M = MDF.Chi(psi, truncation=False)

        f = 0.0
        for j, Mj in enumerate(M):
            if j in M_d:
                f += abs(M_d[j] - Mj) ** 2 / ( 1 - Mj**2 + 1e-10 )
            else:
                f += 0.1 * abs(Mj) ** 2

        return np.real(f)

    results = minimize(CostF, phi, args=(MDF), tol=0.0001 )

    psi_hat = results.x / np.linalg.norm(results.x)
    psi_hat = psi_hat.reshape(2, -1)
    psi_hat = psi_hat[0] + 1j * psi_hat[1]

    return psi_hat 


def NearSparseTomography_v2(phi, MDF: Mean_Direct_Fidelity):
    """
    phi : initial condition for the search
    MDF : FastFidelity or RandomMeasurements class
    """
    phi = np.array([phi.real, phi.imag]).reshape(-1)

    def CostF(psi, MDF):
        psi = psi.reshape(2, MDF.d, -1)
        psi = psi[0] + 1j * psi[1]
        psi = psi @ psi.T.conj()
        psi = psi /np.trace( psi )

        M_d = MDF.Measures
        M = InnerProductMatrices(psi, MDF.NQ * [MDF.Sigmamu]).reshape(
            -1
        ).real / np.sqrt(MDF.d)
        f = 0
        for j, Mj in enumerate(M):
            if j in M_d:
                f += abs(M_d[j] - Mj) ** 2 / ( 1 - Mj**2 + 1e-10 )
            else:
                f += 0.1 * abs(Mj)**2
        return np.real(f).squeeze()

    # print( phi )
    # print("cost in", CostF(phi, MDF))
    results = minimize(CostF, phi, args=(MDF), tol=0.01)

    psi_hat = results.x
    # print("cost out", CostF(psi_hat, MDF))
    psi_hat = psi_hat.reshape(2, MDF.d, -1)
    psi_hat = psi_hat[0] + 1j * psi_hat[1]

    # psi_hat = psi_hat - eigenproj@psi_hat
    # psi_hat = psi_hat / np.linalg.norm( psi_hat, axis=0 )
    psi_hat = psi_hat @ psi_hat.T.conj()
    # psi_hat = eigenval * eigenproj + (1 - eigenval) * psi_hat / np.trace(psi_hat)
    # psi_hat = eigenproj + psi_hat @ psi_hat.T.conj()
    psi_hat = psi_hat /np.trace( psi_hat )

    return psi_hat


#############################################


def SelfGuidedTomography(
    infidelity,
    guess,
    num_iter,
    callback=lambda x, i: None,
    postprocessing=None,
    progressbar=False,
):
    GAINS = {
        "a": 3.0,
        "b": 0.2,
        "A": 0.0,
        "s": 1.0,
        "t": 1 / 6,
    }

    def update(guess, update):
        guess = guess + update
        guess = guess / np.linalg.norm(guess)
        if postprocessing is not None:
            guess = postprocessing(guess)
        return guess

    optimizer = CSPSA(
        callback=callback,
        gains=GAINS,
        apply_update=update,
    )

    results = optimizer.run(infidelity,
                            guess,
                            progressbar=progressbar,
                            num_iter=num_iter,
                            )

    return results
