import numpy as np
from scipy.optimize import minimize
from .LinearAlgebra import InnerProductMatrices
from .cspsa import CSPSA
from .FastFidelity import Mean_Direct_Fidelity
# import cvxpy as cp

# def to_base(number, base, fill=None):
#     """Converts a non-negative number to a list of digits in the given base.

#     The base must be an integer greater than or equal to 2 and the first digit
#     in the list of digits is the most significant one.
#     """
#     if not number:
#         digits = [0]

#     else:
#         digits = []
#         while number:
#             digits.append(number % base)
#             number //= base
#     if fill:
#         for _ in range( fill-len(digits) ):
#             digits.append( 0 )

#     return list(reversed(digits))

# def exp_val_cvx( rho, j, MDF ):
#     ExpVal = cp.trace(LocalProduct_cvx(rho,
#                         [MDF.Sigmamu[k] 
#                         for k in to_base(j,4,MDF.NQ)]
#                         )) 
#     return ExpVal 

# def NearSparseTomography_v3(phi, MDF: Mean_Direct_Fidelity):

#     dim = phi.shape[0]
#     rho = cp.Variable((dim, dim), hermitian=True)
#     constraints = [rho >> 0] 
#     # constraints += [ cp.trace(rho) == 1 ] 
#     constraints += [ exp_val_cvx(rho,j,MDF)==MDF.Measures[j] 
#                     for j in MDF.Measures ] 
#     objective = cp.Minimize( cp.normNuc(rho) )
#     problema = cp.Problem( objective, constraints )
#     result = problema.solve( solver=cp.SCS, max_iters=100)
#     return rho.value #/cp.trace(rho.value)


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
        M   = MDF.Chi(psi, truncation=False)

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

def PositiveMatrix2CholeskyVector(rho):
    rho = rho + 1e-8*np.eye(rho.shape[0])
#     tm  = Cholesky(rho)
    tm  = np.linalg.cholesky(rho)
    return Triangular2Vector(tm)  

def Triangular2Vector(tm):
    d          = len(tm)
    idx        = 0
    cur_length = d
    t          = np.zeros(d**2)   
    for j in range(d):
        t[np.arange(idx,idx+cur_length)] = np.real(np.diag(tm,j))
        idx = idx + cur_length
        if j>0:
            t[np.arange(idx,idx+cur_length)] = np.imag(np.diag(tm,j))
            idx = idx + cur_length
        cur_length = cur_length -1
    return t    

def Vector2Triangular(t):
    d          = int(np.sqrt(len(t)))
    idx        = 0
    cur_length = d
    tm         = np.zeros([d,d])
    for j in range(int(d)):
        tm  = tm + 1*np.diag(t[np.arange(idx,idx+cur_length)],j)
        idx = idx + cur_length
        if j>0:
            tm     = tm + 1j*np.diag(t[np.arange(idx,idx+cur_length)],j)
            idx    = idx + cur_length
        cur_length = cur_length - 1
    return tm

def CholeskyVector2PositiveMatrix( t, mark = 0 ):
    # mark = 0, no trace constraint
    # mark > 0, trace = mark
    tm  = Vector2Triangular(t)
    rho = np.dot(tm.conj().transpose(),tm)
    rho = ( (mark==0) + mark*(mark>0) )*rho/( (mark==0) + np.trace(rho)*(mark>0) ) 
    return 0.5*( rho + rho.conj().T ) 

def NearSparseTomography_v2(phi, MDF: Mean_Direct_Fidelity):
    """
    phi : initial condition for the search
    MDF : FastFidelity or RandomMeasurements class
    """

    def CostF(t, MDF): 
        rho = CholeskyVector2PositiveMatrix( t, 1 ) 
        M_d = MDF.Measures
        M = InnerProductMatrices(rho, 
                                MDF.NQ * [MDF.Sigmamu]
                                ).reshape(-1).real / np.sqrt(MDF.d)
        f = 0 #- 0.1 * np.sum(M**2) 
        for j, Mj in enumerate(M): 
            if j in M_d:
                f += abs(M_d[j] - Mj) ** 2 / ( 1 - Mj**2 + 1e-10 )
            else:
                f += 0.1 * abs(Mj)**2 
        return np.real(f).squeeze() 

    t = PositiveMatrix2CholeskyVector( np.outer(phi,phi.conj()) )
    results = minimize(CostF, t, args=(MDF), tol=0.01)
    rho = CholeskyVector2PositiveMatrix( results.x, 1 ) 

    return rho


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
