import numpy as np


def Outer2Kron(A, Dims):
    # From vec(A) outer vec(B) to A kron B
    N = len(Dims)
    Dim = A.shape
    A = np.transpose(
        A.reshape(2 * Dims), np.array([range(N), range(N, 2 * N)]).T.flatten()
    ).flatten()
    return A.reshape(Dim)


def Kron2Outer(A, Dims):
    # From A kron B to vec(A) outer vec(B)
    N = len(Dims)
    Dim = A.shape
    A = np.transpose(
        A.reshape(np.kron(np.array([1, 1]), Dims)),
        np.array([range(0, 2 * N, 2), range(1, 2 * N, 2)]).flatten(),
    ).flatten()
    return A.reshape(Dim)


def Process2Choi(Process):
    dim = int(np.sqrt(Process[0, :].size))
    Process = np.transpose(Process.reshape([dim, dim, dim, dim]), [0, 2, 1, 3]).reshape(
        [dim**2, dim**2]
    )
    return Process


def LocalProduct(Psi, Operators, Dims=[]):
    """
    Calculate the product (A1xA2x...xAn)|psi>
    """
    sz = Psi
    if not Dims:
        Dims = [Operators[k].shape[-1] for k in range(len(Operators))]
    N = len(Dims)
    for k in range(N):
        Psi = ((Operators[k] @ Psi.reshape(Dims[k], -1)).T).flatten()
    return Psi


def InnerProductMatrices(X, B, Vectorized=False):
    """
    Calculate the inner product tr( X [B1xB2x...xBn])
    """
    X = np.array(X)

    if isinstance(B, list):
        B = B.copy()
        nsys = len(B)
        nops = []
        Dims = []
        if Vectorized == False:
            for j in range(nsys):
                B[j] = np.array(B[j])
                if B[j].ndim == 2:
                    B[j] = np.array([B[j]])
                nops.append(B[j].shape[0])
                Dims.append(B[j].shape[1])
                B[j] = B[j].reshape(nops[j], Dims[j] ** 2)
        elif Vectorized == True:
            for j in range(nsys):
                nops.append(B[j].shape[0])
                Dims.append(int(np.sqrt(B[j].shape[1])))

        if X.ndim == 2:
            TrXB = LocalProduct(Outer2Kron(X.flatten(), Dims), B)
        elif X.ndim == 3:
            TrXB = []
            for j in range(X.shape[0]):
                TrXB.append(LocalProduct(Outer2Kron(X[j].flatten(), Dims), B))
        elif X.ndim == 1:
            TrXB = LocalProduct(Outer2Kron(X, Dims), B)

        return np.array(TrXB).reshape(nops)

    elif isinstance(B, np.ndarray):
        if B.ndim == 2 and Vectorized == False:
            return np.trace(X @ B)

        elif B.ndim == 4:
            nsys = B.shape[0]
            nops = nsys * [B[0].shape[0]]
            Dims = nsys * [B[0].shape[1]]
            B = B.reshape(nsys, nops[0], Dims[0] ** 2)

        elif B.ndim == 3:
            if Vectorized == False:
                nsys = 1
                nops = B.shape[0]
                Dims = [B.shape[1]]
                B = B.reshape(nsys, nops, Dims[0] ** 2)
            if Vectorized == True:
                nsys = B.shape[0]
                nops = nsys * [B[0].shape[0]]
                Dims = nsys * [int(np.sqrt(B[0].shape[1]))]
        if X.ndim == 2:
            TrXB = LocalProduct(Outer2Kron(X.flatten(), Dims), B)
        elif X.ndim == 3:
            TrXB = []
            for j in range(X.shape[0]):
                TrXB.append(LocalProduct(Outer2Kron(X[j].flatten(), Dims), B))
        elif X.ndim == 1:
            TrXB = LocalProduct(Outer2Kron(X, Dims), B)

        return np.array(TrXB).reshape(nops)


def LinearCombinationMatrices(c, B, Vectorized=False):
    B = np.array(B)
    nsys = len(B)
    # nops = [ np.array(B[k]).shape[0] for k in range( nsys ) ]

    if Vectorized == False:
        Dims = [np.array(B[k]).shape[1] for k in range(nsys)]
        Bv = []
        for k in range(len(B)):
            Bv.append(B[k].reshape(-1, Dims[k] ** 2).T)
    else:
        Dims = [np.sqrt(np.array(B[k]).shape[1]) for k in range(nsys)]
        Bv = B

    Lambda = Kron2Outer(LocalProduct(c.flatten(), Bv), Dims).reshape(np.prod(Dims), -1)

    return Lambda


def QuantitiesRho(rho, Pi, p_ex, Vectorized=False):
    """
    Calculate  the log-likelihood multinomial function
    """
    p_th = InnerProductMatrices(rho, Pi, Vectorized=Vectorized)
    f = np.sum(-p_ex * np.log(p_th + 1e-12))
    return f


def VectorizeVectors(Vectors):
    if isinstance(Vectors, list):
        Vectors = np.array(Vectors)

    if Vectors.ndim == 2:
        Vectors = np.array(
            [np.outer(Vectors[j], Vectors[j].conj()) for j in range(Vectors.shape[0])]
        )
        return Vectors.reshape([Vectors.shape[0], -1]).T
    elif Vectors.ndim == 3:
        return Vectors.reshape([Vectors.shape[0], -1]).T


def HermitianPart(A):
    return 0.5 * (A + A.conj().T)


def Complex2Real(A):
    return np.array([np.real(A), np.imag(A)])


def Real2Complex(A):
    return A[0, :] + 1j * A[1, :]


def PartialTrace(rho, Systems, Subsystem):
    # Partial Trace, only works for bipartite systems
    rho = (
        rho.reshape(Systems + Systems)
        .transpose(0, 2, 1, 3)
        .reshape(np.array(Systems) ** 2)
    )

    if Subsystem == 0:
        rho = (np.eye(Systems[Subsystem]).reshape(1, -1) @ rho).flatten()

    elif Subsystem == 1:
        rho = (rho @ (np.eye(Systems[Subsystem]).reshape(-1, 1))).flatten()
    rho = rho.reshape(2 * [int(np.sqrt(rho.size))])

    #     if Subsystem == 0:
    #         rho = np.trace(rho.reshape(Systems+Systems), axis1=0, axis2=2)
    #     elif Subsystem == 1:
    #         rho = np.trace(rho.reshape(Systems+Systems), axis1=1, axis2=3)
    return rho
