import numpy as np
from itertools import product
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import random as rd
from .LinearAlgebra import InnerProductMatrices, LinearCombinationMatrices


class Mean_Direct_Fidelity:
    def __init__(self, NQ) -> None:
        self.NQ = NQ
        self.d = 2**NQ
        self.num_exp_meas = 0

        sigma0 = np.array([[1, 0], [0, 1]])
        sigma1 = np.array([[0, 1], [1, 0]])
        sigma2 = -np.array([[0, -1j], [1j, 0]])
        sigma3 = np.array([[1, 0], [0, -1]])
        Sigmamu = [sigma0, sigma1, sigma2, sigma3]
        self.Measures = {0: 1 / np.sqrt(self.d)}
        self.Sigmamu = Sigmamu

        SigmaL = ["I", "X", "Y", "Z"]
        ChainSigmaL = list(product(SigmaL, repeat=NQ))
        W = []
        for j in range(len(ChainSigmaL)):
            W.append("".join(ChainSigmaL[j]))

        self.W = W  # List of the form [II,IX, XI,IY,..]

    # ----------------------------------------------------------------
    def Expectationvalue(self, j, QuantumState, estimator=Estimator(), shots=1000):
        """
        QuantumState: QuantumState  as vector
        j: Index of the Pauli operator
        """
        job = estimator.run(
            QuantumState,
            SparsePauliOp.from_list([(self.W[j], 1)]),
            run_options={"shots": shots},
        )
        return (1 / np.sqrt(self.d)) * (job.result().values[0])

    # ----------------------------------------------------------------
    def FastTensorProd(self, A, x):
        # Given a list of matrices A = [A1,...,ANQ] and a vector x,
        # return (A1 o ... o ANQ)x where o is the Kronecker product
        L = len(x)
        x = x.reshape(L // 2, 2)
        for a in range(len(A) - 1):
            x = x @ A[-a - 1].T
        x = x.reshape(2, L // 2)
        x = A[0] @ x
        return x.flatten()

    # -----------------------------------------------------------------
    def Chi(self, x, truncation=None):
        """
        x: Pure state as vector
        """

        xc = x.conjugate()
        # alpha=(1/(self.d)**(0.5))
        # Chi = []
        # for A in product(self.Sigmamu, repeat=self.NQ):
        #     chi=(1/np.sqrt(self.d))* np.dot(xc,self.FastTensorProd(A,x))
        #     if truncation and np.sqrt(self.d)*np.abs(chi)<alpha:
        #         Chi.append(0)
        #     else:
        #         Chi.append(chi)
        Chix = InnerProductMatrices(np.outer(x, xc), self.NQ * [self.Sigmamu]).reshape(
            -1
        ) / np.sqrt(self.d)
        if truncation is not None:
            Chix[np.abs(Chix) < truncation] = 0
        return np.array(Chix)

    def truncation(self, x, alpha):
        xc = x.conjugate()
        Chi = InnerProductMatrices(np.outer(x, xc), self.NQ * [self.Sigmamu]).reshape(
            -1
        ) / np.sqrt(self.d)
        Chi[np.abs(Chi) < alpha] = 0
        return LinearCombinationMatrices(
            Chi, self.NQ * [np.array(self.Sigmamu).conj()]
        ) / np.sqrt(self.d)

    def DensityMatrix(self):
        coeff = np.array([self.Measures.get(j, 0) for j in range(4**self.NQ)])
        rho = LinearCombinationMatrices(
            coeff, self.NQ * [np.array(self.Sigmamu).conj()]
        ) / np.sqrt(self.d)
        return rho

    # -----------------------------------------------------------------
    def MeanFidelity(
        self,
        Nrepetitions,
        Npoints,
        x,
        QuantumState,
        estimator=Estimator(),
        shots=1000,
        truncation=None,
        stop_measuring=None,
    ):
        """
        Nrepetitions : Size to perform an average
        Npoints      : Number of points for the MonteCarlo integration
        x            : Pure state as vector
        QuantumState : Mixed state as QuantumCircuit
        """

        if stop_measuring is None:
            stop_measuring = lambda x: False

        x = x / np.linalg.norm(x)
        Chix = self.Chi(x, truncation)
        sums = []
        for i in range(Nrepetitions):
            kreduce = rd.choices(
                range(self.d**2), weights=(Chix.real) ** 2, k=Npoints
            )  # Montecarlo approximation with respect the probability Chix^2
            self._kreduce = kreduce
            for j in set(kreduce):
                if j not in self.Measures:  # Check if we already measure the operator j
                    if stop_measuring(self.Measures):
                        self.Measures[j] = 0
                    else:
                        self.Measures[j] = self.Expectationvalue(
                            j, QuantumState, estimator, shots
                        )
                        self.num_exp_meas += 1
            sum = 0
            for j in set(kreduce):  # Montecarlo sum
                sum += (
                    (1 / len(kreduce)) * (self.Measures[j] / Chix[j]) * kreduce.count(j)
                )
            sums.append(sum)
        sums = np.array(sums).real
        return np.sum(sums) / len(sums)


# -----------------------------------------------------------------


# ---------------------------
# ---------------------------
class Random_Measurements:
    def __init__(self, NQ) -> None:
        self.NQ = NQ
        self.d = 2**NQ

        sigma0 = np.array([[1, 0], [0, 1]])
        sigma1 = np.array([[0, 1], [1, 0]])
        sigma2 = -np.array([[0, -1j], [1j, 0]])
        sigma3 = np.array([[1, 0], [0, -1]])
        Sigmamu = [sigma0, sigma1, sigma2, sigma3]
        self.Measures = {0: 1 / np.sqrt(self.d)}
        self.Sigmamu = Sigmamu

        SigmaL = ["I", "X", "Y", "Z"]
        ChainSigmaL = list(product(SigmaL, repeat=NQ))
        W = []
        for j in range(len(ChainSigmaL)):
            W.append("".join(ChainSigmaL[j]))

        self.W = W  # List of the form [II,IX, XI,IY,..]

    def Expectationvalue(self, j, QuantumState, estimator=Estimator(), shots=1000):
        """
        QuantumState: QuantumState  as vector
        j: Index of the Pauli operator
        """
        job = estimator.run(
            QuantumState,
            SparsePauliOp.from_list([(self.W[j], 1)]),
            run_options={"shots": shots},
        )
        return (1 / np.sqrt(self.d)) * (job.result().values[0])

    def RandomMeasurements(
        self,
        num,
        QuantumState,
        estimator=Estimator(),
        shots=1000,
    ):
        kreduce = rd.sample(list(range(self.d**2)), num)

        for j in set(kreduce):
            if j not in self.Measures:  # Check if we already measure the operator j
                self.Measures[j] = self.Expectationvalue(
                    j, QuantumState, estimator, shots
                )

    def Chi(self, x, truncation=False):
        """
        x: Pure state as vector
        """
        xc = x.conjugate()
        alpha = 1 / (self.d) ** (0.5)
        # Chi = []
        # for A in product(self.Sigmamu, repeat=self.NQ):
        #     chi=(1/np.sqrt(self.d))* np.dot(xc,self.FastTensorProd(A,x))
        #     if truncation and np.sqrt(self.d)*np.abs(chi)<alpha:
        #         Chi.append(0)
        #     else:
        #         Chi.append(chi)
        Chi = InnerProductMatrices(np.outer(x, xc), self.NQ * [self.Sigmamu]).reshape(
            -1
        ) / np.sqrt(self.d)
        if truncation:
            Chi[np.sqrt(self.d) * np.abs(Chi) < alpha] = 0
        return np.array(Chi)
