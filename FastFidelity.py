import numpy as np
from itertools import product
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import random as rd


class Mean_Direct_Fidelity:

    def __init__(self, 
                NQ) -> None:
        
        self.NQ    = NQ
        self.d     = 2**NQ 

        sigma0 = np.array([[1, 0], [0, 1]])
        sigma1 = np.array([[0, 1], [1, 0]])
        sigma2 = np.array([[0, -1j], [1j, 0]])
        sigma3 = np.array([[1, 0], [0, -1]])
        Sigmamu=[sigma0, sigma1, sigma2, sigma3]
        self.Measures={}
        self.Sigmamu = Sigmamu 

        SigmaL=['I','X','Y','Z']
        ChainSigmaL=list(product(SigmaL, repeat=NQ))
        W=[]
        for j in range(len(ChainSigmaL)):
            W.append(''.join(ChainSigmaL[j])) 

        self.W = W #List of the form [II,IX, XI,IY,..]
#----------------------------------------------------------------    
    def Expectationvalue(self,
                            j, 
                            QuantumState,
                            estimator = Estimator(),
                            shots = 1000): 
        """ 
        QuantumState: QuantumState  as vector
        j: Index of the Pauli operator
        """
        job = estimator.run(QuantumState, 
                            SparsePauliOp.from_list([(self.W[j], 1)]), 
                            run_options = { 'shots' : shots } )
        
        return (1/np.sqrt(self.d))*(job.result().values[0])
#----------------------------------------------------------------        
    def FastTensorProd(self,A,x):
        #Given a list of matrices A = [A1,...,ANQ] and a vector x,
        #return (A1 o ... o ANQ)x where o is the Kronecker product
        L=len(x)
        x=x.reshape(L//2, 2)
        for a in range(len(A)-1):
            x=x@A[-a-1].T
        x=x.reshape(2,L//2)
        x=A[0]@x
        return x.flatten()
#-----------------------------------------------------------------    
    def Chi(self,x,truncation=False):
        """ 
        x: Pure state as vector
        """
        xc=x.conjugate()
        alpha=(1/(self.d)**(0.5))
        Chi = []
        for A in product(self.Sigmamu, repeat=self.NQ):
            chi=(1/np.sqrt(self.d))* np.dot(xc,self.FastTensorProd(A,x))
            if truncation and  np.sqrt(self.d)*np.abs(chi)<alpha:
                Chi.append(0)
            else:
                Chi.append(chi)     
        return np.array(Chi)
#-----------------------------------------------------------------    
    def MeanFidelity(self, 
                        Nrepetitions, 
                        Npoints, 
                        x, 
                        QuantumState,
                        estimator = Estimator(),
                        shots     = 1000,
                        truncation=False ):
        """
        Nrepetitions : Size to perform an average
        Npoints      : Number of points for the MonteCarlo integration 
        x            : Pure state as vector
        QuantumState : Mixed state as QuantumCircuit
        """
        Chix = self.Chi(x,
                        truncation)
        sums = []
        for i in range(Nrepetitions):
            kreduce = rd.choices(range( self.d**2), 
                                weights=(Chix.real)**2, 
                                k=Npoints) # Montecarlo approximation with respect the probability Chix^2
            for j in set(kreduce):
                if j not in self.Measures: #Check if we already measure the operator j
                    self.Measures[j] = self.Expectationvalue(j, 
                                                            QuantumState,
                                                            estimator,
                                                            shots) 
            sum = 0
            for j in set(kreduce): #Montecarlo sum
                sum += (1 / len(kreduce)) * (self.Measures[j]/ Chix[j])*kreduce.count(j)
            sums.append(sum)  
        sums = np.array(sums).real
        return np.sum(sums) / len(sums)
#-----------------------------------------------------------------    


