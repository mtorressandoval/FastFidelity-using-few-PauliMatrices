# %%
import os

# %%
from qiskit_aer.primitives import Estimator
from qiskit_aer import AerSimulator 
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import  random_clifford, state_fidelity 
from tomography import SelfGuidedTomography, Mean_Direct_Fidelity, NearSparseTomography, NearSparseTomography_v2

# %%
from qiskit_aer.noise import NoiseModel, depolarizing_error 
error_1 = depolarizing_error( 1e-3, 1 )
error_2 = depolarizing_error( 5e-3, 2 )
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['h', 's', 'sd', 'x'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

# %%
NQs = [2,3,4,5,6,7,8]
L = len(NQs)
N = 100
shots = 1000
num_iter = 10

simulator =Estimator(backend_options={'shots':shots,
                                    'method':"stabilizer",
                                    'noise_model':noise_model,
                                    },
                    transpile_options={'optimization_level':0},
                    ) 

# %%
from joblib import Parallel, delayed

def simulate( NQ, simulator=simulator ):

    d = 2**NQ

    Omega  = random_clifford( NQ ).to_circuit() 

    #####
    qasm = AerSimulator(method='density_matrix',
                        noise_model=noise_model)
    Omega_density = Omega.copy()
    Omega_density.save_density_matrix()
    OmegaM_noisy = np.array( qasm.run(Omega_density).result().data()['density_matrix'] )
    #####

    # I_th = lambda x: 1 - np.vdot( x, OmegaM_noisy@x )/(np.linalg.norm( x ))**2

    MDF = Mean_Direct_Fidelity(NQ)

    stop_measuring = lambda x : ( np.linalg.norm( list( x.values() ) ) > 0.99 )

    I_ex = lambda x : 1 - MDF.MeanFidelity(1,  
                                        2*NQ**2,  
                                        x,  
                                        Omega, 
                                        simulator, 
                                        truncation= None, 
                                        stop_measuring = stop_measuring, 
                                        )

    def I_th( x ):

        x = x/np.linalg.norm( x )
        Is = []
        I_1 = 1 - np.vdot( x, OmegaM_noisy@x )/(np.linalg.norm( x ))**2
        Is.append( I_1 )

        F = 1- I_ex( x )
        FF = np.min( [(d*F-1)/(d-1), 1] )
        rho_white = FF * np.outer( x, x.conj() ) + (1-FF)*np.eye(d) /d
        rho_white = 0.5*rho_white+.5*rho_white.T.conj()
        rho_white = rho_white / np.trace( rho_white )
        I_2 = 1 - state_fidelity( rho_white , OmegaM_noisy ) 
        Is.append( I_2 )

        for rank in [3, d//2, d ]:
            phi = np.eye( d, rank, dtype=complex)
            rho = NearSparseTomography_v2( phi, MDF=MDF )
            rho = 0.5*rho + 0.5*rho.T.conj()
            rho = rho / np.trace( rho )
            I_3 = 1 - state_fidelity( rho, OmegaM_noisy ) 
            Is.append( I_3 )

        return Is

    Fidelities = []
    Measures = []
    Last=[]
    def callback( i, x ):
        Last.append(x)
        Fidelities.append( I_th(x) )
        Measures.append(len(MDF.Measures))
        return None

    # first level 
    psi0 = np.random.rand(d) + 1j * np.random.rand(d)
    psi0 = psi0 / np.linalg.norm(psi0)
    guess = psi0 

    postprocessing = lambda x : NearSparseTomography( x, MDF=MDF )

    SelfGuidedTomography( I_ex, 
                            guess, 
                            num_iter=num_iter, 
                            callback = callback,
                            postprocessing = postprocessing,
                            )

    Results=[np.array(Fidelities), 
                np.array(Measures), 
                np.array(Last),
                ] 

    return Results  #j index the average over the Hilbert space 


# %%
# simulate(2, simulator )

# %%
iterator = np.repeat(NQs, N).flatten()

R = Parallel(n_jobs=-1, verbose=11)(delayed(simulate)(int(NQ)) for NQ in iterator) 

# %%
Fids, Meas, _ = [[[R[i + j * N][r] for i in range(N)] for j in range(len(NQs))]
                    for r in range(3)]

Fids = np.real(Fids)
Meas = np.array(Meas)

# %%
# Save data. Ax order Fids and Meas are: [NQ, Nrun, Niter]

filename = "sgqt_for_mixed.npz"

data = {
    "N": [N],
    "shots": [shots],
    "Niter": [num_iter],
    "NQs": NQs,
    "Fids": Fids,
    "Meas": Meas,
    "info": ["N, shots, and Niter are single integers in one array each.",
             "NQs is the array of number of qubits.",
             "Fids and Meas have shapes (len(NQs), N, Niter)"],
}

# Make directory "data" if not present
os.makedirs("data", exist_ok=True)

# Save
filepath = os.path.join("data", filename)
np.savez(filepath, **data)

# %%



