# Self-guided quantum tomography (legacy, frozen)

This folder keeps the original tomography code of the repository, unchanged,
so that every program still runs.  The project no longer develops it: the
paper and the new package `fastfidelity/` study only the estimation of the
stabilizer fidelity.

Contents
- `tomography/`: package with `Mean_Direct_Fidelity` (DFE with Qiskit Aer
  `Estimator`), `Random_Measurements`, `NearSparseTomography(_v2)`,
  `SelfGuidedTomography` and the `cspsa` optimizer.
- `SGQT_for_mixed.py`, `self_guided_tomo_*.ipynb`, `test_MLE_sparse.ipynb`,
  `test_bayes.ipynb`, `superfast_computation.ipynb`,
  `Direct_Fidelity_Estimation.ipynb`: experiments and examples.
- `data_plotter*.ipynb`: plots of the `.npz` files written by the experiments
  in `data/` (not versioned).

Run the notebooks and scripts from this folder (they import `tomography`
from the working directory).  Checked with qiskit 2.6.2 and qiskit-aer 0.18.2
(`tests/test_legacy.py`).

Removed during the reorganisation
- `FastFidelity with Stochastic Optimization.ipynb`: older copy of
  `self_guided_tomo_with_postprocessing.ipynb`.
- `self_guided_tomo_for_mixed_states.ipynb`: duplicate of `SGQT_for_mixed.py`.
- `near_sparse_tomography_random_measures.ipynb`: did not run
  (`psi_out, results = NearSparseTomography(...)`, but the function returns
  one array).
