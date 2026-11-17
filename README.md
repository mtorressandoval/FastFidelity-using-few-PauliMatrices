# FastFidelityEstimation: stabilizer fidelity with Pauli measurements on one copy

This repository contains

* `fastfidelity/` – direct fidelity estimation (DFE) for stabilizer targets,
  without and with a Pauli store, and the **self-guided estimation protocol**
  that estimates the stabilizer fidelity
  F_STAB(ρ) = max_φ ⟨φ|ρ|φ⟩ of an unknown state ρ (pure or mixed) using only
  measurements of Pauli strings on single copies.
* `experiments/` – the scripts that produce every CSV and figure in `results/`
  (E1–E5 of the paper, n ≤ 8), plus `experiments/large/` with prepared runs for
  large n (stabilizer mixtures up to n = 100, Aer MPS and GPU samplers).
* `paper/` – the manuscript (REVTeX 4.2) and its bibliography.
* `tests/` – exact, statistical and control tests (`pytest`).
* `prototypes/` – the original prototypes used as reference (`vecinos_con_almacen.py`,
  `estabilizador_cercano.py`, `ejemplo_estado_denso.py`) and their output.
* `tomography_sgqt/` – the previous tomography code (self-guided quantum
  tomography), frozen and still working; see its README.
* `docs/` – research plan, writing profile and development notes.

## Installation

```bash
pip install -r requirements.txt        # qiskit 2.6.2, qiskit-aer 0.18.2
pytest -q                               # ~5 min (FF_STAT_REPS=1000 for a quick run)
```

## Minimal example

```python
import numpy as np
from fastfidelity import SpectrumBackend, StabTableau, self_guided_estimation
from fastfidelity.reference import near_stabilizer

rng = np.random.default_rng(1)
psi, phi = near_stabilizer(6, 0.9, rng)            # unknown state, F_STAB = 0.9
backend = SpectrumBackend(psi)                     # simulates Pauli measurements
res = self_guided_estimation(backend, start=None, store="store", seed=7)
print(res.F_hat, res.ci, res.certified, res.search_shots)
```

`store=None` gives the DFE without store; `store="store"` the DFE with store
(stratified estimator: stored Paulis of the group enter deterministically, the
rest of the group is sampled); `substrings=True` lets every shot of a string
feed all its substrings.  `fastfidelity.blocks.hierarchical_search` gives a
warm start from 2^k blocks of at most 8 qubits (`FrameBackend` defines the blocks
in the frame of a known Clifford circuit).  `local_restart=k` builds, from the stored substrings of weight <= k, a
candidate with k-local generators and jumps to it when it improves (exploratory,
off by default).  By default the protocol confirms sign flips whose
advantage is below the multiple-comparison margin (`confirm_flips=True`); for
large n use `N=125*n` and `refresh_on_retry=True`.  Backends: `SpectrumBackend` (dense ρ, n ≲ 11),
`StabilizerMixtureBackend` (analytic, hundreds of qubits),
`TDopedBackend` and `DepolarizedStabilizerBackend` (exact, large n),
`QiskitSamplerBackend` (Qiskit `StatevectorSampler` or Aer `SamplerV2` with
statevector, density-matrix, MPS or GPU methods).

## Reproducing the paper

```bash
cd experiments
python sim6_families.py "T-doped shallow" 10 15    # Fig. 2 (one call per family and n)
python sim2_extension.py 9 15 T-doped:1,2,3 --dense         # Fig. 3, n = 9 and 10
python sim2_extension.py 9 15 depolarized:0.1,0.3,0.5 --dense
python sim2_extension.py 10 15 T-doped:1,2,3 --dense
python sim2_extension.py 10 15 depolarized:0.1,0.3,0.5 --dense
python sim4_blocks.py 10 3 local2,global2,random                 # block warm start (exploratory, not in the paper)
python sim4_blocks.py 10 3 random frame
python sim4_blocks.py 16 3 local2,global2 blocks --scaled
python sim4_blocks.py 16 3 random frame --scaled
python sim5_local_restart.py 10 10 neighbors,random\ pairs,deep\ random local,plain   # local restart (exploratory)
python summary_local_restart.py                    # results/sim5_summary.txt, fig_local_restart.png
python worked_examples_fig.py      # worked examples of Appendix E (Fig. 3)
python fig1_schematic.py; NMAX=14 python make_figures.py   # Figs. 1-2 + results/summary.txt
cd ../paper && latexmk -pdf main.tex
```

`experiments/archive/` keeps the first round of simulations (store savings with a
start close to the optimum, success versus tau and p for n <= 5, dense states and
an ablation of the parameters) and the runs up to 16 qubits started close to
the optimum with N = 125 n (`archive/sim3_large.py`); their CSV files are in
`results/archive/`.
All randomness is seeded; on one CPU core the experiments take about one hour.
The main numbers are collected in `results/summary.txt`; the full test log is
`results/pytest_output.txt`.
