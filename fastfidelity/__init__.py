"""fastfidelity: direct fidelity estimation (DFE) with a Pauli store and the
self-guided estimation protocol for the stabilizer fidelity.

The legacy tomography code (self-guided quantum tomography, Mean_Direct_Fidelity)
lives unchanged in the folder ``tomography_sgqt``.
"""
from . import paulis
from .tableau import StabTableau
from .backends import (pauli_spectrum, SpectrumBackend, StabilizerMixtureBackend,
                       QiskitSamplerBackend)
from .dfe import Estimate, FreshDFE, StoreDFE, make_estimator
from .search import self_guided_estimation, SearchResult, THETA, COS2_PI8
from . import reference

__version__ = "0.2.0"
