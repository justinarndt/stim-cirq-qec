"""
adaptive_qec: Hybrid Stim + Cirq Adaptive QEC Stack
====================================================

Combines:
- Stim: High-speed stabilizer sampling for large-scale Monte Carlo
- Cirq: Full density matrix simulation for coherent noise modeling  
- realtime-qec: Syndrome feedback, drift suppression, pulse remediation

Author: Justin Arndt
"""

from adaptive_qec.hybrid.stim_cirq_bridge import StimCirqBridge
from adaptive_qec.hybrid.adaptive_sampler import HybridAdaptiveSampler
from adaptive_qec.hybrid.realtime_surface import AdaptiveSurfaceCode
from adaptive_qec.feedback.controller import SyndromeFeedbackController
from adaptive_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
from adaptive_qec.remediation.pulse_synthesis import PulseSynthesizer

__version__ = "1.0.0"
__author__ = "Justin Arndt"

__all__ = [
    "StimCirqBridge",
    "HybridAdaptiveSampler",
    "AdaptiveSurfaceCode",
    "SyndromeFeedbackController",
    "HamiltonianLearner",
    "PulseSynthesizer",
]
