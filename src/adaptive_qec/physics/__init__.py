"""
Physics module for realistic quantum hardware modeling.

Contains:
- LeakageTracker: Models transmon |2⟩ state leakage and seepage
- BurstErrorDetector: Detects cosmic ray and burst error events
- CosmicRaySimulator: Simulates high-energy impact events
"""

from adaptive_qec.physics.leakage import LeakageTracker, LeakageState
from adaptive_qec.physics.burst_detector import BurstErrorDetector, BurstEvent
from adaptive_qec.physics.cosmic_ray import CosmicRaySimulator, CosmicRayImpact, inject_cosmic_ray

__all__ = [
    "LeakageTracker", "LeakageState",
    "BurstErrorDetector", "BurstEvent", 
    "CosmicRaySimulator", "CosmicRayImpact", "inject_cosmic_ray"
]
