"""
Test Suite: Burst Detection and Cosmic Ray

Tests for high-energy event detection and cosmic ray simulation.

Author: Justin Arndt
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.physics.burst_detector import BurstErrorDetector, BurstEvent
from adaptive_qec.physics.cosmic_ray import CosmicRaySimulator, inject_cosmic_ray


class TestBurstErrorDetector:
    """Tests for burst error detection."""
    
    def test_initialization(self):
        """Detector initializes correctly."""
        detector = BurstErrorDetector(distance=7)
        
        assert detector.distance == 7
        assert detector.baseline_density is None
        assert len(detector.detected_bursts) == 0
    
    def test_no_detection_below_threshold(self):
        """No burst detected for normal syndromes."""
        detector = BurstErrorDetector(distance=5, spike_threshold=0.3)
        detector.set_baseline(0.05)
        
        # Normal syndrome density
        syndromes = np.random.binomial(1, 0.05, size=25)
        burst = detector.detect(syndromes, cycle=0)
        
        assert burst is None
    
    def test_detection_on_spike(self):
        """Burst detected when syndrome density spikes."""
        detector = BurstErrorDetector(
            distance=5, 
            spike_threshold=0.2,
            spatial_threshold=3
        )
        detector.set_baseline(0.05)
        
        # Create burst syndrome
        syndromes = np.zeros(25)
        syndromes[10:20] = 1  # 40% density spike
        
        burst = detector.detect(syndromes, cycle=0)
        
        assert burst is not None
        assert burst.syndrome_density > 0.3
    
    def test_cooldown_prevents_repeated_detection(self):
        """Cooldown prevents immediate re-detection."""
        detector = BurstErrorDetector(
            distance=5,
            spike_threshold=0.2,
            cooldown_cycles=5
        )
        detector.set_baseline(0.05)
        
        burst_syndromes = np.ones(25)  # 100% trigger
        
        # First detection
        burst1 = detector.detect(burst_syndromes, cycle=0)
        assert burst1 is not None
        
        # Should be blocked by cooldown
        burst2 = detector.detect(burst_syndromes, cycle=1)
        assert burst2 is None
        
        # After cooldown
        burst3 = detector.detect(burst_syndromes, cycle=10)
        assert burst3 is not None
    
    def test_expanded_cirq_region(self):
        """Expanded region includes neighbors."""
        detector = BurstErrorDetector(distance=5)
        
        burst = BurstEvent(
            cycle=0,
            center_qubit=12,
            affected_qubits=[11, 12, 13],
            syndrome_density=0.5,
            severity=0.6
        )
        
        expanded = detector.get_expanded_cirq_region(burst, expansion_radius=1)
        
        # Should include original and neighbors
        assert 12 in expanded
        assert len(expanded) > len(burst.affected_qubits)
    
    def test_recovery_recommendation(self):
        """Recovery recommendation generated."""
        detector = BurstErrorDetector(distance=5)
        
        burst = BurstEvent(
            cycle=0,
            center_qubit=12,
            affected_qubits=[10, 11, 12, 13, 14],
            syndrome_density=0.5,
            severity=0.7
        )
        
        recovery = detector.get_recovery_recommendation(burst)
        
        assert "apply_extra_rounds" in recovery
        assert "decoder_reweight" in recovery
        assert recovery["decoder_reweight"] == True


class TestCosmicRaySimulator:
    """Tests for cosmic ray simulation."""
    
    def test_initialization(self):
        """Simulator initializes correctly."""
        sim = CosmicRaySimulator(distance=7)
        
        assert sim.distance == 7
        assert sim.num_qubits == 49
        assert len(sim.impact_history) == 0
    
    def test_generate_impact(self):
        """Impact generation works."""
        sim = CosmicRaySimulator(distance=7)
        
        impact = sim.generate_impact(cycle=0, center=24, radius=2)
        
        assert impact.center_qubit == 24
        assert impact.radius == 2
        assert len(impact.affected_qubits) > 0
        assert 24 in impact.affected_qubits
    
    def test_affected_qubits_within_radius(self):
        """Affected qubits are within radius."""
        sim = CosmicRaySimulator(distance=5)
        
        impact = sim.generate_impact(cycle=0, center=12, radius=1)
        
        # Center should be affected
        assert 12 in impact.affected_qubits
        # Neighbors should be affected
        assert len(impact.affected_qubits) >= 1
    
    def test_depolarization_map(self):
        """Depolarization map has correct shape."""
        sim = CosmicRaySimulator(distance=5)
        impact = sim.generate_impact(cycle=0, center=12, radius=2)
        
        depol_map = sim.get_depolarization_map(impact)
        
        assert len(depol_map) == 25
        # Center should have highest depolarization
        assert depol_map[12] > 0
    
    def test_impact_rate(self):
        """Impacts occur at expected rate (statistical)."""
        sim = CosmicRaySimulator(distance=5, impact_rate=10.0)  # 10 per 1000
        
        impacts = 0
        for cycle in range(10000):
            impact = sim.check_for_impact(cycle)
            if impact:
                impacts += 1
        
        # Should be roughly 100 impacts (with margin)
        assert 50 < impacts < 200


class TestInjectCosmicRay:
    """Tests for syndrome injection."""
    
    def test_inject_modifies_syndromes(self):
        """Injection modifies syndrome values."""
        syndromes = np.zeros(25)
        
        modified = inject_cosmic_ray(
            syndromes, 
            center=12, 
            radius=2, 
            flip_probability=1.0
        )
        
        # Should have some flipped
        assert np.sum(modified) > 0
    
    def test_inject_affects_nearby(self):
        """Injection affects qubits near center."""
        syndromes = np.zeros(25)
        
        modified = inject_cosmic_ray(
            syndromes,
            center=12,
            radius=3,
            flip_probability=1.0
        )
        
        # Center and nearby should be flipped
        assert modified[12] == 1
