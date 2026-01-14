"""
Test Suite: Stim ↔ Cirq Bridge

Tests for circuit conversion, DEM extraction, and coherent noise injection.

Author: Justin Arndt
"""

import pytest
import numpy as np
import stim
import cirq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.hybrid.stim_cirq_bridge import (
    StimCirqBridge, NoiseModel, CoherentNoiseInjector
)


class TestNoiseModel:
    """Tests for unified noise model."""
    
    def test_default_values(self):
        """Default noise parameters are reasonable."""
        noise = NoiseModel()
        assert 0 < noise.depolarizing < 0.1
        assert 0 < noise.measurement < 0.1
        assert noise.coherent_overrotation >= 0
    
    def test_custom_values(self):
        """Custom noise values are preserved."""
        noise = NoiseModel(
            depolarizing=0.005,
            measurement=0.02,
            coherent_overrotation=0.01,
            zz_crosstalk=0.001
        )
        assert noise.depolarizing == 0.005
        assert noise.coherent_overrotation == 0.01


class TestStimCirqBridge:
    """Tests for Stim ↔ Cirq bridge."""
    
    @pytest.fixture
    def bridge(self):
        return StimCirqBridge(distance=3, rounds=3)
    
    def test_initialization(self, bridge):
        """Bridge initializes correctly."""
        assert bridge.distance == 3
        assert bridge.rounds == 3
        assert len(bridge.qubits) == 9  # 3x3
    
    def test_qubit_layout(self, bridge):
        """Qubits are grid qubits."""
        for q in bridge.qubits:
            assert isinstance(q, cirq.GridQubit)
    
    def test_cirq_circuit_generation(self, bridge):
        """Cirq circuit is generated successfully."""
        circuit = bridge.build_cirq_surface_code()
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit) > 0
    
    def test_cirq_circuit_with_noise(self, bridge):
        """Cirq circuit includes noise when specified."""
        noise = NoiseModel(depolarizing=0.01, coherent_overrotation=0.02)
        circuit = bridge.build_cirq_surface_code(noise)
        
        # Check that circuit has gates
        assert len(circuit) > 0
    
    def test_stim_circuit_generation(self, bridge):
        """Stim circuit is generated successfully."""
        noise = NoiseModel()
        stim_circuit = bridge.cirq_to_stim(None, noise)
        
        assert isinstance(stim_circuit, stim.Circuit)
        assert len(stim_circuit) > 0
    
    def test_dem_extraction(self, bridge):
        """DEM is extracted from Stim circuit."""
        noise = NoiseModel()
        stim_circuit = bridge.cirq_to_stim(None, noise)
        dem = bridge.stim_to_dem(stim_circuit)
        
        assert isinstance(dem, stim.DetectorErrorModel)
    
    def test_stim_sampling(self, bridge):
        """Stim sampling returns valid arrays."""
        noise = NoiseModel()
        stim_circuit = bridge.cirq_to_stim(None, noise)
        
        detection_events, observable_flips = bridge.sample_stim(stim_circuit, shots=100)
        
        assert detection_events.shape[0] == 100
        assert observable_flips.shape[0] == 100
    
    def test_syndrome_density_computation(self, bridge):
        """Syndrome density is computed correctly."""
        detection_events = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
        density = bridge.compute_syndrome_density(detection_events)
        
        assert 0 <= density <= 1
        assert abs(density - 0.5) < 0.01  # 6/12 = 0.5
    
    def test_cirq_coherent_simulation(self, bridge):
        """Cirq density matrix simulation runs."""
        circuit = bridge.build_cirq_surface_code()
        
        # Small subsystem for testing
        sub_circuit = cirq.Circuit([cirq.H(bridge.qubits[0])])
        result = bridge.simulate_cirq_coherent(sub_circuit)
        
        assert result.final_density_matrix is not None


class TestCoherentNoiseInjector:
    """Tests for coherent noise injection."""
    
    @pytest.fixture
    def qubits(self):
        return [cirq.GridQubit(0, i) for i in range(3)]
    
    @pytest.fixture
    def base_circuit(self, qubits):
        return cirq.Circuit([cirq.H(q) for q in qubits])
    
    def test_add_overrotation(self, base_circuit, qubits):
        """Over-rotation gates are added."""
        new_circuit = CoherentNoiseInjector.add_overrotation(
            base_circuit, qubits, angle=0.1
        )
        assert len(new_circuit) > len(base_circuit)
    
    def test_add_zz_crosstalk(self, base_circuit, qubits):
        """ZZ crosstalk gates are added."""
        pairs = [(qubits[0], qubits[1]), (qubits[1], qubits[2])]
        new_circuit = CoherentNoiseInjector.add_zz_crosstalk(
            base_circuit, pairs, strength=0.01
        )
        assert len(new_circuit) > len(base_circuit)
    
    def test_add_amplitude_damping(self, base_circuit, qubits):
        """Amplitude damping channel is added."""
        new_circuit = CoherentNoiseInjector.add_amplitude_damping(
            base_circuit, qubits, gamma=0.01
        )
        assert len(new_circuit) > len(base_circuit)


class TestDistanceScaling:
    """Tests for different code distances."""
    
    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_bridge_at_distance(self, distance):
        """Bridge works at different distances."""
        bridge = StimCirqBridge(distance=distance, rounds=3)
        
        assert len(bridge.qubits) == distance ** 2
        
        stim_circuit = bridge.cirq_to_stim(None, NoiseModel())
        det, obs = bridge.sample_stim(stim_circuit, shots=10)
        
        assert det.shape[0] == 10
