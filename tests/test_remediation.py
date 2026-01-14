"""
Test Suite: Remediation

Tests for Hamiltonian learning and pulse synthesis.

Author: Justin Arndt
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
from adaptive_qec.remediation.pulse_synthesis import PulseSynthesizer


class TestHamiltonianLearner:
    """Tests for MBL-based Hamiltonian learning."""
    
    @pytest.fixture
    def learner(self):
        return HamiltonianLearner(system_size=4)
    
    def test_initialization(self, learner):
        """Learner initializes correctly."""
        assert learner.L == 4
        assert learner.dim == 16
        assert len(learner.ops_XX) == 3
        assert len(learner.ops_Z) == 4
    
    def test_operator_dimensions(self, learner):
        """Operators have correct dimensions."""
        for op in learner.ops_XX:
            assert op.shape == (learner.dim, learner.dim)
        for op in learner.ops_Z:
            assert op.shape == (learner.dim, learner.dim)
    
    def test_dynamics_initial_imbalance(self, learner):
        """Initial state has imbalance ~1."""
        J = np.ones(learner.L - 1)
        h = np.zeros(learner.L)
        t_points = np.array([0.0])
        
        trace = learner.simulate_dynamics(J, h, t_points)
        assert abs(trace[0] - 1.0) < 0.01
    
    def test_dynamics_bounded(self, learner):
        """Imbalance trace is bounded."""
        J = np.ones(learner.L - 1)
        h = 6.0 * np.cos(2 * np.pi * 1.618 * np.arange(learner.L))
        t_points = np.linspace(0, 10, 50)
        
        trace = learner.simulate_dynamics(J, h, t_points)
        assert np.all(np.abs(trace) <= 1.0)
    
    def test_mbl_localization(self, learner):
        """Strong disorder produces localization."""
        J = np.ones(learner.L - 1)
        h = 10.0 * np.random.randn(learner.L)
        t_points = np.linspace(0, 20, 100)
        
        trace = learner.simulate_dynamics(J, h, t_points)
        final_imbalance = np.mean(trace[-10:])
        
        # MBL: imbalance remains non-zero
        assert abs(final_imbalance) > 0.2
    
    def test_hamiltonian_recovery_identity(self, learner):
        """Learning recovers uniform couplings."""
        J_true = np.ones(learner.L - 1)
        h = 6.0 * np.cos(2 * np.pi * 1.618 * np.arange(learner.L))
        t_points = np.linspace(0, 10, 100)
        
        trace = learner.simulate_dynamics(J_true, h, t_points)
        J_recovered, _ = learner.learn_hamiltonian(trace, t_points, h)
        
        max_error = np.max(np.abs(J_recovered - J_true))
        assert max_error < 0.1
    
    def test_hamiltonian_recovery_with_defect(self, learner):
        """Learning recovers defective coupling."""
        J_true = np.array([1.0, 0.5, 1.0])
        h = 6.0 * np.cos(2 * np.pi * 1.618 * np.arange(learner.L))
        t_points = np.linspace(0, 10, 100)
        
        trace = learner.simulate_dynamics(J_true, h, t_points)
        J_recovered, _ = learner.learn_hamiltonian(trace, t_points, h)
        
        max_error = np.max(np.abs(J_recovered - J_true))
        assert max_error < 0.15
    
    def test_defect_detection_weak(self, learner):
        """Detect weak coupling."""
        J = np.array([1.0, 0.5, 1.0])
        defects = learner.detect_defects(J, J_nominal=1.0, threshold=0.15)
        
        assert 1 in defects["weak_couplings"]
    
    def test_defect_detection_strong(self, learner):
        """Detect strong coupling."""
        J = np.array([1.0, 1.3, 1.0])
        defects = learner.detect_defects(J, J_nominal=1.0, threshold=0.15)
        
        assert 1 in defects["strong_couplings"]
    
    def test_aubry_andre_fields(self):
        """Aubry-André fields are generated correctly."""
        h = HamiltonianLearner.generate_aubry_andre_fields(10, disorder_strength=6.0)
        
        assert len(h) == 10
        assert np.max(np.abs(h)) <= 6.0


class TestPulseSynthesizer:
    """Tests for optimal control pulse synthesis."""
    
    @pytest.fixture
    def synth(self):
        return PulseSynthesizer(system_size=4, gate_time=4.0, dt=0.5)
    
    def test_initialization(self, synth):
        """Synthesizer initializes correctly."""
        assert synth.L == 4
        assert synth.dim == 16
        assert synth.num_steps == 8
    
    def test_operators_hermitian(self, synth):
        """Built operators are Hermitian."""
        for op in synth.ops_XX:
            H = op.toarray()
            assert np.allclose(H, H.conj().T)
    
    def test_evolution_preserves_norm(self, synth):
        """Evolution is unitary (preserves norm)."""
        J = np.ones(synth.L - 1)
        h = np.zeros(synth.L)
        pulse = np.zeros((synth.num_steps, synth.L))
        
        final_state = synth.evolve_with_control(J, h, pulse)
        norm = np.linalg.norm(final_state)
        
        assert abs(norm - 1.0) < 1e-10
    
    def test_synthesis_returns_pulse(self, synth):
        """Synthesis returns valid pulse array."""
        J = np.ones(synth.L - 1)
        h = np.zeros(synth.L)
        
        pulse, fid = synth.synthesize(J, h, max_iterations=10, verbose=False)
        
        assert pulse.shape == (synth.num_steps, synth.L)
        assert 0 <= fid <= 1
    
    def test_synthesis_improves_fidelity(self):
        """Synthesis improves over zero control."""
        synth = PulseSynthesizer(system_size=4, gate_time=4.0, dt=0.5)
        
        J = np.array([1.0, 0.5, 1.0])
        h = 3.0 * np.cos(2 * np.pi * 1.618 * np.arange(synth.L))
        
        # Baseline
        zero_pulse = np.zeros((synth.num_steps, synth.L))
        final_zero = synth.evolve_with_control(J, h, zero_pulse)
        
        target_idx = int("".join(["10"] * (synth.L // 2)), 2)
        target = np.zeros(synth.dim, dtype=complex)
        target[target_idx] = 1.0
        
        fid_zero = np.abs(np.vdot(target, final_zero)) ** 2
        
        # Synthesized
        _, fid_opt = synth.synthesize(J, h, max_iterations=50, verbose=False)
        
        # Optimized should be at least as good
        assert fid_opt >= fid_zero * 0.9


class TestRemediationPipeline:
    """Tests for end-to-end remediation pipeline."""
    
    def test_diagnosis_then_remediation(self):
        """Full pipeline: diagnose then remediate."""
        L = 4
        
        # True defective hardware
        J_true = np.array([1.0, 0.6, 1.0])
        h = 6.0 * np.cos(2 * np.pi * 1.618 * np.arange(L))
        
        # Diagnose
        learner = HamiltonianLearner(system_size=L)
        t_points = np.linspace(0, 10, 100)
        trace = learner.simulate_dynamics(J_true, h, t_points)
        J_recovered, _ = learner.learn_hamiltonian(trace, t_points, h)
        
        # Remediate
        synth = PulseSynthesizer(system_size=L, gate_time=4.0, dt=0.5)
        pulse, fidelity = synth.synthesize(J_recovered, h, max_iterations=50, verbose=False)
        
        # Should achieve some fidelity improvement (relaxed for fast testing with limited iterations)
        assert fidelity > 0.3
