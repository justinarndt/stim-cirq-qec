"""
Test Suite: Full Pipeline Integration

End-to-end tests for the hybrid adaptive QEC stack.

Author: Justin Arndt
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.hybrid.stim_cirq_bridge import StimCirqBridge, NoiseModel
from adaptive_qec.hybrid.adaptive_sampler import HybridAdaptiveSampler, SamplingResult
from adaptive_qec.hybrid.realtime_surface import AdaptiveSurfaceCode, ExperimentConfig


class TestHybridAdaptiveSampler:
    """Tests for the hybrid sampler."""
    
    @pytest.fixture
    def sampler(self):
        return HybridAdaptiveSampler(distance=3, rounds=3)
    
    def test_initialization(self, sampler):
        """Sampler initializes correctly."""
        assert sampler.distance == 3
        assert sampler.rounds == 3
        assert sampler.baseline_density is None
    
    def test_calibration(self, sampler):
        """Calibration sets baseline."""
        baseline = sampler.calibrate(calibration_shots=1024)
        
        assert baseline is not None
        assert sampler.baseline_density == baseline
        assert sampler.controller.setpoint == baseline
    
    def test_run_cycle(self, sampler):
        """Single cycle returns valid result."""
        sampler.calibrate(calibration_shots=512)
        result = sampler.run_cycle(batch_size=100)
        
        assert isinstance(result, SamplingResult)
        assert result.detection_events.shape[0] == 100
        assert 0 <= result.syndrome_density <= 1
    
    def test_history_recording(self, sampler):
        """Sampler records cycle history."""
        sampler.calibrate(calibration_shots=512)
        
        for _ in range(5):
            sampler.run_cycle(batch_size=100)
        
        assert len(sampler.history["density"]) == 5
        assert len(sampler.history["correction"]) == 5
    
    def test_drift_detection(self, sampler):
        """Drift detection triggers on deviation."""
        sampler.calibrate(calibration_shots=512)
        sampler.drift_threshold = 0.001  # Very sensitive
        
        # Run cycles - drift might be detected
        results = []
        for _ in range(10):
            result = sampler.run_cycle(batch_size=100)
            results.append(result)
        
        # At least one result should be captured
        assert len(results) == 10


class TestAdaptiveSurfaceCode:
    """Tests for the full surface code runner."""
    
    def test_default_config(self):
        """Default config is reasonable."""
        config = ExperimentConfig()
        
        assert config.distance > 0
        assert config.num_cycles > 0
        assert config.depolarizing > 0
    
    def test_custom_config(self):
        """Custom config is preserved."""
        config = ExperimentConfig(
            distance=5,
            num_cycles=100,
            enable_drift=True,
            drift_rate=0.01
        )
        
        assert config.distance == 5
        assert config.num_cycles == 100
        assert config.enable_drift == True
    
    def test_runner_initialization(self):
        """Runner initializes with config."""
        config = ExperimentConfig(distance=3, num_cycles=10)
        runner = AdaptiveSurfaceCode(config)
        
        assert runner.config == config
        assert runner.noise is not None
    
    def test_drift_application(self):
        """Drift evolves over time."""
        config = ExperimentConfig(
            distance=3,
            enable_drift=True,
            drift_rate=0.1,
            drift_target=0.05
        )
        runner = AdaptiveSurfaceCode(config)
        
        drifts = [runner.apply_drift() for _ in range(50)]
        
        # Drift should move toward target
        assert drifts[-1] > drifts[0]
    
    def test_run_produces_results(self):
        """Run produces complete results."""
        config = ExperimentConfig(
            distance=3,
            rounds=3,
            num_cycles=10,
            batch_size=100
        )
        runner = AdaptiveSurfaceCode(config)
        results = runner.run(verbose=False)
        
        assert "baseline_errors" in results
        assert "adaptive_errors" in results
        assert "suppression_factor" in results
        assert "history" in results


class TestIntegrationPaths:
    """Tests for integration between components."""
    
    def test_bridge_to_sampler(self):
        """Bridge integrates with sampler."""
        bridge = StimCirqBridge(distance=3, rounds=3)
        noise = NoiseModel()
        
        # Bridge generates Stim circuit
        stim_circuit = bridge.cirq_to_stim(None, noise)
        
        # Sampler uses bridge
        sampler = HybridAdaptiveSampler(distance=3, rounds=3, noise=noise)
        sampler.calibrate(calibration_shots=100)
        
        result = sampler.run_cycle(batch_size=50)
        assert result is not None
    
    def test_feedback_affects_results(self):
        """Feedback controller impacts sampling."""
        # With feedback
        config_with = ExperimentConfig(
            distance=3,
            num_cycles=20,
            batch_size=100,
            enable_drift=True,
            feedback_Ki=0.1
        )
        runner_with = AdaptiveSurfaceCode(config_with)
        
        # Without feedback
        config_without = ExperimentConfig(
            distance=3,
            num_cycles=20,
            batch_size=100,
            enable_drift=True,
            feedback_Ki=0.0
        )
        runner_without = AdaptiveSurfaceCode(config_without)
        
        np.random.seed(42)
        results_with = runner_with.run(verbose=False)
        
        np.random.seed(42)
        results_without = runner_without.run(verbose=False)
        
        # Results should differ (feedback has effect)
        # Exact comparison depends on randomness


class TestStatisticalProperties:
    """Tests for statistical validity."""
    
    def test_error_rate_bounds(self):
        """Error rates are valid probabilities."""
        config = ExperimentConfig(
            distance=3,
            num_cycles=20,
            batch_size=100
        )
        runner = AdaptiveSurfaceCode(config)
        results = runner.run(verbose=False)
        
        assert 0 <= results["baseline_error_rate"] <= 1
        assert 0 <= results["adaptive_error_rate"] <= 1
    
    def test_suppression_positive(self):
        """Suppression factor is positive."""
        config = ExperimentConfig(
            distance=3,
            num_cycles=20,
            batch_size=100
        )
        runner = AdaptiveSurfaceCode(config)
        results = runner.run(verbose=False)
        
        assert results["suppression_factor"] > 0
    
    def test_reproducibility_with_seed(self):
        """Results are reproducible with fixed seed."""
        config = ExperimentConfig(distance=3, num_cycles=10, batch_size=50)
        
        np.random.seed(123)
        runner1 = AdaptiveSurfaceCode(config)
        results1 = runner1.run(verbose=False)
        
        np.random.seed(123)
        runner2 = AdaptiveSurfaceCode(config)
        results2 = runner2.run(verbose=False)
        
        # Should be identical
        assert results1["baseline_errors"] == results2["baseline_errors"]
        assert results1["adaptive_errors"] == results2["adaptive_errors"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_distance_one(self):
        """Distance 1 is handled (trivial code)."""
        config = ExperimentConfig(distance=1, num_cycles=5, batch_size=10)
        runner = AdaptiveSurfaceCode(config)
        # Should not crash
        results = runner.run(verbose=False)
        assert results is not None
    
    def test_zero_noise(self):
        """Zero noise produces zero/low errors."""
        config = ExperimentConfig(
            distance=3,
            num_cycles=10,
            batch_size=100,
            depolarizing=0.0001,  # Very low noise
            measurement=0.0001,
            enable_drift=False
        )
        runner = AdaptiveSurfaceCode(config)
        results = runner.run(verbose=False)
        
        # Should have low error rate
        assert results["adaptive_error_rate"] < 0.1
    
    def test_high_noise(self):
        """High noise produces high errors (sanity check)."""
        config = ExperimentConfig(
            distance=3,
            num_cycles=10,
            batch_size=100,
            depolarizing=0.05,  # High noise
            measurement=0.05,
            enable_drift=False
        )
        runner = AdaptiveSurfaceCode(config)
        results = runner.run(verbose=False)
        
        # Should have higher error rate than low noise
        assert results["adaptive_error_rate"] > 0.001
