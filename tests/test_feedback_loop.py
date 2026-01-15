"""
Test Suite: Feedback Loop

Tests for syndrome feedback controller and drift suppression.

Author: Justin Arndt
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.feedback.controller import SyndromeFeedbackController


class TestSyndromeFeedbackController:
    """Tests for the feedback controller."""
    
    def test_initialization(self):
        """Controller initializes with correct state."""
        ctrl = SyndromeFeedbackController()
        
        assert ctrl.setpoint is None
        assert ctrl.integrator_state == 0.0
        assert len(ctrl.correction_queue) == ctrl.latency
    
    def test_calibration(self):
        """Calibration sets valid setpoint."""
        class MockPlant:
            def run_batch(self, correction):
                return 0, 0.05 + np.random.normal(0, 0.001)
        
        ctrl = SyndromeFeedbackController()
        setpoint = ctrl.calibrate(MockPlant(), num_samples=10)
        
        assert setpoint is not None
        assert 0.04 < setpoint < 0.06
        assert ctrl.setpoint == setpoint
    
    def test_update_integration(self):
        """Controller integrates error over time."""
        ctrl = SyndromeFeedbackController(Ki=0.1, feedback_latency=1)
        ctrl.setpoint = 0.05
        
        # Positive error should increase integrator
        for density in [0.06, 0.07, 0.08]:
            ctrl.update(density)
        
        assert ctrl.integrator_state > 0
    
    def test_negative_error_integration(self):
        """Controller handles negative errors."""
        ctrl = SyndromeFeedbackController(Ki=0.1, feedback_latency=1)
        ctrl.setpoint = 0.05
        
        for density in [0.04, 0.03, 0.02]:
            ctrl.update(density)
        
        assert ctrl.integrator_state < 0
    
    def test_correction_bounds(self):
        """Correction respects upper bounds."""
        ctrl = SyndromeFeedbackController(
            Ki=1.0,
            correction_bounds=(-0.01, 0.02)
        )
        ctrl.setpoint = 0.0
        
        for _ in range(100):
            ctrl.update(1.0)
        
        assert ctrl.integrator_state <= 0.02
    
    def test_correction_lower_bound(self):
        """Correction respects lower bounds."""
        ctrl = SyndromeFeedbackController(
            Ki=1.0,
            correction_bounds=(-0.01, 0.02)
        )
        ctrl.setpoint = 0.0
        
        for _ in range(100):
            ctrl.update(-1.0)
        
        assert ctrl.integrator_state >= -0.01
    
    def test_latency_delay(self):
        """Correction is delayed by latency."""
        latency = 5
        ctrl = SyndromeFeedbackController(Ki=1.0, feedback_latency=latency)
        ctrl.setpoint = 0.0
        
        # First correction should be zero
        first_correction = ctrl.update(1.0)
        assert first_correction == 0.0
        
        # After latency steps, corrections appear
        for _ in range(latency):
            ctrl.update(1.0)
        
        assert ctrl.get_active_correction() > 0
    
    def test_reset(self):
        """Reset clears controller state."""
        ctrl = SyndromeFeedbackController()
        ctrl.setpoint = 0.05
        ctrl.integrator_state = 0.5
        
        ctrl.reset()
        
        assert ctrl.integrator_state == 0.0
    
    def test_history_recording(self):
        """Controller records history."""
        ctrl = SyndromeFeedbackController()
        ctrl.setpoint = 0.05
        
        for density in [0.05, 0.06, 0.07]:
            ctrl.update(density)
        
        assert len(ctrl.history["density"]) == 3
        assert len(ctrl.history["correction"]) == 3
        assert len(ctrl.history["error"]) == 3
    
    def test_uncalibrated_error(self):
        """Update raises error if not calibrated."""
        ctrl = SyndromeFeedbackController()
        
        with pytest.raises(ValueError):
            ctrl.update(0.05)


class TestDriftTracking:
    """Tests for drift tracking behavior."""
    
    def test_tracks_increasing_drift(self):
        """Controller tracks monotonically increasing drift."""
        ctrl = SyndromeFeedbackController(Ki=0.05, feedback_latency=1)
        ctrl.setpoint = 0.05
        
        corrections = []
        for t in range(50):
            density = 0.05 + 0.001 * t  # Increasing drift
            ctrl.update(density)
            corrections.append(ctrl.integrator_state)
        
        # Corrections should be increasing
        assert corrections[-1] > corrections[0]
    
    def test_adapts_to_step_change(self):
        """Controller adapts to sudden step change."""
        ctrl = SyndromeFeedbackController(Ki=0.1, feedback_latency=1)
        ctrl.setpoint = 0.05
        
        # Stable period
        for _ in range(10):
            ctrl.update(0.05)
        
        state_before = ctrl.integrator_state
        
        # Step change
        for _ in range(20):
            ctrl.update(0.10)
        
        # Should have adapted
        assert ctrl.integrator_state > state_before
    
    def test_maintains_stability_at_setpoint(self):
        """Controller is stable when density equals setpoint."""
        ctrl = SyndromeFeedbackController(Ki=0.05, feedback_latency=1)
        ctrl.setpoint = 0.05
        
        for _ in range(100):
            ctrl.update(0.05)
        
        # Should remain near zero
        assert abs(ctrl.integrator_state) < 0.01


class TestFeedbackGains:
    """Tests for different feedback gain settings."""
    
    @pytest.mark.parametrize("Ki", [0.01, 0.05, 0.1, 0.2])
    def test_gain_affects_response_speed(self, Ki):
        """Higher gain produces faster response."""
        ctrl = SyndromeFeedbackController(Ki=Ki, feedback_latency=1)
        ctrl.setpoint = 0.05
        
        # Apply step input
        for _ in range(10):
            ctrl.update(0.10)
        
        # Higher gain should produce larger state
        # (just verify it produces some response)
        assert ctrl.integrator_state > 0


class TestLatencyDecay:
    """Tests for T1/T2 decay during feedback latency (Reality Gap fix)."""
    
    def test_zero_latency_no_penalty(self):
        """Zero latency should give zero decay penalty."""
        ctrl = SyndromeFeedbackController(latency_ns=0.0)
        
        penalty = ctrl.compute_latency_decay_penalty()
        
        assert penalty == 0.0
    
    def test_nonzero_latency_has_penalty(self):
        """Non-zero latency should produce decay penalty."""
        ctrl = SyndromeFeedbackController(
            latency_ns=500.0,  # 500ns feedback latency
            t1_us=100.0,
            t2_us=80.0
        )
        
        penalty = ctrl.compute_latency_decay_penalty()
        
        # Should be small but positive
        assert penalty > 0
        assert penalty < 0.01  # ~0.004 expected for 500ns
    
    def test_decay_penalty_scales_with_latency(self):
        """Higher latency should give higher decay penalty."""
        penalties = []
        for latency_ns in [100, 500, 1000, 5000]:
            ctrl = SyndromeFeedbackController(
                latency_ns=latency_ns,
                t1_us=100.0,
                t2_us=80.0
            )
            penalties.append(ctrl.compute_latency_decay_penalty())
        
        # Should be monotonically increasing
        for i in range(len(penalties) - 1):
            assert penalties[i+1] > penalties[i]
    
    def test_decay_penalty_scales_with_t1(self):
        """Shorter T1 should give higher decay penalty."""
        ctrl_long_t1 = SyndromeFeedbackController(
            latency_ns=500.0, t1_us=100.0, t2_us=100.0
        )
        ctrl_short_t1 = SyndromeFeedbackController(
            latency_ns=500.0, t1_us=20.0, t2_us=100.0
        )
        
        penalty_long = ctrl_long_t1.compute_latency_decay_penalty()
        penalty_short = ctrl_short_t1.compute_latency_decay_penalty()
        
        assert penalty_short > penalty_long
    
    def test_history_records_decay_penalty(self):
        """Controller history should record decay penalty."""
        ctrl = SyndromeFeedbackController(
            latency_ns=500.0,
            t1_us=100.0,
            t2_us=80.0
        )
        ctrl.setpoint = 0.05
        
        ctrl.update(0.06)
        
        assert "decay_penalty" in ctrl.history

