"""
Test Suite: Leakage Modeling

Tests for transmon leakage to |2⟩ state and seepage recovery (LRU).

Author: Justin Arndt
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.physics.leakage import LeakageTracker, LeakageState


class TestLeakageTracker:
    """Tests for the LeakageTracker class."""
    
    def test_initialization(self):
        """Tracker initializes with correct state."""
        tracker = LeakageTracker(num_qubits=25)
        
        assert tracker.num_qubits == 25
        assert tracker.get_num_leaked() == 0
        assert len(tracker.history) == 0
    
    def test_no_leakage_at_zero_rate(self):
        """No leakage occurs when rate is zero."""
        tracker = LeakageTracker(num_qubits=100, leakage_rate=0.0)
        
        for _ in range(100):
            tracker.run_cycle()
        
        assert tracker.get_num_leaked() == 0
        assert tracker.total_leakage_events == 0
    
    def test_leakage_accumulates(self):
        """Leakage accumulates over cycles."""
        tracker = LeakageTracker(
            num_qubits=49,  # d=7
            leakage_rate=0.01,  # 1% per gate
            seepage_rate=0.0   # No recovery
        )
        
        leaked_counts = []
        for _ in range(50):
            tracker.run_cycle(num_gates_per_qubit=5)
            leaked_counts.append(tracker.get_num_leaked())
        
        # With no seepage, leaked count should be monotonically increasing
        assert leaked_counts[-1] > leaked_counts[0]
    
    def test_seepage_recovers_qubits(self):
        """Seepage can recover leaked qubits."""
        tracker = LeakageTracker(
            num_qubits=25,
            leakage_rate=0.0,  # No new leakage
            seepage_rate=1.0   # 100% seepage (extreme)
        )
        
        # Manually leak all qubits
        tracker.leaked[:] = True
        assert tracker.get_num_leaked() == 25
        
        # Run cycle - should recover all
        tracker.apply_seepage()
        assert tracker.get_num_leaked() == 0
    
    def test_equilibrium_with_seepage(self):
        """With seepage, leakage reaches equilibrium."""
        tracker = LeakageTracker(
            num_qubits=100,
            leakage_rate=0.005,
            seepage_rate=0.1  # Higher seepage = lower equilibrium
        )
        
        for _ in range(500):
            tracker.run_cycle()
        
        # Should reach some equilibrium < all qubits
        final_leaked = tracker.get_num_leaked()
        assert 0 < final_leaked < 50  # Not all, not none
    
    def test_error_contribution_zero_when_no_leakage(self):
        """Error contribution is zero with no leaked qubits."""
        tracker = LeakageTracker(num_qubits=49)
        
        assert tracker.get_leakage_error_contribution() == 0.0
    
    def test_error_contribution_increases_with_leakage(self):
        """Error contribution increases with more leaked qubits."""
        tracker = LeakageTracker(num_qubits=49)
        
        contributions = []
        for num_leaked in [0, 5, 10, 20, 40]:
            tracker.leaked[:] = False
            tracker.leaked[:num_leaked] = True
            contributions.append(tracker.get_leakage_error_contribution())
        
        # Should be monotonically increasing
        for i in range(len(contributions) - 1):
            assert contributions[i+1] >= contributions[i]
    
    def test_history_recording(self):
        """History is recorded correctly."""
        tracker = LeakageTracker(num_qubits=25)
        
        for _ in range(10):
            tracker.run_cycle()
        
        assert len(tracker.history) == 10
        assert tracker.history[-1].cycle == 10
    
    def test_reset_clears_state(self):
        """Reset clears all leakage state."""
        tracker = LeakageTracker(num_qubits=25, leakage_rate=0.1)
        
        for _ in range(20):
            tracker.run_cycle()
        
        assert tracker.get_num_leaked() > 0
        
        tracker.reset()
        
        assert tracker.get_num_leaked() == 0
        assert tracker.total_leakage_events == 0
        assert len(tracker.history) == 0
    
    def test_statistics(self):
        """Statistics dict contains expected keys."""
        tracker = LeakageTracker(num_qubits=25)
        tracker.run_cycle()
        
        stats = tracker.get_statistics()
        
        assert "num_qubits" in stats
        assert "leakage_rate" in stats
        assert "seepage_rate" in stats
        assert "current_leaked" in stats
        assert "error_contribution" in stats


class TestLeakageIntegration:
    """Integration tests for leakage in sampling."""
    
    def test_leakage_creates_error_floor(self):
        """Leakage creates an error floor at high distances."""
        # This is a conceptual test - with leakage, even perfect QEC
        # will have some error floor from leaked qubits
        tracker = LeakageTracker(
            num_qubits=225,  # d=15
            leakage_rate=0.001,
            seepage_rate=0.01
        )
        
        # Run many cycles to reach equilibrium
        for _ in range(1000):
            tracker.run_cycle()
        
        error_floor = tracker.get_leakage_error_contribution()
        
        # Should have non-zero error floor
        assert error_floor > 0
        # But should be bounded (not catastrophic)
        assert error_floor < 0.5
