"""
Tests for Frequency Analysis and Stability Margins

Tests the Bode plot generation and phase/gain margin computation
for the syndrome feedback controller.

Author: Justin Arndt
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.feedback.frequency_analysis import (
    FrequencyAnalyzer,
    StabilityMargins,
    analyze_controller_stability
)


class TestFrequencyAnalyzer:
    """Tests for FrequencyAnalyzer class."""
    
    def test_initialization(self):
        """Analyzer initializes with correct parameters."""
        analyzer = FrequencyAnalyzer(Ki=0.05, latency_s=500e-9)
        
        assert analyzer.Ki == 0.05
        assert analyzer.latency_s == 500e-9
        assert analyzer.plant_gain == 1.0
    
    def test_open_loop_transfer_function(self):
        """Open-loop TF has expected shape."""
        analyzer = FrequencyAnalyzer()
        freq = np.logspace(-1, 5, 100)
        
        L = analyzer.open_loop_tf(freq)
        
        assert len(L) == 100
        assert np.all(np.isfinite(L))
        
    def test_open_loop_low_frequency_gain(self):
        """At low frequency, integrator has high gain."""
        analyzer = FrequencyAnalyzer(Ki=0.05)
        freq = np.array([0.1])
        
        L = analyzer.open_loop_tf(freq)
        
        # Integrator: |Ki/s| = Ki/ω → large at low freq
        assert np.abs(L[0]) > 1.0
    
    def test_open_loop_high_frequency_rolloff(self):
        """At high frequency, gain rolls off."""
        analyzer = FrequencyAnalyzer(Ki=0.05)
        freq = np.array([1e5])
        
        L = analyzer.open_loop_tf(freq)
        
        # Should have low gain at high freq
        assert np.abs(L[0]) < 0.1
    
    def test_closed_loop_transfer_function(self):
        """Closed-loop TF is bounded."""
        analyzer = FrequencyAnalyzer()
        freq = np.logspace(-1, 5, 100)
        
        T = analyzer.closed_loop_tf(freq)
        
        # Closed-loop should be bounded (no poles in RHP for stable)
        assert np.all(np.abs(T) < 10)
    
    def test_compute_margins_returns_dataclass(self):
        """compute_margins returns StabilityMargins."""
        analyzer = FrequencyAnalyzer()
        
        margins = analyzer.compute_margins()
        
        assert isinstance(margins, StabilityMargins)
        assert hasattr(margins, 'phase_margin_deg')
        assert hasattr(margins, 'gain_margin_db')
        assert hasattr(margins, 'stable')
    
    def test_default_controller_is_stable(self):
        """Default Ki=0.05, latency=500ns is stable."""
        analyzer = FrequencyAnalyzer(Ki=0.05, latency_s=500e-9)
        
        margins = analyzer.compute_margins()
        
        assert margins.stable
        assert margins.phase_margin_deg > 0
    
    def test_high_ki_reduces_phase_margin(self):
        """Higher Ki reduces phase margin."""
        analyzer_low = FrequencyAnalyzer(Ki=0.01)
        analyzer_high = FrequencyAnalyzer(Ki=0.20)
        
        margins_low = analyzer_low.compute_margins()
        margins_high = analyzer_high.compute_margins()
        
        # Higher Ki should have lower phase margin
        assert margins_low.phase_margin_deg >= margins_high.phase_margin_deg
    
    def test_higher_latency_reduces_phase_margin(self):
        """Higher latency reduces phase margin."""
        analyzer_fast = FrequencyAnalyzer(latency_s=100e-9)
        analyzer_slow = FrequencyAnalyzer(latency_s=2000e-9)
        
        margins_fast = analyzer_fast.compute_margins()
        margins_slow = analyzer_slow.compute_margins()
        
        # Higher latency should have lower phase margin
        assert margins_fast.phase_margin_deg >= margins_slow.phase_margin_deg


class TestStabilitySweep:
    """Tests for stability analysis sweep."""
    
    def test_analyze_returns_dict(self):
        """analyze_controller_stability returns dict."""
        results = analyze_controller_stability([0.01, 0.05])
        
        assert isinstance(results, dict)
        assert 0.01 in results
        assert 0.05 in results
    
    def test_result_contains_margins(self):
        """Each result has margin values."""
        results = analyze_controller_stability([0.05])
        
        assert 'phase_margin' in results[0.05]
        assert 'gain_margin' in results[0.05]
        assert 'stable' in results[0.05]
        assert 'safe' in results[0.05]
    
    def test_safety_threshold_applied(self):
        """Safe requires PM > 45° and GM > 6dB."""
        results = analyze_controller_stability([0.05])
        
        res = results[0.05]
        expected_safe = res['phase_margin'] > 45 and res['gain_margin'] > 6
        assert res['safe'] == expected_safe
