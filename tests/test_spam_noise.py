"""
Tests for SPAM Noise Injection

Tests the SPAM noise model and injection functions for
diagnostic robustness testing.

Author: Justin Arndt
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.diagnostics.spam_noise import (
    SPAMNoiseModel,
    inject_readout_noise,
    inject_state_prep_error,
    compute_imbalance_with_spam,
    generate_spam_sweep
)


class TestSPAMNoiseModel:
    """Tests for SPAMNoiseModel dataclass."""
    
    def test_default_values(self):
        """Default values are reasonable."""
        model = SPAMNoiseModel()
        
        assert model.readout_bias == 0.01
        assert model.readout_variance == 0.005
        assert model.prep_error == 0.005
        assert model.asymmetric == False
    
    def test_custom_values(self):
        """Custom values are preserved."""
        model = SPAMNoiseModel(readout_bias=0.05, prep_error=0.02)
        
        assert model.readout_bias == 0.05
        assert model.prep_error == 0.02


class TestInjectReadoutNoise:
    """Tests for readout noise injection."""
    
    def test_zero_bias_preserves_measurements(self):
        """Zero bias should mostly preserve measurements."""
        model = SPAMNoiseModel(readout_bias=0.0, readout_variance=0.0)
        measurements = np.array([0, 1, 0, 1, 0, 1])
        
        noisy = inject_readout_noise(measurements, model)
        
        # Should be mostly unchanged (stochastic, but very low error)
        assert np.sum(noisy != measurements) < 2
    
    def test_high_bias_flips_measurements(self):
        """High bias should flip many measurements."""
        model = SPAMNoiseModel(readout_bias=0.5, readout_variance=0.0)
        measurements = np.zeros(1000, dtype=int)
        
        noisy = inject_readout_noise(measurements, model)
        
        # Should flip about half
        flip_rate = np.mean(noisy != measurements)
        assert 0.3 < flip_rate < 0.7
    
    def test_output_is_binary(self):
        """Output should be 0 or 1."""
        model = SPAMNoiseModel(readout_bias=0.1)
        measurements = np.random.randint(0, 2, 100)
        
        noisy = inject_readout_noise(measurements, model)
        
        assert set(np.unique(noisy)).issubset({0, 1})
    
    def test_asymmetric_different_rates(self):
        """Asymmetric mode uses different 0→1 and 1→0 rates."""
        model = SPAMNoiseModel(asymmetric=True, bias_01=0.1, bias_10=0.3)
        
        zeros = np.zeros(1000, dtype=int)
        ones = np.ones(1000, dtype=int)
        
        noisy_zeros = inject_readout_noise(zeros, model)
        noisy_ones = inject_readout_noise(ones, model)
        
        flip_01 = np.mean(noisy_zeros)  # 0→1 rate
        flip_10 = 1 - np.mean(noisy_ones)  # 1→0 rate
        
        # 0→1 rate should be lower than 1→0
        assert flip_01 < flip_10


class TestComputeImbalanceWithSPAM:
    """Tests for imbalance trace with SPAM noise."""
    
    def test_zero_spam_preserves_imbalance(self):
        """Zero SPAM should preserve imbalance."""
        model = SPAMNoiseModel(readout_bias=0.0, prep_error=0.0, 
                              readout_variance=0.0)
        imbalance = np.array([1.0, 0.8, 0.6, 0.4])
        
        noisy = compute_imbalance_with_spam(imbalance, model)
        
        np.testing.assert_array_almost_equal(imbalance, noisy, decimal=1)
    
    def test_imbalance_bounded(self):
        """Output imbalance should be in [-1, 1]."""
        model = SPAMNoiseModel(readout_bias=0.1)
        imbalance = np.array([1.0, -1.0, 0.5])
        
        noisy = compute_imbalance_with_spam(imbalance, model)
        
        assert np.all(noisy >= -1)
        assert np.all(noisy <= 1)


class TestGenerateSPAMSweep:
    """Tests for SPAM sweep generation."""
    
    def test_default_sweep(self):
        """Default sweep has 5 levels."""
        models = generate_spam_sweep()
        
        assert len(models) == 5
        assert all(isinstance(m, SPAMNoiseModel) for m in models)
    
    def test_custom_levels(self):
        """Custom levels work correctly."""
        levels = [0.0, 0.01, 0.05]
        models = generate_spam_sweep(levels)
        
        assert len(models) == 3
        assert models[0].readout_bias == 0.0
        assert models[1].readout_bias == 0.01
        assert models[2].readout_bias == 0.05
