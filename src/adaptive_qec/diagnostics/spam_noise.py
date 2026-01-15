"""
SPAM Noise Injection for Diagnostic Robustness Testing

Simulates State Preparation And Measurement (SPAM) errors to test
the robustness of Hamiltonian learning methods.

Author: Justin Arndt
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SPAMNoiseModel:
    """
    SPAM noise model for readout and state prep errors.
    
    Parameters
    ----------
    readout_bias : float
        Systematic bias in readout (0-1). A bias of 0.01 means
        measuring |0⟩ has 1% chance of returning |1⟩.
    readout_variance : float
        Random noise in readout probability.
    prep_error : float
        State preparation error rate.
    asymmetric : bool
        If True, 0→1 and 1→0 errors can have different rates.
    bias_01 : float
        P(measure 1 | true 0) if asymmetric.
    bias_10 : float
        P(measure 0 | true 1) if asymmetric.
    """
    readout_bias: float = 0.01
    readout_variance: float = 0.005
    prep_error: float = 0.005
    asymmetric: bool = False
    bias_01: float = 0.01
    bias_10: float = 0.01


def inject_readout_noise(
    measurements: np.ndarray,
    spam_model: SPAMNoiseModel
) -> np.ndarray:
    """
    Inject SPAM noise into measurement outcomes.
    
    Parameters
    ----------
    measurements : np.ndarray
        Clean measurement outcomes (0 or 1).
    spam_model : SPAMNoiseModel
        SPAM noise parameters.
        
    Returns
    -------
    np.ndarray
        Noisy measurement outcomes.
    """
    noisy = measurements.copy().astype(float)
    
    if spam_model.asymmetric:
        # Different error rates for 0→1 vs 1→0
        mask_0 = measurements == 0
        mask_1 = measurements == 1
        
        flip_01 = np.random.random(np.sum(mask_0)) < spam_model.bias_01
        flip_10 = np.random.random(np.sum(mask_1)) < spam_model.bias_10
        
        noisy[mask_0] = np.where(flip_01, 1, 0)
        noisy[mask_1] = np.where(flip_10, 0, 1)
    else:
        # Symmetric readout bias
        flip_prob = spam_model.readout_bias + \
                    np.random.normal(0, spam_model.readout_variance, measurements.shape)
        flip_prob = np.clip(flip_prob, 0, 1)
        
        flip_mask = np.random.random(measurements.shape) < flip_prob
        noisy = np.where(flip_mask, 1 - measurements, measurements)
    
    return noisy.astype(int)


def inject_state_prep_error(
    initial_state: np.ndarray,
    spam_model: SPAMNoiseModel
) -> np.ndarray:
    """
    Inject state preparation errors.
    
    Parameters
    ----------
    initial_state : np.ndarray
        Ideal initial state vector (density matrix diagonal).
    spam_model : SPAMNoiseModel
        SPAM noise parameters.
        
    Returns
    -------
    np.ndarray
        State with preparation errors applied.
    """
    # Simple depolarizing prep error
    error_mask = np.random.random(len(initial_state)) < spam_model.prep_error
    noisy_state = initial_state.copy()
    noisy_state[error_mask] = 0.5  # Mixed state
    return noisy_state / np.sum(noisy_state)


def compute_imbalance_with_spam(
    imbalance_clean: np.ndarray,
    spam_model: SPAMNoiseModel
) -> np.ndarray:
    """
    Apply SPAM noise to imbalance trace.
    
    The imbalance I(t) = <N_odd - N_even> is affected by SPAM:
    - Readout bias adds systematic offset
    - Prep error reduces initial imbalance
    
    Parameters
    ----------
    imbalance_clean : np.ndarray
        Clean imbalance trace I(t).
    spam_model : SPAMNoiseModel
        SPAM noise parameters.
        
    Returns
    -------
    np.ndarray
        Noisy imbalance trace.
    """
    # Prep error reduces initial imbalance
    prep_factor = 1 - 2 * spam_model.prep_error
    
    # Readout bias adds offset and reduces contrast
    readout_factor = 1 - 2 * spam_model.readout_bias
    offset = spam_model.readout_bias - spam_model.readout_bias  # Asymmetric case
    
    # Apply noise
    imbalance_noisy = imbalance_clean * prep_factor * readout_factor
    
    # Add stochastic noise
    noise = np.random.normal(0, spam_model.readout_variance, len(imbalance_clean))
    imbalance_noisy += noise
    
    return np.clip(imbalance_noisy, -1, 1)


def generate_spam_sweep(
    bias_levels: list = [0.0, 0.005, 0.01, 0.02, 0.05]
) -> list:
    """
    Generate a sweep of SPAM noise models for robustness testing.
    
    Parameters
    ----------
    bias_levels : list
        Readout bias levels to test.
        
    Returns
    -------
    list
        List of SPAMNoiseModel instances.
    """
    models = []
    for bias in bias_levels:
        models.append(SPAMNoiseModel(
            readout_bias=bias,
            readout_variance=bias / 2,
            prep_error=bias / 2
        ))
    return models
