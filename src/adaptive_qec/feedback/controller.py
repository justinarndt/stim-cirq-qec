"""
Syndrome-Informed Feedback Controller

Real-time drift tracking using syndrome density measurements.

Author: Justin Arndt
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque


class SyndromeFeedbackController:
    """
    Closed-loop controller for adaptive QEC.
    
    Tracks drift via syndrome density and provides correction
    signals for decoder weight updates.
    
    Includes realistic feedback latency modeling with T1/T2 decay
    during the processing window.
    """
    
    def __init__(
        self,
        Ki: float = 0.05,
        feedback_latency: int = 10,
        correction_bounds: Tuple[float, float] = (-0.02, 0.15),
        latency_ns: float = 0.0,
        t1_us: float = 100.0,
        t2_us: float = 80.0
    ):
        """
        Parameters
        ----------
        Ki : float
            Integral gain for controller.
        feedback_latency : int
            Number of cycles delay for correction (discrete latency).
        correction_bounds : tuple
            (min, max) bounds for correction signal.
        latency_ns : float
            Physical feedback latency in nanoseconds. During this time,
            qubits idle and experience T1/T2 decay.
        t1_us : float
            T1 relaxation time in microseconds.
        t2_us : float
            T2 dephasing time in microseconds.
        """
        self.Ki = Ki
        self.latency = feedback_latency
        self.bounds = correction_bounds
        self.latency_ns = latency_ns
        self.t1_us = t1_us
        self.t2_us = t2_us
        
        self.integrator_state = 0.0
        self.setpoint: Optional[float] = None
        self.correction_queue = deque([0.0] * feedback_latency, maxlen=feedback_latency)
        
        self.history = {"density": [], "correction": [], "error": [], "decay_penalty": []}
    
    def compute_latency_decay_penalty(self) -> float:
        """
        Compute additional error probability from T1/T2 decay during latency.
        
        During feedback processing, qubits idle and experience decoherence.
        This returns the probability of an error occurring during this window.
        
        Returns
        -------
        float
            Decay-induced error probability (0 to ~1).
        """
        if self.latency_ns <= 0:
            return 0.0
        
        # Convert latency from ns to us
        idle_time_us = self.latency_ns / 1000.0
        
        # T1 decay: probability of relaxation during idle
        # P(decay) = 1 - exp(-t/T1)
        t1_decay = 1 - np.exp(-idle_time_us / self.t1_us)
        
        # T2 decay: dephasing contributes to error
        # T2 < T1 typically, so this is often the dominant term
        t2_decay = 1 - np.exp(-idle_time_us / self.t2_us)
        
        # Combined decay penalty (approximately additive for small rates)
        # Factor of 0.5 because only some decay events cause logical errors
        decay_penalty = 0.5 * (t1_decay + t2_decay)
        
        return min(decay_penalty, 1.0)
    
    def calibrate(self, plant, num_samples: int = 10) -> float:
        """Calibrate setpoint from baseline measurements."""
        densities = []
        for _ in range(num_samples):
            _, density = plant.run_batch(correction=0.0)
            densities.append(density)
        
        self.setpoint = np.mean(densities)
        self.integrator_state = 0.0
        self.correction_queue = deque([0.0] * self.latency, maxlen=self.latency)
        
        return self.setpoint
    
    def update(self, measured_density: float) -> float:
        """Compute correction signal."""
        if self.setpoint is None:
            raise ValueError("Controller not calibrated")
        
        error = measured_density - self.setpoint
        self.integrator_state += error * self.Ki
        self.integrator_state = np.clip(self.integrator_state, *self.bounds)
        
        self.correction_queue.append(self.integrator_state)
        correction = self.correction_queue[0]
        
        self.history["density"].append(measured_density)
        self.history["correction"].append(correction)
        self.history["error"].append(error)
        
        return correction
    
    def get_active_correction(self) -> float:
        """Get currently active correction."""
        return self.correction_queue[0]
    
    def reset(self):
        """Reset controller state."""
        self.integrator_state = 0.0
        self.correction_queue = deque([0.0] * self.latency, maxlen=self.latency)
        self.history = {"density": [], "correction": [], "error": []}
