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
    """
    
    def __init__(
        self,
        Ki: float = 0.05,
        feedback_latency: int = 10,
        correction_bounds: Tuple[float, float] = (-0.02, 0.15)
    ):
        self.Ki = Ki
        self.latency = feedback_latency
        self.bounds = correction_bounds
        
        self.integrator_state = 0.0
        self.setpoint: Optional[float] = None
        self.correction_queue = deque([0.0] * feedback_latency, maxlen=feedback_latency)
        
        self.history = {"density": [], "correction": [], "error": []}
    
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
