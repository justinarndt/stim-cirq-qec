"""
Hybrid Adaptive Sampler

Combines Stim's high-speed sampling with Cirq's coherent noise modeling
and realtime-qec's feedback controller for drift-adaptive QEC.

Author: Justin Arndt
"""

import stim
import cirq
import numpy as np
import pymatching
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

from adaptive_qec.hybrid.stim_cirq_bridge import StimCirqBridge, NoiseModel
from adaptive_qec.feedback.controller import SyndromeFeedbackController
from adaptive_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
from adaptive_qec.physics.leakage import LeakageTracker


@dataclass
class SamplingResult:
    """Results from a hybrid sampling cycle."""
    detection_events: np.ndarray
    observable_flips: np.ndarray
    predictions: np.ndarray
    logical_errors: int
    syndrome_density: float
    correction: float
    decay_penalty: float = 0.0  # T1/T2 decay during latency
    leakage_contribution: float = 0.0  # Error from leaked qubits
    num_leaked_qubits: int = 0  # Current leaked qubit count
    coherent_state: Optional[np.ndarray] = None


class HybridAdaptiveSampler:
    """
    Hybrid Stim + Cirq sampler with real-time feedback.
    
    Architecture
    ------------
    1. **Stim Layer**: Fast Pauli-frame sampling for bulk syndrome extraction
    2. **Cirq Layer**: Density matrix simulation for coherent hotspots
    3. **Feedback Layer**: Syndrome density tracking and drift correction
    4. **Decoding Layer**: Adaptive MWPM with dynamic edge weights
    
    The sampler switches between modes based on detected conditions:
    - Normal: Pure Stim sampling (fastest)
    - Drift detected: Stim + feedback correction
    - Coherent hotspot: Cirq simulation on affected qubits
    
    Theory
    ------
    Under stationary Pauli noise, Stim achieves O(N) sampling. When
    drift or coherent effects are detected via syndrome density deviation,
    we selectively invoke Cirq for higher-fidelity modeling on hotspots.
    """
    
    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        noise: Optional[NoiseModel] = None,
        feedback_Ki: float = 0.05,
        feedback_latency: int = 10,
        drift_threshold: float = 0.02,
        latency_ns: float = 0.0,
        t1_us: float = 100.0,
        t2_us: float = 80.0
    ):
        """
        Parameters
        ----------
        distance : int
            Surface code distance.
        rounds : int
            QEC syndrome measurement rounds per cycle.
        noise : NoiseModel, optional
            Initial noise parameters.
        feedback_Ki : float
            Integral gain for feedback controller.
        feedback_latency : int
            Feedback loop delay in rounds.
        drift_threshold : float
            Syndrome density deviation threshold for drift detection.
        latency_ns : float
            Physical feedback latency in nanoseconds. During this time,
            qubits idle and experience T1/T2 decay.
        t1_us : float
            T1 relaxation time in microseconds.
        t2_us : float
            T2 dephasing time in microseconds.
        """
        self.distance = distance
        self.rounds = rounds
        self.noise = noise or NoiseModel()
        self.drift_threshold = drift_threshold
        self.latency_ns = latency_ns
        self.t1_us = t1_us
        self.t2_us = t2_us
        
        # Core components
        self.bridge = StimCirqBridge(distance=distance, rounds=rounds)
        self.controller = SyndromeFeedbackController(
            Ki=feedback_Ki,
            feedback_latency=feedback_latency,
            latency_ns=latency_ns,
            t1_us=t1_us,
            t2_us=t2_us
        )
        
        # Build circuits
        self.stim_circuit = self.bridge.cirq_to_stim(None, self.noise)
        self.dem = self.bridge.stim_to_dem(self.stim_circuit)
        self.decoder = pymatching.Matching.from_stim_circuit(self.stim_circuit)
        
        # Leakage tracking (Reality Gap fix)
        num_data_qubits = distance ** 2
        self.leakage_tracker = LeakageTracker(
            num_qubits=num_data_qubits,
            leakage_rate=self.noise.leakage_rate,
            seepage_rate=self.noise.seepage_rate
        )
        
        # State
        self.baseline_density: Optional[float] = None
        self.cycle_count = 0
        self.history: Dict[str, List] = {
            "density": [],
            "correction": [],
            "logical_errors": [],
            "drift_detected": [],
            "leaked_qubits": [],
            "leakage_contribution": []
        }
    
    def calibrate(self, calibration_shots: int = 10240) -> float:
        """
        Calibrate baseline syndrome density.
        
        Parameters
        ----------
        calibration_shots : int
            Number of shots for calibration.
            
        Returns
        -------
        float
            Baseline syndrome density setpoint.
        """
        detection_events, _ = self.bridge.sample_stim(
            self.stim_circuit,
            shots=calibration_shots
        )
        
        self.baseline_density = self.bridge.compute_syndrome_density(detection_events)
        self.controller.setpoint = self.baseline_density
        
        return self.baseline_density
    
    def run_cycle(
        self,
        batch_size: int = 1024,
        use_cirq_hotspots: bool = False
    ) -> SamplingResult:
        """
        Run one hybrid sampling cycle.
        
        Parameters
        ----------
        batch_size : int
            Number of shots to sample.
        use_cirq_hotspots : bool
            Whether to use Cirq for coherent hotspot analysis.
            
        Returns
        -------
        SamplingResult
            Complete results from this cycle.
        """
        self.cycle_count += 1
        
        # Get current correction from controller
        correction = self.controller.get_active_correction()
        
        # Apply correction by adjusting effective noise
        effective_noise = NoiseModel(
            depolarizing=max(1e-6, self.noise.depolarizing - correction),
            measurement=self.noise.measurement,
            reset=self.noise.reset,
            coherent_overrotation=self.noise.coherent_overrotation,
            zz_crosstalk=self.noise.zz_crosstalk
        )
        
        # Generate circuit with corrected noise
        stim_circuit = self.bridge.cirq_to_stim(None, effective_noise)
        
        # Sample with Stim
        detection_events, observable_flips = self.bridge.sample_stim(
            stim_circuit,
            shots=batch_size
        )
        
        # Compute syndrome density
        density = self.bridge.compute_syndrome_density(detection_events)
        
        # Update feedback controller
        new_correction = self.controller.update(density)
        
        # Detect drift
        drift_detected = False
        if self.baseline_density is not None:
            deviation = abs(density - self.baseline_density)
            drift_detected = deviation > self.drift_threshold
        
        # Optional: Cirq hotspot analysis for coherent effects
        coherent_state = None
        if use_cirq_hotspots and drift_detected:
            # Build small Cirq circuit for hotspot analysis
            cirq_circuit = self.bridge.build_cirq_surface_code(effective_noise)
            result = self.bridge.simulate_cirq_coherent(cirq_circuit)
            coherent_state = result.final_density_matrix
        
        # Decode
        decoder = pymatching.Matching.from_stim_circuit(stim_circuit)
        predictions = decoder.decode_batch(detection_events)
        
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if observable_flips.ndim > 1:
            observable_flips = observable_flips.flatten()
        
        logical_errors = np.sum(predictions != observable_flips)
        
        # Apply T1/T2 decay penalty during feedback latency (Reality Gap fix)
        # This models qubit decoherence while the FPGA processes syndrome data
        decay_penalty = self.controller.compute_latency_decay_penalty()
        if decay_penalty > 0:
            # Each qubit has independent probability of decay-induced error
            # For d qubits, expected additional errors per shot
            num_data_qubits = self.distance ** 2
            expected_decay_errors = decay_penalty * num_data_qubits * batch_size
            # Add as Poisson-sampled additional errors
            decay_induced_errors = np.random.poisson(expected_decay_errors)
            logical_errors += decay_induced_errors
        
        # Apply leakage modeling (Reality Gap fix)
        # Track qubit leakage to |2⟩ state and seepage recovery
        self.leakage_tracker.run_cycle(num_gates_per_qubit=5)
        leakage_contribution = self.leakage_tracker.get_leakage_error_contribution()
        num_leaked = self.leakage_tracker.get_num_leaked()
        
        # Add leakage-induced errors
        if leakage_contribution > 0:
            leakage_errors = np.random.poisson(leakage_contribution * batch_size)
            logical_errors += leakage_errors
        
        # Record history
        self.history["density"].append(density)
        self.history["correction"].append(correction)
        self.history["logical_errors"].append(logical_errors)
        self.history["drift_detected"].append(drift_detected)
        self.history["leaked_qubits"].append(num_leaked)
        self.history["leakage_contribution"].append(leakage_contribution)
        
        return SamplingResult(
            detection_events=detection_events,
            observable_flips=observable_flips,
            predictions=predictions,
            logical_errors=logical_errors,
            syndrome_density=density,
            correction=correction,
            decay_penalty=decay_penalty,
            leakage_contribution=leakage_contribution,
            num_leaked_qubits=num_leaked,
            coherent_state=coherent_state
        )
    
    def run_experiment(
        self,
        num_cycles: int,
        batch_size: int = 1024,
        verbose: bool = True
    ) -> Dict:
        """
        Run full adaptive QEC experiment.
        
        Parameters
        ----------
        num_cycles : int
            Number of QEC cycles to run.
        batch_size : int
            Shots per cycle.
        verbose : bool
            Print progress updates.
            
        Returns
        -------
        dict
            Experiment results including suppression factor.
        """
        if self.baseline_density is None:
            self.calibrate()
        
        total_errors_adaptive = 0
        total_errors_baseline = 0
        
        for cycle in range(num_cycles):
            # Adaptive run
            result = self.run_cycle(batch_size)
            total_errors_adaptive += result.logical_errors
            
            # Baseline run (no correction)
            stim_circuit = self.bridge.cirq_to_stim(None, self.noise)
            det, obs = self.bridge.sample_stim(stim_circuit, batch_size)
            decoder = pymatching.Matching.from_stim_circuit(stim_circuit)
            pred = decoder.decode_batch(det)
            if pred.ndim > 1:
                pred = pred.flatten()
            if obs.ndim > 1:
                obs = obs.flatten()
            total_errors_baseline += np.sum(pred != obs)
            
            if verbose and cycle % 100 == 0:
                print(f"Cycle {cycle}: Baseline={total_errors_baseline}, "
                      f"Adaptive={total_errors_adaptive}")
        
        suppression = total_errors_baseline / max(total_errors_adaptive, 1)
        
        return {
            "baseline_errors": total_errors_baseline,
            "adaptive_errors": total_errors_adaptive,
            "suppression_factor": suppression,
            "cycles": num_cycles,
            "history": self.history
        }
