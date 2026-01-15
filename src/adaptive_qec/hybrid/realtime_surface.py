"""
Real-Time Adaptive Surface Code

Production-ready surface code runner with full hybrid stack:
- Stim bulk sampling
- Cirq coherent hotspots
- Syndrome feedback control
- MBL-based diagnostics
- Pulse remediation

Author: Justin Arndt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import time

from adaptive_qec.hybrid.stim_cirq_bridge import StimCirqBridge, NoiseModel
from adaptive_qec.hybrid.adaptive_sampler import HybridAdaptiveSampler
from adaptive_qec.feedback.controller import SyndromeFeedbackController
from adaptive_qec.diagnostics.hamiltonian_learner import HamiltonianLearner


@dataclass 
class ExperimentConfig:
    """Configuration for adaptive surface code experiment."""
    distance: int = 7
    rounds: int = 5
    num_cycles: int = 1000
    batch_size: int = 1024
    
    # Noise (Willow Specs)
    depolarizing: float = 1e-3    # ~0.1% avg gate error
    measurement: float = 1.5e-2   # ~1.5% readout error
    reset: float = 0.005
    coherent_overrotation: float = 0.0
    zz_crosstalk: float = 0.0
    
    # Drift
    enable_drift: bool = True
    drift_rate: float = 0.005
    drift_target: float = 0.03
    
    # Feedback
    feedback_Ki: float = 0.05
    feedback_latency: int = 10
    
    # Physical latency and decoherence (Willow Ground Truth)
    latency_ns: float = 600.0     # Conservative decoder+FPGA latency
    t1_us: float = 68.0           # Willow Average T1
    t2_us: float = 75.0           # Conservative estimate
    gate_time_ns: float = 25.0    # Typical transmon gate time
    readout_time_ns: float = 500.0


class AdaptiveSurfaceCode:
    """
    Full adaptive surface code implementation.
    
    Combines all components into a unified runner that:
    1. Builds hybrid Stim/Cirq circuits
    2. Applies time-varying drift
    3. Tracks syndrome density in real-time
    4. Adapts decoder weights via feedback
    5. Optionally diagnoses hardware and remediates
    
    This is the production entry point for running adaptive QEC.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Parameters
        ----------
        config : ExperimentConfig, optional
            Experiment configuration. Uses defaults if not specified.
        """
        self.config = config or ExperimentConfig()
        
        # Initialize noise model
        self.noise = NoiseModel(
            depolarizing=self.config.depolarizing,
            measurement=self.config.measurement,
            reset=self.config.reset,
            coherent_overrotation=self.config.coherent_overrotation,
            zz_crosstalk=self.config.zz_crosstalk
        )
        
        # Initialize sampler with latency parameters
        self.sampler = HybridAdaptiveSampler(
            distance=self.config.distance,
            rounds=self.config.rounds,
            noise=self.noise,
            feedback_Ki=self.config.feedback_Ki,
            feedback_latency=self.config.feedback_latency,
            latency_ns=self.config.latency_ns,
            t1_us=self.config.t1_us,
            t2_us=self.config.t2_us
        )
        
        # Drift state
        self.current_drift = 0.0
    
    def apply_drift(self) -> float:
        """
        Simulate Ornstein-Uhlenbeck drift.
        
        Returns
        -------
        float
            Current drift value.
        """
        if not self.config.enable_drift:
            return 0.0
        
        # OU process: dX = θ(μ - X)dt + σdW
        theta = self.config.drift_rate
        mu = self.config.drift_target
        sigma = 1e-5
        
        dx = theta * (mu - self.current_drift) + np.random.normal(0, sigma)
        self.current_drift += dx
        self.current_drift = np.clip(self.current_drift, -0.02, 0.05)
        
        # Update noise model
        self.noise.depolarizing = self.config.depolarizing + self.current_drift
        self.sampler.noise = self.noise
        
        return self.current_drift
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Run the full adaptive surface code experiment.
        
        Parameters
        ----------
        verbose : bool
            Print progress updates.
            
        Returns
        -------
        dict
            Complete experiment results.
        """
        print("=" * 60)
        print("ADAPTIVE SURFACE CODE EXPERIMENT")
        print(f"Distance: {self.config.distance}, Cycles: {self.config.num_cycles}")
        print("=" * 60)
        
        # Calibration
        print("\nCalibrating baseline syndrome density...")
        baseline = self.sampler.calibrate()
        print(f"Baseline density: {baseline:.5f}")
        
        # Run experiment
        print(f"\nRunning {self.config.num_cycles} cycles...")
        start_time = time.time()
        
        history = {
            "cycle": [],
            "drift": [],
            "density": [],
            "correction": [],
            "errors_baseline": [],
            "errors_adaptive": []
        }
        
        cum_errors_baseline = 0
        cum_errors_adaptive = 0
        
        for cycle in range(self.config.num_cycles):
            # Apply drift
            drift = self.apply_drift()
            
            # Run adaptive cycle
            result = self.sampler.run_cycle(self.config.batch_size)
            cum_errors_adaptive += result.logical_errors
            
            # Run baseline (no correction)
            stim_circuit = self.sampler.bridge.cirq_to_stim(None, self.noise)
            det, obs = self.sampler.bridge.sample_stim(stim_circuit, self.config.batch_size)
            import pymatching
            decoder = pymatching.Matching.from_stim_circuit(stim_circuit)
            pred = decoder.decode_batch(det)
            if pred.ndim > 1:
                pred = pred.flatten()
            if obs.ndim > 1:
                obs = obs.flatten()
            cum_errors_baseline += np.sum(pred != obs)
            
            # Record history
            if cycle % 50 == 0:
                history["cycle"].append(cycle)
                history["drift"].append(drift)
                history["density"].append(result.syndrome_density)
                history["correction"].append(result.correction)
                history["errors_baseline"].append(cum_errors_baseline)
                history["errors_adaptive"].append(cum_errors_adaptive)
            
            if verbose and cycle % 200 == 0:
                print(f"  Cycle {cycle:4d} | Drift: {drift:.4f} | "
                      f"Baseline: {cum_errors_baseline:,} | "
                      f"Adaptive: {cum_errors_adaptive:,}")
        
        duration = time.time() - start_time
        suppression = cum_errors_baseline / max(cum_errors_adaptive, 1)
        
        # Results
        results = {
            "config": self.config,
            "baseline_errors": cum_errors_baseline,
            "adaptive_errors": cum_errors_adaptive,
            "suppression_factor": suppression,
            "baseline_error_rate": cum_errors_baseline / (self.config.num_cycles * self.config.batch_size),
            "adaptive_error_rate": cum_errors_adaptive / (self.config.num_cycles * self.config.batch_size),
            "duration_seconds": duration,
            "history": history
        }
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Baseline Error Rate:  {results['baseline_error_rate']*100:.2f}%")
        print(f"Adaptive Error Rate:  {results['adaptive_error_rate']*100:.2f}%")
        print(f"Suppression Factor:   {suppression:.1f}x")
        print(f"Runtime:              {duration:.1f}s")
        
        return results
    
    def plot_results(self, results: Dict, save_path: Optional[Path] = None):
        """Generate results visualization."""
        history = results["history"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top left: Cumulative errors
        ax = axes[0, 0]
        ax.plot(history["cycle"], history["errors_baseline"], 'r-', 
                linewidth=2, label="Baseline")
        ax.plot(history["cycle"], history["errors_adaptive"], 'b-',
                linewidth=3, label="Adaptive")
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Cumulative Logical Errors")
        ax.set_title(f"Drift Suppression: {results['suppression_factor']:.1f}x")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Top right: Drift vs correction
        ax = axes[0, 1]
        ax.plot(history["cycle"], history["drift"], 'r-', label="Drift")
        ax.plot(history["cycle"], history["correction"], 'g-', label="Correction")
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Value")
        ax.set_title("Drift Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom left: Syndrome density
        ax = axes[1, 0]
        ax.plot(history["cycle"], history["density"], 'b-', alpha=0.7)
        ax.axhline(results["config"].depolarizing, color='gray', 
                   linestyle='--', label="Baseline")
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Syndrome Density")
        ax.set_title("Syndrome Density Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Summary table
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        EXPERIMENT SUMMARY
        ==================
        
        Distance:           d={results['config'].distance}
        Cycles:             {results['config'].num_cycles:,}
        Batch Size:         {results['config'].batch_size}
        
        Baseline Errors:    {results['baseline_errors']:,}
        Adaptive Errors:    {results['adaptive_errors']:,}
        
        Suppression:        {results['suppression_factor']:.1f}x
        Runtime:            {results['duration_seconds']:.1f}s
        """
        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
