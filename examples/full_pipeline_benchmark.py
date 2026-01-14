#!/usr/bin/env python3
"""
Example: Full Pipeline Benchmark with Ablations

End-to-end benchmark comparing:
- Static MWPM (baseline)
- Feedback only
- Diagnostics only
- Remediation only
- Full adaptive stack

Author: Justin Arndt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.hybrid.stim_cirq_bridge import StimCirqBridge, NoiseModel
from adaptive_qec.hybrid.adaptive_sampler import HybridAdaptiveSampler
from adaptive_qec.hybrid.realtime_surface import AdaptiveSurfaceCode, ExperimentConfig
import pymatching
import stim


def run_ablation_study(
    distance: int = 7,
    num_cycles: int = 1000,
    batch_size: int = 512,
    num_seeds: int = 3,
    save_plots: bool = True
):
    """
    Run ablation study comparing different components.
    
    Parameters
    ----------
    distance : int
        Code distance.
    num_cycles : int
        QEC cycles.
    batch_size : int
        Shots per cycle.
    num_seeds : int
        Statistical seeds.
    save_plots : bool
        Save result plots.
    """
    print("=" * 70)
    print("FULL PIPELINE ABLATION STUDY")
    print(f"Distance: {distance}, Cycles: {num_cycles}, Seeds: {num_seeds}")
    print("=" * 70)
    
    # Define configurations
    configs = {
        "Static MWPM": {"enable_drift": False, "feedback_Ki": 0.0},
        "Drift Only": {"enable_drift": True, "feedback_Ki": 0.0},
        "Feedback (Ki=0.02)": {"enable_drift": True, "feedback_Ki": 0.02},
        "Feedback (Ki=0.05)": {"enable_drift": True, "feedback_Ki": 0.05},
        "Full Stack": {"enable_drift": True, "feedback_Ki": 0.05},
    }
    
    results = {name: {"error_rate": [], "suppression": [], "time": []} for name in configs}
    
    for name, params in configs.items():
        print(f"\n--- {name} ---")
        
        for seed in range(num_seeds):
            np.random.seed(42 + seed)
            
            config = ExperimentConfig(
                distance=distance,
                rounds=5,
                num_cycles=num_cycles,
                batch_size=batch_size,
                depolarizing=0.001,
                measurement=0.01,
                enable_drift=params["enable_drift"],
                drift_rate=0.005,
                drift_target=0.02,
                feedback_Ki=params["feedback_Ki"]
            )
            
            start = time.time()
            runner = AdaptiveSurfaceCode(config)
            result = runner.run(verbose=False)
            elapsed = time.time() - start
            
            results[name]["error_rate"].append(result["adaptive_error_rate"])
            results[name]["suppression"].append(result["suppression_factor"])
            results[name]["time"].append(elapsed)
            
            print(f"  Seed {seed}: Error={result['adaptive_error_rate']:.4f}, "
                  f"Suppression={result['suppression_factor']:.1f}x, Time={elapsed:.1f}s")
    
    # Compute statistics
    summary = {}
    for name in configs:
        summary[name] = {
            "error_mean": np.mean(results[name]["error_rate"]),
            "error_std": np.std(results[name]["error_rate"]),
            "suppression_mean": np.mean(results[name]["suppression"]),
            "suppression_std": np.std(results[name]["suppression"]),
            "time_mean": np.mean(results[name]["time"])
        }
    
    # Results table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS (Mean ± Std)")
    print("=" * 70)
    print(f"{'Config':<25} {'Error Rate':<18} {'Suppression':<15} {'Time':<10}")
    print("-" * 70)
    
    for name in configs:
        s = summary[name]
        print(f"{name:<25} {s['error_mean']:.4f} ± {s['error_std']:.4f}   "
              f"{s['suppression_mean']:.1f}x ± {s['suppression_std']:.1f}   "
              f"{s['time_mean']:.1f}s")
    
    # Plot
    if save_plots:
        plot_ablation_results(summary, configs)
    
    return summary


def run_scaling_benchmark(
    distances: list = [3, 5, 7, 9, 11],
    num_shots: int = 10000,
    save_plots: bool = True
):
    """
    Benchmark time scaling vs distance.
    """
    print("\n" + "=" * 70)
    print("SCALING BENCHMARK: Time vs Distance")
    print("=" * 70)
    
    timings = {"stim_only": [], "hybrid": []}
    
    for d in distances:
        print(f"\n--- Distance d={d} ---")
        
        # Pure Stim timing
        start = time.time()
        stim_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=5,
            after_clifford_depolarization=0.001
        )
        sampler = stim_circuit.compile_detector_sampler()
        det, obs = sampler.sample(shots=num_shots, separate_observables=True)
        decoder = pymatching.Matching.from_stim_circuit(stim_circuit)
        _ = decoder.decode_batch(det)
        stim_time = time.time() - start
        timings["stim_only"].append(stim_time)
        
        # Hybrid timing
        config = ExperimentConfig(
            distance=d, rounds=5,
            num_cycles=100, batch_size=num_shots // 100
        )
        
        start = time.time()
        runner = AdaptiveSurfaceCode(config)
        runner.run(verbose=False)
        hybrid_time = time.time() - start
        timings["hybrid"].append(hybrid_time)
        
        overhead = (hybrid_time - stim_time) / stim_time * 100
        print(f"  Stim: {stim_time*1000:.1f}ms, Hybrid: {hybrid_time*1000:.1f}ms, "
              f"Overhead: {overhead:.1f}%")
    
    if save_plots:
        plot_scaling_results(distances, timings)
    
    return timings


def plot_ablation_results(summary, configs):
    """Plot ablation study results."""
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(configs.keys())
    error_means = [summary[n]["error_mean"] for n in names]
    error_stds = [summary[n]["error_std"] for n in names]
    suppression_means = [summary[n]["suppression_mean"] for n in names]
    
    # Error rate comparison
    ax = axes[0]
    bars = ax.bar(range(len(names)), error_means, yerr=error_stds,
                  capsize=5, color=['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6'],
                  edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel("Logical Error Rate", fontsize=11)
    ax.set_title("Ablation: Error Rate by Component", fontsize=13, fontweight='bold')
    
    # Suppression comparison
    ax = axes[1]
    colors = ['#27ae60' if s > 10 else '#f39c12' if s > 5 else '#e74c3c' for s in suppression_means]
    bars = ax.bar(range(len(names)), suppression_means, color=colors,
                  edgecolor='black', linewidth=1.5)
    ax.axhline(10, color='red', linestyle='--', label='Target: 10x')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel("Suppression Factor", fontsize=11)
    ax.set_title("Ablation: Suppression by Component", fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "ablation_study.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: docs/ablation_study.png")
    plt.close()


def plot_scaling_results(distances, timings):
    """Plot scaling benchmark results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(distances, [t*1000 for t in timings["stim_only"]], 
              'o-', linewidth=2, markersize=8, label="Stim Only")
    ax.loglog(distances, [t*1000 for t in timings["hybrid"]],
              's-', linewidth=2, markersize=8, label="Hybrid (This Work)")
    
    ax.set_xlabel("Code Distance", fontsize=12)
    ax.set_ylabel("Time (ms, log scale)", fontsize=12)
    ax.set_title("Scaling: Time vs Distance", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Compute average overhead
    overheads = [(h - s) / s * 100 for s, h in zip(timings["stim_only"], timings["hybrid"])]
    avg_overhead = np.mean(overheads)
    ax.annotate(f"Average Overhead: {avg_overhead:.1f}%",
                xy=(0.6, 0.1), xycoords='axes fraction',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "scaling_benchmark.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: docs/scaling_benchmark.png")
    plt.close()


if __name__ == "__main__":
    # Run ablation study
    run_ablation_study()
    
    # Run scaling benchmark
    run_scaling_benchmark()
