#!/usr/bin/env python3
"""
Example: Willow-Like Drift Suppression

Demonstrates adaptive QEC on d=7-11 surface codes with Willow-style
exponential suppression under realistic Ornstein-Uhlenbeck drift.

This replicates the core claim: maintains exponential suppression
(λ > 2 per distance increase) even under 20% noise drift.

Author: Justin Arndt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.hybrid.realtime_surface import AdaptiveSurfaceCode, ExperimentConfig


def run_willow_like_experiment(
    distances: list = [5, 7, 9, 11],
    num_cycles: int = 500,
    batch_size: int = 512,
    num_seeds: int = 5,
    save_plots: bool = True
):
    """
    Run Willow-like exponential suppression benchmark across distances.
    
    Parameters
    ----------
    distances : list
        Code distances to test.
    num_cycles : int
        QEC cycles per seed.
    batch_size : int
        Shots per cycle.
    num_seeds : int
        Statistical seeds for confidence intervals.
    save_plots : bool
        Whether to save result plots.
    """
    print("=" * 70)
    print("WILLOW-LIKE DRIFT SUPPRESSION STUDY")
    print(f"Distances: {distances}, Cycles: {num_cycles}, Seeds: {num_seeds}")
    print("=" * 70)
    
    results = {d: {"baseline": [], "adaptive": []} for d in distances}
    
    for d in distances:
        print(f"\n--- Distance d={d} ---")
        
        for seed in range(num_seeds):
            np.random.seed(42 + seed)
            
            config = ExperimentConfig(
                distance=d,
                rounds=5,
                num_cycles=num_cycles,
                batch_size=batch_size,
                depolarizing=1e-3,
                measurement=1.5e-2,
                enable_drift=True,
                drift_rate=0.005,
                drift_target=0.02  # 20% drift
            )
            
            runner = AdaptiveSurfaceCode(config)
            result = runner.run(verbose=False)
            
            results[d]["baseline"].append(result["baseline_error_rate"])
            results[d]["adaptive"].append(result["adaptive_error_rate"])
            
            print(f"  Seed {seed}: Baseline={result['baseline_error_rate']:.4f}, "
                  f"Adaptive={result['adaptive_error_rate']:.4f}, "
                  f"Suppression={result['suppression_factor']:.1f}x")
    
    # Compute statistics
    summary = {}
    for d in distances:
        baseline_mean = np.mean(results[d]["baseline"])
        baseline_std = np.std(results[d]["baseline"])
        adaptive_mean = np.mean(results[d]["adaptive"])
        adaptive_std = np.std(results[d]["adaptive"])
        suppression = baseline_mean / max(adaptive_mean, 1e-10)
        
        summary[d] = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "adaptive_mean": adaptive_mean,
            "adaptive_std": adaptive_std,
            "suppression": suppression
        }
    
    # Results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Mean ± Std over {} seeds)".format(num_seeds))
    print("=" * 70)
    print(f"{'Distance':<10} {'Baseline':<20} {'Adaptive':<20} {'Suppression':<15}")
    print("-" * 70)
    for d in distances:
        s = summary[d]
        print(f"d={d:<7} {s['baseline_mean']:.4f} ± {s['baseline_std']:.4f}   "
              f"{s['adaptive_mean']:.4f} ± {s['adaptive_std']:.4f}   "
              f"{s['suppression']:.1f}x")
    
    # Check exponential suppression
    print("\n--- Exponential Suppression Check ---")
    for i in range(len(distances) - 1):
        d1, d2 = distances[i], distances[i+1]
        lambda_factor = summary[d1]["adaptive_mean"] / max(summary[d2]["adaptive_mean"], 1e-10)
        print(f"λ (d={d1}→d={d2}): {lambda_factor:.2f}")
    
    # Plot
    if save_plots:
        plot_willow_results(summary, distances, num_seeds)
    
    return summary


def plot_willow_results(summary, distances, num_seeds):
    """Generate Willow-style exponential suppression plot."""
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Log error rate vs distance
    ax = axes[0]
    baseline_means = [summary[d]["baseline_mean"] for d in distances]
    baseline_stds = [summary[d]["baseline_std"] for d in distances]
    adaptive_means = [summary[d]["adaptive_mean"] for d in distances]
    adaptive_stds = [summary[d]["adaptive_std"] for d in distances]
    
    ax.errorbar(distances, baseline_means, yerr=baseline_stds,
                marker='o', linewidth=2, capsize=5, label="Baseline (no adaptation)")
    ax.errorbar(distances, adaptive_means, yerr=adaptive_stds,
                marker='s', linewidth=2, capsize=5, label="Adaptive (this work)")
    
    ax.set_yscale('log')
    ax.set_xlabel("Code Distance", fontsize=12)
    ax.set_ylabel("Logical Error Rate (log scale)", fontsize=12)
    ax.set_title("Exponential Suppression Under 20% Drift", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(distances)
    
    # Right: Suppression factor
    ax = axes[1]
    suppressions = [summary[d]["suppression"] for d in distances]
    colors = ['#27ae60' if s > 10 else '#e74c3c' for s in suppressions]
    
    bars = ax.bar(distances, suppressions, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(10, color='red', linestyle='--', label='Target: 10x')
    ax.set_xlabel("Code Distance", fontsize=12)
    ax.set_ylabel("Suppression Factor", fontsize=12)
    ax.set_title("Drift Suppression Improvement", fontsize=14, fontweight='bold')
    ax.set_xticks(distances)
    ax.legend()
    
    for bar, s in zip(bars, suppressions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{s:.0f}x', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "willow_drift_suppression.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: docs/willow_drift_suppression.png")
    plt.close()


if __name__ == "__main__":
    run_willow_like_experiment()
