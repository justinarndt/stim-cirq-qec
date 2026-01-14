#!/usr/bin/env python3
"""Run d=13 and d=15 experiments for extended results."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.hybrid.realtime_surface import AdaptiveSurfaceCode, ExperimentConfig


def run_extended_distance(distance, num_seeds=5, num_cycles=500):
    print(f"\n{'='*60}")
    print(f"EXTENDED DISTANCE EXPERIMENT: d={distance}")
    print(f"{'='*60}")
    
    results = []
    for seed in range(num_seeds):
        np.random.seed(42 + seed)
        config = ExperimentConfig(
            distance=distance,
            rounds=5,
            num_cycles=num_cycles,
            batch_size=512,
            depolarizing=0.001,
            measurement=0.01,
            enable_drift=True,
            drift_rate=0.005,
            drift_target=0.02,
            feedback_Ki=0.05
        )
        runner = AdaptiveSurfaceCode(config)
        result = runner.run(verbose=False)
        results.append(result)
        print(f"  Seed {seed}: Baseline={result['baseline_error_rate']:.4f}, "
              f"Adaptive={result['adaptive_error_rate']:.6f}, "
              f"Suppression={result['suppression_factor']:.1f}x")
    
    baseline_mean = np.mean([r['baseline_error_rate'] for r in results])
    baseline_std = np.std([r['baseline_error_rate'] for r in results])
    adaptive_mean = np.mean([r['adaptive_error_rate'] for r in results])
    adaptive_std = np.std([r['adaptive_error_rate'] for r in results])
    suppression_mean = np.mean([r['suppression_factor'] for r in results])
    suppression_std = np.std([r['suppression_factor'] for r in results])
    
    print(f"\n--- d={distance} SUMMARY ---")
    print(f"Baseline:    {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"Adaptive:    {adaptive_mean:.6f} ± {adaptive_std:.6f}")
    print(f"Suppression: {suppression_mean:.1f}x ± {suppression_std:.1f}x")
    
    return {
        "distance": distance,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "adaptive_mean": adaptive_mean,
        "adaptive_std": adaptive_std,
        "suppression_mean": suppression_mean,
        "suppression_std": suppression_std
    }


if __name__ == "__main__":
    results = []
    
    # Run d=13
    results.append(run_extended_distance(13))
    
    # Run d=15 if time permits  
    results.append(run_extended_distance(15))
    
    print("\n" + "="*60)
    print("EXTENDED RESULTS TABLE")
    print("="*60)
    print(f"{'Distance':<10} {'Baseline':<15} {'Adaptive':<15} {'Suppression':<15}")
    print("-"*60)
    for r in results:
        print(f"d={r['distance']:<8} {r['baseline_mean']:.2%} ± {r['baseline_std']:.2%}  "
              f"{r['adaptive_mean']:.4%} ± {r['adaptive_std']:.4%}  "
              f"{r['suppression_mean']:.0f}x ± {r['suppression_std']:.0f}x")
