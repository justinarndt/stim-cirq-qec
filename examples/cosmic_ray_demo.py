"""
Cosmic Ray Recovery Demo

Demonstrates detection and recovery from cosmic ray burst events.
This is the "killer demo" showing dynamic Cirq region expansion.

Author: Justin Arndt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.physics.burst_detector import BurstErrorDetector, BurstEvent
from adaptive_qec.physics.cosmic_ray import CosmicRaySimulator, inject_cosmic_ray


def run_cosmic_ray_demo():
    """
    Demonstrate cosmic ray detection and recovery.
    
    Shows:
    1. Normal operation with occasional cosmic ray impacts
    2. Burst detection triggering on high syndrome density
    3. Dynamic expansion of Cirq simulation region
    4. Recovery within 20 cycles
    """
    print("=" * 60)
    print("COSMIC RAY RECOVERY DEMO")
    print("=" * 60)
    
    # Setup
    distance = 7
    num_cycles = 500
    
    # Initialize components
    detector = BurstErrorDetector(
        distance=distance,
        spike_threshold=0.25,
        spatial_threshold=3
    )
    detector.set_baseline(0.05)
    
    simulator = CosmicRaySimulator(
        distance=distance,
        impact_rate=2.0,  # Higher rate for demo (2 per 1000 cycles)
        typical_radius=2,
        max_depol=0.5
    )
    
    # Tracking
    syndrome_densities = []
    impacts = []
    detections = []
    expanded_regions = []
    recovery_cycles = []
    
    print(f"\nRunning {num_cycles} cycles with cosmic ray simulation...")
    print(f"Distance: d={distance} ({distance**2} qubits)")
    print(f"Expected impacts: ~{num_cycles * simulator.impact_rate / 1000:.1f}")
    
    for cycle in range(num_cycles):
        # Simulate normal syndrome density
        base_density = 0.05 + np.random.normal(0, 0.01)
        syndromes = np.random.binomial(1, base_density, size=distance**2)
        
        # Check for cosmic ray impact
        impact = simulator.check_for_impact(cycle)
        if impact:
            impacts.append(cycle)
            # Apply the impact to syndromes
            depol_map = simulator.get_depolarization_map(impact)
            for q in impact.affected_qubits:
                if np.random.random() < depol_map[q]:
                    syndromes[q] = 1
            print(f"\n  COSMIC RAY at cycle {cycle}!")
            print(f"    Center: qubit {impact.center_qubit}")
            print(f"    Affected: {len(impact.affected_qubits)} qubits")
        
        # Compute current density
        current_density = np.mean(syndromes)
        syndrome_densities.append(current_density)
        
        # Check for burst detection
        burst = detector.detect(syndromes, cycle)
        if burst:
            detections.append(cycle)
            
            # Get expanded Cirq region
            expanded = detector.get_expanded_cirq_region(burst, expansion_radius=2)
            expanded_regions.append((cycle, expanded))
            
            # Get recovery recommendation
            recovery = detector.get_recovery_recommendation(burst)
            recovery_cycles.append(cycle + recovery["cycles_until_recovery"])
            
            print(f"  BURST DETECTED at cycle {cycle}")
            print(f"    Severity: {burst.severity:.2f}")
            print(f"    Expanded Cirq region: {len(expanded)} qubits")
            print(f"    Expected recovery: {recovery['cycles_until_recovery']} cycles")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total cycles: {num_cycles}")
    print(f"Cosmic ray impacts: {len(impacts)}")
    print(f"Bursts detected: {len(detections)}")
    print(f"Detection rate: {len(detections)/max(len(impacts),1)*100:.0f}%")
    
    if detections:
        avg_expanded = np.mean([len(r[1]) for r in expanded_regions])
        print(f"Average expanded region: {avg_expanded:.0f} qubits")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: Syndrome density over time
    ax = axes[0]
    ax.plot(syndrome_densities, 'b-', alpha=0.7, linewidth=1)
    ax.axhline(0.05, color='green', linestyle='--', label='Baseline', linewidth=2)
    ax.axhline(0.05 + detector.spike_threshold, color='red', linestyle='--', 
               label='Detection Threshold', linewidth=2)
    
    # Mark impacts
    for imp_cycle in impacts:
        ax.axvline(imp_cycle, color='red', alpha=0.5, linewidth=2)
    
    # Mark detections
    for det_cycle in detections:
        ax.axvline(det_cycle, color='orange', alpha=0.7, linewidth=3, linestyle=':')
    
    ax.set_ylabel('Syndrome Density', fontsize=12)
    ax.set_title('Cosmic Ray Detection Demo', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.8)
    
    # Bottom: Expanded region size
    ax = axes[1]
    
    # Plot expanded region timeline
    expansion_timeline = np.zeros(num_cycles)
    for cycle, region in expanded_regions:
        for c in range(cycle, min(cycle + 20, num_cycles)):
            expansion_timeline[c] = max(expansion_timeline[c], len(region))
    
    ax.fill_between(range(num_cycles), expansion_timeline, alpha=0.5, color='orange')
    ax.set_xlabel('QEC Cycle', fontsize=12)
    ax.set_ylabel('Expanded Cirq Qubits', fontsize=12)
    ax.set_title('Dynamic Cirq Region Expansion', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "cosmic_ray_demo.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filepath}")
    
    plt.show()
    
    return {
        "cycles": num_cycles,
        "impacts": len(impacts),
        "detections": len(detections),
        "detection_rate": len(detections)/max(len(impacts),1),
        "expanded_regions": expanded_regions
    }


if __name__ == "__main__":
    results = run_cosmic_ray_demo()
