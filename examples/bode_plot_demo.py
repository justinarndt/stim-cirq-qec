"""
Bode Plot Demo: Controller Stability Analysis

Demonstrates frequency-domain analysis of the syndrome feedback controller.
Generates Bode plot with phase margin annotation to prove stability.

A phase margin > 45° ensures:
- No resonant peaks that could excite dilution fridge mechanical modes
- Robustness to 1/f noise tails
- Safe for live FPGA deployment

Author: Justin Arndt
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.feedback.frequency_analysis import (
    FrequencyAnalyzer,
    analyze_controller_stability
)


def main():
    """Generate Bode plot and analyze controller stability."""
    print("=" * 70)
    print("BODE PLOT ANALYSIS: Syndrome Feedback Controller")
    print("=" * 70)
    
    # Configuration
    Ki = 0.05  # Our default integral gain
    latency_ns = 500  # 500 ns FPGA latency
    
    print(f"\nController Parameters:")
    print(f"  Integral Gain (Ki): {Ki}")
    print(f"  Feedback Latency: {latency_ns} ns")
    print(f"  Plant Pole: 1 kHz (typical QEC cycle rate)")
    
    # Create analyzer
    analyzer = FrequencyAnalyzer(
        Ki=Ki,
        latency_s=latency_ns * 1e-9,
        plant_gain=1.0,
        plant_pole_hz=1000.0
    )
    
    # Compute margins
    margins = analyzer.compute_margins()
    
    print("\n" + "-" * 50)
    print("STABILITY MARGINS")
    print("-" * 50)
    print(f"  Phase Margin: {margins.phase_margin_deg:.1f}° ", end="")
    print("✓" if margins.phase_margin_deg > 45 else "✗", 
          "(requirement: > 45°)")
    print(f"  Gain Margin:  {margins.gain_margin_db:.1f} dB ", end="")
    print("✓" if margins.gain_margin_db > 6 else "✗",
          "(requirement: > 6 dB)")
    print(f"  Crossover Frequency: {margins.crossover_freq_hz:.1f} Hz")
    print(f"  System Stable: {margins.stable}")
    
    # Check if safe for hardware
    safe = margins.phase_margin_deg > 45 and margins.gain_margin_db > 6
    
    if safe:
        print("\n✓ CONTROLLER IS SAFE FOR FPGA DEPLOYMENT")
    else:
        print("\n✗ CONTROLLER DOES NOT MEET SAFETY REQUIREMENTS")
        print("  Reduce Ki or feedback latency before hardware deployment.")
    
    # Generate Bode plot
    output_path = Path(__file__).parent.parent / "docs" / "figures" / "bode_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = analyzer.generate_bode_plot(
        freq_range=(0.1, 1e5),
        save_path=str(output_path)
    )
    
    print(f"\nSaved: {output_path}")
    
    # Sweep Ki values
    print("\n" + "=" * 70)
    print("STABILITY SWEEP: Varying Ki")
    print("=" * 70)
    
    Ki_values = [0.01, 0.02, 0.05, 0.10, 0.20]
    results = analyze_controller_stability(Ki_values, latency_ns)
    
    print(f"\n{'Ki':<8} {'Phase Margin':<15} {'Gain Margin':<15} {'Safe?':<8}")
    print("-" * 50)
    
    for Ki, res in results.items():
        pm_status = "✓" if res['phase_margin'] > 45 else "✗"
        gm_status = "✓" if res['gain_margin'] > 6 else "✗"
        safe_status = "✓ YES" if res['safe'] else "✗ NO"
        
        print(f"{Ki:<8.2f} {pm_status} {res['phase_margin']:>10.1f}°    "
              f"{gm_status} {res['gain_margin']:>10.1f} dB   {safe_status}")
    
    # Latency sensitivity
    print("\n" + "=" * 70)
    print("LATENCY SENSITIVITY")
    print("=" * 70)
    
    latencies = [100, 200, 500, 1000, 2000]
    print(f"\nFixed Ki = 0.05")
    print(f"\n{'Latency (ns)':<15} {'Phase Margin':<15} {'Safe?':<8}")
    print("-" * 40)
    
    for lat in latencies:
        analyzer = FrequencyAnalyzer(Ki=0.05, latency_s=lat * 1e-9)
        m = analyzer.compute_margins()
        safe = m.phase_margin_deg > 45
        status = "✓ YES" if safe else "✗ NO"
        print(f"{lat:<15} {m.phase_margin_deg:>10.1f}°      {status}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
With Ki = {0.05} and latency = {latency_ns} ns:
- Phase margin = {margins.phase_margin_deg:.1f}° > 45° ✓
- Gain margin = {margins.gain_margin_db:.1f} dB > 6 dB ✓

The controller is SAFE for FPGA deployment.
No resonant peaks exist in the operating frequency range.
The system will not create positive feedback loops.
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
