#!/usr/bin/env python3
"""
Example: Coherent Error Remediation

Demonstrates fidelity recovery on hardware with injected coherent defects:
- Over-rotation errors
- ZZ crosstalk
- Weak coupling links

Uses Cirq for coherent noise modeling + MBL diagnostics + pulse synthesis.

Author: Justin Arndt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
from adaptive_qec.remediation.pulse_synthesis import PulseSynthesizer


def run_coherent_remediation_demo(
    system_size: int = 6,
    defect_strength: float = 0.5,
    crosstalk_strength: float = 1.2,
    max_iterations: int = 500,
    save_plots: bool = True
):
    """
    Demonstrate coherent error remediation pipeline.
    
    Parameters
    ----------
    system_size : int
        Number of qubits.
    defect_strength : float
        Weak link coupling (fraction of nominal).
    crosstalk_strength : float
        Enhanced crosstalk coupling (fraction of nominal).
    max_iterations : int
        Optimizer iterations for pulse synthesis.
    save_plots : bool
        Whether to save result plots.
    """
    print("=" * 70)
    print("COHERENT ERROR REMEDIATION DEMO")
    print(f"System: L={system_size}, Defect: {defect_strength}J, Crosstalk: {crosstalk_strength}J")
    print("=" * 70)
    
    # Setup defective hardware
    J_nominal = 1.0
    J_defect = np.ones(system_size - 1) * J_nominal
    J_defect[2] = defect_strength  # Weak link
    J_defect[4] = crosstalk_strength  # Enhanced crosstalk
    
    h_fields = HamiltonianLearner.generate_aubry_andre_fields(system_size, disorder_strength=6.0)
    
    print(f"\nDefective Hardware: {np.round(J_defect, 2)}")
    print(f"Defect at position 2: {defect_strength} (weak)")
    print(f"Crosstalk at position 4: {crosstalk_strength} (enhanced)")
    
    # Step 1: Diagnose with MBL
    print("\n--- Step 1: Hardware Diagnosis (MBL) ---")
    learner = HamiltonianLearner(system_size=system_size)
    
    t_points = np.linspace(0, 10.0, 100)
    trace_exp = learner.simulate_dynamics(J_defect, h_fields, t_points)
    trace_exp += np.random.normal(0, 0.002, size=trace_exp.shape)
    
    J_recovered, fit_error = learner.learn_hamiltonian(trace_exp, t_points, h_fields)
    defects = learner.detect_defects(J_recovered, J_nominal, threshold=0.15)
    
    print(f"Recovered Couplings: {np.round(J_recovered, 3)}")
    print(f"Detected Weak Links: {defects['weak_couplings']}")
    print(f"Detected Strong Links: {defects['strong_couplings']}")
    recovery_error = np.max(np.abs(J_recovered - J_defect))
    print(f"Max Recovery Error: {recovery_error:.2e}")
    
    # Step 2: Remediate with pulse synthesis
    print("\n--- Step 2: Pulse Remediation ---")
    synth = PulseSynthesizer(system_size=system_size, gate_time=8.0, dt=0.2)
    
    # Baseline: no control
    zero_pulse = np.zeros((synth.num_steps, system_size))
    final_std = synth.evolve_with_control(J_defect, h_fields, zero_pulse)
    
    target_idx = int("".join(["10"] * (system_size // 2)), 2)
    target_psi = np.zeros(synth.dim, dtype=complex)
    target_psi[target_idx] = 1.0
    
    fidelity_std = np.abs(np.vdot(target_psi, final_std)) ** 2
    print(f"Baseline Fidelity (no control): {fidelity_std * 100:.2f}%")
    
    # Synthesize remediation pulse
    optimal_pulse, fidelity_opt = synth.synthesize(
        J_recovered, h_fields,
        max_iterations=max_iterations,
        verbose=True
    )
    
    improvement = fidelity_opt / max(fidelity_std, 1e-10)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline Fidelity:    {fidelity_std * 100:.2f}%")
    print(f"Remediated Fidelity:  {fidelity_opt * 100:.2f}%")
    print(f"Improvement Factor:   {improvement:.1f}x")
    print(f"Recovery Success:     {'YES' if fidelity_opt > 0.99 else 'PARTIAL'}")
    
    # Plot
    if save_plots:
        plot_remediation_results(
            J_defect, J_recovered, optimal_pulse, synth,
            fidelity_std, fidelity_opt, defects
        )
    
    return {
        "J_true": J_defect,
        "J_recovered": J_recovered,
        "recovery_error": recovery_error,
        "defects": defects,
        "fidelity_baseline": fidelity_std,
        "fidelity_remediated": fidelity_opt,
        "improvement": improvement
    }


def plot_remediation_results(J_true, J_recovered, pulse, synth, fid_std, fid_opt, defects):
    """Generate remediation results visualization."""
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 12))
    
    L = len(J_true) + 1
    
    # Top left: Hardware diagnosis
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(J_true))
    width = 0.35
    
    ax1.bar(x - width/2, J_true, width, label='True Defects', color='#c0392b', alpha=0.8)
    ax1.bar(x + width/2, J_recovered, width, label='Recovered (MBL)', color='#27ae60', alpha=0.8)
    ax1.axhline(1.0, color='gray', linestyle='--', label='Nominal', linewidth=1)
    
    ax1.set_xlabel("Coupling Index", fontsize=11)
    ax1.set_ylabel("Coupling Strength (J)", fontsize=11)
    ax1.set_title("Step 1: Hardware Diagnosis via MBL", fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"J{i}{i+1}" for i in range(len(J_true))])
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.5)
    
    # Highlight defects
    for i in defects.get('weak_couplings', []):
        ax1.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.1)
    for i in defects.get('strong_couplings', []):
        ax1.axvspan(i - 0.5, i + 0.5, color='blue', alpha=0.1)
    
    # Top right: Control pulse
    ax2 = fig.add_subplot(2, 2, 2)
    time_axis = np.linspace(0, synth.T_gate, synth.num_steps)
    
    for q in range(min(4, L)):
        ax2.plot(time_axis, pulse[:, q], label=f'Qubit {q}', linewidth=2)
    
    ax2.set_xlabel("Time (a.u.)", fontsize=11)
    ax2.set_ylabel("Detuning Δ(t)", fontsize=11)
    ax2.set_title("Step 2: Synthesized Remediation Pulse", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    
    # Bottom left: Fidelity comparison
    ax3 = fig.add_subplot(2, 2, 3)
    bars = ax3.barh(
        ['Standard\n(No Control)', 'Remediated\n(Optimized Pulse)'],
        [fid_std, fid_opt],
        color=['#95a5a6', '#8e44ad'],
        height=0.5
    )
    ax3.axvline(0.99, color='red', linestyle=':', linewidth=2, label='99% Threshold')
    ax3.set_xlabel("Gate Fidelity", fontsize=12)
    ax3.set_title(f"Fidelity Recovery: {fid_std*100:.1f}% → {fid_opt*100:.1f}%", 
                  fontsize=13, fontweight='bold')
    ax3.set_xlim(0, 1.05)
    ax3.legend(loc='lower right', fontsize=10)
    
    for bar, fid in zip(bars, [fid_std, fid_opt]):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{fid*100:.1f}%', va='center', fontsize=12, fontweight='bold')
    
    # Bottom right: Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary = f"""
    COHERENT ERROR REMEDIATION SUMMARY
    ===================================
    
    Hardware Defects:
      • Weak link at position 2: {J_true[2]:.1f}J
      • Crosstalk at position 4: {J_true[4]:.1f}J
    
    Diagnosis Accuracy:
      • Max recovery error: {np.max(np.abs(J_recovered - J_true)):.2e}
      • Weak links detected: {defects['weak_couplings']}
      • Strong links detected: {defects['strong_couplings']}
    
    Remediation Results:
      • Baseline fidelity: {fid_std*100:.2f}%
      • Remediated fidelity: {fid_opt*100:.2f}%
      • Improvement: {fid_opt/max(fid_std, 1e-10):.0f}x
      
    STATUS: {'✅ SUCCESS' if fid_opt > 0.99 else '⚠️ PARTIAL'}
    """
    
    ax4.text(0.05, 0.95, summary, fontsize=11, family='monospace',
             verticalalignment='top', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "coherent_remediation.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: docs/coherent_remediation.png")
    plt.close()


if __name__ == "__main__":
    run_coherent_remediation_demo()
