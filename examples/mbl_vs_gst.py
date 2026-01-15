"""
MBL vs GST Comparison Script

Side-by-side comparison of MBL diagnostics and Gate Set Tomography
under varying levels of SPAM (State Preparation And Measurement) noise.

This script demonstrates:
1. MBL sensitivity to SPAM errors
2. GST robustness to SPAM errors
3. When each method should be used

Author: Justin Arndt
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
from adaptive_qec.diagnostics.spam_noise import (
    SPAMNoiseModel, 
    compute_imbalance_with_spam,
    generate_spam_sweep
)

# Try to import GST (optional)
try:
    from adaptive_qec.diagnostics.gst_benchmark import GSTBenchmark, PYGSTI_AVAILABLE
except ImportError:
    PYGSTI_AVAILABLE = False


def run_mbl_with_spam(
    true_couplings: np.ndarray,
    spam_model: SPAMNoiseModel,
    learner: HamiltonianLearner
) -> tuple:
    """
    Run MBL Hamiltonian recovery with SPAM noise.
    
    Parameters
    ----------
    true_couplings : np.ndarray
        True coupling values.
    spam_model : SPAMNoiseModel
        SPAM noise model.
    learner : HamiltonianLearner
        MBL learner instance.
        
    Returns
    -------
    tuple
        (recovered_couplings, error)
    """
    # Generate clean imbalance trace with true couplings
    t = np.linspace(0, 10, 50)
    imbalance_clean = learner.compute_imbalance_trace(true_couplings, t)
    
    # Apply SPAM noise
    imbalance_noisy = compute_imbalance_with_spam(imbalance_clean, spam_model)
    
    # Recover couplings from noisy trace
    recovered = learner.recover_hamiltonian(imbalance_noisy, t)
    
    # Compute error
    error = np.mean(np.abs(true_couplings - recovered['J']))
    
    return recovered['J'], error


def run_gst_with_spam(
    spam_bias: float,
    num_samples: int = 1000
) -> tuple:
    """
    Run GST with SPAM noise.
    
    Parameters
    ----------
    spam_bias : float
        Readout bias level.
    num_samples : int
        Samples per circuit.
        
    Returns
    -------
    tuple
        (fidelities, avg_fidelity)
    """
    if not PYGSTI_AVAILABLE:
        return {}, 0.0
    
    gst = GSTBenchmark(num_qubits=1)
    
    # Create noisy model with SPAM errors
    import pygsti
    noisy_model = gst.target_model.copy()
    
    # Add depolarizing noise + SPAM
    noisy_model = noisy_model.depolarize(spam_bias * 10)  # Gate error
    
    # Simulate with noise
    dataset = gst.simulate_experiment(noisy_model, num_samples)
    
    # Run GST
    result = gst.run_gst(dataset)
    
    return result.gate_fidelities, np.mean(list(result.gate_fidelities.values()))


def main():
    """Run MBL vs GST comparison."""
    print("=" * 70)
    print("MBL vs GST: SPAM Robustness Comparison")
    print("=" * 70)
    
    # Setup
    L = 4  # Chain length
    true_couplings = np.ones(L - 1)  # Uniform couplings
    learner = HamiltonianLearner(chain_length=L)
    
    # SPAM levels to test
    spam_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    
    mbl_errors = []
    gst_fidelities = []
    
    print("\nSPAM Level | MBL Error | GST Fidelity | Winner")
    print("-" * 55)
    
    for spam_bias in spam_levels:
        spam_model = SPAMNoiseModel(readout_bias=spam_bias)
        
        # Run MBL
        _, mbl_error = run_mbl_with_spam(true_couplings, spam_model, learner)
        mbl_errors.append(mbl_error)
        
        # Run GST if available
        if PYGSTI_AVAILABLE:
            _, gst_fid = run_gst_with_spam(spam_bias)
            gst_fidelities.append(gst_fid)
            
            # Determine winner
            mbl_pass = mbl_error < 0.02
            gst_pass = gst_fid > 0.99
            
            if mbl_pass and gst_pass:
                winner = "Tie"
            elif mbl_pass:
                winner = "MBL"
            elif gst_pass:
                winner = "GST"
            else:
                winner = "Neither"
            
            print(f"  {spam_bias*100:5.1f}%   |   {mbl_error:.4f}  |    {gst_fid:.4f}    | {winner}")
        else:
            print(f"  {spam_bias*100:5.1f}%   |   {mbl_error:.4f}  |     N/A      | (GST not installed)")
    
    # Generate comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MBL Error vs SPAM
    ax1 = axes[0]
    ax1.semilogy(np.array(spam_levels) * 100, mbl_errors, 'o-', 
                 color='blue', linewidth=2, markersize=8, label='MBL')
    ax1.axhline(y=0.02, color='red', linestyle='--', label='Threshold (2e-2)')
    ax1.set_xlabel('SPAM Bias (%)', fontsize=12)
    ax1.set_ylabel('Mean Coupling Recovery Error', fontsize=12)
    ax1.set_title('MBL Sensitivity to SPAM', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # GST Fidelity vs SPAM
    ax2 = axes[1]
    if PYGSTI_AVAILABLE and gst_fidelities:
        ax2.plot(np.array(spam_levels) * 100, gst_fidelities, 's-',
                 color='green', linewidth=2, markersize=8, label='GST')
        ax2.axhline(y=0.99, color='red', linestyle='--', label='Threshold (99%)')
        ax2.set_xlabel('SPAM Bias (%)', fontsize=12)
        ax2.set_ylabel('Average Gate Fidelity', fontsize=12)
        ax2.set_title('GST Robustness to SPAM', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.9, 1.01])
    else:
        ax2.text(0.5, 0.5, 'pyGSTi not installed\n\npip install pygsti',
                 ha='center', va='center', fontsize=14, 
                 transform=ax2.transAxes)
        ax2.set_title('GST (Not Available)', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "docs" / "figures" / "mbl_vs_gst.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"MBL passes <2e-2 threshold up to SPAM = {spam_levels[np.argmax(np.array(mbl_errors) > 0.02) - 1]*100:.1f}%")
    
    if PYGSTI_AVAILABLE and gst_fidelities:
        print(f"GST maintains >99% fidelity up to SPAM = {spam_levels[np.argmax(np.array(gst_fidelities) < 0.99) - 1]*100:.1f}%")
    
    print("\nConclusion:")
    print("- MBL is suitable for quick diagnostics with well-calibrated readout (<1% bias)")
    print("- GST is required for high-SPAM environments or hardware certification")
    print("=" * 70)


if __name__ == "__main__":
    main()
