"""
Gate Set Tomography (GST) Benchmark

Wrapper around pyGSTi for SPAM-robust Hamiltonian/gate characterization.
This provides a baseline comparison for MBL diagnostics.

Author: Justin Arndt
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Optional pyGSTi import
try:
    import pygsti
    from pygsti.models import modelconstruction as mc
    from pygsti.circuits import Circuit
    from pygsti.data import DataSet
    PYGSTI_AVAILABLE = True
except ImportError:
    PYGSTI_AVAILABLE = False


@dataclass
class GSTResult:
    """Results from GST analysis."""
    estimated_gates: Dict[str, np.ndarray]
    estimated_spam: Dict[str, np.ndarray]
    gate_fidelities: Dict[str, float]
    spam_error: float
    chi2_per_dof: float


class GSTBenchmark:
    """
    Gate Set Tomography benchmark for SPAM-robust characterization.
    
    GST is mathematically proven to be robust against SPAM errors,
    making it the gold standard for gate characterization in quantum
    computing hardware.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits (1 or 2 supported).
    gate_set : list
        List of gate names to characterize.
    max_length : int
        Maximum circuit length for GST.
    """
    
    def __init__(
        self,
        num_qubits: int = 1,
        gate_set: list = None,
        max_length: int = 32
    ):
        if not PYGSTI_AVAILABLE:
            raise ImportError(
                "pyGSTi not installed. Install with: pip install pygsti\n"
                "Or install this package with: pip install .[gst]"
            )
        
        self.num_qubits = num_qubits
        self.gate_set = gate_set or ['Gxpi2', 'Gypi2', 'Gi']
        self.max_length = max_length
        
        # Build target model
        self._build_target_model()
    
    def _build_target_model(self):
        """Build the ideal target gate set."""
        if self.num_qubits == 1:
            self.target_model = pygsti.models.create_explicit_model_from_expressions(
                ['Q0'],
                ['Gi', 'Gxpi2', 'Gypi2'],
                ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"],
                prep_labels=['rho0'],
                prep_expressions=['0'],
                effect_labels=['0', '1'],
                effect_expressions=['0', '1']
            )
        else:
            # 2-qubit GST
            self.target_model = pygsti.models.create_explicit_model_from_expressions(
                ['Q0', 'Q1'],
                ['Gi', 'Gxpi2:Q0', 'Gypi2:Q0', 'Gcnot'],
                ["I(Q0):I(Q1)", "X(pi/2,Q0):I(Q1)", 
                 "Y(pi/2,Q0):I(Q1)", "CNOT(Q0,Q1)"]
            )
    
    def generate_circuits(self) -> List:
        """
        Generate GST circuit set.
        
        Returns
        -------
        list
            List of pyGSTi circuits for GST.
        """
        from pygsti.circuits import make_lsgst_experiment_list
        
        fiducials = pygsti.circuits.to_circuits(['{}', 'Gxpi2', 'Gypi2', 'Gxpi2:Gxpi2'])
        germs = pygsti.circuits.to_circuits(['Gi', 'Gxpi2', 'Gypi2', 'Gxpi2:Gypi2'])
        max_lengths = [1, 2, 4, 8, 16, 32][:self.max_length.bit_length()]
        
        circuits = make_lsgst_experiment_list(
            self.target_model,
            fiducials, fiducials,
            germs, max_lengths
        )
        
        return circuits
    
    def simulate_experiment(
        self,
        noisy_model: Optional[object] = None,
        num_samples: int = 1000
    ) -> DataSet:
        """
        Simulate GST experiment with optional noise.
        
        Parameters
        ----------
        noisy_model : object, optional
            pyGSTi model with noise. If None, uses target model.
        num_samples : int
            Number of samples per circuit.
            
        Returns
        -------
        DataSet
            Simulated experiment data.
        """
        circuits = self.generate_circuits()
        model = noisy_model or self.target_model
        
        dataset = pygsti.data.simulate_data(
            model, circuits,
            num_samples=num_samples,
            sample_error='multinomial'
        )
        
        return dataset
    
    def run_gst(
        self,
        dataset: DataSet,
        verbosity: int = 0
    ) -> GSTResult:
        """
        Run GST protocol on dataset.
        
        Parameters
        ----------
        dataset : DataSet
            Experimental data.
        verbosity : int
            Output verbosity.
            
        Returns
        -------
        GSTResult
            Estimated gate set with fidelities.
        """
        from pygsti.protocols import StandardGST
        
        protocol = StandardGST(
            modes=['TP', 'CPTP'],
            gaugeopt_suite='single'
        )
        
        results = protocol.run(
            self.target_model,
            dataset,
            verbosity=verbosity
        )
        
        # Extract estimates
        estimated_model = results.estimates['CPTP'].models['stdgaugeopt']
        
        # Compute gate fidelities
        gate_fidelities = {}
        for gate_name in self.gate_set:
            if gate_name in estimated_model.operations:
                target_gate = self.target_model.operations[gate_name]
                estimated_gate = estimated_model.operations[gate_name]
                fidelity = pygsti.tools.entanglement_fidelity(
                    target_gate, estimated_gate
                )
                gate_fidelities[gate_name] = float(fidelity)
        
        return GSTResult(
            estimated_gates={g: np.array(estimated_model.operations[g]) 
                           for g in self.gate_set if g in estimated_model.operations},
            estimated_spam={},
            gate_fidelities=gate_fidelities,
            spam_error=0.0,
            chi2_per_dof=results.estimates['CPTP'].chi2_per_dof
        )


def compare_mbl_vs_gst(
    true_couplings: np.ndarray,
    mbl_recovered: np.ndarray,
    gst_fidelities: Dict[str, float]
) -> Dict:
    """
    Compare MBL and GST diagnostic results.
    
    Parameters
    ----------
    true_couplings : np.ndarray
        Ground truth coupling values.
    mbl_recovered : np.ndarray
        MBL-recovered coupling values.
    gst_fidelities : dict
        GST gate fidelities.
        
    Returns
    -------
    dict
        Comparison metrics.
    """
    mbl_error = np.mean(np.abs(true_couplings - mbl_recovered))
    mbl_max_error = np.max(np.abs(true_couplings - mbl_recovered))
    
    avg_gst_fidelity = np.mean(list(gst_fidelities.values()))
    
    return {
        "mbl_mean_error": mbl_error,
        "mbl_max_error": mbl_max_error,
        "gst_avg_fidelity": avg_gst_fidelity,
        "gst_min_fidelity": min(gst_fidelities.values()) if gst_fidelities else 0,
        "mbl_passed": mbl_error < 0.02,  # <2e-2 threshold
        "gst_passed": avg_gst_fidelity > 0.99
    }
