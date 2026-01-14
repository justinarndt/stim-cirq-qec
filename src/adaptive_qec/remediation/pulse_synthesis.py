"""
Pulse Synthesis for Fidelity Recovery

Optimal control on defective hardware.

Author: Justin Arndt
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.optimize import minimize
from typing import Tuple, Optional


class PulseSynthesizer:
    """
    Physics-aware optimal control for fidelity recovery on broken hardware.
    """
    
    def __init__(self, system_size: int = 6, gate_time: float = 8.0, dt: float = 0.2):
        self.L = system_size
        self.dim = 2 ** system_size
        self.T_gate = gate_time
        self.dt = dt
        self.num_steps = int(gate_time / dt)
        
        self.sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        self.sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
        self.id = sparse.eye(2)
        
        self._build_operators()
    
    def _build_operators(self):
        self.ops_XX = []
        self.ops_Z = []
        
        for i in range(self.L - 1):
            term = [self.id] * self.L
            term[i] = self.sx
            term[i + 1] = self.sx
            self.ops_XX.append(self._kron_chain(term))
        
        for i in range(self.L):
            term = [self.id] * self.L
            term[i] = self.sz
            self.ops_Z.append(self._kron_chain(term))
    
    def _kron_chain(self, ops):
        result = ops[0]
        for op in ops[1:]:
            result = sparse.kron(result, op, format='csr')
        return result
    
    def evolve_with_control(
        self,
        J_couplings: np.ndarray,
        h_fields: np.ndarray,
        control_pulse: np.ndarray
    ) -> np.ndarray:
        """Evolve system under control pulse."""
        H_drift = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        for i, J in enumerate(J_couplings):
            H_drift += J * self.ops_XX[i]
        for i, h in enumerate(h_fields):
            H_drift += h * self.ops_Z[i]
        
        psi = np.zeros(self.dim, dtype=complex)
        neel_idx = int("".join(["01"] * (self.L // 2)), 2)
        psi[neel_idx] = 1.0
        
        for step in range(self.num_steps):
            H_t = H_drift.copy()
            for i, amp in enumerate(control_pulse[step]):
                H_t += amp * self.ops_Z[i]
            psi = splinalg.expm_multiply(-1j * H_t * self.dt, psi)
        
        return psi
    
    def synthesize(
        self,
        J_diagnosed: np.ndarray,
        h_fields: np.ndarray,
        max_iterations: int = 500,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """Synthesize optimal control pulse for fidelity recovery."""
        target_idx = int("".join(["10"] * (self.L // 2)), 2)
        target_psi = np.zeros(self.dim, dtype=complex)
        target_psi[target_idx] = 1.0
        
        initial_controls = np.random.normal(0, 2.0, size=self.num_steps * self.L)
        
        def loss(ctrl_flat):
            ctrl = ctrl_flat.reshape((self.num_steps, self.L))
            final_psi = self.evolve_with_control(J_diagnosed, h_fields, ctrl)
            fidelity = np.abs(np.vdot(target_psi, final_psi)) ** 2
            infidelity = 1.0 - fidelity
            
            diffs = np.diff(ctrl, axis=0)
            smoothness = np.sum(diffs ** 2) * 0.0001
            power = np.sum(ctrl ** 2) * 0.00001
            
            return infidelity * 100 + smoothness + power
        
        if verbose:
            print(f"Synthesizing control pulse ({max_iterations} max iterations)...")
        
        result = minimize(loss, initial_controls, method='L-BFGS-B',
                         options={'maxiter': max_iterations, 'ftol': 1e-6})
        
        optimal_pulse = result.x.reshape((self.num_steps, self.L))
        final_psi = self.evolve_with_control(J_diagnosed, h_fields, optimal_pulse)
        fidelity = np.abs(np.vdot(target_psi, final_psi)) ** 2
        
        if verbose:
            print(f"  Final fidelity: {fidelity * 100:.2f}%")
        
        return optimal_pulse, fidelity
