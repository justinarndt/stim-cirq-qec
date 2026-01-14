"""
Hamiltonian Learning for Hardware Diagnostics

MBL-based inverse problem solving to detect hardware defects.

Author: Justin Arndt
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.optimize import minimize
from typing import Tuple, Optional


class HamiltonianLearner:
    """
    Digital twin for quantum hardware diagnostics via MBL benchmarks.
    """
    
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    
    def __init__(self, system_size: int = 6):
        self.L = system_size
        self.dim = 2 ** system_size
        
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
            full_term = self._kron_chain(term)
            self.ops_XX.append(full_term)
        
        for i in range(self.L):
            term = [self.id] * self.L
            term[i] = self.sz
            full_term = self._kron_chain(term)
            self.ops_Z.append(full_term)
        
        self.imbalance_op = sum([((-1)**i) * self.ops_Z[i] for i in range(self.L)])
    
    def _kron_chain(self, ops: list) -> sparse.csr_matrix:
        result = ops[0]
        for op in ops[1:]:
            result = sparse.kron(result, op, format='csr')
        return result
    
    def simulate_dynamics(
        self,
        J_couplings: np.ndarray,
        h_fields: np.ndarray,
        t_points: np.ndarray
    ) -> np.ndarray:
        """Simulate MBL dynamics and return imbalance trace."""
        H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        for i, J in enumerate(J_couplings):
            H += J * self.ops_XX[i]
        for i, h in enumerate(h_fields):
            H += h * self.ops_Z[i]
        
        psi = np.zeros(self.dim, dtype=complex)
        neel_idx = int("".join(["01"] * (self.L // 2)), 2)
        psi[neel_idx] = 1.0
        
        imbalances = []
        for t in t_points:
            psi_t = splinalg.expm_multiply(-1j * H * t, psi)
            I_t = np.vdot(psi_t, self.imbalance_op.dot(psi_t)).real / self.L
            imbalances.append(I_t)
        
        return np.array(imbalances)
    
    def learn_hamiltonian(
        self,
        experimental_trace: np.ndarray,
        t_points: np.ndarray,
        h_fields: np.ndarray,
        initial_guess: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Recover coupling parameters from experimental data."""
        if initial_guess is None:
            initial_guess = np.ones(self.L - 1)
        
        bounds = [(0.0, 2.0) for _ in range(self.L - 1)]
        
        def loss(J_guess):
            sim_trace = self.simulate_dynamics(J_guess, h_fields, t_points)
            return np.mean((sim_trace - experimental_trace) ** 2) * 1e5
        
        result = minimize(loss, initial_guess, method='L-BFGS-B', bounds=bounds,
                         options={'ftol': 1e-9, 'maxiter': 500})
        
        return result.x, result.fun / 1e5
    
    def detect_defects(self, J_recovered: np.ndarray, J_nominal: float = 1.0,
                       threshold: float = 0.1) -> dict:
        """Identify weak/strong couplings."""
        weak = np.where(J_recovered < J_nominal * (1 - threshold))[0]
        strong = np.where(J_recovered > J_nominal * (1 + threshold))[0]
        return {"weak_couplings": weak.tolist(), "strong_couplings": strong.tolist()}
    
    @staticmethod
    def generate_aubry_andre_fields(L: int, disorder_strength: float = 6.0) -> np.ndarray:
        """Generate quasi-periodic on-site fields."""
        beta = HamiltonianLearner.GOLDEN_RATIO
        return disorder_strength * np.cos(2 * np.pi * beta * np.arange(L))
