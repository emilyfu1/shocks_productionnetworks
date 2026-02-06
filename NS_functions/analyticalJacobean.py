"""
Template for analytical Jacobian implementation.

This file provides the mathematical derivation and a template for implementing
the analytical Jacobian of the objective function.

EQUATION:
F(P_vec) = P_vec^(1-ε) - diag(1-γ)@P_VA^(1-ε) - diag(γ)@(Ω@P_vec)^((1-ε)/(1-σ))

Equivalently, for each element i:
F_i(P_vec) = P_vec[i]^(1-ε) - (1-γ[i])*P_VA[i]^(1-ε) - γ[i]*(Σ_k Ω[i,k]*P_vec[k])^α

where α = (1-ε)/(1-σ)

JACOBIAN:
We need to compute J[i,j] = ∂F_i/∂P_vec[j]

DERIVATION:

1. Direct term (i=j):
   ∂/∂P_vec[i] [P_vec[i]^(1-ε)] = (1-ε) * P_vec[i]^(-ε)

2. P_VA term (independent of P_vec):
   ∂/∂P_vec[j] [(1-γ[i])*P_VA[i]^(1-ε)] = 0

3. Composite price term (complex):
   Let Q[i] = Σ_k Ω[i,k]*P_vec[k]  (composite price)
   
   ∂/∂P_vec[j] [γ[i] * Q[i]^α]
   = γ[i] * α * Q[i]^(α-1) * ∂Q[i]/∂P_vec[j]
   = γ[i] * α * Q[i]^(α-1) * Ω[i,j]

FINAL FORMULA:

For i = j (diagonal):
J[i,i] = (1-ε) * P_vec[i]^(-ε) - γ[i] * α * Q[i]^(α-1) * Ω[i,i]

For i ≠ j (off-diagonal):
J[i,j] = -γ[i] * α * Q[i]^(α-1) * Ω[i,j]

Where:
- α = (1-ε)/(1-σ)
- Q[i] = (Ω @ P_vec)[i]
"""

import numpy as np
import scipy.sparse as sp
from typing import Union
from .newton_solver import SolverParams


def analytical_jacobian_template(P_vec: np.ndarray, params: SolverParams) -> np.ndarray:
    """
    Analytical Jacobian of the objective function F(P_vec).
    
    Args:
        P_vec: Current price vector (n,)
        params: SolverParams object containing equation parameters
        
    Returns:
        Jacobian matrix J where J[i,j] = ∂F_i/∂P_vec[j]
    """
    n = len(P_vec)
    
    # Extract parameters
    epsilon = params.epsilon
    sigma = params.sigma
    gamma = params.gamma
    Omega = params.Omega
    
    # Compute exponents
    exp_price = 1 - epsilon  # For P_vec terms
    exp_composite = (1 - epsilon) / (1 - sigma)  # α
    
    # Compute composite prices Q[i] = (Ω @ P_vec)[i]
    if sp.issparse(Omega):
        Q = Omega @ P_vec
    else:
        Q = Omega @ P_vec
    
    # Initialize Jacobian
    J = np.zeros((n, n))
    
    # Compute diagonal terms: ∂F_i/∂P_vec[i]
    for i in range(n):
        # Direct term: (1-ε) * P_vec[i]^(-ε)
        direct_term = exp_price * (P_vec[i] ** (-epsilon))
        
        # Composite term contribution: -γ[i] * α * Q[i]^(α-1) * Ω[i,i]
        if Q[i] > 0:  # Avoid issues with Q=0
            if sp.issparse(Omega):
                omega_ii = Omega[i, i]
            else:
                omega_ii = Omega[i, i]
            
            composite_term = -gamma[i] * exp_composite * (Q[i] ** (exp_composite - 1)) * omega_ii
        else:
            composite_term = 0.0
        
        J[i, i] = direct_term + composite_term
    
    # Compute off-diagonal terms: ∂F_i/∂P_vec[j] for i≠j
    for i in range(n):
        if Q[i] > 0:
            Q_power = Q[i] ** (exp_composite - 1)
            coeff = -gamma[i] * exp_composite * Q_power
            
            for j in range(n):
                if i != j:
                    if sp.issparse(Omega):
                        omega_ij = Omega[i, j]
                    else:
                        omega_ij = Omega[i, j]
                    
                    J[i, j] = coeff * omega_ij
    
    return J


def analytical_jacobian_vectorized(P_vec: np.ndarray, params: SolverParams) -> np.ndarray:
    """
    Vectorized version of analytical Jacobian (faster for dense matrices).
    
    Args:
        P_vec: Current price vector (n,)
        params: SolverParams object
        
    Returns:
        Jacobian matrix J
    """
    n = len(P_vec)
    
    # Extract parameters
    epsilon = params.epsilon
    sigma = params.sigma
    gamma = params.gamma
    Omega = params.Omega
    
    # Compute exponents
    exp_price = 1 - epsilon
    exp_composite = (1 - epsilon) / (1 - sigma)
    
    # Compute composite prices
    if sp.issparse(Omega):
        Q = Omega @ P_vec
        Omega_dense = Omega.toarray()
    else:
        Q = Omega @ P_vec
        Omega_dense = Omega
    
    # Avoid numerical issues with Q=0
    Q_safe = np.where(Q > 0, Q, 1.0)
    Q_power = Q_safe ** (exp_composite - 1)
    Q_power = np.where(Q > 0, Q_power, 0.0)
    
    # Compute Jacobian
    # J[i,j] = -γ[i] * α * Q[i]^(α-1) * Ω[i,j] for all i,j
    J = -np.outer(gamma * exp_composite * Q_power, np.ones(n)) * Omega_dense
    
    # Add diagonal correction for direct term
    # J[i,i] += (1-ε) * P_vec[i]^(-ε)
    diagonal_correction = exp_price * (P_vec ** (-epsilon))
    J[np.arange(n), np.arange(n)] += diagonal_correction
    
    return J


def analytical_jacobian_sparse(P_vec: np.ndarray, params: SolverParams) -> sp.csr_matrix:
    """
    Sparse version of analytical Jacobian (for very large sparse Omega).
    
    Args:
        P_vec: Current price vector (n,)
        params: SolverParams object
        
    Returns:
        Jacobian matrix J as sparse CSR matrix
    """
    n = len(P_vec)
    
    # Extract parameters
    epsilon = params.epsilon
    sigma = params.sigma
    gamma = params.gamma
    Omega = params.Omega
    
    # Compute exponents
    exp_price = 1 - epsilon
    exp_composite = (1 - epsilon) / (1 - sigma)
    
    # Compute composite prices
    if sp.issparse(Omega):
        Q = Omega @ P_vec
    else:
        Q = Omega @ P_vec
        Omega = sp.csr_matrix(Omega)
    
    # Avoid numerical issues
    Q_safe = np.where(Q > 0, Q, 1.0)
    Q_power = Q_safe ** (exp_composite - 1)
    Q_power = np.where(Q > 0, Q_power, 0.0)
    
    # Create sparse Jacobian
    # J[i,j] = -γ[i] * α * Q[i]^(α-1) * Ω[i,j]
    row_scaling = -gamma * exp_composite * Q_power
    
    # Create diagonal matrix for row scaling
    D = sp.diags(row_scaling, format='csr')
    
    # J = D @ Omega
    J = D @ Omega
    
    # Add diagonal correction
    diagonal_correction = exp_price * (P_vec ** (-epsilon))
    J.setdiag(J.diagonal() + diagonal_correction)
    
    return J


# Example usage
if __name__ == "__main__":
    """
    Test the analytical Jacobian against numerical approximation.
    """
    from newton_solver import numerical_jacobian, SolverParams
    
    # Create small test problem
    n = 10
    epsilon = 0.5
    sigma = 0.3
    gamma = np.random.uniform(0.3, 0.7, n)
    Omega = sp.random(n, n, density=0.3, format='csr')
    Omega = Omega.toarray()
    Omega = Omega / Omega.sum(axis=1, keepdims=True)
    P_VA = np.random.uniform(0.8, 1.2, n)
    P_vec = np.random.uniform(0.9, 1.1, n)
    
    params = SolverParams(epsilon, sigma, gamma, Omega, P_VA)
    
    # Compute analytical Jacobian
    print("Computing analytical Jacobian...")
    J_analytical = analytical_jacobian_template(P_vec, params)
    
    # Compute numerical Jacobian
    print("Computing numerical Jacobian...")
    J_numerical = numerical_jacobian(P_vec, params, step_size=1e-7)
    
    # Compare
    diff = np.abs(J_analytical - J_numerical)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    rel_diff = np.max(diff / (np.abs(J_numerical) + 1e-10))
    
    print(f"\nComparison:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Average absolute difference: {avg_diff:.6e}")
    print(f"  Max relative difference: {rel_diff:.6e}")
    
    if max_diff < 1e-5:
        print("\n✓ Analytical Jacobian matches numerical approximation!")
    else:
        print("\n✗ Warning: Large difference detected. Check implementation.")
    
    # Test vectorized version
    print("\nTesting vectorized version...")
    J_vectorized = analytical_jacobian_vectorized(P_vec, params)
    diff_vec = np.abs(J_analytical - J_vectorized)
    print(f"  Max difference from template version: {np.max(diff_vec):.6e}")
    
    # Test sparse version
    print("\nTesting sparse version...")
    params_sparse = SolverParams(epsilon, sigma, gamma, sp.csr_matrix(Omega), P_VA)
    J_sparse = analytical_jacobian_sparse(P_vec, params_sparse)
    J_sparse_dense = J_sparse.toarray()
    diff_sparse = np.abs(J_analytical - J_sparse_dense)
    print(f"  Max difference from template version: {np.max(diff_sparse):.6e}")