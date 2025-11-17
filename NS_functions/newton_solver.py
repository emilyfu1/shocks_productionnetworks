"""
Flexible Newton-Raphson solver for systems of nonlinear equations
with configurable Jacobian update strategies.

Designed for solving equations of the form:
P_vec**(1-epsilon) = diag(1-gamma) @ P_VA**(1-epsilon) + diag(gamma) @ (Omega @ P_vec)**((1-epsilon)/(1-sigma))
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Callable, Dict, Tuple, Optional, Union
import warnings


class SolverParams:
    """Container for equation parameters."""
    def __init__(self, epsilon: float, sigma: float, gamma: np.ndarray, 
                 Omega: Union[np.ndarray, sp.spmatrix], P_VA: np.ndarray):
        self.epsilon = epsilon
        self.sigma = sigma
        self.gamma = gamma
        self.Omega = Omega
        self.P_VA = P_VA
        
        # Pre-compute constants for efficiency
        self.one_minus_gamma = 1 - gamma
        self.exp_price = 1 - epsilon
        self.exp_composite = (1 - epsilon) / (1 - sigma)


def objective_function(P_vec: np.ndarray, params: SolverParams) -> np.ndarray:
    """
    Compute the residual function F(P_vec).
    
    F(P_vec) = P_vec^(1-ε) - diag(1-γ)@P_VA^(1-ε) - diag(γ)@(Ω@P_vec^(1-sigma))^((1-ε)/(1-σ))
    
    Args:
        P_vec: Current price vector
        params: SolverParams object containing equation parameters
        
    Returns:
        Residual vector
    """
    # Left side
    lhs = P_vec ** params.exp_price
    
    # Right side - first term
    rhs_term1 = params.one_minus_gamma * (params.P_VA ** params.exp_price)
    
    # Right side - second term
    if sp.issparse(params.Omega):
        composite_price = params.Omega @ P_vec** (1-params.sigma)
    else:
        composite_price = params.Omega @ P_vec** (1-params.sigma)
    
    rhs_term2 = params.gamma * (composite_price ** params.exp_composite)
    
    # Residual
    residual = lhs - rhs_term1 - rhs_term2
    
    return residual


def numerical_jacobian(P_vec: np.ndarray, params: SolverParams, 
                       step_size: float = 1e-7) -> Union[np.ndarray, sp.spmatrix]:
    """
    Compute Jacobian using central differences.
    
    Args:
        P_vec: Point at which to evaluate Jacobian
        params: SolverParams object
        step_size: Step size for finite differences
        
    Returns:
        Jacobian matrix (n x n)
    """
    n = len(P_vec)
    F_center = objective_function(P_vec, params)
    
    # Pre-allocate Jacobian
    jacobian = np.zeros((n, n))
    
    for i in range(n):
        # Forward perturbation
        P_forward = P_vec.copy()
        h = step_size * max(abs(P_vec[i]), 1.0)  # Adaptive step size
        P_forward[i] += h
        F_forward = objective_function(P_forward, params)
        
        # Backward perturbation
        P_backward = P_vec.copy()
        P_backward[i] -= h
        F_backward = objective_function(P_backward, params)
        
        # Central difference
        jacobian[:, i] = (F_forward - F_backward) / (2 * h)
    
    return jacobian


def analytical_jacobian(P_vec: np.ndarray, params: SolverParams,
                       user_function: Optional[Callable] = None) -> Union[np.ndarray, sp.spmatrix]:
    """
    Compute Jacobian analytically using user-provided function.
    
    Args:
        P_vec: Point at which to evaluate Jacobian
        params: SolverParams object
        user_function: User-provided function that computes Jacobian
        
    Returns:
        Jacobian matrix (n x n)
    """
    if user_function is None:
        raise ValueError("Analytical Jacobian requested but no function provided")
    
    return user_function(P_vec, params)


def newton_raphson_solver(
    P_vec_init: np.ndarray,
    params: SolverParams,
    derivative_option: int = 4,
    update_every: int = 1,
    analytical_jac_func: Optional[Callable] = None,
    tol_residual: float = 1e-8,
    tol_step: float = 1e-8,
    max_iter: int = 100,
    step_size: float = 1e-7,
    verbose: bool = False
) -> Dict:
    """
    Flexible Newton-Raphson solver with configurable Jacobian update strategy.
    
    Args:
        P_vec_init: Initial guess for P_vec
        params: SolverParams object containing equation parameters
        derivative_option: Jacobian computation strategy
            1 = Once at beginning (around known values)
            2 = Once per solve (Broyden-like, computed at initial guess)
            3 = Every 'update_every' iterations
            4 = Analytical (every iteration, using user function)
        update_every: For option 3, update Jacobian every x iterations
        analytical_jac_func: For option 4, user-provided analytical Jacobian function
        tol_residual: Convergence tolerance for ||F(P_vec)||
        tol_step: Convergence tolerance for ||delta_P||
        max_iter: Maximum number of iterations
        step_size: Step size for numerical derivatives
        verbose: Print iteration information
        
    Returns:
        Dictionary containing:
            - 'solution': Final P_vec
            - 'residual_norm': Final residual norm
            - 'iterations': Number of iterations
            - 'converged': Whether solver converged
            - 'convergence_type': 'residual' or 'step' or 'both'
            - 'warning': Any warnings generated
    """
    P_vec = P_vec_init.copy()
    n = len(P_vec)
    converged = False
    warning_msg = None
    convergence_type = None
    
    # Compute initial Jacobian based on derivative option
    if derivative_option in [1, 2]:
        # Compute once at the beginning
        if verbose:
            print(f"Computing Jacobian once at beginning (option {derivative_option})")
        jacobian = numerical_jacobian(P_vec, params, step_size)
        jacobian_fixed = True
    elif derivative_option == 3:
        if verbose:
            print(f"Will update Jacobian every {update_every} iterations (option 3)")
        jacobian = numerical_jacobian(P_vec, params, step_size)
        jacobian_fixed = False
    elif derivative_option == 4:
        if verbose:
            print("Using analytical Jacobian every iteration (option 4)")
        jacobian = analytical_jacobian(P_vec, params, analytical_jac_func)
        jacobian_fixed = False
    else:
        raise ValueError(f"Invalid derivative_option: {derivative_option}. Must be 1, 2, 3, or 4.")
    
    # Newton-Raphson iterations
    for iteration in range(max_iter):
        # Compute residual
        F = objective_function(P_vec, params)
        residual_norm = np.linalg.norm(F)

        if np.isnan(residual_norm):
            error()
        
        if verbose:
            print(f"Iteration {iteration}: ||F|| = {residual_norm:.6e}")
        
        # Check convergence - residual
        residual_converged = residual_norm < tol_residual
        
        if residual_converged:
            converged = True
            convergence_type = 'residual'
            break
        
        # Update Jacobian if needed
        if not jacobian_fixed:
            if derivative_option == 3 and iteration % update_every == 0:
                if verbose:
                    print(f"  Updating Jacobian at iteration {iteration}")
                jacobian = numerical_jacobian(P_vec, params, step_size)
            elif derivative_option == 4:
                jacobian = analytical_jacobian(P_vec, params, analytical_jac_func)
                #jacobian_test = numerical_jacobian(P_vec, params, step_size)
        
        # Solve for Newton step: J * delta_P = -F
        try:
            if sp.issparse(jacobian):
                delta_P = spsolve(sp.csr_matrix(jacobian), -F)
            else:
                delta_P = np.linalg.solve(jacobian, -F)
        except np.linalg.LinAlgError:
            warning_msg = "Singular Jacobian encountered"
            if verbose:
                print(f"  WARNING: {warning_msg}")
            break
        
        # Update solution
        P_vec_new = P_vec + delta_P
        P_vec_new[P_vec_new<0] = 0
        delta_P = P_vec_new - P_vec
        P_vec = P_vec_new
        
        # Check convergence - step size
        step_norm = np.linalg.norm(delta_P)
        step_converged = step_norm < tol_step
        
        if step_converged:
            converged = True
            if residual_converged:
                convergence_type = 'both'
            else:
                convergence_type = 'step'
                warning_msg = f"Converged on step size (||delta|| = {step_norm:.6e}) but residual (||F|| = {residual_norm:.6e}) above tolerance"
                warnings.warn(warning_msg)
            break
    
    # Final residual computation
    F_final = objective_function(P_vec, params)
    residual_norm_final = np.linalg.norm(F_final)
    
    if not converged:
        warning_msg = f"Maximum iterations ({max_iter}) reached without convergence"
        warnings.warn(warning_msg)
    
    results = {
        'solution': P_vec,
        'residual_norm': residual_norm_final,
        'iterations': iteration + 1 if converged else max_iter,
        'converged': converged,
        'convergence_type': convergence_type,
        'warning': warning_msg
    }
    
    return results


def batch_counterfactual_solver(
    P_vec_baseline: np.ndarray,
    params_baseline: SolverParams,
    weights: np.ndarray,
    indices_to_perturb: Optional[np.ndarray] = None,
    derivative_option: int = 2,
    update_every: int = 1,
    analytical_jac_func: Optional[Callable] = None,
    tol_residual: float = 1e-8,
    tol_step: float = 1e-8,
    max_iter: int = 100,
    step_size: float = 1e-7,
    verbose: bool = False
) -> Dict:
    """
    Solve multiple counterfactual scenarios where each element of P_VA is set to 1.
    
    Args:
        P_vec_baseline: Baseline consistent P_vec
        params_baseline: Baseline SolverParams
        indices_to_perturb: Indices of P_VA to perturb (default: all)
        derivative_option: Jacobian update strategy (see newton_raphson_solver)
        update_every: For option 3, update frequency
        analytical_jac_func: For option 4, analytical Jacobian function
        tol_residual: Convergence tolerance for residual
        tol_step: Convergence tolerance for step size
        max_iter: Maximum iterations per solve
        step_size: Step size for numerical derivatives
        verbose: Print progress information
        
    Returns:
        Dictionary containing:
            - 'counterfactual_solutions': List of P_vec solutions for each counterfactual
            - 'counterfactual_indices': Indices that were perturbed
            - 'residual_norms': Final residual norms for each solve
            - 'iterations': Iteration counts for each solve
            - 'convergence_status': Convergence status for each solve
            - 'warnings': Any warnings for each solve
    """
    n = len(P_vec_baseline)

    weights = weights/weights.sum()
    
    if indices_to_perturb is None:
        indices_to_perturb = np.arange(n)
    
    num_counterfactuals = len(indices_to_perturb)
    
    # Storage for results
    counterfactual_solutions = []
    residual_norms = []
    iterations_list = []
    convergence_status = []
    warnings_list = []
    
    if verbose:
        print(f"Starting batch counterfactual solver for {num_counterfactuals} scenarios")
        print(f"Derivative option: {derivative_option}")
    
    for idx, i in enumerate(indices_to_perturb):
        if verbose and (idx % 50 == 0):
            print(f"Solving counterfactual {idx+1}/{num_counterfactuals} (P_VA[{i}] = 1)")
        
        # Create counterfactual P_VA
        P_VA_counterfactual = params_baseline.P_VA.copy()
        P_VA_counterfactual[i] = 1.0
        
        # Create new params object
        params_cf = SolverParams(
            epsilon=params_baseline.epsilon,
            sigma=params_baseline.sigma,
            gamma=params_baseline.gamma,
            Omega=params_baseline.Omega,
            P_VA=P_VA_counterfactual
        )
        
        # Solve using baseline P_vec as initial guess
        result = newton_raphson_solver(
            P_vec_init=P_vec_baseline,
            params=params_cf,
            derivative_option=derivative_option,
            update_every=update_every,
            analytical_jac_func=analytical_jac_func,
            tol_residual=tol_residual,
            tol_step=tol_step,
            max_iter=max_iter,
            step_size=step_size,
            verbose=False  # Suppress individual solve verbosity
        )
        
        counterfactual_solutions.append(weights.T @ result['solution'])
        residual_norms.append(result['residual_norm'])
        iterations_list.append(result['iterations'])
        if result['residual_norm']!='step':
            convergence_status.append(result['converged'])
        else:
            convergence_status.append(False)
        warnings_list.append(result['warning'])
    
    if verbose:
        num_converged = sum(convergence_status)
        avg_iterations = np.mean(iterations_list)
        print(f"\nBatch solve complete:")
        print(f"  Converged: {num_converged}/{num_counterfactuals}")
        print(f"  Average iterations: {avg_iterations:.1f}")
        print(f"  Max iterations: {max(iterations_list)}")
    
    return {
        'counterfactual_solutions': counterfactual_solutions,
        'counterfactual_indices': indices_to_perturb,
        'residual_norms': np.array(residual_norms),
        'iterations': np.array(iterations_list),
        'convergence_status': np.array(convergence_status),
        'warnings': warnings_list
    }

def batch_counterfactual_solver_SD_only(
    P_vec_baseline: np.ndarray,
    params_baseline: SolverParams,
    weights: np.ndarray,
    basis_for_perturb: np.ndarray,
    indices_to_perturb: Optional[np.ndarray] = None,
    derivative_option: int = 2,
    update_every: int = 1,
    analytical_jac_func: Optional[Callable] = None,
    tol_residual: float = 1e-8,
    tol_step: float = 1e-8,
    max_iter: int = 100,
    step_size: float = 1e-7,
    verbose: bool = False
) -> Dict:
    """
    Solve multiple counterfactual scenarios where each element of P_VA is set to 1.
    
    Args:
        P_vec_baseline: Baseline consistent P_vec
        params_baseline: Baseline SolverParams
        indices_to_perturb: Indices of P_VA to perturb (default: all)
        basis_for_perturb: Column in database with which to determine basis for perturbation
        derivative_option: Jacobian update strategy (see newton_raphson_solver)
        update_every: For option 3, update frequency
        analytical_jac_func: For option 4, analytical Jacobian function
        tol_residual: Convergence tolerance for residual
        tol_step: Convergence tolerance for step size
        max_iter: Maximum iterations per solve
        step_size: Step size for numerical derivatives
        verbose: Print progress information
        
    Returns:
        Dictionary containing:
            - 'counterfactual_solutions': List of P_vec solutions for each counterfactual
            - 'counterfactual_indices': Indices that were perturbed
            - 'residual_norms': Final residual norms for each solve
            - 'iterations': Iteration counts for each solve
            - 'convergence_status': Convergence status for each solve
            - 'warnings': Any warnings for each solve
    """
    n = len(P_vec_baseline)

    #weights = weights/weights.sum()
    
    if indices_to_perturb is None:
        indices_to_perturb = np.arange(n)
    
    num_counterfactuals = len(indices_to_perturb)
    
    # Storage for results
    counterfactual_solutions = []
    residual_norms = []
    iterations_list = []
    convergence_status = []
    warnings_list = []
    
    if verbose:
        print(f"Starting batch counterfactual solver for {num_counterfactuals} scenarios")
        print(f"Derivative option: {derivative_option}")
    
    for idx, i in enumerate(indices_to_perturb):
        if verbose and (idx % 50 == 0):
            print(f"Solving counterfactual {idx+1}/{num_counterfactuals} (P_VA[{i}] = 1)")
        
        # Create counterfactual P_VA
        P_VA_counterfactual = params_baseline.P_VA.copy()
        P_VA_counterfactual[basis_for_perturb==i] = 1.0
        
        # Create new params object
        params_cf = SolverParams(
            epsilon=params_baseline.epsilon,
            sigma=params_baseline.sigma,
            gamma=params_baseline.gamma,
            Omega=params_baseline.Omega,
            P_VA=P_VA_counterfactual
        )
        
        # Solve using baseline P_vec as initial guess
        result = newton_raphson_solver(
            P_vec_init=P_vec_baseline,
            params=params_cf,
            derivative_option=derivative_option,
            update_every=update_every,
            analytical_jac_func=analytical_jac_func,
            tol_residual=tol_residual,
            tol_step=tol_step,
            max_iter=max_iter,
            step_size=step_size,
            verbose=False  # Suppress individual solve verbosity
        )
        
        counterfactual_solutions.append(weights.T @ result['solution'])
        residual_norms.append(result['residual_norm'])
        iterations_list.append(result['iterations'])
        if result['residual_norm']!='step':
            convergence_status.append(result['converged'])
        else:
            convergence_status.append(False)
        warnings_list.append(result['warning'])
    
    if verbose:
        num_converged = sum(convergence_status)
        avg_iterations = np.mean(iterations_list)
        print(f"\nBatch solve complete:")
        print(f"  Converged: {num_converged}/{num_counterfactuals}")
        print(f"  Average iterations: {avg_iterations:.1f}")
        print(f"  Max iterations: {max(iterations_list)}")
    
    return {
        'counterfactual_solutions': counterfactual_solutions,
        'counterfactual_indices': indices_to_perturb,
        'residual_norms': np.array(residual_norms),
        'iterations': np.array(iterations_list),
        'convergence_status': np.array(convergence_status),
        'warnings': warnings_list
    }