"""
Core mathematical functions for the network of agents simulation.

This module implements the mathematical framework described in the research paper,
including opinion dynamics, network evolution, and similarity calculations.
"""

import numpy as np
from typing import Tuple

def row_normalize(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Row-normalize a matrix: R(M) = diag(M1 + ε1))^(-1) M
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter to prevent division by zero
        
    Returns:
        Row-normalized matrix
    """
    return M / (M.sum(axis=1, keepdims=True) + epsilon)

def calculate_DN(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate row-wise difference matrix DN(M) := R(D(M))
    
    Where d_ij(M) = ||m_i^T - m_j^T||_1 (L1-norm of difference between row vectors)
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter for normalization
        
    Returns:
        Row-normalized difference matrix
    """
    M_2d = M.reshape(-1, 1) if M.ndim == 1 else M
    D_M = np.linalg.norm(M_2d[:, np.newaxis, :] - M_2d[np.newaxis, :, :], ord=1, axis=2)
    np.fill_diagonal(D_M, 0)
    return row_normalize(D_M, epsilon)

def calculate_SN(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate row-wise similarity matrix SN(M) := R(1 - (I + DN(M)))
    Diagonal entries are zero before normalization.
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter for normalization
        
    Returns:
        Row-normalized similarity matrix
    """
    M_2d = M.reshape(-1, 1) if M.ndim == 1 else M
    DN_M = calculate_DN(M_2d, epsilon)
    n = DN_M.shape[0]
    return row_normalize(np.ones_like(DN_M) - (np.eye(n) + DN_M), epsilon)

def calculate_W(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate weighting matrix per Eq. (2):
    W(X[k], A[k]) := SN(X[k]) ◦ A[k] + (I - diag([SN(X[k]) ◦ A[k]]1))
    
    Args:
        X_k: Opinion vector at time k (single topic)
        A_k: Adjacency matrix at time k
        epsilon: Small positive parameter
        
    Returns:
        Weighting matrix W
    """
    X_k_2d = X_k.reshape(-1, 1) if X_k.ndim == 1 else X_k
    SN_Xk = calculate_SN(X_k_2d, epsilon)
    W_temp = SN_Xk * A_k
    W_Xk_Ak = W_temp + (np.eye(X_k_2d.shape[0]) - np.diag(W_temp.sum(axis=1)))
    assert np.allclose(W_Xk_Ak.sum(axis=1), np.ones_like(W_Xk_Ak.sum(axis=1)), atol=1e-8), "W is not row-stochastic within tolerance"
    return W_Xk_Ak

def update_opinions(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Update opinions: X[k+1] = W(X[k], A[k])X[k]
    
    Args:
        X_k: Opinion vector at time k (single topic)
        A_k: Adjacency matrix at time k
        epsilon: Small positive parameter
        
    Returns:
        Updated opinion vector X[k+1]
    """
    return np.dot(calculate_W(X_k, A_k, epsilon), X_k)

def calculate_S_hat(X_k: np.ndarray, theta: int, epsilon: float) -> np.ndarray:
    """
    Calculate probabilistic similarity matrix Ŝ[k] := R(SN^◦θ(X[k]))
    
    Args:
        X_k: Opinion vector at time k (single topic)
        theta: Positive integer parameter for edge formation
        epsilon: Small positive parameter
        
    Returns:
        Probabilistic similarity matrix Ŝ[k]
    """
    X_k_2d = X_k.reshape(-1, 1) if X_k.ndim == 1 else X_k
    return row_normalize(np.power(calculate_SN(X_k_2d, epsilon), theta), epsilon)

def update_opinions_pure_degroot(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Pure DeGroot opinion update: X[k+1] = W X[k]
    
    Implements the classic DeGroot model where:
    - W is a fixed row-stochastic matrix derived from the adjacency matrix A
    - W[i,j] = A[i,j] / (Σ_k A[i,k] + ε) for numerical stability
    - No network evolution occurs (A remains constant)
    
    Mathematical foundation:
    The DeGroot model assumes agents update their opinions by taking a weighted
    average of their neighbors' opinions, where weights are proportional to
    connection strength and normalized to sum to 1 for each agent.
    
    Convergence properties:
    - If the network is strongly connected and aperiodic, opinions converge
    - The consensus value is a weighted average of initial opinions
    - Convergence rate depends on the second largest eigenvalue of W
    
    Args:
        X_k: Opinion vector at time k (single topic)
        A_k: Fixed adjacency matrix (symmetric, hollow)
        epsilon: Small positive parameter for numerical stability
        
    Returns:
        Updated opinion vector X[k+1]
    """
    n = A_k.shape[0]
    
    # Create row-stochastic weight matrix W from adjacency matrix A
    # For isolated nodes (degree 0), keep opinion unchanged (W[i,i] = 1)
    # For connected nodes, W[i,j] = A[i,j] / Σ_k A[i,k]
    row_sums = A_k.sum(axis=1, keepdims=True)
    
    # Initialize W as identity matrix (isolated nodes keep their opinion)
    W = np.eye(n)
    
    # For connected nodes, use adjacency-based weights
    connected_mask = row_sums.flatten() > 0
    if np.any(connected_mask):
        W[connected_mask] = A_k[connected_mask] / row_sums[connected_mask]
    
    # Ensure W is row-stochastic (each row sums to 1)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10), "W is not row-stochastic"
    
    # DeGroot update: X[k+1] = W X[k]
    return np.dot(W, X_k)

def create_connected_degroot_network(n_agents: int, connectivity: float = 0.3) -> np.ndarray:
    """
    Create a connected network for DeGroot model.
    
    Args:
        n_agents: Number of agents
        connectivity: Target connectivity (0.0 to 1.0)
        
    Returns:
        Symmetric, hollow adjacency matrix
    """
    # Start with empty network
    A = np.zeros((n_agents, n_agents))
    
    # Add edges with probability based on connectivity
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.random.rand() < connectivity:
                A[i, j] = 1
                A[j, i] = 1
    
    return A

def update_edges(A_k: np.ndarray, X_k: np.ndarray, theta: int, epsilon: float, update_probability: float = 1.0) -> np.ndarray:
    """
    Lazy edge update per Eq. (4): for each unordered pair (i<j), with
    probability `update_probability` resample the edge using a single γ; otherwise
    keep the previous value. Resampling rule:
    a_ij[k+1] = a_ji[k+1] = 1 if γ < max(ŝ_ij[k], ε) and i ≠ j; else 0.

    Args:
        A_k: Adjacency matrix at time k
        X_k: Opinion vector at time k (single topic)
        theta: Positive integer parameter for edge formation
        epsilon: Small positive parameter

    Returns:
        Updated symmetric, hollow adjacency matrix A[k+1]
    """
    n = A_k.shape[0]
    S_hat_k = calculate_S_hat(X_k, theta, epsilon)

    # Start from current adjacency and selectively resample edges
    A_next = A_k.copy()

    # Iterate over unordered pairs (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            # With probability update_probability, resample this pair per Eq. (4)
            if np.random.rand() < update_probability:
                gamma = np.random.rand()
                # Use symmetric per-pair probability for undirected edge formation
                threshold = max(S_hat_k[i, j], S_hat_k[j, i], epsilon)
                if gamma < threshold:
                    A_next[i, j] = 1
                    A_next[j, i] = 1
                else:
                    A_next[i, j] = 0
                    A_next[j, i] = 0

    # Enforce hollowness explicitly
    np.fill_diagonal(A_next, 0)

    # Lightweight sanity checks
    assert np.allclose(A_next, A_next.T), "A_next is not symmetric"
    assert np.all(np.diag(A_next) == 0), "A_next diagonal is not zero"

    return A_next
