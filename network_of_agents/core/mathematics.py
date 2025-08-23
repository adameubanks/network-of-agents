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
    row_sums = M.sum(axis=1, keepdims=True) + epsilon
    return (1 / row_sums) * M


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
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    
    M_expanded = M[:, np.newaxis, :]
    M_broadcast = M[np.newaxis, :, :]
    
    D_M = np.linalg.norm(M_expanded - M_broadcast, ord=1, axis=2)
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
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    
    DN_M = calculate_DN(M, epsilon)
    # Construct 1 - (I + DN(M)) so that diagonal is 0 pre-normalization
    n = DN_M.shape[0]
    temp_matrix = np.ones_like(DN_M) - (np.eye(n) + DN_M)
    
    return row_normalize(temp_matrix, epsilon)


def calculate_W(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate weighting matrix W(X[k], A[k]) := SN(X[k]) ◦ A[k] + (I - diag([SN(X[k]) ◦ A[k]]1))
    
    Args:
        X_k: Opinion vector at time k (single topic)
        A_k: Adjacency matrix at time k
        epsilon: Small positive parameter
        
    Returns:
        Weighting matrix W
    """
    if X_k.ndim == 1:
        X_k_2d = X_k.reshape(-1, 1)
    else:
        X_k_2d = X_k
    
    SN_Xk = calculate_SN(X_k_2d, epsilon)
    # Apply epsilon as minimum threshold to prevent edge weights from becoming too small
    SN_Xk = np.maximum(SN_Xk, epsilon)
    W_temp = SN_Xk * A_k
    
    row_sums_W_temp = W_temp.sum(axis=1)
    
    identity_matrix = np.eye(X_k_2d.shape[0])
    diag_correction = np.diag(row_sums_W_temp)
    
    W_Xk_Ak = W_temp + (identity_matrix - diag_correction)
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
    W_k = calculate_W(X_k, A_k, epsilon)
    X_next = np.dot(W_k, X_k)
    
    return X_next


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
    if X_k.ndim == 1:
        X_k_2d = X_k.reshape(-1, 1)
    else:
        X_k_2d = X_k
    
    SN_Xk = calculate_SN(X_k_2d, epsilon)
    SN_powered = np.power(SN_Xk, theta)
    S_hat_k = row_normalize(SN_powered, epsilon)
    return S_hat_k


def update_edges(A_k: np.ndarray, X_k: np.ndarray, theta: int, epsilon: float) -> np.ndarray:
    """
    Update edges a_ij[k+1] based on probabilistic similarity
    
    Args:
        A_k: Adjacency matrix at time k
        X_k: Opinion vector at time k (single topic)
        theta: Positive integer parameter for edge formation
        epsilon: Small positive parameter
        
    Returns:
        Updated adjacency matrix A[k+1]
    """
    n = A_k.shape[0]
    S_hat_k = calculate_S_hat(X_k, theta, epsilon)
    
    A_next = np.zeros((n, n))
    
    # Follow the exact mathematical approach from the paper
    for i in range(n):
        for j in range(n):
            if i != j:  # Ensure i ≠ j
                # Generate single random sample γ ∼ U[0,1]
                gamma = np.random.rand()
                # Check condition: γ < max(ŝ_ij[k], ε)
                threshold = max(S_hat_k[i, j], epsilon)
                
                if gamma < threshold:
                    # Set both a_ij[k+1] = 1 and a_ji[k+1] = 1
                    A_next[i, j] = 1
                    A_next[j, i] = 1
                else:
                    # Set both a_ij[k+1] = 0 and a_ji[k+1] = 0
                    A_next[i, j] = 0
                    A_next[j, i] = 0
    
    return A_next
