import numpy as np
from scipy.linalg import eigh

def kaprekar_tau_laplacian_verified():
    """EXACT construction: w_k = sqrt(N_k * N_{k+1})"""
    N_tau = np.array([383, 576, 2400, 1272, 1518, 1656, 2184], dtype=float)
    n = len(N_tau)
    
    # Edge weights: geometric mean (THE MATCH)
    w = np.sqrt(N_tau[:-1] * N_tau[1:])
    
    # Tridiagonal path adjacency
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = w[i]
        A[i+1, i] = w[i]
    
    # Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    degrees = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    eigenvalues = eigh(L, eigvals_only=True)
    mu1 = eigenvalues[1]
    
    target = 0.1624262417339861
    assert abs(mu1 - target) < 1e-12, f"μ₁ mismatch: {mu1} != {target}"
    
    print(f"✅ μ₁ = {mu1:.16f} EXACT MATCH (w=√(N_i N_{i+1}))")
    return L, eigenvalues

L, evs = kaprekar_tau_laplacian_verified()
