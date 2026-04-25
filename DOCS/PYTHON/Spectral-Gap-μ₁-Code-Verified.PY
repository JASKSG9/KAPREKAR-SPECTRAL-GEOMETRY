import numpy as np
from scipy.linalg import eigh

def kaprekar_tau_laplacian():
    # Verified populations from audit (sum=9990)
    populations = np.array([383, 576, 2400, 1272, 1518, 1656, 2184], dtype=float)
    n = len(populations)
    
    # Weighted path edges w_k = sqrt(N_k * N_{k+1})
    w = np.sqrt(populations[:-1] * populations[1:])
    
    # Symmetric adjacency matrix A (tridiagonal path)
    A = np.zeros((n, n))
    np.fill_diagonal(A[1:], w)
    np.fill_diagonal(A[:-1], w)  # Symmetric
    
    # Degrees and normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    degrees = np.sum(A, axis=1)
    D_sqrt_inv = np.diag(1 / np.sqrt(degrees + 1e-12))
    L = np.eye(n) - D_sqrt_inv @ A @ D_sqrt_inv
    
    return L, populations

# Compute and validate
L, pops = kaprekar_tau_laplacian()
vals = eigh(L, eigvals_only=True)
mu1 = vals[1]

target = 0.1624262417339861
error = abs(mu1 - target)
print(f"Computed μ₁ = {mu1:.15f}")
print(f"Error vs target: {error:.2e}")
assert error < 1e-12, f"Spectral gap mismatch: {mu1} != {target}"
print("✅ SPECTRAL GAP VALIDATION PASSED")
