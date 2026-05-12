```python
# MAY11-KSB_TOY.PY
# Kaprekar Spectral Biology — Minimal Toy Model
# ==============================================
# This script demonstrates:
#   1. The 7‑node Kaprekar Laplacian and its exact SUSY pairing.
#   2. A simple non‑normal operator and the observable splitting Ω.
#
# Requirements: numpy, matplotlib (pip install numpy matplotlib)

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

# ====================  1. KAPREKAR 7‑NODE LAPLACIAN  ====================
print("=== 1. KAPREKAR 7‑NODE LAPLACIAN ===")

# Multiplicities (fibre sizes) for each τ level
N_tau = np.array([383, 576, 2400, 1272, 1518, 1656, 2184], dtype=float)
n = 7  # number of nodes

# Build weighted adjacency from consecutive τ levels
W = np.zeros((n, n))
for i in range(n - 1):
    w = np.sqrt(N_tau[i] * N_tau[i + 1])
    W[i, i + 1] = w
    W[i + 1, i] = w

# Normalized graph Laplacian: L_norm = D^{-1/2} (D - W) D^{-1/2}
D = np.diag(W.sum(1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-15))
L_norm = D_inv_sqrt @ (D - W) @ D_inv_sqrt

# Diagonalize
vals, vecs = la.eigh(L_norm)

# Check SUSY pairing: μ_k + μ_{6-k} should equal 2
for k in range(3):
    print(f" μ_{k} + μ_{6-k} = {vals[k] + vals[6-k]:.4f}  (expected 2.0)")

# Spectral zeta at s=1: sum over nonzero eigenvalues of 1/μ
Z_K = np.sum(1.0 / vals[vals > 1e-10])
print(f" Z_K(1) = {Z_K:.4f}\n")

# ====================  2. NON‑NORMAL OPERATOR & Ω  ====================
print("=== 2. NON‑NORMAL OPERATOR & OBSERVABLE SPLITTING Ω ===")

def build_operator(g, N=8):
    """Create a simple non‑normal operator (tridiagonal with asymmetry g)."""
    L = np.zeros((N, N), dtype=complex)
    for i in range(N):
        L[i, i] = -(i + 1)           # real diagonal
        if i + 1 < N:
            L[i, i + 1] = 1.0 + g    # upper diagonal
            L[i + 1, i] = 1.0 - g    # lower diagonal
    return L

def measure(L, obs_vec, z_vals):
    """Compute the 'spectral measure' μ(z) = Re(⟨obs| (z I - L)^{-1} |obs⟩)."""
    N = L.shape[0]
    mu = np.zeros_like(z_vals, dtype=float)
    for j, z in enumerate(z_vals):
        R = la.inv(z * np.eye(N) - L)
        mu[j] = np.real(np.conj(obs_vec).T @ R @ obs_vec)
    return mu

def overlap(mu1, mu2):
    """Overlap index between two measure vectors."""
    num = np.trapz(np.minimum(np.abs(mu1), np.abs(mu2)))
    den = np.sqrt(np.trapz(np.abs(mu1)**2) * np.trapz(np.abs(mu2)**2) + 1e-30)
    return num / den

# Parameter sweep: g in [0, 0.95]
g_vals = np.linspace(0.0, 0.95, 20)
omega_vals = []

z_grid = 1j * np.linspace(0.1, 3.0, 50)   # imaginary‑axis evaluation points
N_op = 8

for g in g_vals:
    L = build_operator(g, N_op)

    # Random normalized observables (projection directions)
    A = np.random.randn(N_op); A /= la.norm(A)
    B = np.random.randn(N_op); B /= la.norm(B)

    mu_A = measure(L, A, z_grid)
    mu_B = measure(L, B, z_grid)

    omega_vals.append(overlap(mu_A, mu_B))

print(f"Ω range: [{min(omega_vals):.3f}, {max(omega_vals):.3f}]")

# ====================  3. QUICK PLOT  ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Kaprekar spectrum
ax1.bar(range(n), vals, color=plt.cm.viridis(np.linspace(0, 1, n)))
ax1.set_title("Kaprekar Laplacian eigenvalues")
ax1.set_xlabel("index k"); ax1.set_ylabel("μ_k")

# Ω vs g
ax2.plot(g_vals, omega_vals, 'ko-', markersize=5)
ax2.set_ylim(0.7, 1.05)
ax2.set_title("Observable splitting Ω vs non‑normality g")
ax2.set_xlabel("g"); ax2.set_ylabel("Ω(A,B)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("toy_ksb_orsm.png", dpi=150)
print("Plot saved to toy_ksb_orsm.png")
plt.show()
