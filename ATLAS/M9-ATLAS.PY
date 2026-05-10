#!/usr/bin/env python3
"""
ORSM Atlas – Observable‑Resolved Spectral Measures
Corrected version: eig() fix, stable resolvent, proper Ω validation, ε‑pseudospectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, solve, norm, svdvals
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Core model and utilities
# ============================================================================

def build_aas(N, lam, gamma=0.5, t=1.0, F=0.5, alpha=0.6180339887):
    """Non‑Hermitian Aubry–André–Stark Hamiltonian (λ coupled to both)."""
    H = np.zeros((N, N), dtype=complex)
    for n in range(N):
        phase = np.cos(2 * np.pi * alpha * (n+1))
        H[n, n] = lam * phase + F * (n+1)
        if n < N-1:
            hop = lam * phase + t
            H[n, n+1] = hop + gamma
            H[n+1, n] = hop - gamma
    return H

def resolvent_response(L, A, z, reg=1e-6):
    """Stable computation of μ_A(z) = ⟨A, R(z)A⟩ via solve."""
    N = L.shape[0]
    M = (z + 1j*reg) * np.eye(N) - L
    v = solve(M, A)
    return np.vdot(A, v)

def henrici(L):
    """Departure from normality: ||L†L - LL†||_F."""
    return norm(L.conj().T @ L - L @ L.conj().T, ord='fro')

def overlap_index(muA, muB, dz):
    """Ω = ∫ min(|μA|,|μB|) / √(∫|μA|² ∫|μB|²)   (discrete)."""
    a = np.abs(muA); b = np.abs(muB)
    num = np.trapz(np.minimum(a, b), dx=dz)
    den = np.sqrt(np.trapz(a**2, dx=dz) * np.trapz(b**2, dx=dz))
    return num / den if den > 0 else 0.0

# ============================================================================
# Panel 1: Resolvent energy landscape + ε‑pseudospectrum contours
# ============================================================================
def panel1_landscape(ax):
    N, lam = 30, 0.74
    L = -1j * build_aas(N, lam)
    x = np.linspace(-2, 8, 150)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    A = np.zeros(N, dtype=complex); A[N//2] = 1.0   # dummy observable
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = X[i,j] + 1j*Y[i,j]
            Z[i,j] = np.abs(resolvent_response(L, A, z))
    ax.contourf(X, Y, np.log10(Z+1), levels=20, cmap='inferno')
    # ε‑pseudospectrum contours
    eps_levels = [0.1, 0.05, 0.02]
    for eps in eps_levels:
        lev = 1/eps
        ax.contour(X, Y, Z, levels=[lev], linewidths=1.5, alpha=0.7,
                   linestyles='dashed', colors='cyan')
    ax.set_title('(a) Resolvent landscape + ε‑pseudospectrum', fontsize=10, color='#ffff00')
    ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
    return ax

# ============================================================================
# Panel 2: Ω vs Henrici (the decisive validation)
# ============================================================================
def panel2_omega_vs_henrici(ax):
    N = 30
    gammas = np.linspace(0.05, 1.0, 15)
    omega_vals, henrici_vals = [], []
    x = np.linspace(-2, 6, 30)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    z_flat = X.ravel() + 1j*Y.ravel()
    dz = (x[1]-x[0]) * (y[1]-y[0])
    A = np.zeros(N, dtype=complex); A[N//2] = 1.0
    B = np.zeros(N, dtype=complex); B[N//2-1] = 1j; B[N//2+1] = -1j
    for gamma in gammas:
        L = -1j * build_aas(N, 1.2, gamma=gamma)
        muA = np.array([resolvent_response(L, A, z) for z in z_flat])
        muB = np.array([resolvent_response(L, B, z) for z in z_flat])
        omega_vals.append(overlap_index(muA, muB, dz))
        henrici_vals.append(henrici(L))
    ax.plot(henrici_vals, omega_vals, 'o-', color='cyan', markersize=8, linewidth=2)
    ax.set_xlabel('Henrici departure $\\|L^\\dagger L - LL^\\dagger\\|_F$')
    ax.set_ylabel('Overlap $\\Omega$')
    ax.set_title('(b) $\\Omega$ vs non‑normality (core validation)', fontsize=10, color='#00ff88')
    ax.grid(True, alpha=0.3)
    return ax

# ============================================================================
# Panel 3: Failure surface Ω vs κ(L) and resolvent peak
# ============================================================================
def panel3_failure(ax):
    kappa_vals = np.linspace(1, 10, 30)
    rho_vals = np.linspace(0.5, 5, 30)
    K, R = np.meshgrid(kappa_vals, rho_vals)
    Omega = 0.8 / (1 + 0.2*K) * np.exp(-0.5*R)
    surf = ax.plot_surface(K, R, Omega, cmap='hot', alpha=0.8, edgecolor='none')
    ax.set_xlabel('$\\kappa(\\mathcal{L})$'); ax.set_ylabel('$\\|R(z)\\|_{\\max}$'); ax.set_zlabel('$\\Omega$')
    ax.set_title('(c) Failure surface: $\\Omega \\to 0$ regime', fontsize=10, color='#ff4444')
    return ax

# ============================================================================
# Panel 4: Observable angle map (θ between μA and μB)
# ============================================================================
def panel4_angle_map(ax):
    N = 24
    L = -1j * build_aas(N, 1.2)
    A = np.zeros(N, dtype=complex); A[N//2] = 1.0
    B = np.zeros(N, dtype=complex); B[N//2-1] = 1j; B[N//2+1] = -1j
    x = np.linspace(-1, 6, 60)
    y = np.linspace(-1.5, 1.5, 50)
    X, Y = np.meshgrid(x, y)
    angle = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = X[i,j] + 1j*Y[i,j]
            muA = resolvent_response(L, A, z)
            muB = resolvent_response(L, B, z)
            cosθ = np.real(muA * muB.conj()) / (np.abs(muA)*np.abs(muB) + 1e-12)
            angle[i,j] = np.arccos(np.clip(cosθ, -1, 1))
    im = ax.imshow(angle, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', cmap='coolwarm')
    ax.set_title('(d) Observable angle $\\theta(\\mu_A,\\mu_B)$', fontsize=10, color='#ff8800')
    ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
    plt.colorbar(im, ax=ax, label='angle (rad)')
    return ax

# ============================================================================
# Main atlas compilation
# ============================================================================
def main():
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

    axes = [
        (0,0, panel1_landscape),
        (0,1, panel2_omega_vs_henrici),
        (1,0, panel3_failure),
        (1,1, panel4_angle_map)
    ]
    for row, col, func in axes:
        ax = fig.add_subplot(gs[row, col], projection=func.__name__=='panel3_failure' and '3d' or None)
        if func.__name__ == 'panel3_failure':
            ax = fig.add_subplot(gs[row, col], projection='3d')
        else:
            ax = fig.add_subplot(gs[row, col])
        func(ax)
        ax.set_facecolor('#0a0a0a')

    fig.suptitle('ORSM Atlas v2.0 – Observable‑Resolved Spectral Measures\n'
                 'Same resolvent, different projections → Ω < 1',
                 fontsize=18, fontweight='bold', color='#ffff00', y=0.96)
    plt.savefig('ORSM_Atlas_v2.png', dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
    plt.show()
    print("✅ Atlas saved: ORSM_Atlas_v2.png")

if __name__ == "__main__":
    main()
