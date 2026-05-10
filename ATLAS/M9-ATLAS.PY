#!/usr/bin/env python3
"""
ORSM Atlas v3.0 – Observable‑Resolved Spectral Measures
CORRECTED: Panel 1 now shows true resolvent norm ‖(zI-L)^{-1}‖.
Panel 3 now shows Ω vs κ(L) from actual sweeps (not synthetic).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, solve, norm, svd
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

def resolvent_norm(L, z, reg=1e-6):
    """True resolvent norm: ‖(zI - L)^{-1}‖₂ via SVD."""
    N = L.shape[0]
    M = (z + 1j*reg) * np.eye(N) - L
    s = svd(M, compute_uv=False, overwrite_a=True)
    return 1.0 / np.min(s) if np.min(s) > 0 else 1e12

def resolvent_response(L, A, z, reg=1e-6):
    """Observable‑resolved μ_A(z) = ⟨A, (zI-L)^{-1} A⟩."""
    N = L.shape[0]
    M = (z + 1j*reg) * np.eye(N) - L
    v = solve(M, A, assume_a='gen')
    return np.vdot(A, v)

def henrici(L):
    """Departure from normality: ‖L†L - LL†‖_F."""
    return norm(L.conj().T @ L - L @ L.conj().T, ord='fro')

def condition_number(L):
    """Eigenvector condition number κ(L) = ‖V‖ ‖V^{-1}‖."""
    evals, V = eig(L)
    try:
        return norm(V, 2) * norm(np.linalg.inv(V), 2)
    except:
        return 1e6

def overlap_index(muA, muB, dz):
    """Ω = ∫ min(|μA|,|μB|) / √(∫|μA|² ∫|μB|²)."""
    a = np.abs(muA); b = np.abs(muB)
    num = np.trapz(np.minimum(a, b), dx=dz)
    den = np.sqrt(np.trapz(a**2, dx=dz) * np.trapz(b**2, dx=dz))
    return num / den if den > 0 else 0.0

# ============================================================================
# Panel 1: TRUE resolvent norm + ε‑pseudospectrum contours
# ============================================================================
def panel1_resolvent_norm(ax):
    N, lam = 30, 0.74
    L = -1j * build_aas(N, lam)
    x = np.linspace(-2, 8, 80)
    y = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = X[i, j] + 1j*Y[i, j]
            Z[i, j] = resolvent_norm(L, z)
    # Clip for visualization
    Z = np.clip(Z, 1, 1e4)
    ax.contourf(X, Y, np.log10(Z), levels=25, cmap='inferno')
    # ε‑pseudospectrum contours (true ε = 1/‖R‖)
    eps_levels = [0.1, 0.05, 0.02]
    for eps in eps_levels:
        lev = 1.0 / eps
        if lev < Z.max():
            ax.contour(X, Y, Z, levels=[lev], linewidths=1.5, alpha=0.8,
                       linestyles='dashed', colors='cyan')
    ax.set_title('(a) True resolvent norm $\\|(zI-\\mathcal{L})^{-1}\\|$ + $\\varepsilon$-pseudospectrum',
                 fontsize=9, color='#ffff00')
    ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
    return ax

# ============================================================================
# Panel 2: Ω vs Henrici (core validation)
# ============================================================================
def panel2_omega_vs_henrici(ax):
    N = 30
    gammas = np.linspace(0.05, 1.0, 12)
    omega_vals, henrici_vals = [], []
    # Use rectangular contour (correct for non‑normal eigenvalues)
    x = np.linspace(-2, 6, 20)
    y = np.linspace(-1.5, 1.5, 15)
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
# Panel 3: Ω vs κ(L) computed from actual AAS sweeps (replaces synthetic surface)
# ============================================================================
def panel3_omega_vs_kappa(ax):
    N = 30
    lams = np.linspace(0.2, 2.5, 15)
    omega_vals, kappa_vals = [], []
    x = np.linspace(-2, 6, 20)
    y = np.linspace(-1.5, 1.5, 15)
    X, Y = np.meshgrid(x, y)
    z_flat = X.ravel() + 1j*Y.ravel()
    dz = (x[1]-x[0]) * (y[1]-y[0])
    A = np.zeros(N, dtype=complex); A[N//2] = 1.0
    B = np.zeros(N, dtype=complex); B[N//2-1] = 1j; B[N//2+1] = -1j
    for lam in lams:
        L = -1j * build_aas(N, lam, gamma=0.5)
        muA = np.array([resolvent_response(L, A, z) for z in z_flat])
        muB = np.array([resolvent_response(L, B, z) for z in z_flat])
        omega_vals.append(overlap_index(muA, muB, dz))
        kappa_vals.append(condition_number(L))
    ax.plot(kappa_vals, omega_vals, 's-', color='magenta', markersize=8, linewidth=2)
    ax.set_xlabel('Condition number $\\kappa(\\mathcal{L})$')
    ax.set_ylabel('Overlap $\\Omega$')
    ax.set_title('(c) $\\Omega$ vs $\\kappa(\\mathcal{L})$ (AAS sweep, $\\lambda$ varied)',
                 fontsize=10, color='#ff8800')
    ax.grid(True, alpha=0.3)
    return ax

# ============================================================================
# Panel 4: Observable angle map (θ between μA and μB) – unchanged, correct
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
            z = X[i, j] + 1j*Y[i, j]
            muA = resolvent_response(L, A, z)
            muB = resolvent_response(L, B, z)
            cosθ = np.real(muA * muB.conj()) / (np.abs(muA)*np.abs(muB) + 1e-12)
            angle[i, j] = np.arccos(np.clip(cosθ, -1, 1))
    im = ax.imshow(angle, extent=[x[0], x[-1], y[0], y[-1]],
                   origin='lower', aspect='auto', cmap='coolwarm')
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

    # Panel 1
    ax1 = fig.add_subplot(gs[0, 0])
    panel1_resolvent_norm(ax1)
    ax1.set_facecolor('#0a0a0a')

    # Panel 2
    ax2 = fig.add_subplot(gs[0, 1])
    panel2_omega_vs_henrici(ax2)
    ax2.set_facecolor('#0a0a0a')

    # Panel 3
    ax3 = fig.add_subplot(gs[1, 0])
    panel3_omega_vs_kappa(ax3)
    ax3.set_facecolor('#0a0a0a')

    # Panel 4
    ax4 = fig.add_subplot(gs[1, 1])
    panel4_angle_map(ax4)
    ax4.set_facecolor('#0a0a0a')

    fig.suptitle('ORSM Atlas v3.0 – Observable‑Resolved Spectral Measures\n'
                 'True resolvent norm | Observable splitting | Ω vs κ | Angle map',
                 fontsize=18, fontweight='bold', color='#ffff00', y=0.96)
    plt.savefig('ORSM_Atlas_v3.png', dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
    plt.show()
    print("✅ Atlas saved: ORSM_Atlas_v3.png")

if __name__ == "__main__":
    main()
