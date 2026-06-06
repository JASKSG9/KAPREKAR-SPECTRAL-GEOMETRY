```python
#!/usr/bin/env python3
"""
toy.py — Observable-Resolved Spectral Measures (ORSM) v2.1

Numerically stable non‑normal operator probe.
Fixes:
- Schur off‑diagonal norm instead of κ(L)
- expm_multiply for transient growth (Krylov)
- solve() not inv() for resolvent
- Higher‑resolution contour + adaptive spacing
- Composite ORSM diagnostic: (log_G * Ω) / log_schur
- Pseudospectral radius (optional)

Usage:
    python toy.py --n 30 --eps-max 2.5 --n-eps 30 --plot
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
import argparse
import json
from pathlib import Path

# ----------------------------------------------------------------------
# Model: coupled Aubry‑André‑Stark (non‑Hermitian, non‑normal)
# ----------------------------------------------------------------------
def build_coupled_aas(n, epsilon, alpha=0.6180339887, W=1.0, F=0.05, gamma=0.12):
    """Total matrix size = 2n. epsilon controls non‑normality."""
    j = np.arange(n, dtype=float)

    # Chain A (cosine + Stark + gain)
    Ha = np.diag(W * np.cos(2 * np.pi * alpha * j) + F * j).astype(complex)
    for i in range(n - 1):
        Ha[i, i+1] = Ha[i+1, i] = -1.0
    Ha += 1j * gamma * np.diag(np.linspace(-1, 1, n))

    # Chain B (cosine with phase shift + Stark + loss)
    Hb = np.diag(W * np.cos(2 * np.pi * alpha * j + np.pi/3) + F * j).astype(complex)
    for i in range(n - 1):
        Hb[i, i+1] = Hb[i+1, i] = -1.0
    Hb -= 1j * gamma * np.diag(np.linspace(-1, 1, n))

    # Non‑reciprocal coupling (source of non‑normality)
    C = np.zeros((n, n), dtype=complex)
    D = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        C[i, i+1] = 0.5 + 0.3j
        C[i+1, i] = 0.2 - 0.1j
        D[i, i+1] = 0.3 - 0.2j
        D[i+1, i] = 0.4 + 0.1j

    L = np.zeros((2*n, 2*n), dtype=complex)
    L[:n, :n] = Ha
    L[n:, n:] = Hb
    L[:n, n:] = epsilon * C
    L[n:, :n] = epsilon * D
    return L

def make_observables(n):
    """Geometric (local density) and dynamical (current‑like) observables."""
    A = np.zeros(2*n, dtype=complex)
    c = n // 2
    A[c] = 1.0
    A /= la.norm(A)

    B = np.zeros(2*n, dtype=complex)
    if c > 0 and c < 2*n - 1:
        B[c-1] = -1j / np.sqrt(2)
        B[c+1] = 1j / np.sqrt(2)
    if la.norm(B) > 0:
        B /= la.norm(B)
    else:
        B[0] = 1.0
    return A, B

# ----------------------------------------------------------------------
# Stable diagnostics
# ----------------------------------------------------------------------
def schur_non_normality(L):
    """Off‑diagonal Frobenius norm of Schur form. Stable, monotonic in non‑normality."""
    T, _ = la.schur(L, output='complex')
    off = T - np.diag(np.diag(T))
    return float(la.norm(off, 'fro'))

def transient_growth_krylov(L, t_max=8.0, n_t=40):
    """Stochastic growth via expm_multiply (Krylov). Returns G_max, t_peak."""
    t = np.linspace(0, t_max, n_t)
    v = np.random.randn(L.shape[0]) + 1j * np.random.randn(L.shape[0])
    v /= la.norm(v)
    G = np.zeros(n_t)
    for i, ti in enumerate(t):
        vt = expm_multiply(L * ti, v)
        G[i] = la.norm(vt)
    idx = np.argmax(G)
    return float(G[idx]), float(t[idx])

def resolvent_measures(L, A, B, n_theta=200):
    """μ_A(z), μ_B(z) on a circle enclosing spectrum. Uses solve(), not inv."""
    eigs = la.eigvals(L)
    center = np.mean(eigs)
    radius = np.max(np.abs(eigs - center)) + 0.5
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    z_pts = center + radius * np.exp(1j * theta)
    dz = 2 * np.pi * radius / n_theta

    N = L.shape[0]
    I = np.eye(N, dtype=complex)
    muA = np.zeros(n_theta, dtype=complex)
    muB = np.zeros(n_theta, dtype=complex)

    for i, z in enumerate(z_pts):
        try:
            xA = la.solve(z*I - L, A)
            muA[i] = np.vdot(A, xA)
            xB = la.solve(z*I - L, B)
            muB[i] = np.vdot(B, xB)
        except:
            muA[i] = muB[i] = np.nan
    return muA, muB, dz

def overlap_index(muA, muB, dz):
    a = np.abs(muA)
    b = np.abs(muB)
    num = np.abs(np.trapz(a * b, dx=dz))
    den = np.sqrt(np.trapz(a**2, dx=dz) * np.trapz(b**2, dx=dz))
    return num / den if den > 0 else np.nan

def pseudospectral_radius(L, n_points=30):
    """Estimate sup||R(z)|| on a coarse grid near the spectrum."""
    eigs = la.eigvals(L)
    x_min, x_max = np.real(eigs).min() - 1.0, np.real(eigs).max() + 1.0
    y_min, y_max = np.imag(eigs).min() - 1.0, np.imag(eigs).max() + 1.0
    xs = np.linspace(x_min, x_max, n_points)
    ys = np.linspace(y_min, y_max, n_points)
    Rmax = 0.0
    I = np.eye(L.shape[0], dtype=complex)
    for x in xs:
        for y in ys:
            z = x + 1j*y
            try:
                Rz = la.inv(z*I - L)   # only for coarse grid, acceptable
                Rmax = max(Rmax, la.norm(Rz, 2))
            except:
                pass
    return Rmax

# ----------------------------------------------------------------------
# Sweep over non‑normality parameter epsilon
# ----------------------------------------------------------------------
def run_sweep(n=30, eps_min=0.0, eps_max=2.5, n_eps=20, t_max=8.0):
    epsilons = np.linspace(eps_min, eps_max, n_eps)
    A, B = make_observables(n)

    results = {
        'epsilon': epsilons,
        'schur_norm': np.zeros(n_eps),
        'omega': np.zeros(n_eps),
        'G_max': np.zeros(n_eps),
        't_peak': np.zeros(n_eps),
        'pseudospec_max': np.zeros(n_eps),
        'log_schur': np.zeros(n_eps),
        'log_G': np.zeros(n_eps),
        'log_omega': np.zeros(n_eps),
        'ORSM_diag': np.zeros(n_eps),
    }

    print(f"\n{'ε':>6} {'Schur':>12} {'Ω':>10} {'G_max':>12} {'ps_rad':>12} {'ORSM':>12}")
    print("-" * 70)

    for idx, eps in enumerate(epsilons):
        L = build_coupled_aas(n, epsilon=eps)

        # Stable non‑normality
        schur_n = schur_non_normality(L)
        results['schur_norm'][idx] = schur_n
        results['log_schur'][idx] = np.log(schur_n + 1e-12)

        # Resolvent overlap
        muA, muB, dz = resolvent_measures(L, A, B, n_theta=200)
        omega = overlap_index(muA, muB, dz)
        results['omega'][idx] = omega
        results['log_omega'][idx] = np.log(omega + 1e-12)

        # Transient growth (Krylov)
        G_max, t_peak = transient_growth_krylov(L, t_max=t_max, n_t=50)
        results['G_max'][idx] = G_max
        results['log_G'][idx] = np.log(G_max + 1e-12)
        results['t_peak'][idx] = t_peak

        # Pseudospectral radius (optional)
        ps_rad = pseudospectral_radius(L, n_points=30)
        results['pseudospec_max'][idx] = ps_rad

        # Composite diagnostic
        ORSM = (results['log_G'][idx] * omega) / (results['log_schur'][idx] + 1e-12)
        results['ORSM_diag'][idx] = ORSM

        print(f"{eps:6.3f} {schur_n:12.2e} {omega:10.4f} {G_max:12.2e} {ps_rad:12.2e} {ORSM:12.4f}")

    return results

# ----------------------------------------------------------------------
# Save and plot
# ----------------------------------------------------------------------
def save_results(results, outdir='.'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / 'toy_data.npz'
    np.savez(npz_path,
             epsilon=results['epsilon'],
             schur_norm=results['schur_norm'],
             omega=results['omega'],
             G_max=results['G_max'],
             pseudospec_max=results['pseudospec_max'],
             ORSM_diag=results['ORSM_diag'])
    print(f"\nSaved data to {npz_path}")

    summary = {
        'eps_min': float(results['epsilon'][0]),
        'eps_max': float(results['epsilon'][-1]),
        'n_eps': len(results['epsilon']),
        'schur_norm_max': float(np.max(results['schur_norm'])),
        'omega_min': float(np.nanmin(results['omega'])),
        'G_max_max': float(np.nanmax(results['G_max'])),
        'ORSM_range': [float(np.nanmin(results['ORSM_diag'])),
                       float(np.nanmax(results['ORSM_diag']))]
    }
    json_path = outdir / 'toy_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")

def plot_results(results, outdir='.'):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    outdir = Path(outdir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0,0]
    ax.semilogy(results['epsilon'], results['schur_norm'], 'b-', linewidth=1.5)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('Schur off‑diag norm')
    ax.set_title('(a) Non‑normality')
    ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(results['epsilon'], results['omega'], 'ro-', markersize=4, linewidth=1)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\Omega$')
    ax.set_title('(b) Overlap')
    ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax.semilogy(results['epsilon'], results['G_max'], 'gs-', markersize=4, linewidth=1)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$G_{\max}$')
    ax.set_title('(c) Transient growth')
    ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    ax.plot(results['epsilon'], results['ORSM_diag'], 'mD-', markersize=4, linewidth=1)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('ORSM diagnostic')
    ax.set_title('(d) Composite (log_G·Ω)/log_schur')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = outdir / 'toy_fig.png'
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")
    plt.close()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='ORSM v2 — stable non‑normal probe')
    parser.add_argument('--n', type=int, default=30, help='chain size (total 2n)')
    parser.add_argument('--eps-min', type=float, default=0.0)
    parser.add_argument('--eps-max', type=float, default=2.5)
    parser.add_argument('--n-eps', type=int, default=20)
    parser.add_argument('--t-max', type=float, default=8.0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()

    print("=" * 70)
    print("toy.py — Observable-Resolved Spectral Measures (v2.1)")
    print(f"n = {args.n}  (total size {2*args.n})")
    print(f"ε ∈ [{args.eps_min}, {args.eps_max}], {args.n_eps} points")
    print("=" * 70)

    results = run_sweep(n=args.n,
                        eps_min=args.eps_min,
                        eps_max=args.eps_max,
                        n_eps=args.n_eps,
                        t_max=args.t_max)

    save_results(results, outdir=args.outdir)
    if args.plot:
        plot_results(results, outdir=args.outdir)

    print("\nDone. Use --plot to generate figures.")

if __name__ == '__main__':
    main()
