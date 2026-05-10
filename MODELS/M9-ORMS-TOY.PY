#!/usr/bin/env python3
"""
M9-TOY.PY – Observable‑Resolved Spectral Measures (ORSM)
Minimal self‑contained simulation: build operator, compute Ω, transient growth, ε‑pseudospectrum.
"""

import numpy as np
import scipy.linalg as la
import json

def build_operator(n, lam, gamma=0.5, t=1.0, F=0.5, alpha=0.6180339887):
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        phase = np.cos(2 * np.pi * alpha * (i+1))
        H[i, i] = lam * phase + F * (i+1)
        if i < n-1:
            hop = lam * phase + t
            H[i, i+1] = hop + gamma
            H[i+1, i] = hop - gamma
    return H

def resolvent_response(L, A, z, reg=1e-6):
    N = L.shape[0]
    M = (z + 1j*reg) * np.eye(N) - L
    v = la.solve(M, A)
    return np.vdot(A, v)

def overlap_index(muA, muB, dz):
    a = np.abs(muA); b = np.abs(muB)
    num = np.trapz(np.minimum(a, b), dx=dz)
    den = np.sqrt(np.trapz(a**2, dx=dz) * np.trapz(b**2, dx=dz))
    return num / den if den > 0 else 0.0

def transient_growth(L, t_max=8, n_t=60):
    t = np.linspace(0, t_max, n_t)
    growth = [la.norm(la.expm(L * ti), 2) for ti in t]
    return t, growth

def main():
    n, lam = 30, 1.2
    L = -1j * build_operator(n, lam)

    # Observables: density at center (A) and current (B)
    A = np.zeros(n, dtype=complex); A[n//2] = 1.0
    B = np.zeros(n, dtype=complex); B[n//2-1] = 1j; B[n//2+1] = -1j

    # Contour around pseudospectrum
    evals = la.eigvals(L)
    center = np.mean(evals)
    radius = np.max(np.abs(evals - center))
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    z_pts = center + 1.5*radius * np.exp(1j * theta)
    dz = 2*np.pi*1.5*radius / 100

    muA = np.array([resolvent_response(L, A, z) for z in z_pts])
    muB = np.array([resolvent_response(L, B, z) for z in z_pts])
    omega = overlap_index(muA, muB, dz)

    # Transient growth
    t, growth = transient_growth(L, t_max=8, n_t=80)

    print("=" * 60)
    print("ORSM Toy Model Results")
    print("=" * 60)
    print(f"System size: {n}")
    print(f"λ = {lam}")
    print(f"Overlap index Ω = {omega:.4f}")
    print(f"Max transient growth = {max(growth):.2f}x at t ≈ {t[np.argmax(growth)]:.1f}")
    print(f"Henrici departure = {la.norm(L.conj().T @ L - L @ L.conj().T, ord='fro'):.3f}")

    # Save minimal manifest
    manifest = {
        "n": n, "lam": lam,
        "omega": omega,
        "G_max": float(max(growth)),
        "henrici": float(la.norm(L.conj().T @ L - L @ L.conj().T, ord='fro'))
    }
    with open("toy_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n✅ Manifest saved: toy_manifest.json")

if __name__ == "__main__":
    main()
