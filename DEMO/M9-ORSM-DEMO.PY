# %% [markdown]
# # ORSM Demo: Observable‑Resolved Spectral Measures
# 
# **Same resolvent, different projections → Ω < 1**
# 
# This notebook walks through the core mechanism using the `M9-TOY.py` model.

# %% [markdown]
# ## 1. Setup and imports

# %%
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# %% [markdown]
# ## 2. Build the non‑Hermitian Aubry–André–Stark operator

# %%
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

n, lam = 30, 1.2
L = -1j * build_operator(n, lam)

# %% [markdown]
# ## 3. Define two observables: density (A) and current (B)

# %%
A = np.zeros(n, dtype=complex); A[n//2] = 1.0
B = np.zeros(n, dtype=complex); B[n//2-1] = 1j; B[n//2+1] = -1j

# %% [markdown]
# ## 4. Compute observable‑resolved resolvent measures μ_A(z), μ_B(z)

# %%
def resolvent_response(L, A, z, reg=1e-6):
    N = L.shape[0]
    M = (z + 1j*reg) * np.eye(N) - L
    v = la.solve(M, A)
    return np.vdot(A, v)

# Contour enclosing the pseudospectrum
evals = la.eigvals(L)
center = np.mean(evals)
radius = np.max(np.abs(evals - center))
theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
z_pts = center + 1.5*radius * np.exp(1j * theta)

muA = np.array([resolvent_response(L, A, z) for z in z_pts])
muB = np.array([resolvent_response(L, B, z) for z in z_pts])

# %% [markdown]
# ## 5. Overlap index Ω

# %%
def overlap_index(muA, muB, dz):
    a = np.abs(muA); b = np.abs(muB)
    num = np.trapz(np.minimum(a, b), dx=dz)
    den = np.sqrt(np.trapz(a**2, dx=dz) * np.trapz(b**2, dx=dz))
    return num / den if den > 0 else 0.0

dz = 2*np.pi*1.5*radius / 100
omega = overlap_index(muA, muB, dz)
print(f"Overlap Ω = {omega:.4f}")

# %% [markdown]
# ## 6. Visualize μ_A(z) and μ_B(z) on the contour

# %%
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(z_pts.real, z_pts.imag, 'k--', alpha=0.5, label='contour')
plt.scatter(z_pts.real, z_pts.imag, c=np.abs(muA), cmap='viridis', s=50)
plt.colorbar(label='|μ_A(z)|')
plt.title('|μ_A(z)| on contour')
plt.xlabel('Re(z)'); plt.ylabel('Im(z)')

plt.subplot(1,2,2)
plt.plot(z_pts.real, z_pts.imag, 'k--', alpha=0.5, label='contour')
plt.scatter(z_pts.real, z_pts.imag, c=np.abs(muB), cmap='plasma', s=50)
plt.colorbar(label='|μ_B(z)|')
plt.title('|μ_B(z)| on contour')
plt.xlabel('Re(z)'); plt.ylabel('Im(z)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Transient growth validation

# %%
t = np.linspace(0, 8, 80)
growth = [la.norm(la.expm(L * ti), 2) for ti in t]

plt.figure()
plt.semilogy(t, growth, 'm-', lw=2)
plt.xlabel('t'); plt.ylabel('||e^{Lt}||')
plt.title(f'Transient growth: max = {max(growth):.2f}x at t ≈ {t[np.argmax(growth)]:.1f}')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 8. Conclusion
# 
# - Ω < 1 → observables **induce distinct spectral measures** on the same resolvent.
# - Transient growth confirms non‑normal amplification.
# - This is the **Observable‑Induced Spectral Splitting** (ORSM) mechanism.

# %% [markdown]
# ---
# **ORSM v1.0 – James A. Skaggs, 2026**
