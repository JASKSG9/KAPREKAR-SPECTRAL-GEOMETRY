import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from a21_core.sparse_engine import build_sparse_W, normalized_laplacian_sparse

# -----------------------------
# PRECONDITIONER
# -----------------------------
def jacobi_precondition(L):
    d = L.diagonal()
    d = 1.0 / np.sqrt(np.abs(d) + 1e-12)
    M = sp.diags(d)
    return M @ L @ M

# -----------------------------
# STABLE MU1 SOLVER
# -----------------------------
def compute_mu1_precond(L):
    Lp = jacobi_precondition(L)
    vals, vecs = spla.eigsh(Lp, k=2, which='SM')
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:,idx]
    return vals[1], vecs[:,1]

# -----------------------------
# ENTROPY OF PROFILE
# -----------------------------
def profile_entropy(pi):
    p = pi / np.sum(pi)
    return -np.sum(p * np.log(p + 1e-12))

# -----------------------------
# PERTURBATION ROBUSTNESS
# -----------------------------
def perturbation_penalty(pi, trials=5, eps=0.02):
    mus = []

    for _ in range(trials):
        noise = 1 + eps*np.random.randn(len(pi))
        p2 = np.clip(pi * noise, 1e-8, None)

        W = build_sparse_W(p2)
        L = normalized_laplacian_sparse(W)
        mu,_ = compute_mu1_precond(L)
        mus.append(mu)

    return np.std(mus)

# -----------------------------
# FULL PRECONDITIONED INVERSE
# -----------------------------
def inverse_solve_preconditioned(n, mu_target, ipr_target=None, steps=300, lr=0.03):

    pi = np.random.rand(n) + 1.0

    for step in range(steps):

        W = build_sparse_W(pi)
        L = normalized_laplacian_sparse(W)
        mu, v = compute_mu1_precond(L)

        ipr = np.sum(v**4) / (np.sum(v**2)**2)
        H = profile_entropy(pi)
        R = perturbation_penalty(pi, trials=3)

        loss = (mu - mu_target)**2

        if ipr_target is not None:
            loss += 0.5*(ipr - ipr_target)**2

        # smoothness
        loss += 0.01*np.sum(np.diff(pi)**2)

        # entropy support (maximize entropy -> subtract)
        loss -= 0.003*H

        # perturbation robustness
        loss += 0.2*R

        grad = np.zeros_like(pi)

        for i in range(1, n-1):
            grad[i] = (v[i]-v[i-1])**2 - (v[i+1]-v[i])**2

        grad *= 2*(mu - mu_target)

        grad += 0.02*np.gradient(np.gradient(pi))

        pi -= lr * grad
        pi = np.clip(pi, 1e-6, None)

        if step % 20 == 0:
            print(
                f"step {step} | μ1={mu:.6f} | IPR={ipr:.6f} | H={H:.6f} | R={R:.6f} | loss={loss:.6f}"
            )

    return pi
