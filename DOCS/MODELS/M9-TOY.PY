import numpy as np
import networkx as nx
from scipy.linalg import expm
import matplotlib.pyplot as plt

np.random.seed(0)

# -----------------------------
# 1. BUILD 3-BLOCK TOY MODEL
# -----------------------------
N1, N2, N3 = 10, 10, 10
N = N1 + N2 + N3

A = np.zeros((N, N))

# Block A (dense SCC)
for i in range(N1):
    for j in range(N1):
        if i != j:
            A[i,j] = 1

# Block B (weak SCC)
for i in range(N1, N1+N2):
    for j in range(N1, N1+N2):
        if i != j and np.random.rand() < 0.3:
            A[i,j] = 1

# Block C (chain)
for i in range(N1+N2, N-1):
    A[i, i+1] = 1

# Directed coupling A → B → C
eps = 0.2
A[0:N1, N1:N1+N2] = eps
A[N1:N1+N2, N1+N2:N] = eps

# -----------------------------
# 2. NORMALIZED LAPLACIAN
# -----------------------------
def normalized_laplacian(W):
    d = W.sum(axis=1)
    d[d == 0] = 1
    D_inv_sqrt = np.diag(1 / np.sqrt(d))
    return np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt

L0 = normalized_laplacian(A)

# -----------------------------
# 3. SCC + CONDENSATION
# -----------------------------
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
sccs = list(nx.strongly_connected_components(G))
C = nx.condensation(G)

print("Number of SCCs:", len(sccs))

# -----------------------------
# 4. μ₁(k) + IPR(k)
# -----------------------------
def compute_mu1_ipr(L):
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    mu1 = vals[1]
    v = vecs[:,1]
    ipr = np.sum(np.abs(v)**4)
    return mu1, ipr

k_vals = np.linspace(0.01, 3, 50)
mu_vals = []
ipr_vals = []

for k in k_vals:
    Wk = expm(-k * L0)
    Lk = normalized_laplacian(Wk)
    mu1, ipr = compute_mu1_ipr(Lk)
    mu_vals.append(mu1)
    ipr_vals.append(ipr)

# -----------------------------
# 5. ER NULL (ANALYTIC)
# -----------------------------
p = np.sum(A > 0) / (N*N)

def mu_er(k):
    return 1 - np.exp(-k * (1 - 2/np.sqrt(p*N)))

def sigma_er(k):
    return np.exp(-k) / np.sqrt(N)

Z_vals = [(m - mu_er(k)) / sigma_er(k) for m,k in zip(mu_vals, k_vals)]

# -----------------------------
# 6. SCC vs CONDENSATION SPLIT
# -----------------------------
# Build block-restricted Laplacians

def subgraph_laplacian(nodes):
    idx = list(nodes)
    W = A[np.ix_(idx, idx)]
    return normalized_laplacian(W)

mu_scc = []
for comp in sccs:
    Ls = subgraph_laplacian(comp)
    vals = np.linalg.eigvalsh(Ls)
    if len(vals) > 1:
        mu_scc.append(vals[1])

# condensation graph Laplacian
C_adj = nx.to_numpy_array(C)
L_cond = normalized_laplacian(C_adj)
vals_cond = np.linalg.eigvalsh(L_cond)
mu_cond = vals_cond[1] if len(vals_cond) > 1 else 0

# -----------------------------
# 7. PLOTS (FINAL OUTPUT)
# -----------------------------
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("μ1(k)")
plt.plot(k_vals, mu_vals)

plt.subplot(2,2,2)
plt.title("IPR(k)")
plt.plot(k_vals, ipr_vals)

plt.subplot(2,2,3)
plt.title("Z(k)")
plt.plot(k_vals, Z_vals)

plt.subplot(2,2,4)
plt.title("SCC vs Condensation μ1")
plt.bar(range(len(mu_scc)), mu_scc, label="SCC")
plt.axhline(mu_cond, color='r', linestyle='--', label='Condensation')
plt.legend()

plt.tight_layout()
plt.show()

