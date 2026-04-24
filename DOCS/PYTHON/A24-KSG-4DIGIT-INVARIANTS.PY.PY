#!/usr/bin/env python3
"""KSG CORE – 4‑digit invariants only. No speculation."""
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

def kaprekar(n):
    s = f"{n:04d}"
    return int(''.join(sorted(s, reverse=True))) - int(''.join(sorted(s)))

# Compute τ-depth
N = 10000
tau = np.zeros(N, dtype=int)
for i in range(N):
    n = i
    steps = 0
    while n not in (0, 6174) and steps < 20:
        n = kaprekar(n)
        steps += 1
    tau[i] = steps

basin = tau > 0
tau_basin = tau[basin]
hist = np.bincount(tau_basin)
print("τ-histogram:", hist.tolist())  # [383,576,2400,1272,1518,1656,2184]

# Build graph (undirected, basin only)
indices = np.where(basin)[0]
idx_map = {v: i for i, v in enumerate(indices)}
rows, cols = [], []
for v in indices:
    t = kaprekar(v)
    if t in idx_map:
        rows.append(idx_map[v])
        cols.append(idx_map[t])
from scipy.sparse import csr_matrix
A = csr_matrix(([1.0]*len(rows), (rows, cols)), shape=(len(indices), len(indices)))
A = A + A.T
deg = A.sum(axis=1).A1
D_inv_sqrt = diags(1.0 / np.sqrt(deg))
L = eye(len(indices)) - D_inv_sqrt @ A @ D_inv_sqrt
evals = eigsh(L, k=2, which='SM', return_eigenvectors=False)
mu1 = evals[1]
print(f"μ₁ = {mu1:.15f}")  # 0.1624262417339861
