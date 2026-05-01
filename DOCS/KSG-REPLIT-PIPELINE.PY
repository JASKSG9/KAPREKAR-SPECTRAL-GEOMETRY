#!/usr/bin/env python3
"""
KSG-MINIMAL-PIPELINE.py
Verifies core invariants for Kaprekar map (d=4, b=10 and b=16)
Output: τ-distribution, image size, attractors, spectral gap
"""

from collections import Counter, defaultdict
import numpy as np
import time
import sys

# ============================================================
# KAPREKAR MAPS
# ============================================================

def K_10(n: int) -> int:
    """Kaprekar step for base-10, 4-digit (padded)."""
    s = f"{n:04d}"
    desc = int("".join(sorted(s, reverse=True)))
    asc = int("".join(sorted(s)))
    return desc - asc

def K_16(n: int) -> int:
    """Kaprekar step for base-16, 4-digit hex (padded)."""
    s = f"{n:04x}"
    desc = int("".join(sorted(s, reverse=True)), 16)
    asc = int("".join(sorted(s)), 16)
    return desc - asc

# ============================================================
# ATTRACTOR DETECTION
# ============================================================

def find_attractors(K, N: int, max_iter: int = 100):
    """Return set of attractor nodes (fixed points + cycle nodes)."""
    attractors = set()
    for start in range(N):
        if start in attractors:
            continue
        seen = {}
        x = start
        for step in range(max_iter):
            if x in seen:
                # cycle detected
                cycle_start = seen[x]
                cycle_nodes = []
                y = x
                for _ in range(step - cycle_start):
                    cycle_nodes.append(y)
                    y = K(y)
                attractors.update(cycle_nodes)
                break
            if x in attractors:
                break
            seen[x] = step
            x = K(x)
    return attractors

# ============================================================
# TAU DEPTH (DISTANCE TO ATTRACTOR)
# ============================================================

def compute_tau(K, N: int, attractors: set, max_steps: int = 100):
    """Return dict {n: tau(n)} for all n in range(N)."""
    tau = {}
    for n in range(N):
        if n in attractors:
            tau[n] = 0
            continue
        x = n
        steps = 0
        visited = set()
        while x not in attractors and steps < max_steps:
            if x in visited:
                steps = -1  # should not happen if attractors correct
                break
            visited.add(x)
            x = K(x)
            steps += 1
        tau[n] = steps if steps >= 0 else 0
    return tau

# ============================================================
# QUOTIENT GRAPH SPECTRAL GAP
# ============================================================

def spectral_gap(tau: dict, N: int):
    """Build quotient graph from tau shells, return μ₁ (second smallest eigenvalue)."""
    # Group states by tau
    shells = defaultdict(list)
    for n, t in tau.items():
        shells[t].append(n)

    tau_values = sorted(shells.keys())
    if len(tau_values) < 2:
        return 0.0

    # Build transition matrix between shells
    trans = defaultdict(lambda: defaultdict(int))
    for n in range(N):
        src = tau[n]
        dst = tau.get(K_10(n) if N == 10000 else K_16(n), -1)
        if src >= 0 and dst >= 0:
            trans[src][dst] += 1

    # Symmetrized adjacency
    m = len(tau_values)
    idx = {v: i for i, v in enumerate(tau_values)}
    A = np.zeros((m, m))
    for i_val in tau_values:
        for j_val in tau_values:
            if trans[i_val][j_val] > 0 or trans[j_val][i_val] > 0:
                A[idx[i_val], idx[j_val]] = 1

    # Normalized Laplacian
    D = A.sum(axis=1)
    D[D == 0] = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    L = np.eye(m) - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals = np.sort(np.linalg.eigvalsh(L))
    mu1 = eigvals[1] if len(eigvals) > 1 else 0.0
    return mu1

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(name: str, K_func, N: int):
    print(f"\n{'='*60}")
    print(f"KSG PIPELINE: {name}")
    print(f"{'='*60}")

    # Step 1: Image set
    start = time.time()
    image = set()
    for n in range(N):
        image.add(K_func(n))
    img_size = len(image)
    print(f"[1] Image size |K(Ω)| = {img_size}")
    print(f"    Time: {time.time()-start:.2f}s")

    # Step 2: Attractors
    start = time.time()
    attractors = find_attractors(K_func, N)
    print(f"[2] Attractors: {sorted(attractors)}")
    print(f"    Time: {time.time()-start:.2f}s")

    # Step 3: τ-distribution
    start = time.time()
    tau = compute_tau(K_func, N, attractors)
    tau_counts = Counter(tau.values())
    print(f"[3] τ-distribution:")
    for t in sorted(tau_counts.keys()):
        cnt = tau_counts[t]
        pct = 100 * cnt / N
        bar = '█' * int(pct / 2)
        print(f"    τ={t}: {cnt:6d} ({pct:5.2f}%) {bar}")
    tau_max = max(tau_counts.keys())
    print(f"    τ_max = {tau_max}")

    # Step 4: Quotient spectral gap
    start = time.time()
    mu1 = spectral_gap(tau, N)
    print(f"[4] Quotient spectral gap μ₁ = {mu1:.10f}")
    print(f"    Time: {time.time()-start:.2f}s")

    # Summary
    print(f"\n✅ {name} verification complete.")
    return {
        "name": name,
        "N": N,
        "img_size": img_size,
        "attractors": sorted(attractors),
        "tau_max": tau_max,
        "mu1": mu1
    }

# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    results = []
    results.append(run_pipeline("Base-10 (d=4)", K_10, 10000))
    results.append(run_pipeline("Base-16 (d=4)", K_16, 65536))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['name']:15} |Ω|={r['N']:6} |Im|={r['img_size']:3} τ_max={r['tau_max']} 


https://replit.com/@QuanTarius/Kaprekar-Pipeline?s=app


https://4690d789-5ded-4a3b-a005-137878d51479-00-1akkq7gu3cmz5.riker.replit.dev/explorer
