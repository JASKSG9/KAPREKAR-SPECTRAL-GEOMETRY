import numpy as np
from itertools import combinations_with_replacement
from math import factorial
from scipy import sparse
from scipy.sparse.linalg import eigs
from numpy.linalg import pinv

# =========================
# 1. STATE SPACE (BAGS)
# =========================

def generate_bags(d=5):
    return sorted(set(combinations_with_replacement(range(10), d)))

def bag_to_int(b):
    return tuple(b)

# =========================
# 2. KAPREKAR MAP
# =========================

def kaprekar(n, d):
    s = f"{n:0{d}d}"
    hi = int("".join(sorted(s, reverse=True)))
    lo = int("".join(sorted(s)))
    return hi - lo

def bag_representative(b, d):
    # convert bag → canonical integer
    return int("".join(map(str, b)))

def bagify(n, d):
    return tuple(sorted(map(int, f"{n:0{d}d}")))

# =========================
# 3. BUILD QUOTIENT MAP
# =========================

def build_map(bags, d):
    bag_next = {}
    for b in bags:
        n = bag_representative(b, d)
        nxt = kaprekar(n, d)
        bag_next[b] = bagify(nxt, d)
    return bag_next

# =========================
# 4. INDEXING
# =========================

def index_bags(bags):
    return {b:i for i,b in enumerate(bags)}

# =========================
# 5. TRANSITION MATRIX
# =========================

def build_P(bags, idx, nxt):
    n = len(bags)
    rows, cols = [], []

    for b in bags:
        i = idx[b]
        j = idx[nxt[b]]
        rows.append(i)
        cols.append(j)

    data = np.ones(len(rows))
    P = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return P

# =========================
# 6. LAPLACIAN
# =========================

def laplacian(P):
    I = sparse.eye(P.shape[0])
    return I - P

# =========================
# 7. BASINS
# =========================

def compute_basins(bags, nxt):
    basin = {}
    for b in bags:
        seen = set()
        cur = b
        while cur not in seen:
            seen.add(cur)
            cur = nxt[cur]
        basin[b] = cur
    return basin

# =========================
# 8. TAU FIELD
# =========================

def compute_tau(bags, nxt, cycles):
    tau = {}
    cycle_set = set(cycles)

    for b in bags:
        depth = 0
        cur = b
        seen = set()

        while cur not in cycle_set and cur not in seen:
            seen.add(cur)
            cur = nxt[cur]
            depth += 1

        tau[b] = depth

    return tau

# =========================
# 9. SPECTRUM
# =========================

def spectral_gap(L, k=2):
    vals, _ = eigs(L, k=k, which='SM')
    vals = np.real(vals)
    vals.sort()
    return vals[1]  # μ₁

# =========================
# 10. RESISTANCE
# =========================

def resistance_matrix(L):
    L_dense = L.toarray()
    L_pinv = pinv(L_dense)
    return L_pinv

# =========================
# 11. MAIN PIPELINE
# =========================

def run(d=5):

    bags = generate_bags(d)
    idx = index_bags(bags)

    nxt = build_map(bags, d)

    P = build_P(bags, idx, nxt)
    L = laplacian(P)

    basins = compute_basins(bags, nxt)

    cycles = list(set(basins.values()))

    tau = compute_tau(bags, nxt, cycles)

    mu1 = spectral_gap(L)

    R = resistance_matrix(L)

    # =========================
    # OUTPUT SNAPSHOT
    # =========================

    print("BAGS:", len(bags))
    print("CYCLES:", len(cycles))
    print("μ₁ (global):", mu1)

    print("\nTAU SAMPLE:")
    for i, b in enumerate(list(bags)[:10]):
        print(b, "τ =", tau[b])

    print("\nRESISTANCE MATRIX SHAPE:", R.shape)

    return {
        "bags": bags,
        "P": P,
        "L": L,
        "basins": basins,
        "tau": tau,
        "mu1": mu1,
        "resistance": R
    }

# =========================
# RUN
# =========================

if __name__ == "__main__":
    results = run(d=5)
