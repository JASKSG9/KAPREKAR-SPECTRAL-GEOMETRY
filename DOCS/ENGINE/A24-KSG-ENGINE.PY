import itertools
import numpy as np
from collections import defaultdict
from math import factorial

# =========================================================
# 1. DIGIT BAG SPACE (QUOTIENT CONSTRUCTION)
# =========================================================

def digit_bag(n, d):
    s = str(n).zfill(d)
    return tuple(sorted(int(x) for x in s))

def all_bags(d):
    # compositions of digits 0-9 with repetition allowed
    # restricted by integer partitions in digit space
    bags = set()
    for combo in itertools.combinations_with_replacement(range(10), d):
        bags.add(tuple(sorted(combo)))
    return list(bags)

# =========================================================
# 2. KAPREKAR MAP
# =========================================================

def kaprekar(n, d):
    s = str(n).zfill(d)
    a = int("".join(sorted(s, reverse=True)))
    b = int("".join(sorted(s)))
    return a - b

def induced_map(bag, d):
    # representative reconstruction (canonical lift)
    n = int("".join(map(str, bag)))
    return digit_bag(kaprekar(n, d), d)

# =========================================================
# 3. FIBER WEIGHT (MULTINOMIAL)
# =========================================================

def fiber_weight(bag):
    counts = defaultdict(int)
    for x in bag:
        counts[x] += 1

    denom = 1
    for v in counts.values():
        denom *= factorial(v)

    return factorial(len(bag)) // denom

# =========================================================
# 4. QUOTIENT GRAPH
# =========================================================

def build_quotient_graph(bags, d):
    index = {b:i for i,b in enumerate(bags)}
    N = len(bags)

    P = np.zeros((N, N), dtype=np.float64)

    for b in bags:
        i = index[b]
        nb = induced_map(b, d)
        j = index[nb]
        P[i, j] = 1.0

    return P, index

# =========================================================
# 5. BASIN DETECTION
# =========================================================

def compute_basins(P):
    N = P.shape[0]
    visited = set()
    basins = []

    for i in range(N):
        if i in visited:
            continue

        path = []
        cur = i

        while cur not in path:
            path.append(cur)
            cur = np.argmax(P[cur])

        cycle_start = cur
        cycle = path[path.index(cycle_start):]

        basin = set(path)
        visited |= basin
        basins.append((cycle, basin))

    return basins

# =========================================================
# 6. τ DEPTH FIELD
# =========================================================

def tau_depth(P, max_iter=50):
    N = P.shape[0]
    tau = np.zeros(N)

    for i in range(N):
        cur = i
        seen = set()
        depth = 0

        while cur not in seen and depth < max_iter:
            seen.add(cur)
            cur = np.argmax(P[cur])
            depth += 1

        tau[i] = depth

    return tau

# =========================================================
# 7. SPECTRAL STRUCTURE (NON-NORMAL)
# =========================================================

def spectral_analysis(P):
    eigvals = np.linalg.eigvals(P)
    svals = np.linalg.svd(P, compute_uv=False)

    return {
        "eigenvalues": np.sort_complex(eigvals),
        "singular_values": np.sort(svals)[::-1]
    }

# =========================================================
# 8. MAIN PIPELINE
# =========================================================

def run(d=5):
    print(f"\n🧠 KAPREKAR QUOTIENT ENGINE (d={d})")

    # build integer sample space (reduced sampling for feasibility)
    N = 10**(d//2)
    sample = list(range(N))

    bags = list({digit_bag(n, d) for n in sample})

    print(f"Quotient nodes: {len(bags)}")

    # weights
    weights = {b: fiber_weight(b) for b in bags}

    # graph
    P, index = build_quotient_graph(bags, d)

    # basins
    basins = compute_basins(P)
    print(f"Basins detected: {len(basins)}")

    # tau field
    tau = tau_depth(P)
    print(f"Max τ-depth: {int(np.max(tau))}")

    # spectral
    spec = spectral_analysis(P)
    print(f"Top singular value: {spec['singular_values'][0]}")

    # measure stats
    total_w = sum(weights.values())
    entropy_proxy = -sum((w/total_w)*np.log(w/total_w) for w in weights.values())

    print(f"Fiber entropy proxy: {entropy_proxy:.4f}")

    return {
        "bags": bags,
        "weights": weights,
        "basins": basins,
        "tau": tau,
        "spectral": spec
    }

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run(d=5)
