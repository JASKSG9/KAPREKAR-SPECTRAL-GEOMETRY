# ============================================================
# GMFLOW-KAPREKAR-D5: FULL + QUOTIENT + NEAR-INVOLUTION
# ============================================================

import numpy as np
from collections import defaultdict, deque
from itertools import combinations_with_replacement
from math import factorial

# ============================================================
# 1. KAPREKAR MAP
# ============================================================

def kaprekar_step(n, d=5):
    s = str(n).zfill(d)
    return int(''.join(sorted(s, reverse=True))) - int(''.join(sorted(s)))

# ============================================================
# 2. FULL SYSTEM BUILD
# ============================================================

def build_full_system(d=5):
    N = 10**d
    T = np.array([kaprekar_step(i, d) for i in range(N)])

    rev = defaultdict(list)
    for i, j in enumerate(T):
        rev[j].append(i)

    cycles = [
        (0,),
        (74943,62964,71973,83952),
        (53955,59994),
        (63954,61974,82962,75933)
    ]

    basin = -np.ones(N, dtype=int)
    depth = -np.ones(N, dtype=int)

    q = deque()
    for cid, cyc in enumerate(cycles):
        for x in cyc:
            basin[x] = cid
            depth[x] = 0
            q.append(x)

    while q:
        x = q.popleft()
        for p in rev[x]:
            if depth[p] == -1:
                depth[p] = depth[x] + 1
                basin[p] = basin[x]
                q.append(p)

    weights = np.array([np.sum(basin == i) for i in range(4)]) / N
    tau = np.array([
        depth[basin == i].mean() if np.any(basin == i) else 0
        for i in range(4)
    ])

    C_abstime = 1.0 / np.sum(weights / np.maximum(tau, 1e-12))

    return T, basin, depth, weights, tau, C_abstime

# ============================================================
# 3. QUOTIENT SYSTEM (MULTISET)
# ============================================================

def kaprekar_on_multiset(ms):
    desc = int(''.join(map(str, sorted(ms, reverse=True))))
    asc  = int(''.join(map(str, sorted(ms))))
    res  = desc - asc
    return tuple(sorted(map(int, str(res).zfill(5))))

def multiset_weight(ms):
    counts = [ms.count(d) for d in set(ms)]
    denom = 1
    for c in counts:
        denom *= factorial(c)
    return factorial(5) // denom

def build_multiset_space():
    multisets = list(combinations_with_replacement(range(10), 5))
    index = {ms: i for i, ms in enumerate(multisets)}
    return multisets, index

# ============================================================
# 4. FIBER PROFILE (CRITICAL OBJECT)
# ============================================================

def build_fiber_profile(basin_full):
    """
    Returns:
        fiber_map: ms -> {cycle_id: total_weight}
    """
    fiber_map = defaultdict(lambda: defaultdict(int))

    for n in range(100000):
        s = str(n).zfill(5)
        ms = tuple(sorted(map(int, s)))
        cid = basin_full[n]
        fiber_map[ms][cid] += 1

    return fiber_map

# ============================================================
# 5. NEAR-INVOLUTION CONSTRUCTION
# ============================================================

def construct_near_involution(fiber_map):
    """
    Build Φ: multisets in basin C2 ↔ C4

    Strategy:
        - Extract multisets that map to C2 and C4
        - Match by closest combinatorial weight + digit spread
    """

    C2, C4 = 1, 3  # dominant cycles

    ms_C2 = []
    ms_C4 = []

    for ms, dist in fiber_map.items():
        if dist.get(C2, 0) > 0:
            ms_C2.append(ms)
        elif dist.get(C4, 0) > 0:
            ms_C4.append(ms)

    def feature(ms):
        # invariant features
        spread = max(ms) - min(ms)
        entropy = len(set(ms))
        weight = multiset_weight(ms)
        return np.array([spread, entropy, weight])

    feats_C2 = [(ms, feature(ms)) for ms in ms_C2]
    feats_C4 = [(ms, feature(ms)) for ms in ms_C4]

    used = set()
    Phi = {}

    for ms2, f2 in feats_C2:
        best = None
        best_dist = 1e18

        for j, (ms4, f4) in enumerate(feats_C4):
            if j in used:
                continue

            d = np.linalg.norm(f2 - f4)
            if d < best_dist:
                best_dist = d
                best = j

        if best is not None:
            ms4 = feats_C4[best][0]
            Phi[ms2] = ms4
            Phi[ms4] = ms2
            used.add(best)

    return Phi

# ============================================================
# 6. VALIDATION METRICS
# ============================================================

def evaluate_involution(Phi, fiber_map):
    """
    Measure how well Φ preserves mass
    """
    C2, C4 = 1, 3

    errors = []

    for ms, ms_phi in Phi.items():
        w1 = fiber_map[ms].get(C2, 0) + fiber_map[ms].get(C4, 0)
        w2 = fiber_map[ms_phi].get(C2, 0) + fiber_map[ms_phi].get(C4, 0)

        if w1 + w2 > 0:
            errors.append(abs(w1 - w2) / max(w1, w2))

    return np.mean(errors), np.max(errors)

# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def run():
    print("=" * 60)
    print("GMFLOW KAPREKAR d=5")
    print("=" * 60)

    T, basin, depth, weights, tau, C = build_full_system()

    print("\n--- FULL SYSTEM ---")
    print("weights:", weights)
    print("tau:", tau)
    print("C_abstime:", C)

    print("\n--- BUILD FIBER PROFILE ---")
    fiber_map = build_fiber_profile(basin)

    print("multisets:", len(fiber_map))

    print("\n--- CONSTRUCT NEAR-INVOLUTION ---")
    Phi = construct_near_involution(fiber_map)

    mean_err, max_err = evaluate_involution(Phi, fiber_map)

    print("\n--- INVOLUTION QUALITY ---")
    print("mean relative error:", mean_err)
    print("max relative error:", max_err)

    print("\n--- SUMMARY ---")
    print("Φ size:", len(Phi))
    print("This is a measure-level symmetry, not dynamical.")

if __name__ == "__main__":
    run()
