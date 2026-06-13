#!/usr/bin/env python3
"""
TOY.PY — Kaprekar Quotient Simulator (Q₅₄ model)

Post-audit minimal implementation:
- no spectral assumptions
- no unnecessary lifting
- direct sigma simulation
"""

from itertools import combinations_with_replacement

# ----------------------------
# Basic transformations
# ----------------------------

def sorted_digits(n: int):
    return tuple(sorted(map(int, f"{n:04d}")))


def kaprekar_step(sig):
    """Classical Kaprekar subtraction on sorted tuple."""
    high = int("".join(map(str, sig[::-1])))
    low  = int("".join(map(str, sig)))
    return sorted_digits(high - low)


def to_gap(sig):
    a0,a1,a2,a3 = sig
    return (a1-a0, a2-a1, a3-a2)


def to_Sg2(sig):
    g1,g2,g3 = to_gap(sig)
    return (g1 + g2 + g3, g2)

# ----------------------------
# Build Q₅₄
# ----------------------------

def build_Q54():
    Q = set()
    for c in combinations_with_replacement(range(10), 4):
        if len(set(c)) == 1:
            continue
        Q.add(to_Sg2(tuple(c)))
    return Q

# ----------------------------
# Direct sigma lift (canonical representative)
# ----------------------------

def lift(S, g2):
    """
    Construct one valid sigma-state matching (S,g2).
    Deterministic canonical lift.
    """
    for a0 in range(10):
        a3 = a0 + S
        if a3 > 9:
            continue

        for a1 in range(a0, a3+1):
            a2 = a1 + g2
            if a2 > a3:
                continue

            sig = (a0,a1,a2,a3)
            if len(set(sig)) > 1:
                return sig

    raise ValueError(f"Unliftable state {(S,g2)}")

# ----------------------------
# One quotient iteration
# ----------------------------

def iterate(Q):
    out = set()
    for (S,g2) in Q:
        sig = lift(S,g2)
        nxt = kaprekar_step(sig)
        out.add(to_Sg2(nxt))
    return out

# ----------------------------
# Filtration test
# ----------------------------

def run_filtration():
    Q = build_Q54()
    print("Initial:", len(Q))

    sizes = [len(Q)]
    for _ in range(6):
        Q = iterate(Q)
        sizes.append(len(Q))
        print("Next:", len(Q))

    expected = [54, 20, 14, 10, 7, 4, 1]

    print("\nExpected:", expected)
    print("Observed:", sizes)

    assert sizes == expected, "Filtration mismatch"
    print("\n✓ Verified Kaprekar filtration chain")

# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":
    run_filtration()
