# LIMITATIONS.md — Structural Constraints and Validity Boundaries

**Version:** 3.0 (Post‑Audit)  
**Date:** 2026-06-13

---

## 1. Scope Boundary

This framework is strictly defined for:

- Base‑10, 4‑digit Kaprekar subtraction dynamics
- Deterministic digit rearrangement systems
- Finite discrete state spaces

It does **not** extend automatically to:

- Arbitrary‑length digit systems
- Non‑decimal bases without re‑derivation
- Continuous dynamical systems
- Probabilistic or noisy variants

---

## 2. State Representation Limitation

All quotient constructions rely on projections:

- Sorted digit projection
- Gap vector representation `(g₁,g₂,g₃)`
- Triangle quotient `(S,g₂)`

These projections are:

- Many‑to‑one (non‑injective)
- Information‑losing with respect to full digit permutations

Consequences:

- Fiber structure encodes hidden degeneracy
- Distinct raw states may share identical quotient images
- Full reversibility is **not** guaranteed

---

## 3. Affine Structure Limitation

The affine atlas description is:

- Piecewise valid only on discrete lattice regions
- Defined by ordering constraints, not smooth geometry
- Non‑differentiable in any continuous extension

Therefore:

- No global linear operator exists
- No single matrix representation applies
- Classical linear algebra methods do **not** globally describe dynamics

---

## 4. Filtration Limitation

The observed collapse sequences:

- Full triangle: `55 → 21 → 15 → 11 → 8 → 5 → 2 → 1`
- Non‑repdigit: `54 → 20 → 14 → 10 → 7 → 4 → 1`

are:

- Proven finite‑state results
- Dependent on chosen quotient convention (inclusion/exclusion of `(0,0)`)
- **Not** asymptotic theorems over infinite systems

Thus they describe **finite dynamical exhaustion**, not universal convergence laws.

---

## 5. Nerode Limitation

The statement `∼_{Nerode} = Identity` holds **only** on the reachable quotient image (31 states in sigma‑space, 21 in `(S,g₂)` space).

This implies:

- No two reachable states share identical future trajectories
- But does **not** imply randomness or chaos
- Does **not** imply maximal entropy or ergodicity

It is a **uniqueness‑of‑orbit property**, not a stochastic property.

---

## 6. Verification Limitation

All results currently depend on:

- Finite enumeration of quotient states (≤ 715)
- Exhaustive trajectory comparison
- Symbolic evaluation of finite affine regions

There is **no**:

- Infinite‑limit argument
- Measure‑theoretic convergence proof
- Topological dynamical systems framework
- Functional analytic operator extension

---

## 7. Generalization Limitation

Known results are **not yet proven** to extend to:

- Base `b ≠ 10`
- Digit length `d ≠ 4`
- Alternative subtraction or digit‑reordering rules

Any such extension requires independent reconstruction of:

- Gap simplex structure
- Quotient projection validity
- Affine atlas partition

---

## 8. Structural Interpretation Limitation

The framework is:

- Purely combinatorial and algebraic
- **Not** physically interpreted
- **Not** claimed to model natural phenomena

Any physical or spectral analogies are:

- Heuristic only
- **Not** part of the formal system
- **Not** required for correctness of results

---

## 9. Computational Limitation

All proofs currently rely on:

- Finite state enumeration
- Deterministic simulation
- Explicit case partitioning

No results require:

- Infinite computation
- Numerical approximation limits
- Floating‑point convergence arguments

---

## 10. Validity Summary

The system is best characterized as:

> **A finite, piecewise‑affine dynamical system over a structured quotient space with provable forward filtration and trivial observational equivalence on reachable states.**

All claims outside this characterization are **exploratory** or **speculative** and are explicitly marked as such in the repository.

---
