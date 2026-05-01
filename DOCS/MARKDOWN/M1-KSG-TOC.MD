# KAPREKAR SPECTRAL GEOMETRY — Complete Documentation Index

## Quick Navigation
- [1. Executive Summary](#1-executive-summary)
- [2. Mathematical Framework](#2-mathematical-framework)
- [3. Computational Results](#3-computational-results)
- [4. Verification Suite](#4-verification-suite)
- [5. Visualization & Diagrams](#5-visualization--diagrams)
- [6. API Reference](#6-api-reference)
- [7. Installation & Usage](#7-installation--usage)
- [8. Contributing](#8-contributing)
- [9. Citation & Attribution](#9-citation--attribution)
- [10. Roadmap & Open Problems](#10-roadmap--open-problems)

## Section Details

### 1. Executive Summary
- Core Invariants (τ-stratification, palindrome algebra, spectral gaps)
- Key Findings (95.6% intrinsic partition recovery, 8/10 perturbation bound validation)
- Impact (protein structure prediction, universal convergence patterns)

### 2. Mathematical Framework
- Functional Graph Decomposition Theorem
- Lyapunov Function Proof (τ-monotonicity)
- Image Graph Spectral Theorem
- Quotient Graph Analysis
- Palindrome Kernel Theorem
- Gateway-Lock Mechanism
- Intrinsic Coordinate Uniqueness
- Perturbation Bound (Fiber-Variance)

### 3. Computational Results
- Base-10 Analysis (d=4): 10,000 states, 136 image nodes, 8 τ-shells
- Base-16 Analysis (d=4): 65,536 states, monocyclic attractor
- Base-19 Analysis (d=4): 130,321 states, bicyclic attractor
- Cross-base Patterns (b ∈ [2,20])
- Palindrome Statistics (90 total, bimodal τ-distribution)
- Protein Contact KSG (CATH 4.2, 89% accuracy)

### 4. Verification Suite
- Test 1: τ-Monotonicity (0 violations)
- Test 2: Palindrome Algebra (0 errors)
- Test 3: Shell Populations (8/8 match)
- Test 4: Image Cardinality (136 verified)
- Test 5: Spectral Gap (μ₁ = 0.1624 ± 1%)
- Test 6: Palindrome Distribution (bimodal confirmed)
- Test 7: Fiber Variance (Δ = 147.43)
- Test 8: Gateway-Lock (90/90 palindromes via 3 gateways)

### 5. Visualization & Diagrams
- ASCII Functional Graph (tree + cycles)
- Mermaid DAG (τ-shells → transitions)
- Spectral Distribution (eigenvalue plot)
- Palindrome Flow Traces (gateway paths)
- Protein Contact Heatmap (CATH classification)
- Perturbation Landscape (10-system benchmark)

### 6. API Reference
- `kaprekar(n, d=4)` — Compute K(n)
- `tau_depth(n, max_iter=10)` — Graph distance to attractor
- `is_palindrome(n, d=4)` — Palindrome check
- `build_quotient_laplacian(shells)` — Construct quotient graph
- `spectral_clustering(image_graph, k=8)` — Recover τ-partition
- `perturbation_bound(Delta, sigma_fiber, N)` — Coarse-graining bound

### 7. Installation & Usage
- Requirements: Python 3.9+, NumPy, SciPy, scikit-learn
- Installation: `pip install kaprekar-spectral-geometry`
- Quick Start: `python verify_ksg_complete.py`
- Advanced: Custom bases, dimensions, partitions

### 8. Contributing
- Fork repository, create feature branch
- Add tests for new results
- Submit pull request with documentation
- Code style: PEP 8, type hints required

### 9. Citation & Attribution
- BibTeX entry provided
- Collaborating AI systems acknowledged
- Research timeline (2026-04-01 to 2026-05-01)

### 10. Roadmap & Open Problems
- OP10: Closed-form N_τ(b,d) formula
- OP11: Bases with no non-trivial cycles
- OP12: Bijection proof for |Image(b,4)|
- OP13: Intrinsic coordinate theorem (general)
- OP14: Collatz conjecture via τ-stratification

@software{ksg2026,
  author = {Node \#10878},
  title = {Kaprekar Spectral Geometry},
  year = {2026},
  url = {https://github.com/JASKSG9/kaprekar-spectral-geometry}
}
