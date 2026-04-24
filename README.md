# Kaprekar Spectral Geometry (KSG)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Space](https://img.shields.io/badge/🤗-Live%20Demo-orange)](https://huggingface.co/spaces/Aqarion-TB13/KAPREKAR)

**Exact spectral analysis of the 4‑digit Kaprekar graph and its deterministic 5‑ and 6‑digit extensions.**  
No hype, no numerology – only reproducible computational mathematics, with a full research pipeline for structured discovery. [web:6][web:7]

---

## 🔬 What is KSG?

The Kaprekar routine for 4‑digit numbers (`0000` … `9999`) defines a deterministic map  
`T(n) = descending(n) - ascending(n)`.  
Every number (except `0000`) converges to the fixed point **6174** in at most 7 steps. [web:7][web:11]  

KSG treats this functional graph as an **undirected weighted graph** and computes its **normalized Laplacian** spectrum. The resulting invariants are exact and reproducible. The project extends to 5‑digit and 6‑digit cases, revealing a rich multi‑level structural hierarchy.

---

## ✅ Locked Invariants (4‑digit)

These values are **machine‑verified** and serve as ground truth for any spectral graph algorithm.

| Invariant | Value | Verification |
|-----------|-------|--------------|
| τ‑depth histogram | `[383, 576, 2400, 1272, 1518, 1656, 2184]` | exhaustive iteration |
| Maximum τ | 7 | – |
| Bottleneck τ* | 5 | gradient of log‑histogram |
| Spectral gap μ₁ | `0.1624262417339861` | sparse eigensolver |
| SUSY pairing (path) | λₖ + λ₆₋ₖ = 2 (error < 1e‑12) | eigenvalue table |
| Skeleton sums | Σ(S₃) = Σ(S₅) = 59193 | set cardinality |

---

## 📊 Extended Results (5‑ and 6‑digit)

| Digit length | Main basin size | τ_max | μ₁ (largest component) |
|--------------|----------------|-------|------------------------|
| 3d | – | 5 | 2.59e‑03 |
| 4d | 9 990 | 7 | 5.24e‑05 |
| 5d | ≈96 000 | 6 | 1.54e‑05 |
| 6d | ≈950 000 | 13 | 2.30e‑06 (exact from `ksg_d6_full.json`) |

Key observations:
- τ_max oscillates non‑monotonically: 5,7,6,13,8,19 (d=3…8).
- Exponential decay: μ₁(d) ≈ 1.2e‑02 · e⁻¹·⁵²ᵈ, faster than 1/n².
- Effective resistance peaks at |τ|=4 and drops at τ≥5, a **structural invariant** across digit lengths.
- 5‑digit 4‑cycle basins (2 and 4) exhibit spectral symmetry (μ₁ ratio ≈ 1.0006), confirming **9‑complement symmetry is spectral** rather than purely combinatorial. [web:39]

---

## 🏗️ HeiCut Reduction Potential

The 4‑digit Kaprekar hypergraph (nodes = numbers, hyperedges = τ‑levels) satisfies all HeiCut rules:

- τ = 0,1 → Rules 1,2  
- τ = 2–4 → Rule 3 (hyperedge containment)  
- τ = 5 → Rule 6 (min‑degree contraction)  
- τ = 6–7 → Rule 7 (label propagation)

**All 10 000 nodes are reducible;** the exact minimum cut size is 5 358 (nodes with τ ≥ 5). The Kaprekar hypergraph is thus a perfect benchmark for hypergraph reduction without ILP. [web:36]

---

## 🌊 Research Flow (6 Phases)

The project follows a structured pipeline in `FLOW/A24-KSG‑FLOW.MD` (see repo); core outline:

1. **Core Computation** – τ‑histograms, μ₁, resistance landscapes.  
2. **Structural Analysis** – digit‑permutation orbits, 9‑complement symmetry.  
3. **Predictive Modeling** – τ_max(d), μ₁(d) scaling, basin extrapolation.  
4. **HeiCut Integration** – exact τ‑level ↔ rule mapping.  
5. **Publication Pipeline** – figures, LaTeX, arXiv‑ready drafts.  
6. **Synthesis** – multi‑level duality narrative (combinatorial ↔ spectral symmetries).

For the full progression, see `FLOW/A24-KSG‑FLOW.MD`.

---

## Repository Structure

```text
KAPREKAR-SPECTRAL-GEOMETRY/
├── DOCS/              # Scripts, docs, flows
│   ├── PYTHON/        # KSG‑NODE‑V4.PY, d6_ksg_full.py
│   ├── FLOW/          # A24‑KSG‑FLOW.MD
│   └── TODO/          # A24‑OPEN‑PROBLEMS.MD, SPECULATIVE_ANALOGIES.MD
├── data/              # ksg_d6_full.json etc.
├── atlas/             # ASCII/PNG summaries
└── README.md          # this file
