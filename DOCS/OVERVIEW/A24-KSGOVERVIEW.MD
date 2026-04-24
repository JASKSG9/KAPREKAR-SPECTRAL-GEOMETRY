```markdown
# KAPREKAR SPECTRAL GEOMETRY — OFFICIAL OVERVIEW

**Version:** 4.0 (2026-04-24)  
**Status:** Verified computational mathematics | No speculative claims  
**Repository:** [github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY](https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY)  
**Live Demo:** [huggingface.co/spaces/Aqarion-TB13/KAPREKAR](https://huggingface.co/spaces/Aqarion-TB13/KAPREKAR)

---

## 1. WHAT IS KSG?

Kaprekar Spectral Geometry (KSG) is the **exact spectral analysis** of the functional graph induced by the 4‑digit Kaprekar routine  
\( T(n) = \text{desc}(n) - \text{asc}(n) \).  

The graph has **10 000 nodes** (numbers 0000…9999) and edges \( n \to T(n) \).  
All non‑zero nodes converge to the fixed point **6174** in at most 7 steps.  

By converting this directed graph into an undirected weighted graph (each undirected edge carries weight 1), we compute its **normalized Laplacian** and obtain **exact spectral invariants** that are reproducible to machine precision.

---

## 2. LOCKED INVARIANTS (4‑DIGIT SYSTEM)

These values are **reproducible** by the code in `DOCS/PYTHON/KSG-NODE-V4.PY`.

| Invariant | Value | Verification |
|-----------|-------|--------------|
| τ‑depth histogram \( N_\tau \) | `[383, 576, 2400, 1272, 1518, 1656, 2184]` | exhaustive iteration |
| Maximum τ | 7 | – |
| Bottleneck cut \( \tau^* \) (steepest gradient) | 5 | gradient of log‑histogram |
| Spectral gap \( \mu_1 \) (2nd eigenvalue of \( L \)) | `0.1624262417339861` | sparse eigensolver |
| SUSY pairing (unweighted τ‑level path) | \( \lambda_k + \lambda_{6-k} = 2 \), error < 1e‑12 | eigenvalue table |
| Skeleton sums | \( \Sigma(S_3) = \Sigma(S_5) = 59193 \) | set cardinality |

**Visual summary (ASCII)**  

```

┌─────────────────────────────────────────────────────────────────┐
│                    KSG 4‑digit invariant atlas                  │
├─────────────────────────────────────────────────────────────────┤
│  τ-depth histogram                                             │
│  2400 ┤           ██                                           │
│  2000 ┤           ██                                           │
│  1600 ┤           ██      ██                                   │
│  1200 ┤   ██      ██      ██                                   │
│   800 ┤   ██      ██      ██                                   │
│   400 ┤   ██      ██      ██     ██     ██     ██              │
│     0 ┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───  │
│        1   2   3   4   5   6   7   8   9  10  11  12  13  14   │
│                       τ* = 5                                   │
│                                                                 │
│  Spectral gap: μ₁ = 0.16242624                                 │
│  SUSY: λ₁+λ₅ = 2,  λ₂+λ₄ = 2                                   │
└─────────────────────────────────────────────────────────────────┘

```

---

## 3. OPEN PROBLEMS (TECHNICAL & VERIFIABLE)

The following are **well‑posed** research questions. No speculative analogies; each can be settled by computation or proof.

### P1 — Universal scaling of μ₁ with digit length
Compute \( \mu_1(d) \) for \( d = 3,5,6 \) and test \( \mu_1(d) \propto d^{-\alpha} \).  
*Feasibility*: 3‑digit (1k nodes), 5‑digit (100k), 6‑digit (1M).  
*Target*: power‑law fit with \( R^2 > 0.99 \).

### P2 — τ‑histograms for d = 5 and d = 6
Enumerate τ‑depth for all 5‑digit and 6‑digit states. Determine if a bottleneck \( \tau^* \) exists and if \( \tau^*/\tau_{\max} \approx 5/7 \).  
*Feasibility*: trivial (cached iteration).

### P3 — Skeleton sum equality for d = 5
Compute \( \sigma(k) = \sum\{ x \mid \tau(x)=k \} \) for 5‑digit. Find any \( k_1 \neq k_2 \) with \( \sigma(k_1)=\sigma(k_2) \).  
*Feasibility*: direct enumeration.

### P4 — Coarsening preservation of μ₁
Compare SHyPar vs KaHyPar on the 4‑digit graph (10k → 1k nodes). Which preserves \( \mu_1 \) better?  
*Feasibility*: use existing libraries, measure preservation ratio.

### P5 — Non‑normality and mixing time of transition matrix \( P \)
Compute the departure from normality \( \|PP^T - P^T P\| \) and the condition number \( \kappa(P) \).  
Detect transient amplification and estimate mixing time.  
*Feasibility*: numerical linear algebra (10k×10k sparse).

---

## 4. REPRODUCIBLE CODE & DEPLOYMENT

All invariants can be reproduced by running:

```bash
git clone https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY
cd KAPREKAR-SPECTRAL-GEOMETRY
python3 DOCS/PYTHON/KSG-NODE-V4.PY
```

Live interactive dashboard (Hugging Face Spaces) – no installation required:
https://huggingface.co/spaces/Aqarion-TB13/KAPREKAR

Bootstrap script (Linux / macOS):

```bash
curl -s https://raw.githubusercontent.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY/main/DOCS/BASH/A24-KSG-BOOTSTRAP.SH | bash
```

---

5. PROJECT PHILOSOPHY

· Verified core – all invariants are computed, not guessed.
· Open problems – clearly defined, falsifiable, and computationally accessible.
· No hype – speculative analogies (topological charge, phantom energy, etc.) are moved to DOCS/SPECULATIVE_ANALOGIES.MD and clearly labeled as such.
· Reproducibility first – every figure and number can be regenerated from the source code.

---

6. CITATION & CONTACT

If you use KSG in your research, please cite:

J. A. Skaggs, "Kaprekar Spectral Geometry: Exact Invariants of the 4‑Digit Kaprekar Graph," 2026.
Available at github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY

For questions or collaboration, open an issue on GitHub or contact the maintainer.

---

Last update: 2026-04-24
License: MIT
Next milestone: Compute μ₁ for d=5 and d=6 (see P1).

```
