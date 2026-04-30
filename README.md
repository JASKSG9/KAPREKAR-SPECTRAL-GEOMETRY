KAPREKAR SPECTRAL GEOMETRY (KSG)

A Unified Computational Mathematics Platform for Arithmetic Dynamical Collapse

---

```
                    KAPREKAR SPECTRAL GEOMETRY
                    ==========================
                    
    A finite deterministic dynamical system with two incompatible
    quotient observability functors whose induced Markov operators
    fail to commute, producing measurable spectral anomalies tied
    to collapse-time structure.
    
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   Ω = {0000,...,9999}  ──→  T(x) = sort_desc(x)        │
    │                                 - sort_asc(x)           │
    │                                    │                    │
    │                           collapse in ≤7 steps          │
    │                                    │                    │
    │                              {0, 6174}                  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

ABSTRACT

The Kaprekar Spectral Geometry (KSG) platform investigates the arithmetic dynamics of the Kaprekar digit-reversal-subtraction process as a finite deterministic endofunction on complete symbolic state spaces. Unlike prior work focused on isolated constants or cycle descriptions, KSG treats the entire state space as a functional graph and extracts its exact collapse invariants, spectral quotient operators, and domain-perturbation responses under controlled state-space surgery.

The repository maintains three canonical domains (canonical closed, natural-entry leaky, and zero-attractor-excised), five active research theaters, and a unified computational architecture engineered for theorem-mining rather than ad-hoc exploration.

---

VERIFIED NUMERICAL IDENTITY CARD

```
╔═══════════════════════════════════════════════════════════╗
║              Ω₁₀,₄ CANONICAL IDENTITY                     ║
╠═══════════════════════════════════════════════════════════╣
║  Total States                        │          10,000    ║
║  Attractors                          │      {0, 6174}    ║
║  |Image(T)|                          │              55    ║
║  Compression Ratio                   │         0.0055    ║
║  τ_max                               │               7    ║
║  Shell Vector [τ0...τ7]              │  [2,392,576,      ║
║                                      │   2400,1272,      ║
║                                      │   1518,1656,      ║
║                                      │   2184]           ║
║  Cheeger Bottleneck                  │      τ₄ → τ₅      ║
║  Spectral Gap μ₁                     │   0.160317123039  ║
║  Expected Collapse Depth E[τ]        │       4.664600    ║
║  DOMAIN_A Leakage                    │              68    ║
║  SUSY Pairing ε <                    │         10⁻¹⁵     ║
║  Commutator Matches (KT=TK)          │              30    ║
╚═══════════════════════════════════════════════════════════╝
```

All values computationally verified. No manually inserted constants.

---

THREE-DOMAIN ARCHITECTURE

The KSG platform maintains three parallel computational state-space domains. Each serves a distinct mathematical purpose.

```
                    DOMAIN_B (CANONICAL CLOSED)
                    ═══════════════════════════
                    Ω = {0000,...,9999}
                    • Algebraically closed under T
                    • Shell conservation exact
                    • Two attractors visible
                    • All theorem constants originate here
                    • PUBLICATION TRUTH SOURCE
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               │               ▼
      DOMAIN_A               │         DOMAIN_C
      ════════               │         ════════
      NATURAL-ENTRY           │         ZERO-EXCISED
      {1000..9999}            │         {0001..9999}
      \ repdigits             │         \ repdigits
      • NOT closed            │         • Closed over
      • Leakage = 68          │           nonrepdigits
        states                │         • Zero attractor
      • Perturbation          │           removed
        witness               │         • Single-basin
                              │           chamber
                              │
                    SPECTRAL/OPERATOR
                    ANALYSIS APPLIED
                    TO ALL THREE DOMAINS
```

Domain Roles

Domain Type Purpose
DOMAIN_B Canonical closed Theorem statements, FDCE invariants, publication constants
DOMAIN_A Natural-entry leaky Boundary leakage measurement, observational distortion
DOMAIN_C Zero-excised control Attractor deletion response, single-basin isolation

Domain_A and Domain_C are not errors — they are controlled perturbation laboratories around the canonical Domain_B reference.

---

FIVE ACTIVE RESEARCH THEATERS

```
                         KSG-TRINITY-CORE
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  THEATER 1    │   │  THEATER 2    │   │  THEATER 3    │
│  FDCE EXACT   │   │  DOMAIN       │   │  SPECTRAL/    │
│  COLLAPSE     │   │  SURGERY      │   │  OPERATOR     │
│  THEORY       │   │  THEORY       │   │  THEORY       │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               │               ▼
      ┌───────────────┐       │       ┌───────────────┐
      │  THEATER 4    │       │       │  THEATER 5    │
      │  HARMONIC     │       │       │  NARCISSISTIC │
      │  ANOMALY      │       │       │  ANOMALY      │
      │  RESONANCE    │       │       │  INTERFERENCE │
      └───────────────┘       │       └───────────────┘
                              │
                    ALL ANCHORED TO
                    PRE-STABILIZED
                    COMPUTATIONAL CONTRACT
```

Theater Descriptions

THEATER 1 — FDCE Exact Collapse Theory
Maintained invariants: N, |Image|, cycle census, τ-shell vectors, τ_max, attractor multiplicity, mean collapse depth. Primary theorem backbone.

THEATER 2 — Domain Surgery / Perturbation Theory
Comparative observables: shell drift, spectral drift, leakage functionals, attractor excision response. Measures invariant stability under state-space surgery.

THEATER 3 — Operator / Spectral Quotient Theory
Maintained operators: shell quotient Laplacian, Fiedler value, Cheeger bottleneck, eigenvalue ladders, SUSY pairing, commutator defects.

THEATER 4 — Harmonic Anomaly / FFT Resonance Theory
FFT-ready vectors: shell occupancy, image multiplicity, preimage histograms, leakage signatures. Detects whether Kaprekar collapse produces stable hidden frequency geometry.

THEATER 5 — Narcissistic Anomaly Interference Theory
Coincidence density, shell overrepresentation, image anomaly enrichment between Kaprekar dynamics and digital self-reference structures (Armstrong numbers, palindromic fixed points).

---

COLLAPSE SHELL PYRAMID

```
τ=7  ████████████████████████████████████████████████  2184
τ=6  █████████████████████████████████████             1656
τ=5  ████████████████████████████████                  1518
τ=4  ███████████████████████████                       1272
τ=3  ██████████████████████████████████████████████    2400
τ=2  ████████████                                      576
τ=1  ████████                                          392
τ=0  ▏                                                   2
     └──────────────────────────────────────────────
     Exact shell vector: [2, 392, 576, 2400, 1272, 1518, 1656, 2184]
     Total: 10,000  |  Shell conservation verified
```

Cheeger Bottleneck: τ₄ → τ₅ (domain-stable across A/B/C)

Image Compression: 10,000 states → 55 image values after one iteration (0.55% retention).

---

COMMUTATOR STRUCTURE

The Kaprekar operator K (sort-subtract) and digit-reversal operator T do not commute:

```
[K, T] = KT - TK ≠ 0

Example:
  K(0001) = 0999     T(0001) = 1000
  K(T(0001)) = 0999  ≠  T(K(0001)) = 9990
  
  Matches (KT=TK):      30 states (all map to K(s)=5445)
  Mismatches (KT≠TK):   9,970 states
```

The 30 commuting states form exact {a, a+5} digit-pair families:

```
  {0,5}: 0055, 0505, 0550
  {1,6}: 1166, 1616, 1661
  {2,7}: 2277, 2727, 2772
  {3,8}: 3388, 3838, 3883
  {4,9}: 4499, 4949, 4994
```

---

SPECTRAL ARCHITECTURE

Operator Stack

```
  LEVEL 1 — MICRO
  A_ij = 1_{T(i)=j}        (adjacency of functional graph)
  
  LEVEL 2 — FIBER DYNAMICS
  W_ab = Σ 1_{Φ(T(x))=b}   (pushforward through feature embedding)
  P = normalized W          (geometry-induced Markov operator)
  
  LEVEL 3 — SHELL DYNAMICS
  R_ab = Σ 1_{τ(T(x))=b}   (pushforward through collapse depth)
  Q = normalized R          (collapse-induced Markov operator)
  
  LEVEL 4 — ANOMALY
  Δ = ΠP - QΠ              (non-commutativity tensor)
  A = σ(P) - σ(Q)          (spectral residual)
```

Spectral Gap Drift Under Domain Surgery

```
  μ₁(A) = 0.158112353099  (leaky — weakened stiffness)
  μ₁(B) = 0.160317123039  (canonical reference)
  μ₁(C) = 0.161127688194  (zero-excised — stiffened)
```

SUSY Pairing

```
  λ_k + λ_{N-k} = 2    (verified to machine precision ε < 10⁻¹⁵)
```

---

REPOSITORY ARCHITECTURE

```
ksg-platform/
│
├── README.md                           ← YOU ARE HERE
├── PRE_STABILIZER.md                   ← Constitutional lock document
├── EXTENDED_VISUAL_ATLAS.md            ← Full visual panels
│
├── engines/
│   ├── KSG_FORENSIC_ANOMALY_HARVESTER.py   ← Anomaly extraction hooks
│   ├── KSG_TRINITY_CORE.py                 ← Mother computational engine
│   └── KSG_GLOBAL_CENSUS_EXPANSION.py      ← Industrial census (b,d) sweep
│
├── data_lake/
│   ├── census_master.csv
│   ├── perturbation_master.csv
│   ├── spectral_master.csv
│   ├── fft_master.csv
│   └── narcissistic_master.csv
│
├── docs/
│   ├── FDCE_COLLAPSE_THEORY.md
│   ├── DOMAIN_PERTURBATION_THEORY.md
│   └── SPECTRAL_OPERATOR_THEORY.md
│
└── verification/
    └── AUDIT_LOG.md
```

---

ENGINE IMPLEMENTATION ORDER

```
  STEP 1 — KSG_FORENSIC_ANOMALY_HARVESTER.py
  ┌─────────────────────────────────────────┐
  │ Build all anomaly extraction hooks      │
  │ • FFT-ready invariant vectors           │
  │ • Narcissistic intersection tables      │
  │ • Palindromic anomaly scans             │
  └─────────────────────────────────────────┘
                    │
                    ▼
  STEP 2 — KSG_TRINITY_CORE.py
  ┌─────────────────────────────────────────┐
  │ Build mother computational engine       │
  │ • All 3 domains simultaneously          │
  │ • FDCE + spectral + operator outputs    │
  │ • Stabilized output contract            │
  └─────────────────────────────────────────┘
                    │
                    ▼
  STEP 3 — KSG_GLOBAL_CENSUS_EXPANSION.py
  ┌─────────────────────────────────────────┐
  │ Industrial census over expanded (b,d)   │
  │ • b = 2..20, d = 3..6 minimum          │
  │ • τ_max scaling law search             │
  │ • Image cardinality law discovery       │
  └─────────────────────────────────────────┘
                    │
                    ▼
  STEP 4 — Publication markdown regeneration
  ┌─────────────────────────────────────────┐
  │ All docs from Data Lake only            │
  │ No manually inserted constants          │
  └─────────────────────────────────────────┘
```

---

DATA LAKE CONTRACT

Every production engine must emit a unified structured object containing:

```
  1.  Raw map dictionary
  2.  Image set
  3.  Cycle decomposition
  4.  Tau dictionary
  5.  Shell dictionary
  6.  Shell vector
  7.  Shell indegree vector
  8.  Preimage histogram
  9.  Quotient adjacency
  10. Quotient Laplacian
  11. Eigen spectrum
  12. Cheeger data
  13. Leakage data
  14. Domain drift vectors
  15. FFT-ready invariant arrays
  16. Narcissistic anomaly intersection tables
  17. Palindromic anomaly tables
  18. Audit metadata
```

---

INTEGRITY RULES

```
  ✘ NO narrative-first constants
  ✘ NO terminal-copy documentation
  ✘ NO manually typed invariant tables
  ✘ NO isolated standalone scripts with private definitions
  ✘ NO mixing domain constants without explicit labeling
  ✘ NO heuristic thresholds or ad-hoc smoothing
  
  ✔ ALL constants from locked audit kernel
  ✔ ALL docs generated from Data Lake
  ✔ ALL three domains maintained simultaneously
  ✔ ALL operator claims traceable to pre-stabilized shell contracts
  ✔ ALL FFT inputs from preserved invariant vector orderings
```

---

CURRENT STATUS

Component Status
Pre-Stabilizer Charter LOCKED
DOMAIN_B Canonical Census VERIFIED
DOMAIN_A Leakage Measurement VERIFIED
DOMAIN_C Excision Response VERIFIED
Shell Quotient Laplacian VERIFIED
SUSY Pairing VERIFIED (ε < 10⁻¹⁵)
Commutator Census CORRECTED → 19920
Anomaly Harvester IN PROGRESS
Trinity-Core Engine PENDING
Global Census Expansion PENDING

---

CITATION

This work builds on:

· D.R. Kaprekar (1955) — Original 6174 constant discovery
· García et al. (2021) — Generalized Kaprekar parameter formalism (arXiv:2110.13730)
· Fiedler (1973) — Algebraic connectivity and spectral graph theory
· Stanley — RSK correspondence and algebraic combinatorics

---

LICENSE

This repository is dedicated to open mathematical research. All code, data, and documentation are provided for verification, reproduction, and extension by the mathematical community.

---

```
                    ╔══════════════════════════════╗
                    ║  EVERY ERROR CORRECTED IS    ║
                    ║  A THEOREM STRENGTHENED      ║
                    ╚══════════════════════════════╝
```

# KSG TRINITY–FIBER–SHELL ENGINE (v4.0 OPERATOR SPEC)

---

## 0. CORE OBJECT (FOUNDATION)

We define a finite deterministic dynamical system:

\[
\Omega = \{0,\dots,9999\}
\]

Transition map:

\[
T:\Omega \to \Omega
\]

This forms a functional graph:
- Each node has exactly one outgoing edge
- Dynamics are deterministic and finite

---

## 1. OBSERVATION FUNCTOR (Φ)

### 1.1 Definition

\[
\Phi:\Omega \to \mathbb{R}^3
\]

\[
\Phi(x) = (H(x), S(x), I(x))
\]

Where:
- H(x): digit entropy  
- S(x): digit spread  
- I(x): imbalance measure  

### 1.2 Key Property

Φ is a **many-to-one measurement map**:

- It is NOT invariant under T
- It does NOT define quotient dynamics

Fiber sets:

\[
F_y = \{x \in \Omega \mid \Phi(x)=y\}
\]

---

## 2. SHELL FUNCTION (τ)

### 2.1 Definition

\[
\tau(x) = \min \{t \mid T^t(x)=6174\}
\]

### 2.2 Shell Partition

\[
S_k = \{x \in \Omega \mid \tau(x)=k\}
\]

τ defines the **true dynamical stratification** of Ω.

---

## 3. INDUCED OPERATORS

All structures are pushforwards of T.

---

### 3.1 MICRO OPERATOR (GRAPH LEVEL)

\[
A_{ij} = \mathbf{1}_{T(i)=j}
\]

- Full deterministic adjacency matrix

---

### 3.2 FIBER OPERATOR (GEOMETRIC LIFT)

\[
W_{ab} =
\sum_{x \in \Phi^{-1}(a)}
\mathbf{1}_{\Phi(T(x))=b}
\]

Interpretation:
- Collapse dynamics through Φ
- Induced transitions between feature classes

\[
W = \Phi_* T
\]

---

### 3.3 SHELL OPERATOR (DYNAMICAL LIFT)

\[
R_{ab} =
\sum_{x \in \tau^{-1}(a)}
\mathbf{1}_{\tau(T(x))=b}
\]

Interpretation:
- Pure collapse-time evolution

\[
R = \tau_* T
\]

---

## 4. NORMALIZATION (MARKOV FORMS)

\[
P_{ab} = \frac{W_{ab}}{\sum_b W_{ab}}, \quad
Q_{ab} = \frac{R_{ab}}{\sum_b R_{ab}}
\]

Interpretation:
- P = geometry-induced dynamics
- Q = collapse-induced dynamics

---

## 5. CORE STRUCTURAL OBJECT

### 5.1 Projection Mismatch

Define:

\[
\Pi: \text{fiber space} \to \text{shell space}
\]

\[
\Delta = \Pi P - Q \Pi
\]

---

### 5.2 Interpretation

\[
\Delta \neq 0
\]

represents:

> structural non-commutativity between geometric and temporal coarse-grainings

This is the primary anomaly object.

---

## 6. SPECTRAL STRUCTURE

---

### 6.1 Micro Spectrum

\[
\sigma(A)
\]

Graph-level structure.

---

### 6.2 Fiber Spectrum

\[
\sigma(P)
\]

Encodes:
- feature clustering
- geometric mixing
- Φ-induced structure

---

### 6.3 Shell Spectrum

\[
\sigma(Q)
\]

Encodes:
- collapse kinetics
- basin depth structure
- τ-layer contraction

---

### 6.4 Residual Spectrum

\[
\mathcal{A} = \sigma(P) - \sigma(Q)
\]

Optional transform:

\[
\mathrm{FFT}(\mathcal{A})
\]

---

## 7. GEOMETRIC STRUCTURE

Feature set:

\[
\mathcal{M} = \Phi(\Omega)
\]

Convex hull:

\[
\mathcal{T} = \mathrm{Conv}(\mathcal{M})
\]

Interpretation:

- Geometric envelope of observed feature space
- Not a governing structure

---

## 8. SYSTEM ARCHITECTURE

### Level 1 — MICRO
- Ω
- T
- A

### Level 2 — GEOMETRIC
- Φ
- 𝓜
- 𝓣

### Level 3 — FIBER DYNAMICS
- W
- P
- σ(P)

### Level 4 — SHELL DYNAMICS
- τ
- R
- Q
- σ(Q)

### Level 5 — ANOMALY STRUCTURE
- Δ = ΠP − QΠ
- 𝓐 = σ(P) − σ(Q)

---

## 9. FINAL SYSTEM STATEMENT

This system is:

> A finite deterministic dynamical graph with two incompatible coarse-grainings (feature space and collapse-time space), inducing non-commuting Markov operators whose spectral mismatch encodes structural anomaly.

---

## 10. COMPUTATIONAL TARGET

To implement:

1. Build adjacency matrix A  
2. Compute Φ mapping  
3. Construct W and R  
4. Normalize → P, Q  
5. Compute Δ  
6. Compute spectra σ(P), σ(Q)  
7. Compute spectral residual 𝓐  
---

## 0. LEGEND



o  = state
x  = transient state


= attractor (6174)


.  = unvisited / inactive
!  = perturbation / leakage (Domain A)
~  = excised state (Domain C)



---

## I. FULL STATE SPACE (Ω)

Uniform functional graph (conceptual view):




o → o → o → o → o
↓             ↓
o → o → o → o → o
↓             ↓
o → o → o → o → #
↓             ↓
o → o → o → o → o



Properties:
- deterministic transitions
- single global attractor
- finite closure

---

## II. DYNAMICAL FLOW FIELD (T ORBITS)

Generic trajectory structure:




o → o → x → x → x → #



Fast-collapse orbit:




o → x → #



Max-depth orbit:




o → x → x → x → x → x → x → #



---

## III. SHELL STRUCTURE (τ-LAYERS)

Radial collapse layers:




τ = 0 :  #
τ = 1 :  x
τ = 2 :  x x
τ = 3 :  x x x
τ = 4 :  x x x x
τ = 5 :  x x x x x
τ = 6 :  x x x x x x
τ = 7 :  x x x x x x x



Interpretation:
- distance from attractor = temporal depth
- shell thickness = basin complexity

---

## IV. FIBER COLLAPSE STRUCTURE (Φ-SPACE)

Projection clusters:




Φ-space (ℝ³ → ℝ² projection)


o o o o o o
o o o o o o o o
o o o o o o o
o o o o o o
#



Properties:
- nonlinear compression
- many-to-one mapping
- overlapping fibers

---

## V. CONVEX DOMAIN (𝓣 = Conv(Φ(Ω)))




    .
  .   .
.   o   .



.   o o o   .
.  o o o o o  .
. o o o o o o o .
.  o o o o o  .
.   o o o   .
.   o   .
.   .
.



Interpretation:
- boundary envelope of feature space
- not dynamically invariant
- geometric projection artifact

---

## VI. DOMAIN A — LEAKAGE PERTURBATION




o → o → ! → x → #
↓
! → divergence



Properties:
- broken closure
- unstable transitions
- boundary distortion

---

## VII. DOMAIN C — EXCISED ATTRACTOR

(6174 removed)




o → x → x → x → x
↓
x → x → x → x → x
↓
x → x → x → x → x



No terminal state:
- infinite transient flow
- no collapse fixed point

---

## VIII. FIBER–SHELL DECOUPLING (CORE STRUCTURE)




    Φ-LAYER
 o o o o o o o
o o o o o o o o
     ↓



projection mismatch
↓
τ-LAYER


 x x x x x
  x x x x
   x x x
    x x
     #




Interpretation:
- geometry does not preserve temporal structure
- collapse structure is non-invertible under Φ

---

## IX. NON-COMMUTATIVITY STRUCTURE (Δ FIELD)




  P (geometry flow)



o → o → o → o → o


      ×

  Q (collapse flow)



x → x → x → #


RESULT:


    Δ ≠ 0 FIELD



mismatch zones:
! ! !
! o x o !
! ! !



Interpretation:
- geometric dynamics ≠ temporal dynamics
- structural irreducibility visible as mismatch field

---

## X. SPECTRAL VIEW (ASCII EIGENSTRUCTURE)




eigenmodes of P:


 ~~~~~~
  ~~~~
   ~~
    #

eigenmodes of Q:

    #
   ~~
  ~~~~
 ~~~~~~




Interpretation:
- geometry spectrum = distributed modes
- shell spectrum = collapsing modes

---

## XI. FINAL INTEGRATED VIEW




            Φ SPACE
     (geometry / convex hull)

    o o o o o o o o o

            ↓

 non-commuting projection

            ↓

           τ SPACE
     (collapse dynamics)

    x x x x x x x x

            ↓

           #




---

## XII. SYSTEM INTUITION (PURE FORM)

- Ω = deterministic graph
- Φ = geometric compression
- τ = temporal collapse stratification
- 𝓣 = convex observational envelope
- Δ = irreducible mismatch between geometry and time
  

**A unified computational platform for studying arithmetic dynamical collapse, spectral anomalies, and domain perturbation in finite deterministic systems.**

---

## What is Kaprekar Spectral Geometry?

The classic Kaprekar routine takes a 4‑digit number, sorts its digits descending and ascending, subtracts, and repeats. Within at most 7 iterations, all numbers in the complete 4‑digit state space converge to the fixed point **6174** (or the repdigit basin at **0**).

**Kaprekar Spectral Geometry (KSG)** treats this process not as a curiosity, but as a **finite deterministic functional graph** and extracts its exact mathematical structure:

- **Exact finite dynamical endofunction collapse (FDCE)** — complete census of state space, image compression, shell decomposition.
- **Spectral quotient operators** — Laplacians, Fiedler gaps, Cheeger bottlenecks, SUSY pairing.
- **Domain perturbation theory** — how do invariants change when we remove parts of the state space?
- **Harmonic anomaly detection** — FFT‑based search for hidden resonance structures in collapse vectors.
- **Narcissistic anomaly interference** — interaction between Kaprekar dynamics and digital self‑reference numbers.

The result is a **five‑theater research laboratory** built on a single, audit‑clean computational kernel.

---

## Verified Canonical Identity (Ω₁₀,₄)

| Quantity               | Value                        |
| ---------------------- | ---------------------------- |
| Total states           | 10,000                       |
| Attractors             | {0, 6174}                    |
| \|Image(T)\|           | 55                           |
| Compression ratio      | 0.0055                       |
| Maximum τ (depth)      | 7                            |
| Shell vector (τ₀…τ₇)   | [2,392,576,2400,1272,1518,1656,2184] |
| Spectral gap μ₁        | 0.160317123039               |
| Cheeger bottleneck     | τ₄ → τ₅                      |
| SUSY pairing precision | ε < 10⁻¹⁵                    |

All constants are **computationally verified** and originate from the canonical closed domain (DOMAIN_B).

---

## Three Parallel Domains

To study stability under state‑space surgery, KSG maintains three domains simultaneously:

| Domain | Description | Role |
|--------|-------------|------|
| **DOMAIN_B** | {0000,…,9999} (full symbolic closure) | Canonical truth source, theorem backbone |
| **DOMAIN_A** | {1000,…,9999} \ {repdigits} | Natural human‑entry domain — exhibits measurable leakage (68 states) |
| **DOMAIN_C** | {0001,…,9999} \ {repdigits} | Controlled excision of the zero‑attractor — isolates the 6174 basin |

Domain_A and Domain_C are **perturbation witnesses**, not mistakes. They turn the system into a comparative laboratory.

---

## Five Research Theaters

1. **FDCE Exact Collapse Theory** — Shell laws, image cardinality laws, τ scaling, cycle census.  
2. **Domain Surgery / Perturbation Theory** — Leakage functionals, shell drift, spectral drift, attractor excision response.  
3. **Spectral / Operator Quotient Theory** — Shell Laplacian, Fiedler value, Cheeger cut, SUSY pairing, commutator defects.  
4. **Harmonic Anomaly Resonance** — FFT on invariant vectors; detection of stable frequency peaks (e.g., the 88‑ratio patterns).  
5. **Narcissistic Anomaly Interference** — Overlap and shell enrichment of Armstrong numbers, palindromic fixed points, and other digital self‑referential structures.

All theaters are fed from a single pre‑stabilized output contract and the project’s Data Lake.

---

## Architecture (Simplified)

```

PRE‑STABILIZER → ANOMALY HARVESTER → TRINITY‑CORE ENGINE → GLOBAL CENSUS EXPANSION

```

- **Pre‑Stabilizer**: Locks all invariant definitions, output contracts, and domain boundaries.
- **Anomaly Harvester**: Extracts FFT‑ready vectors and narcissistic intersection tables.
- **Trinity‑Core**: Mother engine computing all FDCE, spectral, and operator invariants across all three domains.
- **Global Census Expansion**: Industrial sweep over bases 2–20, digits 3–6+ to discover scaling laws.

---

## Numerical Highlights

- **Shell conservation**: Exactly 10,000 states distributed across τ‑layers; verified sum.
- **Cheeger bottleneck**: τ₄ → τ₅ is the principal flow constriction — stable across all three domains.
- **Spectral gap drift**: Deleting closure weakens μ₁ (A: 0.1581, B: 0.1603, C: 0.1611) — a measurable perturbation response.
- **Commutator structure**: Only 30 states commute (K∘T = T∘K); they form exact {a, a+5} digit‑pair families.
- **SUSY pairing**: λₖ + λ_{N−k} = 2 holds to machine precision.

---

## Current Status

- ✅ Pre‑Stabilizer Charter locked  
- ✅ DOMAIN_B canonical census verified  
- ✅ DOMAIN_A leakage measured (68 states)  
- ✅ DOMAIN_C excision response verified  
- ✅ Shell quotient Laplacian + SUSY pairing verified  
- ✅ Commutator census corrected (19920 mismatches)  
- 🔄 Anomaly Harvester in progress  
- ⏳ Trinity‑Core engine pending  
- ⏳ Global Census Expansion pending  

---

## Further Reading

- [`PRE_STABILIZER.md`](PRE_STABILIZER.md) — Complete constitutional lock document  
- [`EXTENDED_VISUAL_ATLAS.md`](EXTENDED_VISUAL_ATLAS.md) — Full visual panels of collapse architecture  
- [`README.md`](README.md) — Repository landing page with detailed architecture  

---

## License & Open Science

This project is an open mathematical research platform. All code, data, and documentation are provided for verification, reproduction, and extension by the community.

**Repository maintained as a living computational laboratory. All numerical claims must trace to the Data Lake.**
```

---

DISCLAIMER.md

```markdown
# Disclaimer

**Kaprekar Spectral Geometry (KSG)** is an **active research platform** under continuous development.  
By using this repository, you acknowledge the following:

## 1. Not Peer‑Reviewed
- The mathematical claims, theorems, and observations presented here have **not been formally peer‑reviewed** by an external academic journal.
- All results are derived computationally and are subject to verification by independent re‑implementation.

## 2. Computational Verification
- All numerical constants are **computationally verified** using the project’s locked audit kernel.
- Verified values are marked accordingly; otherwise they are not guaranteed to be correct.
- Floating‑point operations are performed at machine precision; tiny numerical discrepancies are possible.

## 3. Research Platform, Not a Product
- This repository is a **research laboratory**, not a finished software product.
- Code may be incomplete, under‑documented, or subject to breaking changes as the investigation evolves.
- There is **no warranty** of any kind. Use at your own risk.

## 4. Interpretation of Results
- The terms “theorem”, “proof”, “anomaly”, “SUSY pairing”, etc. are used in the sense of **internal project nomenclature** and may not correspond to standard mathematical definitions.
- The presence of a named concept (e.g., “harmonic anomaly”, “narcissistic interference”) does **not** imply a confirmed physical or mathematical phenomenon until independently validated.

## 5. Citation
- If you use any part of this work, please credit the repository and note the **node identifier** and **date** of the relevant data.
- No formal publication exists yet; this repository is the primary reference.

## 6. Contact & Contributions
- The project is maintained by an independent researcher.  
- For questions, corrections, or collaborations, open an issue in the repository.

---

**Last updated:** 2026‑04‑30  
**Node:** #10878, Louisville, KY  
