# KAPREKAR-SPECTRAL-GEOMETRY
Kaprekar Spectral Geometry (KSG) framework for analyzing finite dynamical systems via hypergraph representations, effective resistance, and Laplacian spectra. Includes SHyPar/KaHyPar coarsening pipelines, τ-basin dynamics, and μ₁ invariant tracking for studying structure-preserving graph reduction and phase transitions in complex systems.
# Kaprekar Spectral Geometry (KSG)

Kaprekar Spectral Geometry (KSG) is a research framework for representing finite dynamical systems as hypergraphs and analyzing them through spectral graph theory, Laplacian operators, and structure-preserving coarsening methods.

---

## 🔷 Core Idea

KSG models dynamical systems (Kaprekar flows) as weighted hypergraphs:

- Nodes → system states
- Hyperedges → transitions / basin relations
- Weights → transition strength or structural similarity

The system is studied through:

- Laplacian spectra (μ₁, Fiedler value)
- Effective resistance metrics
- Hierarchical partitioning (SHyPar / KaHyPar style)
- τ-basin transition structure

---

## 🔷 Key Components

### 1. Hypergraph Construction
Transforms Kaprekar or discrete iterative systems into weighted hypergraphs.

### 2. Spectral Analysis
Uses normalized Laplacian:

L = I - D⁻¹ᐟ² A D⁻¹ᐟ²

Key invariant:
- μ₁ (algebraic connectivity / bottleneck strength)

---

### 3. SHyPar Coarsening (Structure-Preserving)
Flow-based graph reduction:

Hypergraph → resistance estimation → max-flow clustering → node contraction → multilevel coarsening

Goal:
Preserve spectral structure while reducing graph size.

---

### 4. KaHyPar Partitioning
Supports two modes:

- Recursive Bisection → hierarchical structure-preserving splits
- Direct K-way → fast flat partitioning (less structure preservation)

---

### 5. Effective Resistance
Used to detect bottlenecks and basin boundaries:

R(i,j) = (e_i - e_j)^T L⁺ (e_i - e_j)

High resistance edges correspond to τ-basin transitions.

---

## 🔷 Research Objectives

- Study stability of μ₁ under coarsening
- Identify τ-basin transition boundaries (e.g., 5→6 cuts)
- Analyze spectral invariants under graph reduction
- Compare recursive vs direct partitioning effects
- Explore universality of scaling behavior across digit sizes

---

## 🔷 Example Workflow

1. Build Kaprekar hypergraph
2. Compute Laplacian L
3. Extract μ₁ and spectral features
4. Apply SHyPar coarsening
5. Recompute μ₁ on reduced graph
6. Compare spectral invariance
7. Analyze τ-basin transitions

---

## 🔷 Key Hypothesis (Empirical)

Spectral invariants (μ₁, resistance structure) remain approximately stable under SHyPar-style coarsening when bottleneck edges are preserved.

---

## 🔷 Tech Stack

- Python (NumPy, SciPy, NetworkX)
- Sparse linear algebra (eigs / eigsh)
- Hypergraph partitioning (KaHyPar concepts)
- Spectral graph theory tools

---

## 🔷 Status

Experimental / research prototype.

Focus: structural analysis of discrete dynamical systems via spectral geometry.

---# 🔷 ACS2 — Advanced Spectral Collapse System (KSG Core Extension)

ACS2 is the extended analytical layer of Kaprekar Spectral Geometry (KSG), designed to unify dynamical systems, spectral graph theory, and hierarchical hypergraph reduction into a single structure-preserving framework.

It formalizes the evolution of discrete systems (e.g., Kaprekar mappings) as **spectral flow processes on weighted hypergraphs**, where geometry, dynamics, and partition structure co-evolve under Laplacian constraints.

---

## 🔶 Core Principle

ACS2 treats any finite iterative system as:

System → Graph → Hypergraph → Laplacian Manifold → Spectral Flow

Where:

- Nodes = discrete states
- Edges = transition operators
- Hyperedges = multi-state coupling dynamics
- Weights = resistance / flow strength

---

## 🔷 ACS2 Architecture

### 1. State Encoding Layer
Maps dynamical rules into graph form:

- Kaprekar iteration → state transition graph
- Basin structure → clustering topology
- τ-depth → hierarchical node labeling

---

### 2. Spectral Geometry Layer
Transforms graph into spectral domain:

L = I - D⁻¹ᐟ² A D⁻¹ᐟ²

Key observables:
- μ₁ → global connectivity / bottleneck strength
- eigenvectors → basin flow directions
- spectral gap → system stability measure

---

### 3. Hypergraph Expansion Layer
Extends pairwise structure into multi-node interactions:

- captures higher-order Kaprekar transitions
- encodes basin merges and splits
- enables SHyPar-style contraction analysis

---

### 4. Coarsening Engine (SHyPar Mode)
Structure-preserving reduction pipeline:

Hypergraph
  → resistance estimation
  → flow clustering
  → node contraction
  → multilevel reduction

Goal:
Preserve μ₁ and bottleneck topology under compression.

---

### 5. Partition Intelligence Layer (KaHyPar Mode)

Supports two regimes:

- Recursive Bisection → hierarchical spectral preservation
- Direct K-way → flat segmentation for fast partitioning

ACS2 evaluates distortion between both modes to quantify structural loss.

---

## 🔷 Spectral Invariants (ACS2 Core Outputs)

ACS2 tracks the following invariant fields:

- μ₁ (algebraic connectivity)
- Effective resistance matrix R(i,j)
- Bottleneck index B = argmin cuts
- Curvature proxy via second-order Laplacian structure
- τ-transition boundaries in dynamical systems

---

## 🔷 ACS2 Collapse Hypothesis

ACS2 is built on the hypothesis that:

> Complex discrete dynamics reduce to stable spectral signatures under structured coarsening.

Meaning:
Even after aggressive reduction, core invariants (μ₁, resistance topology, basin cuts) remain stable or follow predictable scaling laws.

---

## 🔷 ACS2 Research Targets

- Stability of μ₁ under hypergraph coarsening
- Universality of τ-basin transition points
- Spectral phase transitions under defect injection
- Resistance-based prediction of dynamical bottlenecks
- Scaling laws across digit-size Kaprekar systems

---

## 🔷 ACS2 Output Modes

- Spectral reports (μ₁, eigenstructure)
- Hypergraph visualizations
- Coarsening stability curves
- Basin transition heatmaps
- Partition comparison (SHyPar vs KaHyPar)

---

## 🔷 Status

ACS2 is an experimental research framework bridging:

- Spectral graph theory
- Hypergraph partitioning
- Nonlinear discrete dynamics
- Structure-preserving reduction systems

It is intended for exploratory mathematical physics and computational graph theory research.

## 🔷 License

Open research framework (extendable for academic use).

https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY

https://huggingface.co/spaces/Aqarion-TB13/KAPREKAR/resolve/main/FLOWS/A23-KSG-FLOW.MD
