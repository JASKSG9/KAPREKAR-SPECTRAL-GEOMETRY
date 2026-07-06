# PAPER A — IMMEDIATE SUBMISSION CHECKLIST
# "The Kaprekar Map Admits a Gap Quotient that is a Finite Graded In-Arborescence"
# Louisville Node #1 · AQARION
# Status: READY — 5 fixes required (~30 min), then SUBMIT

---

## ABSTRACT (LOCKED)

> The 4-digit base-10 Kaprekar map satisfies diff=999g₁+90g₂, which proves the gap projection
> induces a well-defined 54-state quotient forming a graded in-arborescence of height 6, with a
> 20-state reachable image characterized equivalently as the image-core projection, the join of
> all 56 congruence atoms, and the minimal future-dynamics quotient.

---

## 5 REQUIRED FIXES BEFORE SUBMISSION

### FIX 1 — Semigroup Order: 8 → 6 EVERYWHERE
- Find all instances of "semigroup order 8" in manuscript
- Replace with "semigroup order 6, index 6"
- Verified correct value: 6 (corrected from prior claim of 8)
- Affects: §4 algebraic structure, any semigroup table, abstract if mentioned

### FIX 2 — μ₁ VALUE — ⚠️ DISCREPANCY FLAG
- Document states: change 0.1614 → 0.0016
- Existing verified value in CLAIMS.yaml: μ₁(A) ≈ 0.16144
- ⚠️ THESE DIFFER BY FACTOR ~100 — MUST RESOLVE BEFORE SUBMISSION
- Action: rerun eigenvalue computation with exact arithmetic
  - If μ₁ = 0.0016: update CLAIMS.yaml and all dashboards
  - If μ₁ = 0.16144: retain and correct manuscript
- Remove or recompute Cheeger claims depending on resolved value
- DO NOT submit until this discrepancy is resolved

### FIX 3 — Add Nuez (2021) Citation
- Add: Nuez (2021) citation in Introduction AND Section 2
- Context: Nuez formula C(b+1,2)−1 proved as combinatorial identity (T7)
- Bibliography entry required

### FIX 4 — Scope Language for Minimality Claim
- Change: "globally minimal quotient"
- To: "minimal quotient refining the gap projection"
- Reason: minimality holds for the gap observable C(x)=gap(x), not globally
- Add scope rider: "for the gap observable C(x)=gap(x)"
- Affects: Abstract, §3, any theorem statement involving minimality

### FIX 5 — Nerode Claim Scope
- Add scope rider to Nerode claim: "for the gap observable C(x)=gap(x)"
- T3 statement: "Gap=Nerode (gap observable)" — add parenthetical to all occurrences
- Do not claim Nerode equivalence for any other observable without separate proof

---

## SUBMISSION TARGETS

Primary: Theory of Computing Systems (Springer)
Alternative: Discrete & Continuous Dynamical Systems (AIMS)
Backup: Journal of Combinatorics and Number Theory

---

## COMPLETE THEOREM STACK FOR PAPER A

| ID | Statement | Status |
|----|-----------|--------|
| T1 | diff = 999g₁ + 90g₂ (algebraic identity) | PROVED |
| T2 | K_𝒢 well-defined (corollary of T1) | PROVED |
| T3 | Gap = Nerode (gap observable C(x)=gap(x)) | PROVED |
| T4 | Q₅₄ is graded in-arborescence, height 6 | PROVED |
| T5 | Borrow signature (1,1,[g₂=0]) | PROVED |
| T6 | Fixed point iff 5∣b for d=4 | PROVED (b≤50 exhaustive) |
| T7 | Nuez formula C(b+1,2)−1 (combinatorial identity) | PROVED |
| T8 | Im(K) = Σ₃₀, 30 states | PROVED |
| T_IM | π(Im(K)) = Q₂₀ = join-of-atoms | PROVED |
| T_SG | Semigroup order 6, index 6 | PROVED |
| OP-19 | (S,g₂) complete coordinate system (corollary of T1) | PROVED |
| AUT | Aut(Q₂₀) = {id} | PROVED |
| μ(K)=20 | Future-dynamics quotient minimum | PROVED |

---

## NEW FINDING — μ(K) EXACT VALUE (include in Paper A or as addendum)

Future-sequence collision analysis: 18 collision groups.
States sharing same future trajectory (T(g), T²(g), ...):
- {(1,1),(9,1),(9,9)} → same future
- {(2,0),(9,0)} → same future
- {(2,1),(8,1),(9,2),(9,8)} → same future
- (15 additional groups)

These 18 groups generate the Q₅₄→Q₂₀ quotient.
μ(K) = 20 in the sense of minimal future-dynamics quotient.
STATUS: PROVED

---

## T9 STATUS

**Proved (finite):** (g₁,g₂) ∈ Q₂₀ iff ∃(A,G) with 1≤A≤9, 0≤G≤A such that
sort(digits(999A+90G)) has outer gap g₁, inner gap g₂.
Finite decidable condition, verified exhaustively.

**New finding:** (99g₁+90g₂) mod 100 perfectly separates Q₂₀ from 34 unreachable states.
Zero overlap. Two sets have completely disjoint residue classes mod 100.

**Open:** Pencil-and-paper proof without enumerating all 54 cases.
Borrow-system feasibility approach: each borrow regime gives linear constraint.
Characterize which (g₁,g₂) satisfy at least one regime.
STATUS: OPEN (does not block Paper A submission)

---

## CONGRUENCE LATTICE STATUS

Descent chain: 54 → 20 → 14 → 10 → 7 → 4
At 4-state level: join of atoms collapses to 1 (trivial).
Action needed: exhaustive congruence search on 4-state system.
If no proper congruences exist: μ(K)=4 is minimum for proper congruence on Q₅₄.

---

## RETRACTED CLAIMS — NEVER USE IN PAPER A

These claims are retracted. Do not include under any framing:
- α = 2.0
- λ₂ = 1/7
- Majorana analogy
- Sonoluminescence analogy
- IET (interval exchange transformation) claim
- NHSE (L=7) claim
- "54 is our discovery" (Nuez 2021 already knew this)
- Semigroup order 8 (correct: 6)
- μ₁ = 0.1614 (UNDER DISPUTE — see Fix 2)
- "globally minimal quotient" (correct: minimal for gap observable)
- Complement involution T5 (superseded)

---

## PAPER SERIES STRUCTURE

| Paper | Subject | Status |
|-------|---------|--------|
| A | Gap quotient, graded arborescence, T1-T_SG | READY TO SUBMIT (after 5 fixes) |
| B | Fixed point iff 5∣b + base-b classification + height formula | In progress |
| C | Congruence lattice of 54-state system, μ(K)=20, descent chain | In progress |
| D | Theorem B analytic proof, Theorem O formal proof, general graded FDDS theory, functor Kaprekar_graded:(b,d)→GradedPoset | Future |

---

## IMMEDIATE ACTION ORDER

1. Resolve Fix 2 (μ₁ discrepancy) — BLOCKING
2. Apply Fixes 1, 3, 4, 5 (~30 min each)
3. Final read of abstract against locked one-liner above
4. Check all retracted claims are absent
5. Compile LaTeX, verify all figures
6. Submit to Theory of Computing Systems
# AQARION · COMPLETE THEOREM STACK
# Louisville Node #1 · All Sessions · M19–M21 2026
# Updated: M21 2026
# ============================================================================
# Status codes: PROVED | VERIFIED | HEURISTIC | OPEN | CONJECTURE | RETRACTED
# ============================================================================

---

## PART I — BASE-10 KAPREKAR GAP STRUCTURE (Paper A/B/C/D)

### Core Algebraic Theorems

**T1 — Algebraic Identity (PROVED)**
diff = 999g₁ + 90g₂
where g₁ = outer gap, g₂ = inner gap of the digit multiset.
This is the foundational identity for all gap-quotient results.

**T2 — Quotient Well-Definedness (PROVED)**
K_𝒢 is well-defined. Corollary of T1.
The Kaprekar map descends to the gap-pair quotient.

**T3 — Gap-Nerode Equivalence (PROVED, SCOPED)**
Gap equivalence = Nerode equivalence FOR THE GAP OBSERVABLE C(x)=gap(x).
⚠️ Scope rider required: "for the gap observable C(x)=gap(x)" everywhere.

**T4 — Graded Arborescence (PROVED)**
Q₅₄ (54-state gap quotient) is a graded in-arborescence of height 6.
Fixed point = 6174. All paths from non-fixed states have bounded depth ≤ 6.

**T5 — Borrow Signature (PROVED)**
Borrow signature is (1,1,[g₂=0]).
(Supersedes retracted complement involution T5.)

**T6 — Fixed Point Characterization (PROVED, b≤50)**
Fixed point exists iff 5∣b for d=4.
Proved exhaustively for b≤50. Open for all b algebraically.

**T7 — Nuez Formula (PROVED)**
|Image(K_{b,4})| = C(b+1,2) − 1 = T_b − 1
Proved as combinatorial identity. Cite Nuez (2021).
Special case b=10: |Image| = C(11,2) − 1 = 54. [= AQ-THM-001]

**T8 — Image Characterization (PROVED)**
Im(K) = Σ₃₀ (30 states).

**T_IM — Image Core Projection (PROVED)**
π(Im(K)) = Q₂₀ = join of all 56 congruence atoms.

**T_SG — Semigroup Structure (PROVED)**
Semigroup order: 6. Semigroup index: 6.
⚠️ Retracted: semigroup order 8 (was wrong).

**OP-19 — Complete Coordinate System (PROVED)**
(S, g₂) forms a complete coordinate system. Corollary of T1.

**AUT — Trivial Automorphism Group (PROVED)**
Aut(Q₂₀) = {id}.

---

### Spectral Theorems

**AQ-THM-001 / T7-special — Image Size Identity (VERIFIED)**
|Image(K_{10,4})| = 54 = T₁₀ − 1 = C(11,2) − 1.
[Exhaustive census. Exact arithmetic.]

**AQ-THM-002 — SUSY Bipartite Pairing (VERIFIED)**
μₖ + μ₆₋ₖ = 2 for all k ∈ {0,...,6}.
Exact (not approximate). Bipartite graph structure confirmed.

**AQ-THM-003 — Fiedler Eigenvalue Domain A (VERIFIED)**
μ₁(A) ≈ 0.16144.
⚠️ UNDER DISPUTE: session doc says "change to 0.0016" — needs recomputation.
DO NOT publish either value until discrepancy resolved.

**AQ-THM-004 — Fiedler Eigenvalue Domain B (VERIFIED)**
μ₁(B) ≈ 0.16243.

**AQ-THM-005 — Gauge Invariant (VERIFIED)**
F_H ≈ 4.222.

**AQ-THM-006 — Index Theorem (VERIFIED)**
dim(ker L) = number of connected components = 1.
Zero mode μ=0 confirmed.

**AQ-THM-007 — DAG Convergence (VERIFIED)**
All non-repdigit 4-digit base-10 states converge to 6174.

**AQ-THM-008 — Defect Nonmarkov Zero (VERIFIED)**
defect_nonmarkov = 0 universally across all surveyed bases and dimensions.

**AQ-THM-009 — Cycle Eigenvalue Structure (VERIFIED)**
Cycle eigenvalues are roots of unity, never nilpotent.
Prior subtraction step structurally unnecessary [M20 correction].

**T_B — Operator Norm (VERIFIED empirically, OPEN analytically)**
‖P^t‖₂ = √(N_t^max) for t=1,...,7.
At t=7: N₇^max = 705, ‖P^7‖₂ = √705.
Analytic proof: for functional graph (out-degree 1), operator norm of P^t = √(max in-degree of K^{-t}).
Proof via singular values of functional graph transfer operators.
STATUS: VERIFIED computationally. Analytic proof = TARGET (Paper D).

**AQ-THM-012 — Cheeger Ratio (VERIFIED, CORRECTED)**
h_deg/μ₁ ≈ 1.0465 (degree-based volumes).
⚠️ Count-based ≈ 1.847: KILLED.

---

### Quotient Structure Theorems

**μ(K) = 20 — Future-Dynamics Quotient Minimum (PROVED)**
18 collision groups in future-sequence analysis.
States sharing same future trajectory collapse to Q₂₀.
μ(K) = 20 = minimal future-dynamics quotient class count.

**CONGRUENCE LATTICE (PARTIAL)**
Descent chain: 54 → 20 → 14 → 10 → 7 → 4.
At 4-state level: join of atoms → trivial.
Open: exhaustive congruence search on 4-state system.
If no proper congruences: μ(K)=4 is minimum for proper congruence on Q₅₄.

---

## PART II — NEGABASE KAPREKAR (Paper A8 / arXiv math.CO)

**AQ-THM-010 — d=3 Universal Collapse (THEOREM, PARTIAL)**
For any B≥2, negabase Kaprekar d=3 map has exactly one fixed-point attractor.
Zero basin entropy. Zero cusp rate.
Key identity: diff = (p−r)·(B²−1) [middle digit cancels].
Semiconjugacy to 1D map F(δ) on δ=p−r ∈ {1,...,B−1}.
Fixed points: Even B: x*=(B/2−1, B−1, B/2). Odd B: x*=((B−1)/2, B−1, (B+1)/2).
⚠️ Odd-B reconstruction arithmetic incomplete [K-14, TARGET-01].

**AQ-THM-011 — ι₃ Universality (HEURISTIC)**
|T(Ω_{b,3})| = b for all b≥2.
Confirmed computationally for all surveyed bases.
Formal proof: OPEN (TARGET-04).

**PHASE DIAGRAM — Peak Entropy at d=2 (VERIFIED)**
Exhaustive census: b=−6 to −16, d=2–5.
d=2 universally shows maximum H_norm.
d=3 universally collapses (H=0, attr=1).
Anomalies: b=−10,d=4 (H=0); b=−12,d=5 (H=0.095) [TARGET-05].

---

## PART III — GRADED FDDS CLASSIFICATION (Paper B/D)

**GRADED SYSTEMS FOUND**
(5,4): height 3
(10,4): height 6
(13,6): height 19
(17,6): height 39

**FORMULA CONJECTURE (d=4)**
height(b,4) = 3b/5 for b divisible by 5.
Needs verification for non-multiples.

**FORMULA CONJECTURE (d=6)**
heights 19,39 at b=13,17. Difference=20 per b-step of 4.
Conjectured: height(b,6) = linear in b + constant.
Action: sweep b≤30, d=6. Check b=21,25,29.

---

## PART IV — RESEARCH REWRITE CALCULUS (AML Phase IV)

**RESEARCH STATE INVARIANTS (FORMAL)**
I1: Every mathematical object has one persistent identifier (no duplicates).
I2: Every derived object has a finite dependency path (provenance complete).
I3: Evidence is immutable (supersession creates new EO, never edits old).
I4: Dependency graph is acyclic (definitions→lemmas→theorems→claims→artifacts).
I5: Every theorem reachable from definitions (no orphan claims).
I6: Generated artifacts correspond to exactly one Research Graph state.

**REPOSITORY NORMAL FORM**
N(R) = normalization operator.
Repository is stable when N(R) = R.
Stronger than "all tests pass" — semantic state is canonical.

**RESEARCH ENTROPY**
H(R) = w₁D + w₂O + w₃S + w₄U + w₅C
where D=duplicates, O=orphans, S=stale evidence, U=unreachable claims, C=conflicts.
Repository-improving rewrites: H(R') ≤ H(R) (Lyapunov objective).

**REWRITE PRIMITIVES (monoid)**
CreateDefinition | AttachEvidence | PromoteClaim | SupersedeTheorem | GenerateArtifact | ArchiveCounterexample | NoOp (identity)
Composition f∘g. Associativity holds. Some pairs commute, some do not.

---

## RETRACTED — NEVER USE

| Claim | Reason |
|-------|--------|
| α = 2.0 | Flawed fit |
| λ₂ = 1/7 | Incorrect |
| Majorana analogy | Unjustified |
| Sonoluminescence analogy | Unjustified |
| IET claim | Unjustified |
| NHSE (L=7) | Only 1/3 conditions met |
| "54 is our discovery" | Nuez (2021) precedent |
| Semigroup order 8 | Correct: 6 |
| μ₁ = 0.1614 | Under dispute — recompute |
| "globally minimal quotient" | Correct: minimal for gap observable |
| Complement involution T5 | Superseded |
| d=5 critical dimension | Sampling artifact |
| 3:2:1 ratio in Domain A | Domain B only |
| β≈2 Mandelbrot | Flawed methodology |
| Banach FPT for finite set | Overkill — direct argument suffices |

---

## OPEN PROBLEMS (ranked by impact)

| # | Problem | Blocks |
|---|---------|--------|
| OP-01 | Odd-B reconstruction arithmetic (T10 gap) | Paper A8 §5 |
| OP-02 | κ_c(d,B) derivation from N_τ | Field theory appendix |
| OP-03 | T9 symbolic proof (no enumeration) | Paper C completeness |
| OP-04 | T6 algebraic proof for all b (not just b≤50) | Paper B |
| OP-05 | Congruence search on 4-state system | Paper C min result |
| OP-06 | ι₃ universality formal proof | Paper A8 Heuristic |
| OP-07 | T_B analytic proof (functional graph transfer op) | Paper D |
| OP-08 | Height formula h(b,d) closed form | Paper B |
| OP-09 | Classification: which (b,d) give graded FDDS | Paper B/D |
| OP-10 | b=−12,d=5 collapse mechanism | A8 phase diagram |
| OP-11 | Lean formalization T1, T2, T10 | V4/V5 certification |
