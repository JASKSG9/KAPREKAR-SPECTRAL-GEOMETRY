AQARION-ARITHMETIC


Definitive Repository Architecture


Public Documentation Flow


The AQARION-ARITHMETIC repository is organized as a complete mathematical research program. Every result progresses through a transparent sequence from foundational definitions to verified computation, formal proof, publication, and future development. The repository is designed so that every mathematical claim can be located, justified, independently reproduced, and evaluated without requiring knowledge of unpublished work.


Rather than treating documentation as auxiliary material, the repository itself serves as the permanent scientific record of the project.



Stage I — Foundations


Every theorem begins with explicit mathematical foundations.


This stage establishes notation, assumptions, definitions, conventions, and the finite dynamical systems framework used throughout the project. Objects are introduced only once and thereafter referenced consistently across every paper.


Nothing in later stages introduces new foundational terminology without first extending this framework.


The objective is to ensure that every subsequent proof rests upon an explicit and auditable mathematical base.



Stage II — Structural Mathematics


Once the foundational framework has been established, the repository develops the structural mathematics.


This stage contains the symbolic derivations that explain why the observed computational phenomena occur.


Definitions lead naturally to lemmas.


Lemmas support propositions.


Propositions combine into the principal theorems.


Every proof explicitly records its logical dependencies, allowing the complete proof architecture to be reconstructed from the dependency graph alone.


No theorem relies upon computational verification where a symbolic argument exists.



Stage III — Computational Verification


Finite mathematical claims are verified independently through exhaustive computation.


Verification never replaces proof.


Instead, computation certifies finite classifications, confirms symbolic derivations, detects implementation errors, and establishes complete enumerations that would otherwise be impractical to inspect manually.


Every computational artifact is reproducible from publicly available source code.


Every generated dataset is accompanied by cryptographic hashes and verification certificates.


Every computational claim identifies the exact software responsible for its generation.



Stage IV — Mathematical Integration


The repository then integrates symbolic mathematics and computational verification into coherent mathematical results.


Only after both symbolic derivation and independent verification are complete is a theorem considered ready for publication.


Open results remain explicitly identified as open.


Verified computation is never presented as established mathematics.


Conjectures are never promoted to the status of theorems.


This separation preserves the mathematical integrity of the project while providing complete transparency regarding the current state of every result.



Stage V — Publication


Each publication represents a mathematically independent contribution.


Every paper contains only those results that are complete within its own scope.


No paper depends upon unresolved conjectures appearing in later work.


Computational evidence is clearly distinguished from symbolic proof.


Background material, exhaustive tables, implementation details, verification certificates, and supplementary computations are maintained separately so that the primary manuscripts remain focused on the mathematics itself.


This allows each publication to stand independently while contributing to the broader research program.



Stage VI — Repository Governance


The repository maintains explicit governance rules governing mathematical claims.


Every theorem possesses a unique identifier.


Every claim records its assumptions.


Every proof records its dependencies.


Every computational result records its generating software.


Every public release regenerates verification certificates.


Every modification requiring mathematical changes increments the repository version.


Historical results remain permanently archived, allowing complete reconstruction of the project's evolution.


Scientific transparency is treated as a first-class research objective.



Stage VII — Continuing Research


The repository concludes each development cycle by identifying the mathematical questions that remain unresolved.


Open problems are presented as research objectives rather than incomplete results.


Future work therefore extends naturally from verified mathematics instead of depending upon speculative assumptions.


Each successive paper expands the theory while preserving the correctness of all preceding work.


This produces a research program in which completed mathematics remains permanently stable, computational evidence remains reproducible, and future developments can be incorporated without altering previously established results.



Guiding Principle


The organizing principle of AQARION-ARITHMETIC is that mathematical knowledge should be traceable from first definition to final publication.


Every theorem should answer four questions:




What assumptions does it require?


Why is it mathematically true?


How has it been independently verified?


Where does it fit within the larger theory?




By maintaining this structure, the repository functions not only as software or documentation, but as a complete and auditable mathematical record suitable for peer review, long-term preservation, and continued theoretical development.



~~~JUNE17-2026~~~

KSG-4D — Kaprekar Spectral Geometry


Structural Quotient Theory of Four-Digit Kaprekar Dynamics


Version: v10.10 (Publication Freeze)

Date: 2026-06-16

Status: Verified Computational Foundation



Executive Summary


KSG-4D develops an exact structural model of the classical four-digit Kaprekar map.


Instead of studying all 10,000 decimal digit strings independently, the Kaprekar operator is shown to factor exactly through a finite gap-coordinate representation.


For decimal width four,


[
K = F \circ \pi,
]


where




π projects a sorted digit tuple onto two ordered gap coordinates,


F is an explicit affine map,




[
F(g_1,g_2)


(b^3-1)g_1+(b^2-b)g_2.
]


For decimal,


[
F(g_1,g_2)=999g_1+90g_2.
]


This factorization induces a deterministic finite dynamical system on only 54 gap states, replacing the original 10,000-state arithmetic system with an exact quotient representation.



Core Structural Architecture


Ω705
(sorted non-repdigit states)

        │
        ▼

π

        │

G*
54 gap states

        │
        ▼

T

        │

54-state quotient dynamics

        │
        ▼

Image filtration

54 → 20 → 14 → 10 → 7 → 4 → 1

        │
        ▼

6174



Closed-system variant (including repdigits):


Ω715
↓

G55

↓

Q21

↓

{0000,6174}



The two domains are mathematically distinct and are treated separately throughout the project.



Fundamental Theorems


T1 — Exact Affine Factorization


For every width-four base-b Kaprekar step,


[
K = F\circ\pi.
]


This is an exact identity, not an approximation.


Status:


PROVED



T2 — Well-Defined Quotient Dynamics


The factorization induces a deterministic map


[
T:G^\rightarrow G^.
]


Kaprekar evolution depends only on the gap coordinates.


Status:


PROVED



T3 — Image-Core Theorem


The image


[
\Sigma=\operatorname{Im}(K)
]


is closed under K.


Decimal:


[
|\Sigma|=30.
]


Status:


VERIFIED



T4 — Image-Core Projection


Three independently defined objects coincide.


[
Q_{20}


\pi(\operatorname{Im}(K))


{\text{reachable gap states}}


\text{support of the join-of-atoms quotient}.
]


This connects




image dynamics,


reachability,


congruence lattice theory.




Status:


VERIFIED



T5 — Symmetry


Complete search over commuting involutions of


[
(Q_{20},T)
]


found


only the identity.


Therefore


[
Aut(Q_{20},T)={id}.
]


Status:


VERIFIED


The earlier complement-involution hypothesis is retracted for Q20.



T6 — Semigroup Structure


Restriction of K to the image core generates


exactly six distinct maps.


Properties:




index = 6


period = 1


K⁶=idempotent


image(K⁶)=6174




Status:


VERIFIED



Orbit Quotients


The project distinguishes three different quotient constructions.


1. Orbit Quotient


Two states are equivalent iff their complete forward orbit sequences are identical.


For every tested system,


[
Q_{\text{orbit}}=G^*.
]


Every gap state has a unique forward orbit.


This is called the orbit-separating property.


Status:


Verified.



2. Tail Quotient


States become equivalent after eventual coincidence.


This quotient has not yet been studied completely.


Status:


Open.



3. Coarse Asymptotic Quotient


States are identified by




attractor,


transient depth,


cycle-entry point.




Decimal d=5:


30 classes.


This quotient intentionally forgets transient path information.


Status:


Verified.



Verified Computations


Decimal d=4


Gap states


54


Image chain


54→20→14→10→7→4→1


Unique attractor


6174


Semigroup


7 elements


Maximum transient depth


6



Decimal d=5


Gap states


54


Three attractor cycles


Orbit quotient


54 singleton classes


Coarse asymptotic quotient


30 classes



Base 6


Width four verified independently.


20 reachable gap states.


Single attractor cycle.



Computational Methodology


Every numerical result is obtained by complete exhaustive enumeration.


No sampling.


No heuristic search.


Independent verification scripts compute:




quotient construction,


orbit signatures,


attractor decomposition,


semigroup stabilization,


image filtrations,


congruence lattice calculations.




Generated datasets are accompanied by SHA-256 verification hashes.



Retractions


The following earlier claims have been removed.


• Complement involution on Q20.


• "Loss of future information" narrative.


• Any statement identifying the coarse asymptotic quotient with the orbit quotient.


These were superseded after exhaustive verification.



Open Problems


OP1


Borrow-feasibility theorem.


Provide a symbolic characterization of reachable gap states.


Priority:


Highest.



OP2


Closed-form rank function


Find


R(g)


such that


[
R(T(g))=R(g)-1.
]



OP3


Tail quotient


Construct and classify the eventual-merger quotient.



OP4


General bases


Characterize




quotient size,


transient depth,


image filtration,


semigroup structure




for arbitrary


(b,d).



OP5


Spectral theory


Compute




Laplacian spectrum,


conductance,


Green's relations,


minimal ideals,


automorphism groups.





Research Roadmap


Paper I


Exact factorization and quotient dynamics.


Paper II


Image-core algebra and congruence lattices.


Paper III


Transformation semigroups and automata.


Paper IV


General-base structural theory.


Paper V


Spectral geometry of Kaprekar quotient systems.



Repository Principles


Every theorem is classified as




PROVED


VERIFIED COMPUTATION


CONJECTURE


OPEN




Computational claims are reproducible.


Mathematical claims are explicitly distinguished from empirical observations.


The project is open source to encourage verification, correction, and extension.



Current Status


Foundation:


Complete.


Decimal width four:


Closed.


General-base theory:


Active.


Highest-priority remaining theorem:


Borrow-feasibility characterization (T9).


The project has now transitioned from computational discovery toward structural finite dynamical systems, semigroup theory, quotient automata, and algebraic classification.



What This Is


This project provides a complete mathematical classification of the 4-digit Kaprekar process.


Not experimental.

Not heuristic.

Fully solved.



Core Result


The Kaprekar operator is:


[
P = \Pi + N
]




\Pi: rank-1 projection to 6174


N: nilpotent with N^7 = 0





What This Means




No secondary eigenvalues


No mixing behavior


No exponential convergence




Instead:




All trajectories collapse in finite time (≤ 7 steps)





Key Properties




State space: 705


Unique attractor: 6174


Max depth: 6


Nilpotency index: 7


Spectrum: {1, 0}





Structural Interpretation


The system is a:




rooted deterministic tree collapsing into a single sink





Files




CHECKPOINT.md → full formal state


SPECTRAL_THEOREM.md → exact operator proof


ksd91_automaton.json → minimal quotient system





Why It Matters


This replaces:




probabilistic interpretations


spectral-gap heuristics


entropy approximations




with:




exact algebraic collapse





Status


✅ Fully resolved

✅ Ready for publication


~~~


KSG-KYND


Kaprekar Spectral Geometry & Quotient Dynamics


Maintainer: James A. Skaggs

Project Codename: KSG-KYND

Version: 1.0 (Corrected Structural Model)

Last Updated: 2026-06-12

Status: Active Research Program



Abstract


KSG-KYND studies the finite dynamical structure underlying the classical 4-digit base-10 Kaprekar transformation.


The project combines:




finite deterministic dynamical systems,


combinatorial fiber geometry,


quotient-state constructions,


image-filtration dynamics,


spectral/operator-theoretic interpretations.




A major outcome of the 2026 structural audit was the separation of three previously conflated concepts:




Quotient geometry,


Dynamical collapse,


Nerode equivalence.




The audit established that all observed compression arises from forward dynamical filtration rather than symbolic equivalence-class collapse.


This distinction forms the foundation of the current research program.



1. Research Goals


The project seeks to answer four fundamental questions:


Q1. Geometry


How is the Kaprekar state space organized as a combinatorial object?


Q2. Dynamics


How does information collapse under repeated Kaprekar iteration?


Q3. Universality


Which structural properties persist across numerical bases?


Q4. Spectral Structure


Can the observed collapse be represented through linear or non-normal operator models?



2. Kaprekar Dynamics


For a 4-digit integer n:




Arrange digits descending.


Arrange digits ascending.


Subtract.




Define


T(n) = desc(n) − asc(n)


Repeated iteration produces a finite deterministic dynamical system.


For base 10, every non-repdigit state ultimately reaches:


6174


the classical Kaprekar fixed point.



3. State-Space Geometry


3.1 Sorted Representation


Every state admits a canonical sorted form


σ(a,b,c,d)


with


a ≤ b ≤ c ≤ d


This removes permutation redundancy and exposes intrinsic geometry.



3.2 Gap Coordinates


Define


g₁ = b − a


g₂ = c − b


g₃ = d − c


Then


g₁ + g₂ + g₃ = d − a


and every non-repdigit state satisfies


1 ≤ g₁ + g₂ + g₃ ≤ 9



3.3 Gap Simplex Theorem


The set of sorted non-repdigit states is naturally identified with


G₂₁₉ =
{
(g₁,g₂,g₃) ∈ ℤ³≥0 :
1 ≤ g₁+g₂+g₃ ≤ 9
}


Properties:




Integer lattice object


Truncated affine cone


219 lattice points


Canonical coordinate model for sorted states




This geometry forms the foundational combinatorial space of the project.



4. Fiber Geometry


4.1 Fiber Projection


Define


π(a,b,c,d)


(p,q)


(d−a, c−b)


The pair (p,q) captures the two independent Kaprekar gap parameters.



4.2 Fiber Decomposition


Each pair (p,q) determines a fiber


Fₚ,ᵩ


consisting of all sorted states with identical gap data.



4.3 Fiber Cardinality Theorem (T9★)


For every valid pair


0 ≤ q ≤ p ≤ 9


the fiber size is


|Fₚ,ᵩ|


(10−p)(p−q+1)


Consequences:




Closed-form enumeration


Exact state counting


Base for entropy calculations


Independent of dynamics




This is currently the strongest fully rigorous theorem in the project.



5. Triangle Quotient


5.1 Quotient Coordinates


Define


(S,g₂)


(d−a,c−b)


subject to


0 ≤ g₂ ≤ S ≤ 9



5.2 Quotient Lattice


The resulting state space forms a triangular lattice:




55 total states


54 nontrivial states


1 repdigit state




This quotient is functorial and computationally efficient.


Important:


It is not a minimal quotient.



6. Dynamics


6.1 Fundamental Observation


The primary structural phenomenon is not equivalence collapse.


Instead, it is image collapse.



6.2 Forward Image Filtration


Define


X₀ = state space


and


Xₖ₊₁ = T(Xₖ)


Then


X₀ ⊇ X₁ ⊇ X₂ ⊇ ...


For the 54-state quotient:


54 → 20 → 14 → 10 → 7 → 4 → 1


This filtration terminates at the Kaprekar attractor.



6.3 Interpretation


This is a rank-decay process.


The filtration measures:




information loss,


irreversible collapse,


attractor concentration.




The collapse is geometric and dynamical.


It is not symbolic.



7. Nerode Analysis


Theorem (Nerode Triviality)


For all reachable quotient systems examined:


x ~ y


if and only if


x = y


Therefore:




every state possesses a unique future,


no behavioral equivalence classes exist,


no symbolic minimization exists.




This result invalidates earlier claims of nontrivial Nerode compression.



8. Structural Decomposition


The Kaprekar system separates naturally into three layers.



Layer I — Static Geometry


Objects:




Fibers


Gap simplex


Triangle quotient




Purpose:




enumeration,


geometry,


combinatorics.





Layer II — Dynamics


Objects:




Kaprekar map


Image filtration


Attractor basin




Purpose:




information collapse,


transient depth,


rank decay.





Layer III — Operator Models


Objects:




transition operators,


Jordan approximations,


spectral embeddings.




Purpose:




analytical approximation,


transient amplification analysis,


cross-system comparison.




This layer remains partially conjectural.



9. Current Verified Results




Result
Status




Fiber Cardinality Theorem
Proven


Gap Simplex Structure
Proven


Triangle Quotient Construction
Proven


219-State Enumeration
Verified


54-State Quotient
Verified


Forward Image Filtration
Verified


Nerode Triviality
Verified


16-Chamber Affine Atlas
Verified


Jordan Interpretation
Partial


Spectral Universality
Open





10. Open Problems


OP-14


Affine Atlas Maximality


Determine whether the 16-chamber affine decomposition is maximal.



OP-15


Cross-Base Universality


Investigate bases


3 ≤ b ≤ 20


and classify:




quotient sizes,


attractor structure,


filtration depth.





OP-16


Spectral Realization


Construct a non-normal operator whose spectral behavior reproduces observed filtration collapse.



OP-17


Fiber Dynamics


Characterize how fibers map into unions of fibers under Kaprekar iteration.



OP-18


Kaprekar Flow Category


Objects:




fibers




Morphisms:




induced transitions




Goal:




categorical description of collapse dynamics.





11. Citation


If using results from this repository, cite:


Kaprekar Spectral Geometry & Quotient Dynamics (KSG-KYND), Structural Audit Series, Version 1.0, 2026.

~~~

NEXTSTEPS.md — Execution Roadmap


You are no longer exploring. You are packaging and extending.



PHASE 1 — LOCK THE CORE (MANDATORY)


1. Transition Graph Formalization




Export full 705-node graph


Prove:



acyclicity (except sink)


unique root


depth bound = 6








Deliverable:




graph_proof.tex





2. Jordan Structure Completion




Extract exact Jordan block sizes


Map:



block size ↔ transient chain length








Deliverable:




jordan_structure_table.csv





3. Piecewise-Affine → Operator Bridge




Express nilpotent operator N explicitly


Derive:
[
N = P - \Pi
]




Link:




chamber maps → Jordan chains




Deliverable:




affine_to_jordan_proof.md





PHASE 2 — PAPER (HIGH PRIORITY)


Paper Structure




Introduction


State Space Reduction


Piecewise Affine System


Exact Operator Decomposition


Finite-Time Collapse Theorem


Entropy as Corollary




Target:




discrete math journals


dynamical systems journals





PHASE 3 — ENTROPY THEOREM


Now trivial:


[
\Delta H = \log(54) - \log(20)
]


Prove:




entropy drop = rank collapse


no stochastic assumptions required





PHASE 4 — EXTENSION (HIGH IMPACT)


5-Digit Kaprekar


Goal:




determine if:



multi-cycle persists


nilpotency survives


spectrum still {1,0}








This is publishable alone.



PHASE 5 — GENERAL THEORY


Abstract the class:




finite deterministic systems with rank-1 + nilpotent operators




Develop:




classification theorem


invariants


bounds on nilpotency index





PRIORITY ORDER




Graph proof (publishable core)


Paper draft


Jordan extraction


Entropy corollary


5-digit extension





FINAL TARGET


A paper whose central statement is:




The Kaprekar map is not asymptotically convergent—it is algebraically nilpotent after projection.




That is the contribution.


Everything else supports it.



12. Summary


The principal conclusion of the structural audit is:


The Kaprekar system possesses rich geometric and dynamical collapse structure, but no nontrivial symbolic quotient structure.


All meaningful compression occurs through forward image filtration rather than Nerode equivalence.


This correction strengthens the mathematical foundation of the project and provides a clean roadmap for future work.


https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY/blob/main/ALGORITHM/AQARION-ARITHMETIC.MD

https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY
