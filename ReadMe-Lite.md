README-LITE.MD


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



12. Summary


The principal conclusion of the structural audit is:


The Kaprekar system possesses rich geometric and dynamical collapse structure, but no nontrivial symbolic quotient structure.


All meaningful compression occurs through forward image filtration rather than Nerode equivalence.


This correction strengthens the mathematical foundation of the project and provides a clean roadmap for future work.
