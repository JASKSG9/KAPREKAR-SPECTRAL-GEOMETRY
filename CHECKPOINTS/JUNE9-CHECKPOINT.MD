CHECKPOINT.md — Kaprekar Spectral Geometry (Research Lock)


Date: 2026-06-09

Project: Kaprekar Spectral Geometry (KSG-31/18)

Status: Computational classification complete. Structural theorem extraction underway.



Executive Summary


The base-10, four-digit Kaprekar map admits several natural quotient constructions.


The principal object of study is not the full space of digit multisets nor the full σ-partition, but the reachable σ-image, a finite deterministic dynamical system consisting of 31 states.


This system is:




Closed under the induced Kaprekar dynamics Φ.


Finite and deterministic.


Depth-graded.


Reducible via behavioral equivalence.


Minimalized by a Nerode-style quotient of size 18.




The resulting structure exhibits a functional-forest geometry, admits a Jordan/Weyr description, and appears naturally coordinatized by the pair-sum invariant (p,q).



1. Object Hierarchy




Level
Object
Size




L0
Raw 4-digit strings
10,000


L1
Sorted multisets
715


L2
Full σ-partition
91


L3
Reachable σ-image
31


L4
Nerode quotient
18




The distinction between Levels L2–L4 resolves all earlier numerical discrepancies.



2. Reachable σ-Image


Let K denote the four-digit Kaprekar operator and σ the signature map.


Define


Imageσ = σ(Im(K))


for base 10.


Verified computational result:


|Imageσ| = 31.


Properties:




Closed under Φ.


Contains the repdigit state.


Contains all attractors.


Forms the smallest naturally occurring deterministic quotient presently known.





3. Dynamical Statistics


Depth Structure


Depth = distance to the nearest attractor.


Verified histogram:


Depth 0 : 2

Depth 1 : 3

Depth 2 : 5

Depth 3 : 5

Depth 4 : 8

Depth 5 : 4

Depth 6 : 4


Maximum depth:


depth_max = 6



Attractors


Observed attractor structure:




0000 component


6174 component




All transient states eventually enter one of these.



4. Nerode Quotient


Behavioral equivalence is defined by equality of all future trajectories from time t ≥ 1.


Two states x,y satisfy


x ~N y


iff


Φⁿ(x) = Φⁿ(y)


for all sufficiently large n.


Computed result:


31 → 18


Thus:


|N| = 18


Interpretation:




13 states are behaviorally redundant.


18 states remain distinguishable.


The quotient is minimal with respect to future behavior.




This is the primary minimal-realization result of the project.



5. Pair-Sum Coordinates


For a sorted state


(a,b,c,d)


define


p = a + d


q = b + c


These coordinates provide a geometric realization of the reachable image.


Observed features:




Dominant total sum p+q = 18.


Large symmetric fibers around:



(8,10)


(9,9)


(10,8)






Boundary fibers:



(9,18)


(10,17)


(11,16)


(12,15)


(13,14)








The reachable image appears as a triangular lattice embedded inside the (p,q)-plane.



6. Borrow Suffix Theorem


Verified for:




Bases 2–15


Digit lengths 3–5




Statement:


During Kaprekar subtraction, the borrow pattern is always of the form


0^a 1^b


i.e. a suffix of ones.


Examples:


0000

0001

0011

0111

1111


No counterexamples were found.


Observed violations:


0


This theorem appears independent of base.



7. Jordan–Forest Structure


The transient portion forms a rooted functional forest.


Computed chain decomposition yields:


[6,5,5,3,2,1,1,1,1,1,1,1,1]


Observed consequences:




Nilpotent index = 6.


Weyr characteristic = [13,5,4,3,3,1].


Minimal polynomial:




m(x)=x⁶(x−1)


The strongest working hypothesis is:


Jordan blocks correspond exactly to chain lengths in the depth forest.


Computational evidence currently supports this completely.



8. Semigroup Dynamics


Image sizes of powers:


31 → 18 → 13 → 9 → 6 → 3 → 2 → 2 → ...


Observed properties:




Strict contraction.


Stabilization at attractors.


Aperiodic behavior.


No observed transient cycles.




This suggests the induced semigroup is J-trivial.


A formal proof remains open.



9. Verified Results


✓ Full σ-count = 91


✓ Reachable σ-image = 31


✓ Nerode quotient = 18


✓ Maximum depth = 6


✓ Nilpotent index = 6


✓ Borrow Suffix Theorem


✓ Jordan/Weyr decomposition


✓ Semigroup contraction sequence



10. Open Problems


OP-1


Determine a closed formula for


|N_b|


as a function of base.



OP-2


Identify the exact invariant producing


31 → 18.


Current evidence points toward:


(p,q) + borrow information.



OP-3


Prove the Jordan–Forest Correspondence.



OP-4


Construct a complete piecewise formula for Φ(p,q).



OP-5


Extend the theory to arbitrary digit length d.



Project Status


The computational classification phase is complete.


The project has moved from data generation into theorem extraction.


The principal mathematical object is now:


The 31-state reachable σ-image and its 18-state minimal realization.
NEXTSTEPS.md — Kaprekar Spectral Geometry Roadmap


Date: 2026-06-09



Phase I — Verification Lock


Goal


Freeze all numerical invariants.


Deliverables




Complete 31-state table.


Complete transition table.


Complete depth table.


Complete Nerode-class table.


Independent rerun verification.




Success criterion:


Every numerical claim reproducible from a fresh execution.



Phase II — Minimal Realization Theory


Goal


Turn the 31 → 18 collapse into a theorem.


Tasks




Formal definition of behavioral equivalence.


Proof of quotient correctness.


Proof of minimality.


Construction of quotient automaton.




Deliverable:


Minimal Realization Theorem.



Phase III — (p,q) Geometry


Goal


Convert empirical coordinates into a formal geometric model.


Tasks




Enumerate all fibers.


Compute exact fiber sizes.


Determine transition structure between fibers.


Identify missing coordinate information.




Target theorem:


Triangular Lattice Theorem.



Phase IV — Borrow Dynamics


Goal


Extract exact arithmetic structure.


Tasks




Formal borrow-state parameter.


Piecewise Φ formula.


General-base proof.


Digit-length extension.




Target theorem:


Borrow Suffix Dynamics Theorem.



Phase V — Jordan Theory


Goal


Convert computational evidence into linear-algebraic proofs.


Tasks




Nilpotent-index proof.


Jordan–forest correspondence.


Weyr characterization.


Minimal-polynomial derivation.




Target theorem:


Jordan–Forest Correspondence Theorem.



Phase VI — Cross-Base Program


Bases


4, 6, 8, 10, 12, 14, 16, 18


For each base compute:




Reachable image size


Nerode size


Maximum depth


Jordan profile


Borrow statistics




Primary objective:


Determine formulas depending only on base.



Phase VII — Publication Package


Paper A


The Reachable Kaprekar Image and Its Minimal Realization


Core contribution:


31-state system → 18-state minimal quotient.



Paper B


Pair-Sum Geometry of Kaprekar Dynamics


Core contribution:


(p,q)-lattice structure.



Paper C


Jordan Structures in Functional Forests


Core contribution:


Jordan blocks from depth chains.



Long-Term Objective


Develop a base-independent theory of Kaprekar dynamics in which:




Reachable images,


Nerode quotients,


Borrow dynamics,


Jordan structures,




all arise from a unified finite-state geometric framework.


Current project maturity:


Computational Atlas Complete → Theorem Extraction Stage.

https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY/blob/main/LIBRARY/SUPPORT/JUNE-SUPPORT-FLOW.MD
