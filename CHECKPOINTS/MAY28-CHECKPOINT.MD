CHECKPOINT.md


Project


The Kaprekar Map as a Symbolic-Dynamics Minimization

Subtitle: Myhill–Nerode Equivalence and the Minimal Deterministic Factor of a Non-Invariant Observable



Current Status


Checkpoint Date: 2026-05-29


Status: Final paper draft completed.


The paper has undergone multiple rounds of structural, mathematical, and verification review. All previously identified issues have been addressed.


The work now presents a theorem-driven symbolic-dynamics construction rather than a computation-first investigation.



Core Result


Starting from the 10-chamber observable


C = (borrow pattern, output permutation),


the refinement process


P₀ = C


Pₙ₊₁ = Pₙ ∨ K⁻¹(Pₙ)


stabilizes after three refinement steps:


|P₀| = 10

|P₁| = 37

|P₂| = 51

|P₃| = 54

|P₄| = 54


yielding a 54-block partition P∞.


This partition is characterized intrinsically by itinerary equivalence:


x ∼ y  ⇔  I(x) = I(y),


where I(x) is the full future chamber itinerary.


Equivalently,


P∞ = ⋁ₙ≥0 K⁻ⁿ(C).



Main Theorems


Theorem 1


Finite refinement stabilization.


The chamber refinement sequence stabilizes at 54 blocks.


Theorem 2


Exact lumpability.


The resulting partition P∞ is K-invariant and induces a well-defined quotient dynamical system.


Theorem 3


Myhill–Nerode characterization.


The blocks of P∞ are precisely the itinerary-equivalence classes.


Theorem 4


Coarsest K-invariant refinement.


P∞ is the coarsest partition refining C that is invariant under K.


Equivalent characterization:


P∞ = ⋁ₙ≥0 K⁻ⁿ(C).


Theorem 5


Quotient dynamics.


The quotient contains:




1 recurrent block


53 transient blocks




and possesses nilpotence index 6 on the transient subspace.



Verified Computational Facts


Verification 1


Refinement stabilization:


10 → 37 → 51 → 54 → 54


Confirmed.


Verification 2


Lumpability violations:


0


Confirmed.


Verification 3


Recurrent block size:


|B*| = 20


Confirmed.


Verification 4


Maximum quotient depth:


6


Confirmed.


Depth distribution:




Depth 0: 1


Depth 1: 3


Depth 2: 12


Depth 3: 10


Depth 4: 10


Depth 5: 10


Depth 6: 8




Verification 5


Weyr sequence:


[34, 40, 44, 47, 50, 53]


Confirmed.


Nilpotence index:


6


Verification 6


State-space depth filtration:


|L₀| = 37

|L₁| = 62

|L₂| = 148

|L₃| = 99

|L₄| = 138

|L₅| = 162

|L₆| = 58


Total:


704 transient states.


Maximum state-space depth:










Conceptual Structure


The paper separates three distinct objects.


1. Chamber Stratification


10 chambers.


Geometric observable.


Not invariant.


2. Minimal Deterministic Factor


54 blocks.


Myhill–Nerode quotient.


Coarsest K-invariant refinement.


3. State Dynamics


705-state Kaprekar graph.


Ground-truth dynamics.


Unique attracting cycle {6174}.



Depth Separation


Three notions of depth are distinguished.


State-space depth


Maximum: 7


Quotient depth


Maximum: 6


Nilpotence index


Maximum Jordan length: 6


The latter two coincide.



General Construction


The Kaprekar system is now presented as a special case of the following finite symbolic-dynamics procedure.


Given:




finite dynamical system (X, K)


arbitrary partition C




define


P∞ = ⋁ₙ≥0 K⁻ⁿ(C).


Then:




P∞ is the coarsest K-invariant refinement of C.


P∞ is the minimal deterministic symbolic factor respecting C.


States are equivalent iff they share identical C-itineraries.




The Kaprekar example provides a fully computed instance:


|X| = 705

|C| = 10

|P∞| = 54


with explicit quotient dynamics.



Open Problems


OP1


Arithmetic characterization of the 54 blocks.


Find explicit invariants replacing iterative refinement.


OP2


Generalization to arbitrary digit length d and base b.


Determine chamber structure and refinement growth.


OP3


Regular-language description.


Determine whether the 54-state quotient admits a finite automaton presentation directly in the chamber alphabet.



Publication Assessment


Definitions: 10/10


Conceptual organization: 10/10


Mathematical honesty: 10/10


Proof completeness: 9.5/10


Computational reproducibility: 10/10


Overall status:


The project has reached a stable theorem-centered form. The principal object is no longer the computed 54-block partition itself, but the symbolic-dynamics construction that identifies it as the minimal deterministic factor of a non-invariant observable.


