KSG-KYND: Exact Quotient Semiconjugacy in Generalized Kaprekar Dynamics

Repository Overview

This repository contains the formal development, computational verification, and manuscript materials for the KSG-KYND program: an exact quotient-state reformulation of generalized Kaprekar dynamics across arbitrary radix systems.


Unlike classical Kaprekar investigations, which focus primarily on isolated constants or trajectory anecdotes in base 10, this project studies the Kaprekar routine as a family of finite deterministic dynamical systems and proves that the full symbolic digit dynamics admit an exact lower-dimensional semiconjugate quotient.


The central object is not the familiar 6174 phenomenon.
The central object is the existence of a finite simplicial quotient state space on which the generalized Kaprekar operator descends exactly.



Mathematical Scope


For fixed base  and digit length , define the generalized Kaprekar map



K_{b,d}: \Omega_{b,d} \to \Omega_{b,d}



on the full symbolic digit state space



\Omega_{b,d} = \{0,1,\dots,b^d-1\}



by:




writing each state as a zero-padded -digit base- string,


sorting digits ascending and descending,


subtracting the ascending arrangement from the descending arrangement,


re-padding to  digits.




The KSG-KYND program constructs an exact quotient coordinate map



Q: \Omega_{b,d} \twoheadrightarrow Q_{b,d}



into a reduced paired-gap lattice state space and studies the descended finite dynamical system



\widetilde K_{b,d}: Q_{b,d} \to Q_{b,d}



satisfying the semiconjugacy relation



Q \circ K_{b,d} = \widetilde K_{b,d} \circ Q.



This converts the generalized Kaprekar routine from a large symbolic digit graph into an exact finite quotient dynamical system of substantially lower combinatorial dimension.



Core Verified Results in This Draft


The current arXiv draft develops and/or audits the following:




formal construction of the quotient paired-gap coordinate system;


exact semiconjugacy theorem between full symbolic dynamics and quotient dynamics;


combinatorial cardinality formula for the quotient simplex lattice;


explicit quotient census tables across tested bases and digit lengths;


lower-dimensional cycle and attractor observations on the quotient systems;


computational identification of chamber-dependent branch behavior in the descended operator.




Only formally proved statements are presented as theorems in the manuscript.
Computational observations and theorem-target conjectures are explicitly separated.



Current Claim Status Policy


This repository follows a strict verification ledger.
Every mathematical assertion is classified as one of:




[T] Formally Proved — deductively established and manuscript-safe;


[V] Computationally Verified — exhaustively checked over declared parameter ranges;


[C] Conjectural — supported by evidence but not yet algebraically derived;


[R] Motivational — explanatory context only.




No universal claim is stated without either proof or explicit verification bounds.
This repository is intentionally maintained as a non-promotional mathematical audit environment.



Repository Layout


KSG-KYND/
├── ARxIV/
│   ├── KSG-KYND-FLOW.md              # principal manuscript draft
│   ├── OVCR_CLAIM_REGISTER.md       # official verified claim register
│   ├── PROOF_LEDGER.md              # theorem-by-theorem proof burden tracking
│   └── NOTES_APPENDIX.md            # non-manuscript exploratory notes
│
├── core/
│   ├── quotient_builder.py          # Q_{b,d} construction and state enumeration
│   ├── descended_operator.py        # explicit K~ computation on quotient states
│   ├── semiconjugacy_verify.py      # machine verification of Q∘K = K~∘Q
│   └── census_runner.py             # multi-base quotient census sweeps
│
├── data/
│   ├── quotient_census_tables.csv
│   └── cycle_reports/
│
├── figures/
│   ├── simplex_lattice_examples/
│   └── quotient_flow_diagrams/
│
└── README.md




Relation to Earlier KSG Work


This repository is mathematically distinct from the earlier KSG-GMFLOW / spectral shell-collapse computational program.




KSG-GMFLOW studies the full functional graph on : shell filtrations, collapse depths, image compression, and spectral observables.


KSG-KYND studies the exact reduced quotient dynamics through semiconjugate factorization.




The former is a computational census framework.
The latter is a structural theorem framework.


They are complementary but should not be confused as the same manuscript project.



Reproducibility


All computational tables appearing in the manuscript are generated directly from the audited core scripts.
No hand-entered numerical constants are permitted in final manuscript tables.


Verification scripts include:




quotient cardinality cross-checks,


semiconjugacy identity sweeps,


quotient cycle census,


parameter-range consistency tests.




All reported computational observations are reproducible from repository code.



Manuscript Status


This repository is under active theorem audit and formalization.


Current stage:




semiconjugacy framework established; claim ledger under refinement; manuscript prose being hardened to referee-safe standards.




This means exploratory notes may coexist with formalized sections, but only verified sections are intended for arXiv release.



Citation (provisional)


If citing this work prior to formal release, reference the repository draft title:


KSG-KYND: Exact Quotient Semiconjugacy and Simplicial Reduction in Generalized Kaprekar Dynamics


Formal bibliographic metadata will be inserted upon manuscript submission.



Integrity Notice


This repository is maintained as a mathematical verification workspace.
