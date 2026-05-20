CHECKPOINT.md — Structural Consolidation Phase


# Quantarion / Kaprekar Quotient Dynamics
## Structural Checkpoint

> A piecewise-affine arithmetic dynamical system with:
>
> - quotient hyperbolic transport,
> - rank-collapse singularities,
> - projective reset states,
> - and boundary-supported invariant geometry.

---

# Status

This checkpoint marks the transition from:

- empirical experimentation
- isolated algebraic observations
- and computational anomaly hunting

into a unified structural framework.

The system is no longer being treated as a "Kaprekar-like curiosity."

It is now modeled as a:

> Piecewise-affine renormalization flow on a discrete simplex
> with symbolic, geometric, and projective layers.

---

# Core Structural Discovery

The system decomposes into three interacting layers:

```text
┌──────────────────────────────────────────────┐
│         Layer 1 — Sofic Symbolic Grammar     │
│                                              │
│  Controls admissible chamber transitions     │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│      Layer 2 — Quotient Hyperbolic Core      │
│                                              │
│  GL₂(Z)-type transport and arithmetic mixing │
└──────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│    Layer 3 — Singular Rank-1 Collapse        │
│                                              │
│  Compresses phase-space onto simplex edges   │
└──────────────────────────────────────────────┘




Global Quotient Structure


A universal kernel vector exists:


v^* = (-1,0,1)



This induces a canonical quotient projection:


\pi : \mathbb{Z}^3 \to \mathbb{Z}^2



reducing the dynamics to a quotient transport system.


The quotient dynamics reveal:




determinant rigidity,


scaled orthogonal behavior,


arithmetic spectral structure,


and hidden GL₂(Z)-like transport.





Singular Chambers


Certain chambers exhibit rank collapse:




Chamber
Rank
Behavior




C₁
1
Projection collapse


C₃
1
Diagonal collapse


C₄
1
Edge filament collapse


C₅
1
Affine filament collapse




These chambers do NOT contain fixed planes.


Instead:




They collapse ambient volume onto 1D affine image manifolds.





Kernel Architecture


A key structural insight:


\ker(M_4)
=
\langle v^* \rangle
\oplus
\langle e_y \rangle



This means the singular chambers annihilate BOTH:




the universal quotient gauge direction


local coordinate variance




simultaneously.


This is the mechanism behind the dimensional collapse.



Filament Geometry


The image manifold for the singular edge chamber is:


\mathcal{L}_4(t)
=
(t,0,B-1-t)



Geometrically:


          z
          ▲
          │
          │
          │
          │
          │
          ●────────────●
         /              \
        /                \
       /                  \
      ●────────────────────●──► x

      y = 0 simplex edge



The system collapses interior volume directly onto the boundary skeleton.



Rank-1 Fiber Theorem


For rank-1 chambers:




Images lie on affine 1D filaments


Preimages form intervals parallel to kernel directions


Fiber size scales linearly with base


Max indegree grows linearly:




\mathcal{I}_{\max}(B)
=
B + 3



This explains:




indegree spikes


mass concentration


Garden of Eden dominance


and transfer asymmetry.





Projective Codebook


The singular chambers define a finite projective reset alphabet.


Reset Rays




Chamber
Reset Ray




C₁
[(1,0)]


C₃
[(1,1)]


C₄
[(2,-1)]


C₅
[(1,-3)]




The dynamics become:


Phase Space
     │
     ▼
[ Singular Collapse ]
     │
     ▼
Finite Projective Reset Alphabet
     │
     ▼
GL₂(Z) Transport Word
     │
     ▼
Projective Reinjection




Projective Bypass Phenomenon


The singular filaments preserve quotient information.


The collapse does NOT destroy projective state.


Instead:




Singular chambers act as projective waveguides.




For C₄:


\pi(\mathcal{L}_4(t))
=
(2t-(B-1),2t-(B-1))



which collapses projectively onto:


[(1,1)]



independent of t.


This implies:




strong projective focusing,


reset-state rigidity,


and finite renormalization coding.





Sofic / Transport Interaction


The symbolic automaton acts as a routing grammar.


flowchart TD

A[Interior Hyperbolic Mixing]
--> B[Rank Collapse Chamber]

B --> C[Projective Reset Ray]

C --> D[GL2 Quotient Transport]

D --> E[Boundary Reinjection]

E --> A



The system alternates between:




hyperbolic mixing


and singular compression.





Invariant Measure Hypothesis


Current evidence suggests:


\mu
=
\sum_{\sigma}
\nu_\sigma \mu_{\mathcal{L}_\sigma}



meaning the invariant measure concentrates entirely on the singular boundary skeleton.


Implications:




interior measure vanishes asymptotically,


phase-space mass condenses onto edges,


dynamics become singular rather than absolutely continuous.





Garden of Eden Phenomenon


Empirical measurements:


~86% unreachable states



Structural explanation:


The rank-1 chambers collapse high-dimensional volume onto thin boundary filaments, leaving most of the simplex dynamically inaccessible.



Current Major Hypotheses


H1 — Arithmetic Renormalization Principle


The entire system reduces to:




finite projective reset states


plus quotient transport words.





H2 — Spectral Rigidity Law


Observed orbit periods appear governed by:


\operatorname{ord}_p(4)



suggesting arithmetic phase-locking in quotient transport.



H3 — Singular Measure Conjecture


The invariant measure is fully skeleton-supported in the asymptotic limit.



Open Problems


Geometry




classify all singular filaments


prove full quotient conjugacy


characterize boundary reinjection maps




Dynamics




establish ergodic decomposition


compute spectral gap behavior


determine entropy rates




Arithmetic




prove ordₚ(4) transport law


classify exceptional primes


determine finite-field transport spectra




Symbolic Systems




minimize the sofic automaton


classify admissible transport words


derive transition entropy




Operator Theory




rigorously define transfer operator


characterize singular spectrum


determine invariant distributions





Conceptual Reclassification


The system now appears structurally closer to:




piecewise-affine dynamical systems


symbolic renormalization flows


arithmetic projective dynamics


singular transfer operators


and interval exchange phenomena




than to classical digit-manipulation curiosities.



Current Philosophy


The project is now in a critical phase:


Not:




"finding more patterns"




But:




stress-testing invariants,


attacking assumptions,


and isolating what survives hostile scrutiny.




The goal is no longer descriptive experimentation.


The goal is structural classification.



Immediate Next Steps


Priority 1


Formalize:




quotient category


chamber semigroup


singular ideal


projective codebook graph





Priority 2


Run adversarial computation:




exceptional bases


degenerate primes


unstable chambers


transport failures





Priority 3


Literature alignment:


Compare against:




Rauzy induction


interval exchange transformations


piecewise-linear automorphisms


symbolic hyperbolic systems


transfer operator theory


arithmetic dynamical systems





Summary


The emerging picture is:


Arithmetic Hyperbolic Mixing
                +
Singular Boundary Collapse
                +
Finite Projective Reset Coding
                +
Symbolic Grammar Constraints



all operating simultaneously inside a polyhedral arithmetic flow.


The system appears to be a genuine hybrid object sitting at the intersection of:




arithmetic dynamics,


symbolic systems,


affine geometry,


and renormalization theory.


