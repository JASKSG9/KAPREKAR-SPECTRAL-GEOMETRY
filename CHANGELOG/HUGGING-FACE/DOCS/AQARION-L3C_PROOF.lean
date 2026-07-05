

🧪 Lean LC3 – Exact Descent Equivalence (Draft Proof)

Theorem (exact_descent_iff)
Let (X, T) be a finite deterministic dynamical system, Π a partition with projection P, and D = (I - P) K P the defect operator. Then

```lean
theorem exact_descent_iff (Π : Partition X) :
  D = 0 ↔ (∀ f : ℝ ^ X, K f ∈ Im(P) ↔ f ∈ Im(P)) :=
```

In words:
D = 0 precisely when the image of the projection P is invariant under the Koopman operator K, i.e., K sends P‑constant functions to P‑constant functions.

---

Proof sketch (Lean‑compatible)

We work in the finite‑dimensional vector space ℝ ^ X.
P is the linear projection onto the subspace V = { f | ∀ x,y, Π x = Π y → f x = f y }.

Part 1: D = 0 → K V ⊆ V

Assume D = 0. For any f ∈ V (so P f = f), we must show K f ∈ V, i.e., P (K f) = K f.

From D = (I - P) K P = 0 we have (I - P) K P = 0.
Apply this to f:

```
(I - P) (K (P f)) = 0
```

Since P f = f, this simplifies to (I - P) (K f) = 0, so K f = P (K f), meaning K f ∈ V. ✔

Part 2: K V ⊆ V → D = 0

Assume K preserves V. For any f, we must show D f = 0.
Write f = f_v + f_h where f_v = P f ∈ V and f_h = (I - P) f ∈ W (the orthogonal complement).
Then

```
D f = (I - P) K (P f) = (I - P) (K f_v).
```

Because K f_v ∈ V (by invariance), we have (I - P) (K f_v) = 0. Hence D f = 0. ✔

Equivalence with the commuting diagram (well‑defined quotient)

We also state:

```lean
theorem descent_commuting_diagram (hDzero : D = 0) :
  π ∘ T = T_quot ∘ π := ...
```

This follows because D = 0 guarantees that the partition is a congruence: x ~ y → T x ~ T y.
The def of T_quot then makes the diagram commute by construction.
The reverse direction (commuting diagram → D = 0) is also true because the diagram enforces that if x ~ y then T x ~ T y, which is exactly the invariance condition.

---

Lean code scaffold (inline)

```lean
import Mathlib.LinearAlgebra.Projection
import Mathlib.LinearAlgebra.FiniteDimensional
open LinearMap

variables (X : Type) [Finite X] (T : X → X) (Π : X → X) [DecidableEq X]

-- projection onto Π‑constant functions
noncomputable def proj_Π : (ℝ ^ X) →ₗ[ℝ] (ℝ ^ X) := ...

-- Koopman operator
noncomputable def koopman : (ℝ ^ X) →ₗ[ℝ] (ℝ ^ X) :=
  { toFun := λ f x => f (T x),
    map_add' := ...,
    map_smul' := ... }

-- defect operator
noncomputable def defect_Π : (ℝ ^ X) →ₗ[ℝ] (ℝ ^ X) :=
  (LinearMap.id - proj_Π) ∘ₗ koopman ∘ₗ proj_Π

theorem exact_descent_iff :
  defect_Π = 0 ↔ (∀ f : ℝ ^ X, koopman f ∈ range (proj_Π) ↔ f ∈ range (proj_Π)) :=
by
  constructor
  · intro hDzero f
    have hKPf : koopman (proj_Π f) ∈ range (proj_Π) := by
      -- from D=0, we have (I-P) K P f = 0 → P K P f = K P f
      have := congrArg (· (f)) hDzero
      simpa [defect_Π, koopman, proj_Π] using this
    -- for f itself
    constructor
    · intro hf_range
      -- f in range(P) means f = P f, then K f = K P f ∈ range(P)
      rcases hf_range with ⟨g, hg⟩
      have hg' : proj_Π g = f := hg
      ...
    · intro hKf_range
      ...
  · intro h_inv
    ext f
    -- decompose f = P f + (I-P)f, use invariance
    ...
```

(Full formalisation requires details about the finite‑dimensional basis and the concrete definition of proj_Π; these can be filled in with standard Mathlib lemmas for projections onto invariant subspaces.)

---
