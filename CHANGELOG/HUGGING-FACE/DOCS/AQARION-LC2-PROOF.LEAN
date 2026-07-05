import Mathlib.LinearAlgebra.Projection
import Mathlib.Analysis.InnerProductSpace.Basic

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

/-- Projection operator onto a subspace W. -/
structure Projection (W : Submodule ℝ V) where
  toLinear : V →ₗ[ℝ] V
  is_proj : toLinear ∘ₗ toLinear = toLinear
  range_eq : LinearMap.range toLinear = W

def P_Π (W : Submodule ℝ V) : Projection W :=
  orthogonalProjection W

/-- Koopman operator K : V → V (linear endomorphism). -/
def K := (V →ₗ[ℝ] V)

/-- Defect operator D = (I - P) K P. -/
def D (P : Projection W) (K : V →ₗ[ℝ] V) : V →ₗ[ℝ] V :=
  (LinearMap.id - P.toLinear) ∘ₗ K ∘ₗ P.toLinear

theorem defect_zero_iff_invariant (P : Projection W) (K : V →ₗ[ℝ] V) :
  D P K = 0 ↔ K ∘ₗ P.toLinear = P.toLinear ∘ₗ K ∘ₗ P.toLinear := by
  unfold D
  rw [LinearMap.comp_assoc, sub_comp, comp_sub, sub_eq_zero]
  exact LinearMap.eq_iff_comp_eq (by rfl)

theorem invariant_iff_range_invariant (P : Projection W) (K : V →ₗ[ℝ] V) :
  K ∘ₗ P.toLinear = P.toLinear ∘ₗ K ∘ₗ P.toLinear ↔
    ∀ x ∈ W, K x ∈ W := by
  constructor
  · intro h x hx
    have : x = P.toLinear x := by
      rw [← P.range_eq]
      exact hx
    rw [this, ← LinearMap.comp_apply, h, LinearMap.comp_apply]
    exact Submodule.mem_range_self _ (K x)
  · intro h
    ext x
    have : P.toLinear x ∈ W := Submodule.mem_range_self _ x
    have : K (P.toLinear x) ∈ W := h _ this
    rw [LinearMap.comp_apply, LinearMap.comp_apply]
    exact (P.range_eq ▸ (P.is_proj ▸ id_apply _) : P.toLinear (K (P.toLinear x)) = K (P.toLinear x)).symm

theorem certification_theorem (P : Projection W) (K : V →ₗ[ℝ] V) :
  D P K = 0 ↔ ∀ x ∈ W, K x ∈ W := by
  rw [defect_zero_iff_invariant, invariant_iff_range_invariant P K]
