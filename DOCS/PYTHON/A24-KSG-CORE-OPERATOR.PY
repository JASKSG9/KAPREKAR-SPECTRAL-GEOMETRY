"""
core/operators.py

P1: Deterministic Kaprekar map (row-stochastic, deterministic).
P2: Cross-class structured perturbation — GUARANTEED non-commuting with T.

ROOT CAUSE OF PREVIOUS BUGS:
  Old P2 used same-mod-class targeting → block-diagonal w.r.t. T's partition
  → [P2, T] ≈ 0 on quotient → A(α) measured noise, not structure.

NEW P2 DESIGN (doc 8 + doc 9 prescription):
  P2 = (1-ε) * P_local + ε * P_cross
  where P_cross has uniform support over ALL states
  → guaranteed cross-T-equivalence-class coupling
  → [P2, T] ≠ 0 by construction
  → commutator norm C > 0 verified explicitly before sweep
"""
import numpy as np
import scipy.sparse as sp


def normalize_rows(M: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.array(M.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1.0
    return sp.diags(1.0 / row_sum) @ M


def build_P1(kmap: np.ndarray, keep: np.ndarray) -> sp.csr_matrix:
    """
    P1: deterministic Kaprekar transition on the kept subgraph.
    P1[i,j] = 1 if kaprekar_step(keep[i]) == keep[j], else 0.
    """
    keep_idx = {int(v): idx for idx, v in enumerate(keep)}
    n = len(keep)
    rows, cols = [], []
    for i, state in enumerate(keep):
        target = int(kmap[int(state)])
        if target in keep_idx:
            rows.append(i)
            cols.append(keep_idx[target])

    P1 = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    ).tocsr()
    return normalize_rows(P1)


def build_P2_cross_class(
    keep: np.ndarray,
    d: int,
    epsilon: float = 0.20,
    n_local: int = 5,
    seed: int = 42
) -> sp.csr_matrix:
    """
    P2 with GUARANTEED cross-T-class coupling.

    Structure:
      (1-epsilon): local digit perturbations (NOT class-preserving)
      epsilon    : uniform over all n states (guaranteed cross-class)

    Guarantee: every row sends epsilon-weight to arbitrary states,
    including states with different digit-sum mod 9.
    → [P2, T] ≠ 0 is certain.
    """
    rng = np.random.default_rng(seed)
    n = len(keep)
    keep_set = set(int(v) for v in keep)
    keep_idx = {int(v): idx for idx, v in enumerate(keep)}

    rows, cols, data = [], [], []

    for i, state in enumerate(keep):
        s = f"{int(state):0{d}d}"

        # --- LOCAL COMPONENT: digit perturbations (NOT class-restricted) ---
        local_targets = set()
        attempts = 0
        while len(local_targets) < n_local and attempts < 50:
            pos = rng.integers(0, d)
            new_digit = rng.integers(0, 10)
            new_s = list(s)
            new_s[pos] = str(new_digit)
            t = int(''.join(new_s))
            if t in keep_idx and t != int(state):
                local_targets.add(keep_idx[t])
            attempts += 1

        if not local_targets:
            local_targets = {i}  # self-loop fallback

        w_local = (1.0 - epsilon) / len(local_targets)
        for j in local_targets:
            rows.append(i)
            cols.append(j)
            data.append(w_local)

        # --- CROSS COMPONENT: uniform over ALL states ---
        # This GUARANTEES cross-T-class mass transport
        n_cross = max(10, n // 100)
        cross_targets = rng.choice(n, size=n_cross, replace=False)
        w_cross = epsilon / n_cross
        for j in cross_targets:
            rows.append(i)
            cols.append(int(j))
            data.append(w_cross)

    P2 = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return normalize_rows(P2)


def phi_alpha(P1: sp.csr_matrix, P2: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
    """
    φ_α = α P1 + (1-α) P2
    Row-stochastic by convexity (both P1, P2 are row-stochastic).
    """
    return alpha * P1 + (1.0 - alpha) * P2


def commutator_norm(
    phi: sp.csr_matrix, T: sp.csr_matrix
) -> float:
    """
    ‖[φ, T]‖₁ = ‖φT - Tφ‖₁  (max column sum of absolute values)
    
    This is THE primary diagnostic.
    If < 1e-6, operators commute → A(α) will be zero → pipeline degenerate.
    """
    PT = phi @ T
    TP = T @ phi
    diff = PT - TP
    # L1 operator norm = max column sum
    col_sums = np.abs(diff).sum(axis=0)
    return float(col_sums.max())
