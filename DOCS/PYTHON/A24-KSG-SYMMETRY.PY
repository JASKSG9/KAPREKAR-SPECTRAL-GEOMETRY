"""
core/symmetry.py

T is a LINEAR PERMUTATION on Ω — digit-reversal.
  T(1234) = 4321, T(0612) = 2160, etc.

Key properties verified:
  T² = I  (involution)
  T is a permutation matrix (orthogonal)
  T does NOT reduce to bucket projection → no dimension collapse bug

PREVIOUS BUG: symmetry.py used bucket projection collapsing 9990-vector
to 9 buckets, causing shape mismatch when composing with phi.
This version keeps T on the FULL state space.
"""
import numpy as np
import scipy.sparse as sp


def build_T_permutation(keep: np.ndarray, d: int) -> sp.csr_matrix:
    """
    Construct T as a sparse permutation matrix on the subgraph.
    
    T maps state i → reverse_digits(i).
    If reverse_digits(i) is not in the kept subgraph, T maps to self (best
    available approximation — flagged in commutator diagnostics).
    
    Returns:
        T_mat: n×n sparse permutation matrix
        n_self: number of self-maps (ideal = 0 for clean T)
    """
    keep_set = set(int(v) for v in keep)
    keep_idx = {int(v): idx for idx, v in enumerate(keep)}
    n = len(keep)

    rows, cols = [], []
    n_cross = 0
    n_self  = 0

    for i, state in enumerate(keep):
        s = f"{int(state):0{d}d}"
        rev_state = int(s[::-1])
        if rev_state in keep_idx:
            j = keep_idx[rev_state]
            n_cross += 1
        else:
            j = i   # self-map fallback
            n_self += 1
        rows.append(i)
        cols.append(j)

    T_mat = sp.coo_matrix(
        (np.ones(n), (rows, cols)), shape=(n, n)
    ).tocsr()

    return T_mat, n_self


def verify_involution(T_mat: sp.csr_matrix, tol: float = 1e-10) -> bool:
    """Check T² ≈ I."""
    T2 = T_mat @ T_mat
    diff = T2 - sp.eye(T_mat.shape[0])
    return float(abs(diff).max()) < tol
