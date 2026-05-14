#!/usr/bin/env python3
"""
OQTS ENGINE v2 — Fully Unified Operator System

Single abstraction:

    X_{t+1} = T(X_t)
    Q(X_t)   = quotient state
    Φ(Q(X_t)) = observables

Everything (Kaprekar, SPATS, ORSM) is a configuration of:
- state space
- transition operator T
- quotient map Q
- observables Φ
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
from dataclasses import dataclass


# ============================================================
# CORE RESULT STRUCTURE
# ============================================================

@dataclass
class OQTSResult:
    mode: str
    trajectory: list
    quotient: list
    observables: dict


# ============================================================
# UNIVERSAL ENGINE
# ============================================================

class OQTS:
    def __init__(self, T, Q, Phi):
        """
        T   : state evolution function
        Q   : quotient map
        Phi : list of observables acting on quotient space
        """
        self.T = T
        self.Q = Q
        self.Phi = Phi

    def run(self, x0, steps=10):
        traj = [x0]
        for _ in range(steps):
            x0 = self.T(x0)
            traj.append(x0)

        qtraj = [self.Q(x) for x in traj]

        obs = {i: [] for i in range(len(self.Phi))}
        for q in qtraj:
            for i, phi in enumerate(self.Phi):
                obs[i].append(phi(q))

        return traj, qtraj, obs


# ============================================================
# KAPREKAR (DISCRETE QUOTIENT DYNAMICS)
# ============================================================

def kaprekar_T(n):
    s = str(n).zfill(4)
    a = int("".join(sorted(s, reverse=True)))
    b = int("".join(sorted(s)))
    return a - b


def kaprekar_Q(n):
    return tuple(sorted(str(n).zfill(4)))


def kaprekar_Phi():
    def entropy(q):
        digits = np.array([int(c) for c in q])
        p = np.bincount(digits, minlength=10) / len(digits)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return [entropy]


def run_kaprekar(seed=6174, steps=15):
    engine = OQTS(kaprekar_T, kaprekar_Q, kaprekar_Phi())
    traj, qtraj, obs = engine.run(seed, steps)
    return OQTSResult("kaprekar", traj, qtraj, obs)


# ============================================================
# SPATS (GEOMETRIC FIELD SYSTEM)
# ============================================================

def spats_T(x):
    from scipy.signal import convolve2d
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return convolve2d(x, kernel, mode="same", boundary="wrap")


def spats_Q(x):
    # symmetry quotient = spectral signature
    f = np.fft.fft2(x)
    return np.sort(np.round(f.real.flatten(), 4))


def spats_Phi():
    def energy(q):
        return np.linalg.norm(q)
    return [energy]


def run_spats(n=16, steps=8):
    x0 = np.random.randn(n, n)

    engine = OQTS(spats_T, spats_Q, spats_Phi())
    traj, qtraj, obs = engine.run(x0, steps)

    return OQTSResult("spats", traj, qtraj, obs)


# ============================================================
# ORSM (NON-NORMAL OPERATOR SYSTEM)
# ============================================================

def build_L(n=20):
    j = np.arange(n)
    H = np.diag(np.cos(j))
    for i in range(n - 1):
        H[i, i+1] = H[i+1, i] = -1
    return H + 1j * np.diag(np.linspace(-1, 1, n))


def orsm_Q(v):
    v = np.array(v)
    return v / (np.linalg.norm(v) + 1e-12)


def orsm_Phi():
    def norm(q):
        return np.linalg.norm(q)
    return [norm]


def run_orsm(n=20, steps=10):
    L = build_L(n)
    v0 = np.random.randn(n) + 1j * np.random.randn(n)
    v0 /= np.linalg.norm(v0)

    def T(v):
        return expm_multiply(L, v)

    engine = OQTS(T, orsm_Q, orsm_Phi())
    traj, qtraj, obs = engine.run(v0, steps)

    return OQTSResult("orsm", traj, qtraj, obs)


# ============================================================
# UNIFIED DRIVER
# ============================================================

def run_all():
    return {
        "kaprekar": run_kaprekar(),
        "spats": run_spats(),
        "orsm": run_orsm()
    }


# ============================================================
# CROSS-SYSTEM INVARIANT COLLAPSE
# ============================================================

def collapse_invariants(results):
    """
    Single scalar invariant:
    measures dispersion of observable pullbacks across all systems
    """

    all_vals = []

    for r in results.values():
        for stream in r.observables.values():
            all_vals.extend(stream)

    all_vals = np.array(all_vals, dtype=float)

    return {
        "OQTS_global_dispersion": float(np.std(all_vals)),
        "OQTS_normalized_entropy": float(
            -np.sum((np.abs(all_vals) + 1e-12) /
                    np.sum(np.abs(all_vals)) *
                    np.log((np.abs(all_vals) + 1e-12) /
                    np.sum(np.abs(all_vals))))
        )
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = run_all()
    invariant = collapse_invariants(results)

    print("\n=== OQTS ENGINE v2 ===")
    print(invariant)

    for k, r in results.items():
        print(f"\n[{k.upper()}]")
        print("trajectory:", len(r.trajectory))
        print("quotient steps:", len(r.quotient))
        print("observable streams:", {i: len(v) for i, v in r.observables.items()})
