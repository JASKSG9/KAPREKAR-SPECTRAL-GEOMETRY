# GLOBAL_CONVERGENCE_THEOREM.md
## KSD Project — Global Convergence of the Kaprekar Gap Map
### Status: Complete.

---

## Theorem Statement

**Theorem (Global Convergence).** *Let $T_{10} : \Omega_{10} \to \Omega_{10}$ be the 4-digit Kaprekar gap map. Then:*

1. *(Fixed points.)* The only fixed points are $(0,0)$ (trivial) and $(6,2)$ (corresponding to 6174).

2. *(Unique nontrivial attractor.)* $(6,2)$ is the unique nontrivial periodic orbit. It is a fixed point (period 1), not a cycle.

3. *(Global convergence of active states.)* Every active state $(g_1,g_2) \in \Omega_{10} \setminus \{(0,0)\}$ satisfies $T_{10}^n(g_1,g_2) = (6,2)$ for all sufficiently large $n$.

4. *(Orbit length bound.)* Every active state reaches $(6,2)$ within 7 iterations.

5. *(Repdigit basin.)* The state $(0,0)$ corresponds to repdigit numbers (all digits equal). Every 4-digit number with a non-constant digit multiset eventually reaches 6174.

---

## Proof

**Part 1** follows from FIXED_POINT_THEOREM_FINAL.md (Theorem 6.1): exhaustive chamber exclusion shows exactly two fixed points.

**Part 2:** The orbit structure computation on all 54 active states identifies the unique cycle node set as $\{(6,2)\}$ — a single fixed point, not a longer cycle.

**Part 3:** Verified computationally: for every $(g_1,g_2) \in \Omega_{10}^*$, the orbit sequence $T_{10}^n(g_1,g_2)$ reaches $(6,2)$ for some $n \leq 7$.

**Part 4:** The maximum orbit length is 7 (states at depth 7 in Domain B). Direct verification:
- The depth distribution $N_\tau = (1, 383, 576, 2400, 1272, 1518, 1656, 2184)$ shows all 9990 Domain B states reach 6174 within 7 steps.
- In the gap quotient, the same bound holds by projection.

**Part 5:** Repdigit numbers have $g_1 = g_2 = 0$, hence $(0,0)$, which is the trivial fixed point. All other 4-digit numbers (with non-constant digit multiset) satisfy $g_1 + g_2 \geq 1$, hence are active states, and converge to 6174. $\square$

---

## Corollary: The Semiconjugacy

The gap map $T_{10}$ is a **semiconjugate** of the Kaprekar map $K_{10}$ via the quotient map $Q: B \to \Omega_{10}^*$:
$$
Q \circ K_{10} = T_{10} \circ Q.
$$

Global convergence of $T_{10}$ to $(6,2)$ therefore implies global convergence of $K_{10}$ to 6174 (in Domain B). $\square$

---

## Spectral Interpretation

The spectral gap $\mu_1 = 0.01140$ of the gap functional graph governs the mixing rate of an associated random walk. A smaller spectral gap corresponds to slower mixing; the small value reflects the near-path topology of the gap graph (mean degree 1.98).

However, the **deterministic** convergence (not random walk) is guaranteed in $\leq 7$ steps regardless of the spectral gap, because the map is a finite deterministic dynamical system with a global attractor.
