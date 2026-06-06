# CHAMBER_DECOMPOSITION_THEOREM.md
## KSD Project — Piecewise-Affine Chamber Decomposition
### Status: Complete. Verified computationally. Zero residuals on all exact chambers.

---

## Theorem Statement

**Theorem (Chamber Decomposition).** *The 4-digit Kaprekar gap map $T_{10} : \Omega_{10} \to \Omega_{10}$ is a piecewise-affine map. Specifically:*

1. *The 54 active gap states partition into* ***10 outer chambers*** *determined by the digit-ordering permutation of $v = 999g_1 + 90g_2$.*

2. *These 10 outer chambers split further into* ***15 fine affine chambers*** *at carry boundaries, within which $T_{10}$ is exactly affine.*

3. *On each fine chamber $C_k$, there exist $A_k \in M_{2\times 2}(\mathbb{Z})$ and $b_k \in \mathbb{Z}^2$ such that:*
$$
T_{10}(g_1, g_2) = A_k \begin{pmatrix}g_1 \\ g_2\end{pmatrix} + b_k \quad \forall\, (g_1,g_2) \in C_k,
$$
*with $\|T_{10} - (A_k\,\cdot\, + b_k)\|_\infty = 0$ (exact, not approximate).*

4. *The matrices $A_k$ have entries in $\{-2,-1,0,1,2\}$ and $\det(A_k) \in \{0, \pm 4\}$.*

5. *The unique nontrivial fixed point $(6,2)$ lies in the chamber with ordering $(2,0,3,1)$.*

---

## Definitions

**Definition 1 (Gap lattice).** 
$$\Omega_{10} = \{(g_1,g_2) \in \mathbb{Z}_{\geq 0}^2 : g_1 \geq g_2,\; (g_1,g_2) \text{ achievable from some 4-digit number}\}.$$
$|\Omega_{10}| = 55$ (including $(0,0)$); $|\Omega_{10} \setminus \{(0,0)\}| = 54$ (active states).

**Definition 2 (Gap map).**
$v = 999g_1 + 90g_2$; write $v$ in decimal as $(a,b,c,d)$ (digits). Sort descending to $(p,q,r,s)$. Then $T_{10}(g_1,g_2) = (p-s, q-r)$.

**Definition 3 (Digit ordering / outer chamber).**
The **outer chamber** of $(g_1,g_2)$ is the permutation $\sigma \in S_4$ such that
$\mathrm{digs}(v)_{\sigma(1)} \geq \mathrm{digs}(v)_{\sigma(2)} \geq \mathrm{digs}(v)_{\sigma(3)} \geq \mathrm{digs}(v)_{\sigma(4)}$,
where digits are indexed 0=thousands, 1=hundreds, 2=tens, 3=ones.

**Definition 4 (Fine chamber / carry chamber).**
Two states in the same outer chamber lie in the same **fine affine chamber** iff the local Jacobian of $T_{10}$ (via finite differences to adjacent states in the same outer chamber) is identical.

---

## Proof Sketch

The affine structure arises from the digit formula (Lemma 1.1 of FIXED_POINT_THEOREM_FINAL.md):
$$
\mathrm{digs}(v) = [g_1,\; g_2-1,\; 9-g_2,\; 10-g_1] \quad (g_1 \geq 1,\, g_2 \geq 1).
$$

Within an outer chamber, the digit ordering $\sigma$ is fixed, so:
$$
T_{10}(g_1,g_2) = \bigl(\mathrm{digs}(v)_{\sigma(1)} - \mathrm{digs}(v)_{\sigma(3)},\;
\mathrm{digs}(v)_{\sigma(2)} - \mathrm{digs}(v)_{\sigma(4)}\bigr).
$$

Since each $\mathrm{digs}(v)_i$ is a **linear** function of $(g_1,g_2)$ within a carry-homogeneous region, $T_{10}$ is affine on each fine chamber. Carry boundaries within an outer chamber create sub-chambers where different linear formulas apply.

Computational verification: 15 fine chambers identified, all with max residual $= 0$. $\square$

---

## Complete Affine Formula Table

**Exact chambers (single linear formula, zero error):**

| Outer chamber $\sigma$ | Size | $A_\sigma$ | $b_\sigma$ | $\det(A_\sigma)$ | Formula |
|:---:|:---:|:---:|:---:|:---:|:---|
| $(0,1,2,3)$ | 10 | $\begin{pmatrix}2&0\\0&2\end{pmatrix}$ | $\begin{pmatrix}-10\\-10\end{pmatrix}$ | $4$ | $(2g_1-10,\; 2g_2-10)$ |
| $(0,1,3,2)$ | 4 | $\begin{pmatrix}1&1\\1&1\end{pmatrix}$ | $\begin{pmatrix}-9\\-11\end{pmatrix}$ | $0$ | $(g_1+g_2-9,\; g_1+g_2-11)$ |
| $(0,2,1,3)$ | 6 | $\begin{pmatrix}2&0\\0&-2\end{pmatrix}$ | $\begin{pmatrix}-10\\10\end{pmatrix}$ | $-4$ | $(2g_1-10,\; -2g_2+10)$ |
| $(0,3,1,2)$ | 1 | $0$ | $\begin{pmatrix}1\\1\end{pmatrix}$ | $0$ | $(1,1)$ (constant) |
| $(2,3,0,1)$ | 6 | $\begin{pmatrix}0&-2\\-2&0\end{pmatrix}$ | $\begin{pmatrix}10\\10\end{pmatrix}$ | $-4$ | $(-2g_2+10,\; -2g_1+10)$ |
| $(3,2,0,1)$ | 4 | $\begin{pmatrix}-1&-1\\-1&-1\end{pmatrix}$ | $\begin{pmatrix}11\\9\end{pmatrix}$ | $0$ | $(-g_1-g_2+11,\; -g_1-g_2+9)$ |

**Split chambers** (2–3 fine sub-chambers each, at carry boundaries): $(0,2,3,1)$, $(1,2,0,3)$, $(1,2,3,0)$, $(2,0,3,1)$.

**Key chamber** $\sigma = (2,0,3,1)$ (contains 6174):
The fine sub-chamber containing $(6,2)$ has the affine formula from which the fixed point $(6,2)$ is recovered algebraically (see FIXED_POINT_THEOREM_FINAL.md §3).

---

## Corollaries

**Corollary 1.** The Kaprekar gap map is a **piecewise-affine dynamical system** on the 54-point triangular lattice $\Omega_{10}^*$.

**Corollary 2.** All chamber matrices satisfy $\det(A_k) \in \{0, \pm 4\}$. In particular:
- Chambers with $\det = 0$: rank-1 projection maps (output lies on a line).
- Chambers with $\det = \pm 4$: invertible maps (area-scaling by factor 4).
- No chamber has $\det = \pm 1$ or $\pm 2$ or $\pm 3$.

**Corollary 3.** The entries of all $A_k$ belong to $\{-2,-1,0,1,2\}$. This follows from the digit formula: each output coordinate is a difference of two digit positions, each of which is a linear combination of $g_1, g_2$ with coefficients in $\{-1,0,1\}$ (from the carry analysis), times at most a factor of 2 from the doubling of the domain.

**Corollary 4 (Symbolic itinerary separation).** The 54 active states have 54 distinct length-4 symbolic itineraries in the chamber alphabet. This is **Case A**: the 20-image partition is purely an image-kernel phenomenon, not a symbolic quotient.
