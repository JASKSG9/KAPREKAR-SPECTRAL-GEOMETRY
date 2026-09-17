# AQARION — TRUTH BASELINE

**Document ID:** AQ-TRUTH-BASELINE-2026-09-17  
**Date:** 2026-09-17  
**Repository:** `JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY`  
**Status:** ACTIVE SOURCE OF TRUTH  
**Purpose:** Current claim-state and definition baseline  
**ARCH-369:** WORK IN PROGRESS — NOT A COMPLETION TARGET

---

## 1. GOVERNANCE RULE

This document defines the current repository truth state.

Older checkpoints, READMEs, theorem registries, API documents, demos,
and generated summaries may contain historical results.

Historical material is not automatically current merely because it remains
in the repository.

When an older document conflicts with this baseline, the older statement
must be treated as:

`HISTORICAL / SUPERSEDED`

until explicitly reconciled.

No historical document may promote a claim by itself.

---

# 2. CANONICAL FOUR-DIGIT KAPREKAR DOMAIN

The canonical four-digit research domain is:

\[
X =
\{1000,\ldots,9999\}
\setminus
\{1111,2222,\ldots,9999\}.
\]

Therefore:

\[
|X| = 9000 - 9 = \boxed{8991}.
\]

Canonical source-state cardinality:

**8991**

Not:

- 8992
- 9990

unless an alternate state domain is explicitly defined.

---

# 3. CANONICAL GAP OBSERVABLE

For a four-digit number with sorted digits

\[
a\ge b\ge c\ge d,
\]

define

\[
\pi(n)=(a-d,b-c).
\]

The canonical gap-state set has:

\[
\boxed{|G|=54}.
\]

The induced transition is denoted

\[
T_G:G\to G.
\]

The canonical semiconjugacy statement is:

\[
\boxed{\pi\circ K=T_G\circ\pi}.
\]

This is the canonical Kaprekar gap quotient.

---

# 4. 54-STATE VS 55-STATE OBJECTS

These objects MUST NOT be conflated.

## 4.1 Canonical 54-state object

The 54-state object is:

**Kaprekar gap quotient**

\[
G=\pi(X).
\]

It is the canonical object for the current Kaprekar arithmetic baseline.

## 4.2 55-state object

A 55-state construction may occur in historical FOQDS,
trace-equivalence, or related quotient work.

It must be labelled explicitly:

**55-state FOQDS / auxiliary construction**

It must not be described as the canonical 54-state gap quotient.

Historical documents containing the 55-state construction remain useful
as research history, but their terminology must be explicit.

---

# 5. DEPTH CONVENTION

For the canonical gap quotient:

\[
\nu_{\mathrm{full}}=7
\]

and

\[
\nu_{\mathrm{quotient}}=6.
\]

The current image chain is:

\[
\boxed{
54\rightarrow20\rightarrow14\rightarrow10\rightarrow7
\rightarrow4\rightarrow1
}
\]

where the terminal state is the attracting fixed state.

Any document using

\[
54\rightarrow20\rightarrow14\rightarrow10\rightarrow7
\rightarrow4\rightarrow2\rightarrow1
\]

must be treated as superseded until independently reconciled.

---

# 6. ATTRACTOR

The canonical four-digit Kaprekar attractor is:

\[
\boxed{6174}.
\]

The corresponding gap state is:

\[
\boxed{(6,2)}.
\]

---

# 7. EVIDENCE TAXONOMY

The current evidence vocabulary is:

| Tag | Meaning |
|---|---|
| `[D]` | Defined |
| `[P]` | Analytically proved |
| `[V]` | Computationally verified |
| `[IV]` | Independently verified |
| `[CV]` | Computationally validated / checked |
| `[PV]` | Proved + independently computationally validated |
| `[F]` | Formally checked by a proof assistant |
| `[CE]` | Counterexample established |
| `[REFUTED]` | Previous claim disproved |
| `[OPEN]` | Unresolved |
| `[CONJ]` | Conjecture |
| `[H]` | Historical / superseded |
| `[Q]` | Quarantined pending reconciliation |

A claim must never receive `[F]` without an actual formal checker receipt.

Absence of Lean does not imply `[BLOCKED]`.

---

# 8. OPERATOR DEFECT

For a partition projection \(P_\Pi\):

\[
\boxed{
D_\Pi=(I-P_\Pi)KP_\Pi
}
\]

is the canonical defect.

The retired expression

\[
(I-P)KP(I-P)
\]

must not be used as the canonical defect.

For the canonical defect:

\[
D_\Pi=0
\]

certifies forward invariance of the observable subspace:

\[
K(V_\Pi)\subseteq V_\Pi.
\]

It does NOT by itself imply:

\[
KP_\Pi=P_\Pi K.
\]

Therefore:

`D=0 => commutation`

is NOT a valid general rule.

---

# 9. CURRENT KAPREKAR STATUS

## Established baseline

- `[D]` four-digit Kaprekar map
- `[D]` canonical non-repdigit domain
- `[V/IV]` 8991-domain computational results where independently reproduced
- `[D]` gap observable
- `[V/IV]` 54-state quotient structure
- `[V/IV]` exact semiconjugacy on the canonical quotient
- `[V/IV]` attractor 6174
- `[V/IV]` maximum full depth 7
- `[V/IV]` quotient depth 6
- `[V/IV]` image chain 54→20→14→10→7→4→1

These labels must be backed by the corresponding receipts/artifacts
before being represented as stronger evidence classes.

---

# 10. CHAMBER STATUS

The chamber count is NOT currently frozen.

The repository contains conflicting historical claims including
10 and 12 chambers.

Therefore:

\[
\boxed{\text{CHAMBER COUNT = OPEN}}
\]

until the discrepancy is independently reconciled.

The following language is prohibited as current truth:

- "exactly 10 chambers"
- "definitively characterized 10 chambers"
- "10-chamber theorem"
- "frozen 10-chamber atlas"

unless the unresolved discrepancy has been closed.

Historical 10-chamber material may remain under `[H]` or `[Q]`.

---

# 11. PUBLICATION STATUS

The repository must not globally describe itself as:

`PUBLICATION-READY`

or:

`SUBMISSION-READY`

as a mathematical status.

Use:

`RESEARCH ACTIVE — CLAIM-SPECIFIC EVIDENCE RECONCILIATION IN PROGRESS`

instead.

Publication decisions are claim-specific and depend on the actual
evidence required for the claims being made.

---

# 12. ARCH-369

ARCH-369 is:

`WORK IN PROGRESS`

It is not a publication-completion milestone.

It must not be used as evidence that the surrounding repository is
complete.

---

# 13. CURRENT PRIORITY

The immediate repository task is:

1. reconcile stale claims;
2. remove contradictory status language;
3. separate historical from current mathematics;
4. reconcile 8991/8992/9990 terminology;
5. separate 54-state and 55-state constructions;
6. quarantine unresolved chamber claims;
7. update theorem/claim registries;
8. update README/API status language;
9. only then resume new construction work.

No new architecture is required to accomplish this cleanup.

---

# 14. PROHIBITED CLAIMS WITHOUT RECONCILIATION

The following must not appear as current verified facts:

- `|Ω| = 9990`
- `|Ω| = 8992`
- canonical quotient = 55 states
- 10 chambers are definitively established
- 12 chambers are definitively established
- generic closure/submodularity theorem is proved
- every old checkpoint is current
- repository is globally publication-ready
- Lean verification exists merely because Lean source exists

---

# 15. STATUS

**AQARION TRUTH BASELINE: ACTIVE**

**ARCH-369: WIP**

**NEW BUILD EXPANSION: PAUSED**

**REPOSITORY RECONCILIATION: PRIORITY**
