# AQARION — CLAIM RECONCILIATION REGISTER

**Document ID:** AQ-CLAIM-RECON-2026-09-17  
**Date:** 2026-09-17  
**Status:** ACTIVE  
**Scope:** Repository-wide stale/contradictory claim cleanup

---

## PURPOSE

This document identifies repository material that must be corrected,
quarantined, or explicitly marked historical.

It is NOT a new mathematical theory.

It is a repository integrity artifact.

---

# CRITICAL-001 — SOURCE STATE CARDINALITY

### Conflicting claims

Historical/current repository material contains:

- 8991
- 8992
- 9990

### Canonical current definition

\[
X=\{1000,\ldots,9999\}\setminus
\{1111,\ldots,9999\}
\]

so

\[
|X|=8991.
\]

### Action

Replace current-domain claims of 8992 or 9990 with 8991.

If an alternate domain is intentionally retained, define it explicitly
before giving its cardinality.

### Affected known files

- `MAIN-API.MD`
- `INVARIANTS.MD`
- `CHECKPOINTS/JUNE6-CHECKPOINT.MD`
- `CHECKPOINTS/JUNE19-CHECKPOINT.MD`
- `DOCS/PYTHON/JUNE6-GROUNDTRUTH.PY`
- `CHANGELOG/HUGGING-FACE/AQARION-SOS/MARKDOWNS/SUPPORT-FLOW.TXT`
- related historical documents

### Status

`OPEN — REPOSITORY-WIDE TEXT RECONCILIATION`

---

# CRITICAL-002 — 54-STATE VS 55-STATE

### Canonical

54-state:

`Kaprekar gap quotient`

### Historical/auxiliary

55-state:

`FOQDS / auxiliary construction`

### Action

Do NOT globally delete 55-state material.

Do explicitly label its mathematical role.

### Affected known files

- `INVARIANTS.MD`
- `THEOREMS-REGISTRY.MD`
- `CHECKPOINTS/MAY24-CHECKPOINT.MD`
- `CHECKPOINTS/MAY26-CHECKPOINT.MD`
- `JUNE13-GROUNDTRUTH.MD`
- `CHECKPOINTS/JUNE6-CHECKPOINT.MD`
- `CHECKPOINTS/JULY11-CHECKPOINT.MD`

### Status

`OPEN — TERMINOLOGY RECONCILIATION`

---

# CRITICAL-003 — IMAGE CHAIN

### Canonical current chain

\[
54\to20\to14\to10\to7\to4\to1
\]

### Superseded chain

\[
54\to20\to14\to10\to7\to4\to2\to1
\]

### Action

The superseded chain must not appear in current-status documents.

### Status

`REPLACE WHERE PRESENT`

---

# CRITICAL-004 — CHAMBER COUNT

### Known conflict

Historical repository material contains claims of:

- 10 chambers
- 12 chambers
- 11 stable chambers in other historical material

The API document explicitly records the 10-vs-12 discrepancy as unresolved.

### Current classification

\[
\boxed{\text{OPEN}}
\]

### Action

Do not promote 10, 11, or 12 as the definitive chamber count.

### Status

`QUARANTINED`

---

# CRITICAL-005 — PUBLICATION-READY LANGUAGE

### Problem

Multiple files contain:

`PUBLICATION-READY`

`SUBMISSION-READY`

or equivalent global completion language.

### Action

Replace current global status with:

`RESEARCH ACTIVE — CLAIM-SPECIFIC EVIDENCE RECONCILIATION IN PROGRESS`

Historical checkpoints may retain their historical language only if
clearly dated and labelled historical.

### Status

`REPLACE CURRENT STATUS LANGUAGE`

---

# CRITICAL-006 — OLD EVIDENCE TAXONOMY

### Problem

Older files use only:

- P
- CV
- P+CV
- O

### Current taxonomy

Use the canonical evidence tags defined in:

`AQARION-TRUTH-BASELINE-2026-09-17.md`

### Action

Do not silently reinterpret old labels.

Map them explicitly or mark them historical.

### Status

`OPEN`

---

# CRITICAL-007 — LEAN STATUS

### Current policy

Lean is optional formalization infrastructure.

### Required distinction

`[F]` means actual formal checker receipt.

No Lean source file alone establishes `[F]`.

Absence of Lean does not block computational or analytic evidence.

### Status

`POLICY ACTIVE`

---

# CRITICAL-008 — GENERIC SUBMODULARITY CLAIMS

Any generic theorem claiming that arbitrary finite closure systems
automatically produce the desired submodularity law must be treated as:

`[Q]` or `[REFUTED]`

until the specific closure construction and proof are independently
established.

The previous generic proof route is not to be reused as established
mathematics.

---

# CRITICAL-009 — SEMIGROUP CLAIMS

The 7-element monogenic semigroup claim may remain as a claim attached
to the canonical 54-state quotient only when its actual convention and
power indexing are specified.

Do not infer other quotient/FOQDS semigroup properties from it.

---

# CRITICAL-010 — HISTORICAL CHECKPOINTS

Files named:

`CHECKPOINTS/*`

are historical research records.

They are not automatically current specifications.

A historical checkpoint may remain unchanged.

The current truth state must instead be maintained by:

`AQARION-TRUTH-BASELINE-2026-09-17.md`

and the current claim registry.

---

# REPOSITORY CLEANUP ORDER

1. Truth baseline
2. Claim registry
3. Theorem registry
4. Main API
5. README
6. current checkpoint
7. historical documents
8. generated/demo material

Do not rewrite historical checkpoints merely to make history look current.

---

# FINAL STATE

This register is itself a governance artifact.

It does not certify the mathematical claims listed above.

It records what must be reconciled before those claims are represented
as current repository truth.
