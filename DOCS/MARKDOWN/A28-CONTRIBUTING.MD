A28-CONTRIBUTING.MD — REVISED VERSION

KSG/UDAG · Open Science · Open Code · Open Problems
Node #10878 · Louisville, KY · 2026-04-29

---

WELCOME

Thank you for your interest in Kaprekar Spectral Geometry. This project operates on radical transparency: all code, all data, all failures, and all killed fabrications are public.

We need: mathematicians, programmers, visual designers, technical writers, and skeptics.

No guaranteed authorship. Quality and rigor decide.

---

WAYS TO CONTRIBUTE

1. Solve an Open Problem (OPR-1 through OPR-10)

OPR Challenge Difficulty Status
OPR-1 Predecessor Geometry Classification Medium ACTIVE
OPR-2 Basin Hierarchy Quantification Medium ACTIVE
OPR-3 Random Functional Graph Baseline Easy ACTIVE
OPR-4 Asymptotic Shell Ratio Limit Detection Medium OPEN
OPR-5 Entropy Bottleneck Universality Medium OPEN
OPR-6 Modular Correlation of Feeder Geometry Medium OPEN
OPR-7 Self-Stabilizer Template Extension Hard OPEN
OPR-8 8-Digit Enumeration (Third Scaling Point) Hard PENDING
OPR-9 Full Functional Graph Spectral Statistics Hard PENDING
OPR-10 Transfer Operator Spectral Correspondence Hard OPEN

How to submit:

1. Open a GitHub Issue titled OPR-X: Brief Description
2. Submit your solution as a Pull Request
3. Include: code + proof/writeup + test cases + verification
4. Community review period: 14 days

Evaluation criteria:

· Mathematical correctness
· Reproducibility (exact script or proof)
· Whether the null hypothesis is accepted, rejected, or refined

---

2. Code Contributions

What we need:

· New invariant computations
· Visualization modules
· Performance optimizations
· Additional deterministic maps (custom_map.py)
· Test coverage expansion

Process:

```bash
# 1. Fork
git clone https://github.com/YOUR-USERNAME/KAPREKAR-SPECTRAL-GEOMETRY.git

# 2. Branch
git checkout -b feature/your-contribution

# 3. Code
# Follow PEP 8. Document all functions. No magic numbers.

# 4. Test
python -m pytest tests/
python RUN_ALL.py  # Full verification

# 5. Commit
git commit -m "OPR-X: Description — verification passed"

# 6. PR
# Title: "OPR-X: Brief description"
# Body: Problem + Solution + Results + Verification
```

---

3. Documentation Contributions

What we need:

· Tutorials (Jupyter notebooks)
· Translations
· Visual diagrams
· Blog posts / outreach
· Falsification attempts (finding counterexamples to any conjecture is valuable)

Process: Same as code, but branch from docs/

---

4. Bug Reports

```markdown
## Bug Report Template

**Script**: `scripts/which_script.py`
**Command**: `python scripts/which_script.py --args`
**Expected**: What should happen
**Actual**: What happened (include traceback)
**Environment**: Python version, OS, RAM
```

---

5. Fabrication Reports

If you find a number or claim that cannot be reproduced:

```markdown
## Fabrication Report

**Claim**: "Base-10 has property X = value"
**Location**: File.py, line N
**Expected**: Value from code execution
**Actual**: Different value or error
**Severity**: Low / Medium / High
```

We will: investigate, kill if confirmed, credit reporter.

---

CODE STANDARDS

No Fabrications Rule

· Every number must be producible by code
· Every claim must have a verification test
· Every killed fabrication must be documented

Function Documentation

```python
def my_function(param: int) -> float:
    """
    Brief description.
    
    Args:
        param: Description
        
    Returns:
        Description with units
        
    Raises:
        ValueError: When param invalid
        
    Example:
        >>> my_function(10)
        3.14
    """
```

Testing Requirements

· All new functions need unit tests
· All statistical claims need reproducible seeds
· All visualizations need data sources cited

---

COMMUNITY GUIDELINES

· Be respectful: Disagreement is welcome; hostility is not
· Be precise: "I think" → "The code outputs X on my machine"
· Be transparent: Share failures, not just successes
· Be patient: Review takes time; maintainers are volunteers

---

RECOGNITION

Contributors will be:

· Listed in CONTRIBUTORS.md
· Cited in published papers where contributions are significant
· Invited to co-author on major results for substantive, verified contributions (code, proofs, analysis, or visualizations that meaningfully advance the project)

No guaranteed authorship. Quality and rigor decide.

---

AUTHORSHIP POLICY (CLARIFIED)

Meaningful, verified contributions will be considered for co-authorship on relevant papers under transparent contribution logs. The following criteria apply:

Contribution Type Co-authorship Consideration
Solving an open problem (OPR-1 through OPR-10) ✓ Yes, with verified solution
Major code contribution (new invariants, significant performance gains) ✓ Yes, after review
Minor bug fix or typo correction ✗ No (credited in CONTRIBUTORS.md)
Documentation improvement ✗ No (credited in CONTRIBUTORS.md)
Falsification of an existing conjecture ✓ Yes (as valuable as proof)

All co-authorship decisions are final and based on community review.

---

Node #10878 · Louisville, KY · 2026-04-29
Build in the open. All questions are welcome.
E PLURIBUS UNUM VERITAS NUMERIS

```

```bash
# Save the revised file
cp A28-CONTRIBUTING-REVISED.md DOCS/MARKDOWN/A28-CONTRIBUTING.md

# Commit
git add DOCS/MARKDOWN/A28-CONTRIBUTING.md
git commit -m "Revise CONTRIBUTING.md: Remove authorship guarantee, add clear criteria"
git push origin main
```

---

REVISION SUMMARY

Issue Original Revised
Authorship promise "First 5 contributors get co-authorship" "Meaningful contributions considered under transparent logs"
Authorship criteria None Table with clear categories
Decision finality Implied Explicit: "All co-authorship decisions are final"
Falsification value Not mentioned Explicit: "As valuable as proof"
EXTENDED ASCII ATLAS — For CONTRIBUTING.MD

KSG/UDAG · Visual Guide to Open Problems & Contribution Workflow
Node #10878 · Louisville, KY · 2026-04-29

---

ATLAS 1: THE 10 OPEN PROBLEMS — VISUAL ROADMAP

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPEN PROBLEMS REGISTRY — PRIORITY MATRIX                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DIFFICULTY                                                                 │
│      ▲                                                                      │
│   Hard│  OPR-7  OPR-8  OPR-9  OPR-10                                        │
│      │   ●──────●──────●──────●                                              │
│      │   │      │      │      │                                             │
│      │   │  OPR-1      OPR-6  │                                             │
│      │   │   ●──────────●     │                                             │
│      │   │   │              │ │                                             │
│Medium│   │   │  OPR-2  OPR-4│ │                                             │
│      │   │   │   ●──────●   │ │                                             │
│      │   │   │   │      │   │ │                                             │
│      │   │   │  OPR-5      │ │                                             │
│      │   │   │   ●─────────┘ │                                             │
│      │   │   │               │                                              │
│  Easy│   │   │  OPR-3        │                                              │
│      │   ▼   ▼   ●───────────┘                                              │
│      │                                                                      │
│      └──────────────────────────────────────────────────────────→ IMPACT    │
│            Low              Medium               High                       │
│                                                                             │
│   LEGEND:                                                                   │
│   ● = Active / Open        │   Priority: High impact + Medium difficulty   │
│   ◇ = Pending (OPR-8,9)    │   → OPR-1, OPR-2, OPR-3 first                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

ATLAS 2: CONTRIBUTION WORKFLOW — ASCII FLOWCHART

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTRIBUTION LIFECYCLE — FROM IDEA TO MERGE              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│   │ 1. SELECT│─────►│ 2. READ  │─────►│ 3. BUILD │─────►│ 4. TEST  │       │
│   │   an OPR │      │  problem │      │   code/  │      │  locally │       │
│   │   from   │      │  & null  │      │   proof  │      │  & audit │       │
│   │  registry│      │ hypothesis│      │          │      │          │       │
│   └────┬─────┘      └────┬─────┘      └────┬─────┘      └────┬─────┘       │
│        │                 │                 │                 │             │
│        ▼                 ▼                 ▼                 ▼             │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│   │ 5. OPEN  │      │ 6. PR    │      │ 7. REVIEW│      │ 8. MERGE │       │
│   │  Issue   │─────►│  Submit  │─────►│ (14 days)│─────►│   &      │       │
│   │  (claim) │      │  with    │      │  by      │      │  release │       │
│   │          │      │  solution│      │community │      │  bounty  │       │
│   └──────────┘      └──────────┘      └────┬─────┘      └──────────┘       │
│                                             │                               │
│                                             ▼                               │
│                                    ┌─────────────────┐                      │
│                                    │ ACCEPTED?       │                      │
│                                    │  ┌─────┐ ┌─────┐ │                      │
│                                    │  │ YES │ │ NO  │ │                      │
│                                    │  └──┬──┘ └──┬──┘ │                      │
│                                    └─────┼───────┼────┘                      │
│                                          │       │                          │
│                                          │       └──────────────┐           │
│                                          ▼                      ▼           │
│                                    ┌──────────┐          ┌──────────┐       │
│                                    │ Co-author│          │ Revise & │       │
│                                    │ credit   │          │ resubmit │       │
│                                    │ + bounty │          │          │       │
│                                    └──────────┘          └──────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

ATLAS 3: OPR-1 PREDECESSOR GEOMETRY — VISUAL TARGET

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPR-1: PREDECESSOR TREE CLASSIFICATION                    │
│                 "Who classifies the feeder basins?" — NO ONE YET             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ATTRACTOR (fixed point or cycle)                  │
│                                      ●                                      │
│                                     /|\                                     │
│                                    / | \                                    │
│                                   /  |  \                                   │
│                     Level 1:   ●───●───●───●  (cycle-adjacent)             │
│                               /|\ /|\ /|\ /|\                               │
│                              / | X | X | X | \                              │
│                    Level 2: ●● ●●● ●●● ●●● ●●  (70-80% of basin)            │
│                            /|\ /|\ /|\ /|\ /|\                              │
│                           /  X   X   X   X  \                               │
│                 Level 3: ●●●●●●●●●●●●●●●●●●●●●  (deep transients)            │
│                                                                             │
│   BASIN OBSERVABLES TO COMPUTE:                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ • Basin size |B(a)|                                                  │   │
│   │ • Maximum depth (shell index)                                       │   │
│   │ • Mean depth                                                        │   │
│   │ • Indegree of attractor (direct feeders)                            │   │
│   │ • Predecessor entropy P_a = -Σ_t p_t log₂ p_t                        │   │
│   │   where p_t = |{x: τ(x)=t}| / |B(a)|                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   EXPECTED OUTPUT: CSV table with one row per attractor across b=2-25       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

ATLAS 4: OPR-3 RANDOM BASELINE — EXPERIMENTAL DESIGN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              OPR-3: RANDOM FUNCTIONAL GRAPH BASELINE (OP0)                  │
│           "Is Kaprekar special, or just another random map?"                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   EXPERIMENTAL DESIGN:                                                      │
│                                                                             │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │  KAPREKAR MAP   │              │ 1000 RANDOM     │                      │
│   │  T(b,d)         │              │ PERMUTATION     │                      │
│   │  (deterministic)│              │ MAPS (uniform)  │                      │
│   └────────┬────────┘              └────────┬────────┘                      │
│            │                                │                               │
│            ▼                                ▼                               │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │ Compute         │              │ Compute         │                      │
│   │ observables:    │              │ observables:    │                      │
│   │ • max_τ         │              │ • max_τ         │                      │
│   │ • B_index       │              │ • B_index       │                      │
│   │ • λ₁            │              │ • λ₁            │                      │
│   │ • attractor cnt │              │ • attractor cnt │                      │
│   └────────┬────────┘              └────────┬────────┘                      │
│            │                                │                               │
│            └──────────────┬─────────────────┘                               │
│                           ▼                                                 │
│                   ┌─────────────────┐                                       │
│                   │ Z-SCORE =        │                                       │
│                   │ (K - μ_rand)/σ_rand│                                    │
│                   └────────┬────────┘                                       │
│                            ▼                                                │
│                   ┌─────────────────┐                                       │
│                   │ |Z| > 3?        │                                       │
│                   │ → ANOMALY       │                                       │
│                   │ |Z| < 2?        │                                       │
│                   │ → NOT SPECIAL   │                                       │
│                   └─────────────────┘                                       │
│                                                                             │
│   OUTPUT: Z-score table for each (b,d), every observable                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

ATLAS 5: CODE CONTRIBUTION TIERS — EFFORT vs REWARD

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTRIBUTION IMPACT MATRIX                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   REWARD                                                                    │
│   (Recognition)                                                            │
│      ▲                                                                      │
│      │                                                                      │
│  High│  ★ Solve OPR-1..10        ★ Add new invariant                      │
│      │  (co-authorship)           (co-authorship)                          │
│      │      ●────────────────────────●                                      │
│      │      │                        │                                      │
│      │      │  ● New visualization   │                                      │
│      │      │    (cited in papers)   │                                      │
│      │      │                        │                                      │
│   Med│      │  ● Bug fix             │                                      │
│      │      │    (CONTRIBUTORS.md)   │                                      │
│      │      │                        │                                      │
│      │      │  ● Typo fix            │                                      │
│  Low│      ▼    (no credit)          │                                      │
│      │                                                                      │
│      └──────────────────────────────────────────────────────→ EFFORT        │
│            Low         Medium         High          Very High               │
│                                                                             │
│   LEGEND:                                                                   │
│   ★ = High-impact, high-reward                                              │
│   ● = Contribution point                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

ATLAS 6: FALSIFICATION TRACK — KILL ZONE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KSG KILL ZONE — FALSIFICATION WELCOME                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Every conjecture has a falsification condition.                           │
│   Finding a counterexample is as valuable as proving it.                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  EXAMPLE FALSIFICATION REPORT:                                      │   │
│   │                                                                      │   │
│   │  Conjecture (OPR-1): Predecessor entropy P_a is invariant within    │   │
│   │                     modular families.                               │   │
│   │                                                                      │   │
│   │  FALSIFICATION: Found two fixed points in same Kay family with       │   │
│   │                  ΔP > 0.5 nats.                                     │   │
│   │                                                                      │   │
│   │  EVIDENCE:                                                           │   │
│   │    b=6, d=4: fixed point 1234 → P=2.3 bits                         │   │
│   │    b=6, d=4: fixed point 4321 → P=3.1 bits                         │   │
│   │    Both are "symmetric even" family per Kay (2024).                 │   │
│   │                                                                      │   │
│   │  VERDICT: Conjecture REFUTED. Need refined classification.          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   HOW TO SUBMIT A FALSIFICATION:                                           │
│   1. Identify the OPR and its falsification condition                      │
│   2. Run the code or construct the proof                                   │
│   3. Open an Issue titled "FALSIFICATION: OPR-X"                          │
│   4. Include counterexample and verification script                        │
│   5. Receive credit in CONTRIBUTORS.md and potential co-authorship         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

ATLAS 7: QUICK REFERENCE — COMMANDS FOR EACH OPR

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONE-LINE EXECUTION FOR EACH OPEN PROBLEM                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   OPR-1 (Predecessor Geometry):                                             │
│   $ python core/op1_predecessor_census.py --bases 2-25                      │
│                                                                             │
│   OPR-2 (Basin Hierarchy):                                                  │
│   $ python core/op2_basin_hierarchy.py --input data_lake/predecessors.csv   │
│                                                                             │
│   OPR-3 (Random Baseline):                                                  │
│   $ python core/op0_random_baseline.py --b=10 --d=4 --trials=1000           │
│                                                                             │
│   OPR-4 (Shell Ratio Limits):                                               │
│   $ python scripts/shell_ratio_analysis.py --all-bases --neutral            │
│                                                                             │
│   OPR-5 (Bottleneck Universality):                                          │
│   $ python scripts/bottleneck_z_test.py --baseline results/random_baseline.csv│
│                                                                             │
│   OPR-6 (Modular Correlation):                                              │
│   $ python scripts/modular_correlation.py --v2 --spearman                   │
│                                                                             │
│   OPR-7 (Self-Stabilizer Template):                                         │
│   $ python scripts/self_stabilizer_search.py --bases 2-25 --digits 3-5      │
│                                                                             │
│   OPR-8 (8-Digit Enumeration):                                              │
│   $ make -C cpp/ enum_d8 && ./enum_d8 --b=10 --d=8 --threads=8             │
│                                                                             │
│   OPR-9 (Full Graph Spectral):                                              │
│   $ python scripts/full_graph_spectrum.py --b=10 --d=4 --sparse             │
│                                                                             │
│   OPR-10 (Transfer Operator):                                               │
│   $ python scripts/transfer_operator_spectrum.py --b=10 --d=4               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

ATLAS 8: REPOSITORY STRUCTURE — WHERE YOUR CODE GOES

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KSG REPOSITORY — CONTRIBUTOR'S MAP                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   KAPREKAR-SPECTRAL-GEOMETRY/                                               │
│   │                                                                         │
│   ├── core/                      ← Core algorithms (your new code here)    │
│   │   ├── kaprekar_engine.py     ← T map, τ-depth                          │
│   │   ├── metrics.py             ← Φ_*, M₁, h, Cheeger                     │
│   │   ├── op1_predecessor_census.py   ← OPR-1, OPR-2                       │
│   │   ├── op0_random_baseline.py      ← OPR-3                              │
│   │   └── yamagami_integration.py      ← OPR-6, OPR-7                      │
│   │                                                                         │
│   ├── scripts/                   ← Executables (run these)                 │
│   │   ├── RUN_ALL.py             ← Master pipeline                         │
│   │   ├── shell_ratio_analysis.py    ← OPR-4                               │
│   │   ├── bottleneck_z_test.py       ← OPR-5                               │
│   │   └── modular_correlation.py     ← OPR-6                               │
│   │                                                                         │
│   ├── cpp/                       ← High-performance (OPR-8)                │
│   │   └── enum_d8.cpp            ← 8-digit enumeration                     │
│   │                                                                         │
│   ├── tests/                     ← Unit tests (must pass)                  │
│   │   ├── test_kaprekar.py                                                  │
│   │   └── test_predecessor.py                                              │
│   │                                                                         │
│   ├── data_lake/                 ← All CSVs (ground truth)                 │
│   │   ├── master_census.csv                                                │
│   │   ├── predecessor_profiles.csv                                         │
│   │   └── random_baseline.csv                                             │
│   │                                                                         │
│   ├── results/                   ← Generated outputs (gitignored)          │
│   │                                                                         │
│   └── DOCS/                      ← Documentation (your PRs here)           │
│       ├── MARKDOWN/              ← All .md files                           │
│       └── manuscripts/           ← LaTeX papers                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

HOW TO USE THIS ATLAS IN CONTRIBUTING.MD

Insert the relevant ASCII sections into A28-CONTRIBUTING.MD under appropriate headings. For example:

· Atlas 1 → Under "Open Problems Registry" heading
· Atlas 2 → Under "Contribution Workflow" heading
· Atlas 3-7 → Under specific OPR descriptions
· Atlas 8 → Under "Repository Structure" heading

These ASCII diagrams provide visual orientation without requiring external image hosting. All are pure text, viewable in any terminal or Markdown renderer.

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  END OF EXTENDED ASCII ATLAS — FOR CONTRIBUTING.MD                           ║
║  Node #10878 · Louisville, KY · 2026-04-29                                    ║
║  Copy-paste any section into CONTRIBUTING.md as needed                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

```bash
# Save the atlas as a separate file or integrate directly
cp EXTENDED_ASCII_ATLAS_CONTRIBUTING.md DOCS/MARKDOWN/A28-CONTRIBUTING-ATLAS.md

# Or append to CONTRIBUTING.md
cat EXTENDED_ASCII_ATLAS_CONTRIBUTING.md >> DOCS/MARKDOWN/A28-CONTRIBUTING.md

git add DOCS/MARKDOWN/
git commit -m "Add extended ASCII atlas for CONTRIBUTING.md — 8 visual guides"
git push origin main
```
