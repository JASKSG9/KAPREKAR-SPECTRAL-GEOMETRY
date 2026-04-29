
# ============================================================
# 5. CHEATSHEET.md
# ============================================================

cheatsheet = """# CHEATSHEET — KSG/UDAG Quick Reference
**One page. All commands. Zero fluff.**

---

## INSTALLATION

```bash
git clone https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY.git
cd KAPREKAR-SPECTRAL-GEOMETRY
pip install -r requirements.txt
```

---

## 30-SECOND START

```bash
python RUN_ALL.py
# → Full census (b=2-30, d=4)
# → Base-10 singularity score
# → Output: results/
```

---

## CORE COMMANDS

### Level 1: Beginner (30s)
```bash
python RUN_ALL.py
```
**Output**: `results/verification_report.txt`

### Level 2: Intermediate (5min)
```bash
python scripts/custom_map.py --map collatz --N 10000
```
**Output**: `results/collatz_vs_kaprekar.csv`

### Level 3: Advanced (30min)
```bash
python scripts/graph_native_invariants.py --bases 2-25 --digits 4
```
**Output**: `results/graph_native_invariants.csv`
**Answers**: Is base-10 anomalous in pure graph metrics?

### Level 4: Expert (1hr)
```bash
python scripts/gm2_observables.py --all-invariants
```
**Output**: `results/gm2_observables.csv`
**Builds**: 3D phase space; primality clustering

### Level 5: Literature (2hr)
```bash
python core/op1_yamagami_kay_bridge.py
```
**Output**: Γ incompleteness scan
**Answers**: How much of Kaprekar geometry do current algebraic papers explain?

---

## KEY INVARIANTS TABLE

| Symbol | Name | Formula | Interpretation |
|--------|------|---------|----------------|
| τ | Shell depth | min{k: Tᵏ(x) ∈ A} | Distance to attractor |
| nₛ | Stabilization | Steps until Kₙ = |A| | Collapse speed |
| E∞ | Entropy loss | log₂N − H∞ | Total info erased |
| max Rₙ | Peak erasure | max(R₀, ..., Rₙₛ) | Strongest pulse |
| M₀ | Merge entropy | Shannon entropy of {rₙ} | Collapse uniformity |
| Φ_* | Conductance | min cut / min(vol) | Bottleneck width |
| M₁ | Return time | mean diag((I−Q)⁻¹) | Relaxation speed |
| h | Entropy rate | −∑πᵢ∑Pᵢⱼ log Pᵢⱼ | Mixing randomness |
| Γ | Incompleteness | 1 − matched/actual | Literature gap measure |

---

## DECISION TREE

```
Q: Is base-10 anomalous in (Φ_*, M₁, h)?
  YES → Run OP11 (algebraic proof)
  NO  → Run OP16 (RAG retrieval)

Q: Do primes/composites/squares cluster?
  YES → Phase separation confirmed
  NO  → Primality not a classifier

Q: Can I add my own map?
  YES → Edit scripts/custom_map.py
      → Define T: [0,N) → [0,N)
      → Run: python custom_map.py --map custom
```

---

## COMMON ERRORS

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: core.kaprekar` | PYTHONPATH | `export PYTHONPATH=$(pwd):$PYTHONPATH` |
| `MemoryError` (base 30, d=4) | Too large | `python RUN_ALL.py --lowmem --bases 2-25` |
| `Heatmaps blank` | Backend | Add `matplotlib.use('Agg')` at top |
| `ARPACK convergence` | Sparse matrix | Increase `maxiter` or use dense fallback |
| `tau-path μ₁ = 0` | Single shell | Skip base; path needs ≥2 shells |

---

## FILE LOCATIONS

| What | Where |
|------|-------|
| Core engine | `core/kaprekar.py` |
| Invariants | `core/metrics.py` |
| Main sweep | `scripts/graph_native_invariants.py` |
| Yamagami bridge | `core/op1_yamagami_kay_bridge.py` |
| Visualizations | `scripts/visualize_phase_space.py` |
| All outputs | `results/` |
| All docs | `docs/` |

---

## OP QUICK REFERENCE

| OP | Command | Time | Output |
|----|---------|------|--------|
| OP2 | `python scripts/op2_b2n_scaling.py` | 2min | `op2_b2n_scaling.csv` |
| OP3 | `python scripts/op3_algebraic.py` | 5min | `op3_algebraic_analysis.csv` |
| OP4 | `python scripts/op4_descent_extension.py` | 3min | `op4_descent_report.txt` |
| OP5 | `python scripts/op5_cheeger_from_shells.py` | 2min | `op5_cheeger_shells.csv` |
| OP8+9 | `python scripts/op8_op9_tech_debt.py` | 1min | `op8_op9_report.txt` |
| OP13 | `python scripts/graph_native_invariants.py` | 30min | `graph_native_invariants.csv` |

---

## ONE-LINE CHECKS

```bash
# Verify base-10 attractor
python -c "from core.kaprekar import build_map; T=build_map(4,10); print(T[6174])"
# Expected: 6174

# Count attractors for base-10
python -c "from core.kaprekar import build_map, find_attractor; T=build_map(4,10); A=find_attractor(T); print(len(A))"
# Expected: 2

# Check shell descent
python -c "from core.kaprekar import build_map, compute_tau; T=build_map(4,10); tau=compute_tau(T); print(all(tau[T[x]]==tau[x]-1 for x in range(len(T)) if tau[x]>0))"
# Expected: True
```
***

*Node #10878 · Louisville, KY · 2026-04-28*
*Print me. Tape me to your monitor.*
"""

with open('/mnt/agents/output/CHEATSHEET.md', 'w') as f:
    f.write(cheatsheet)
print("✅ CHEATSHEET.md")
