#============================================================
# 6. TROUBLESHOOTING.md
# ============================================================

troubleshooting = """# TROUBLESHOOTING — Common Issues & Fixes
**KSG/UDAG · Production Solutions**

---

## CATEGORY 1: INSTALLATION

### Issue: `pip install` fails
```
ERROR: Could not find a version that satisfies the requirement...
```
**Cause**: Python < 3.9 or missing system dependencies
**Fix**:
```bash
# Check Python version
python --version  # Need 3.9+

# Update pip
pip install --upgrade pip

# Install system deps (Ubuntu/Debian)
sudo apt-get install python3-dev libatlas-base-dev

# Install requirements
pip install -r requirements.txt
```

### Issue: `ModuleNotFoundError: No module named 'scipy'`
**Fix**:
```bash
pip install numpy scipy pandas matplotlib seaborn
```

---

## CATEGORY 2: RUNTIME ERRORS

### Issue: `ModuleNotFoundError: core.kaprekar`
```
Traceback:
  File "scripts/graph_native_invariants.py", line 5, in <module>
    from core.kaprekar import build_map
ModuleNotFoundError: No module named 'core'
```
**Cause**: PYTHONPATH not set; running script from wrong directory
**Fix**:
```bash
# Option 1: Run from repo root
python scripts/graph_native_invariants.py

# Option 2: Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/graph_native_invariants.py

# Option 3: Use python -m
python -m scripts.graph_native_invariants
```

---

### Issue: `MemoryError` on large bases
```
MemoryError: Unable to allocate array...
```
**Cause**: Base 30, d=4 requires 810,000 states × adjacency matrix
**Fix**:
```bash
# Use low-memory mode
python RUN_ALL.py --lowmem --bases 2-25

# Or skip large bases
python scripts/graph_native_invariants.py --bases 2-20

# Or increase system swap
sudo swapon /swapfile  # Linux
```

---

### Issue: `ARPACK convergence failure`
```
scipy.sparse.linalg.eigsh: ARPACK error -1: No convergence...
```
**Cause**: Sparse eigenvalue solver fails on ill-conditioned matrix
**Fix**:
```python
# In core/metrics.py, increase maxiter:
eigenvalues = sp_linalg.eigsh(A, k=2, which='SM', 
                              return_eigenvectors=False,
                              maxiter=10000)  # Increase from default

# Or use dense fallback for small matrices:
if N < 5000:
    eigenvalues = np.linalg.eigvalsh(A.toarray())
```

---

### Issue: `tau-path μ₁ = 0`
```
Warning: Fiedler value is zero for base X
```
**Cause**: Single transient shell (τ_max = 1); path has no interior
**Fix**:
```python
# Skip bases with τ_max < 2
if len(shell_sizes) < 3:
    print(f"Base {base}: insufficient shells, skipping")
    continue
```

---

## CATEGORY 3: OUTPUT ISSUES

### Issue: Heatmaps are blank / white
**Cause**: Matplotlib backend issue in headless environment
**Fix**:
```python
# Add at TOP of visualization script
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
import matplotlib.pyplot as plt
```

### Issue: CSV files contain NaN
**Cause**: Division by zero or uninitialized values
**Fix**:
```python
# Check for zero denominators
denom = Q.sum(axis=1, keepdims=True)
denom[denom < 1e-12] = 1.0  # Prevent division by zero
Q = Q / denom
```

### Issue: `results/` directory not created
**Fix**:
```python
import os
os.makedirs('results', exist_ok=True)
```

---

## CATEGORY 4: VERIFICATION FAILURES

### Issue: Shell Descent verification fails
```
AssertionError: τ(T(x)) != τ(x) - 1 for x=...
```
**Diagnosis**:
```python
# Debug: print violating state
for x in range(N):
    if tau[x] > 0:
        tx = T[x]
        if tau[tx] != tau[x] - 1:
            print(f"VIOLATION: x={x}, tau(x)={tau[x]}, tau(T(x))={tau[tx]}")
            break
```
**Likely cause**: `compute_tau` bug or non-deterministic map
**Fix**: Use corrected `compute_shell_depth` from core/op1_yamagami_kay_bridge.py

---

### Issue: Base-10 not unique in your run
```
AssertionError: all_fixed_bases != [10]
```
**Diagnosis**:
```python
# Check cycle detection
attractors = find_attractor(T)
for cyc in attractors:
    print(f"Cycle length: {len(cyc)}, states: {cyc}")
```
**Likely cause**: Cycle detection algorithm error
**Fix**: Use `extract_all_attractors` from core/op1_yamagami_kay_bridge.py

---

## CATEGORY 5: PERFORMANCE

### Issue: OP13 sweep takes > 1 hour
**Cause**: Full eigenvalue computation on large matrices
**Optimization**:
```bash
# Use sparse matrices only
python scripts/graph_native_invariants.py --sparse-only

# Parallelize across bases
python scripts/graph_native_invariants.py --parallel --jobs 4

# Or compute subset
python scripts/graph_native_invariants.py --bases 2-15
```

### Issue: `op1_yamagami_kay_bridge.py` runs forever
**Cause**: Full enumeration for large (b,d) combinations
**Fix**:
```bash
# Limit scan range
python core/op1_yamagami_kay_bridge.py --bases 2-10 --digits 3,4
```

---

## CATEGORY 6: PLATFORM-SPECIFIC

### macOS: `RuntimeError: Python is not installed as a framework`
**Fix**:
```python
import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg' for headless
```

### Windows: `UnicodeEncodeError` in ASCII diagrams
**Fix**:
```python
# Set console encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

### WSL: `DISPLAY` not set
**Fix**:
```bash
export DISPLAY=:0
# Or use Agg backend (see above)
```

---

## DEBUGGING CHECKLIST

```
□ Python version ≥ 3.9?
□ All requirements installed?
□ Running from repo root?
□ PYTHONPATH set correctly?
□ results/ directory exists?
□ Enough RAM for chosen base/digit?
□ Matplotlib backend configured?
□ No killed fabrications in code?
```

---

## STILL STUCK?

1. Check [CHEATSHEET.md](CHEATSHEET.md) for quick commands
2. Check [GitHub Issues](https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY/issues)
3. Run audit: `python scripts/audit_fabrications.py`
4. Open issue with: error message, command used, Python version, OS

---

*Node #10878 · Louisville, KY · 2026-04-28*
"""

with open('/mnt/agents/output/TROUBLESHOOTING.md', 'w') as f:
    f.write(troubleshooting)
print("✅ TROUBLESHOOTING.md")
