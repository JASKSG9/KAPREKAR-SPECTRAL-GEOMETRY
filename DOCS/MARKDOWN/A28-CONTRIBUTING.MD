
# ============================================================
# 4. CONTRIBUTING.md
# ============================================================

contributing = """# CONTRIBUTING — How to Participate in KSG/UDAG
**Open Science · Open Code · Open Problems**

---

## WELCOME

Thank you for your interest in Kaprekar Spectral Geometry. This project operates on **radical transparency**: all code, all data, all failures, and all killed fabrications are public.

**We need**: mathematicians, programmers, visual designers, technical writers, and skeptics.

---

## WAYS TO CONTRIBUTE

### 1. Solve an Open Problem (OP10-OP18)

| OP | Challenge | Bounty | Difficulty |
|----|-----------|--------|------------|
| OP10 | Cheeger closed form | $500 | Medium |
| OP11 | Decimal uniqueness proof | $2,000 | Hard |
| OP12 | B₂/N asymptotics | $1,000 | Medium |
| OP14 | Faithful spectral proxy | $1,500 | Hard |
| OP15 | Collatz mimicry | $800 | Medium |
| OP16 | RAG retrieval engine | $1,200 | Medium |
| OP17 | Collapse category | $3,000 | Very Hard |
| OP18 | Quantum merge entropy | $2,500 | Very Hard |

**How to claim**:
1. Open a GitHub Issue titled `BOUNTY-CLAIM: OP## — Brief Description`
2. Submit your solution as a Pull Request
3. Include: code + proof/writeup + test cases + verification
4. Community review period: 14 days
5. Bounty released upon acceptance

---

### 2. Code Contributions

**What we need**:
- New invariant computations
- Visualization modules
- Performance optimizations
- Additional deterministic maps (custom_map.py)
- Test coverage expansion

**Process**:
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
git commit -m "OP##: Description — verification passed"

# 6. PR
# Title: "OP##: Brief description"
# Body: Problem + Solution + Results + Verification
```

---

### 3. Documentation Contributions

**What we need**:
- Tutorials (Jupyter notebooks)
- Translations
- Visual diagrams
- Blog posts / outreach

**Process**: Same as code, but branch from `docs/`

---

### 4. Bug Reports

```markdown
## Bug Report Template

**Script**: `scripts/which_script.py`
**Command**: `python scripts/which_script.py --args`
**Expected**: What should happen
**Actual**: What happened (include traceback)
**Environment**: Python version, OS, RAM
```

---

### 5. Fabrication Reports

If you find a number or claim that cannot be reproduced:
```markdown
## Fabrication Report

**Claim**: "Base-10 has property X = value"
**Location**: File.py, line N
**Expected**: Value from code execution
**Actual**: Different value or error
**Severity**: Low / Medium / High
```

**We will**: investigate, kill if confirmed, credit reporter.

---

## CODE STANDARDS

### No Fabrications Rule
- Every number must be producible by code
- Every claim must have a verification test
- Every killed fabrication must be documented

### Function Documentation
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

### Testing Requirements
- All new functions need unit tests
- All statistical claims need reproducible seeds
- All visualizations need data sources cited

---

## COMMUNITY GUIDELINES

- **Be respectful**: Disagreement is welcome; hostility is not
- **Be precise**: "I think" → "The code outputs X on my machine"
- **Be transparent**: Share failures, not just successes
- **Be patient**: Review takes time; maintainers are volunteers
---

## RECOGNITION

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Cited in published papers
- Invited to co-author on major results
---

*Node #10878 · Louisville, KY · 2026-04-28*
*Build in the open. All questions are welcome.*
"""

with open('/mnt/agents/output/CONTRIBUTING.md', 'w') as f:
    f.write(contributing)
print("✅ CONTRIBUTING.md")
