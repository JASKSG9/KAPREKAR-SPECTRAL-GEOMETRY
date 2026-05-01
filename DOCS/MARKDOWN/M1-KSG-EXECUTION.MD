# KSG EXECUTION SPECIFICATION — d=4, base‑10

**Node:** 10878 · **Target:** L4‑01 KV260 · **Branch:** `main`

---

## 1. STATE SPACE AND MAP

- `Ω = {0000..9999}` → `|Ω| = 10,000`.  
- `K(n) = desc_digits(n) - asc_digits(n)` → deterministic 4‑digit, base‑10.  
- `Image = K(Ω)` → `|Image| = 55`.  
- Fixed points: `{0, 6174}`.  
- `τ(n) =` steps to `{0, 6174}`; `τ_max = 7`.

---

## 2. τ‑DISTRIBUTION (EXACT)

```
τ=0:  2      {0, 6174}
τ=1:  392    includes repdigits 1111..9999
τ=2:  576
τ=3:  2400
τ=4:  1272   ← Fiedler cut τ=4→5
τ=5:  1518
τ=6:  1656
τ=7:  2184
Sum: 10000 ✓
```

Repdigits are `τ=1`, not `τ=0`.  
Claim `τ=0 = 11` is KILLED.

---

## 3. IMAGE SIZE (d=4, b=2..20)

Formula: `|Image(b,4)| = b(b+1)/2`.

| b  | |Image| | b(b+1)/2 | match |
|----|--------|-----------|-------|
| 2  | 3      | 3         | YES   |
| 3  | 6      | 6         | YES   |
| 4  | 10     | 10        | YES   |
| 5  | 15     | 15        | YES   |
| 6  | 21     | 21        | YES   |
| 7  | 28     | 28        | YES   |
| 8  | 36     | 36        | YES   |
| 9  | 45     | 45        | YES   |
|10  | 55     | 55        | YES   |
|11  | 66     | 66        | YES   |
|12  | 78     | 78        | YES   |
|13  | 91     | 91        | YES   |
|14  |105     |105        | YES   |
|15  |120     |120        | YES   |
|16  |136     |136        | YES   |
|17  |153     |153        | YES   |
|18  |171     |171        | YES   |
|19  |190     |190        | YES   |
|20  |210     |210        | YES   |

---

## 4. ATTRACTOR CENSUS (d=4, b=2..20)

| b  | fixed_points | cycles | only_fixed? |
|----|--------------|--------|------------|
| 2  | 3            | 0      | YES        |
| 3  | 1            | 1      | NO         |
| 4  | 2            | 1      | NO         |
| 5  | 2            | 0      | YES        |
| 6  | 1            | 1      | NO         |
| 7  | 1            | 1      | NO         |
| 8  | 1            | 2      | NO         |
| 9  | 1            | 2      | NO         |
|10  | 2            | 0      | YES        |
|11  | 1            | 2      | NO         |
|12  | 1            | 2      | NO         |
|13  | 1            | 3      | NO         |
|14  | 1            | 1      | NO         |
|15  | 2            | 6      | NO         |
|16  | 1            | 4      | NO         |
|17  | 1            | 8      | NO         |
|18  | 1            | 2      | NO         |
|19  | 1            | 4      | NO         |
|20  | 2            | 1      | NO         |

Only‑fixed bases: `2,5,10` → “base‑10 uniqueness” KILLED.

---

## 5. SPECTRAL MODELS (7‑node, 8‑node)

**7‑node (τ=1..7):**

- `μ₁ = 0.1624262417`.  
- `λₖ + λ₆₋ₖ = 2` (reflection symmetry).  
- Fiedler cut at `τ=4→5`.  
- Cheeger: `h = 0.1700`, `h²/2 = 0.01445 ≤ μ₁ ≤ 2h = 0.3400`.

**8‑node (τ=0..7):**

- `μ₁ = 0.1603171230`.

Scaling law `μ₁(d) = 12.576 / d^{3.137}` is **suspended**.

---

## 6. COARSE‑GRAINING `g*` (k_opt=20)

From 55‑node image graph:

- `W = A Aᵀ + Aᵀ A`.  
- `L_norm = I - D^{-1/2} W D^{-1/2}`.  
- `k_opt = 20`.  
- `g* = KMeans(n_clusters=20)` on first 20 eigenvectors.

- 19/20 clusters: 100% τ‑pure.  
- C9: 960 (τ=3) + 348 (τ=6); `Δ ≈ 0.78`.

---

## 7. INDUCED MARKOV CHAIN `P*`

- States: 20 macrostates (`g*` clusters).  
- Two absorbing classes:  
  - C5 (6174)  
  - C19 (0, repdigits)  
- All others: transient.  
- `γ(P*) = 0` (two eigenvalues = 1).

---

## 8. PALINDROME GATEWAYS (81 non‑repdigit)

| τ | count | Gateway set |
|---|-------|-------------|
| 3 | 0     | bypassed |
| 4 | 36    | {1089,4356,6534,9801} |
| 5 | 9     | {5445} |
| 6 | 36    | {2178,3267,7623,8712} |
| 7 | 0     | unreachable |

Never reach τ=3, τ=7.

---

## 9. EXECUTION COMMANDS (KV260)

```bash
python3 gstar_kaprekar.py --target kv260 --spectral-mask true
python3 palindrome_pulse.py --verify-gateways
python3 cross_base_census.py --digits 4 --bmax 30
python3 scaling_law_mu1.py --fit-range 4,8
```

---

## 10. VERIFICATION BENCHMARKS

| Test              | Tolerance   | Hardware | Frequency |
|-------------------|-------------|----------|-----------|
| τ distribution     | exact       | CPU      | once      |
| Image size (b=2..20) | exact     | CPU      | once      |
| μ₁ (7‑node)        | ≤ 1e‑10     | CPU      | per run   |
| g* purity          | 0% deviation| CPU/GPU  | per run   |
| Palindrome gateways| exact       | CPU      | once      |

---

## 11. DEPENDENCIES

```text
Python 3.8+
numpy ≥ 1.21
scipy ≥ 1.7
scikit‑learn ≥ 1.0
```

---

## 12. KILLED CLAIMS (PUT IN HISTORY)

- “Base‑10 uniqueness”  
- `τ=0 = 11`  
- “SUSY pairing”  
- “810× unfaithful”

---

## 13. OPEN PROBLEMS

- OP12 – Prove `|Image(b,4)| = b(b+1)/2` for all `b≥2`.  
- OP11 – Characterise all bases `b` with only fixed‑point attractors.  
- OP10 – Closed form for `N_τ(b,4)`.  
- OP9 – Why C9 mixes τ=3 and τ=6.

---

## 14. LICENSE AND CITATION

Code: MIT (see `LICENSE.md`).  
Data/Results: CC0 1.0 (public domain).

```bibtex
@misc{ksg_atlas_2026,
  title        = {Kaprekar Spectral Geometry (KSG): 4-digit, base-10 Atlas},
  author       = {{KSG Contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY}},
  note         = {Node 10878, 2026-05-01}
}
```

## 15. ACTIVE DIR STRUCTURE

```text
KAPREKAR-SPECTRAL-GEOMETRY/
├── gstar_kaprekar.py
├── palindrome_pulse.py
├── cross_base_census.py
├── scaling_law_mu1.py
├── data/
│   ├── tau_distribution.json
│   ├── attractor_census.csv
│   └── image_55.csv
└── DOCS/MARKDOWN/M1-KSG-EXECUTION.MD
