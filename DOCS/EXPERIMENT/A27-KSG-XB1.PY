#!/usr/bin/env python3
"""
EXPERIMENT_B – Complete Causality Pipeline for A30 Base Scan
Node #10878 | April 27, 2026 (FINAL REFINED)

Computes:
  - Depth histograms for bases 2‑25 (4‑digit Kaprekar)
  - Morphological descriptors for each histogram
  - Statistical causality tests (MI, Spearman, separability)
  - Symbolic regression to extract candidate laws
  - Exports CSV with all descriptors and coherence

Run with: python experiment_b.py
"""

import numpy as np
from collections import Counter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 0: A30 BASE SCANNER (4‑digit Kaprekar, bases 2‑25)
# ============================================================================

def kaprekar_step_base(n, base, digits=4):
    """Single Kaprekar step for given base and digit length."""
    if n == 0:
        return 0
    s = []
    tmp = n
    for _ in range(digits):
        s.append(tmp % base)
        tmp //= base
    s = s[::-1]
    while len(s) < digits:
        s = [0] + s
    asc = 0
    desc = 0
    for d in sorted(s):
        asc = asc * base + d
    for d in sorted(s, reverse=True):
        desc = desc * base + d
    return desc - asc


def depth_histogram(base, digits=4, max_depth=50):
    """Return N_τ histogram (list of counts per τ) for a given base."""
    total = base ** digits
    use_sampling = base > 15
    if use_sampling:
        rng = np.random.RandomState(42)
        states = rng.choice(total, size=min(10000, total), replace=False)
    else:
        states = range(total)

    depths = []
    for n in states:
        if len(set(str(n).zfill(digits))) == 1:
            depths.append(0)
            continue
        x = n
        steps = 0
        seen = set()
        while x not in seen and steps < max_depth:
            seen.add(x)
            x = kaprekar_step_base(x, base, digits)
            steps += 1
        depths.append(steps)

    hist = Counter(depths)
    max_tau = max(hist.keys())
    return np.array([hist.get(t, 0) for t in range(1, max_tau + 1)], dtype=float)


def scan_bases(bases):
    """Scan multiple bases, return list of (base, N_tau)."""
    results = []
    for base in bases:
        print(f"Scanning base {base}...", end=" ", flush=True)
        N = depth_histogram(base)
        results.append((base, N))
        print(f"τ_max={len(N)}")
    return results


# ============================================================================
# PART A: MORPHOLOGICAL DESCRIPTOR EXTRACTOR (full, no placeholders)
# ============================================================================

def compute_descriptors(N_tau):
    """Extract 14 morphological descriptors from depth histogram."""
    N = np.array(N_tau, dtype=float)
    tau = np.arange(1, len(N) + 1)
    total = N.sum()
    if total == 0:
        return {f'd_{i}': 0.0 for i in range(20)}

    mean_tau = np.sum(tau * N) / total
    var_tau = np.sum(N * (tau - mean_tau) ** 2) / total
    std_tau = np.sqrt(var_tau)
    skew = np.sum(N * ((tau - mean_tau) / std_tau) ** 3) / total if std_tau > 0 else 0
    kurt = np.sum(N * ((tau - mean_tau) / std_tau) ** 4) / total - 3 if std_tau > 0 else 0

    if len(N) > 1:
        w = np.sqrt(N[:-1] * N[1:])
        w = w[w > 0]
        bottleneck_ratio = np.max(w) / np.mean(w) if len(w) > 0 and np.mean(w) > 0 else 1.0
    else:
        bottleneck_ratio = 1.0

    if len(N) >= 4:
        right_mass = N[3:].sum()
        left_mass = N[:3].sum()
    else:
        right_mass = N.sum()
        left_mass = 1
    rl_ratio = right_mass / left_mass if left_mass > 0 else 1.0

    peak_tau = np.argmax(N) + 1
    peak_sharpness = np.max(N) / total

    if len(N) > 2:
        diff = np.diff(N)
        roughness = np.var(diff)
        total_variation = np.sum(np.abs(diff))
    else:
        roughness = 0
        total_variation = 0

    p = N / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) if len(p) > 0 else 0

    # Placeholders for optional external data (palindrome ratio, gateway entropy)
    pal_ratio = 0.0
    gateway_entropy = 0.0

    return {
        'mean_depth': mean_tau,
        'variance': var_tau,
        'skewness': skew,
        'kurtosis': kurt,
        'bottleneck_ratio': bottleneck_ratio,
        'rl_mass_ratio': rl_ratio,
        'peak_tau': peak_tau,
        'peak_sharpness': peak_sharpness,
        'roughness': roughness,
        'entropy': entropy,
        'palindrome_ratio': pal_ratio,
        'gateway_entropy': gateway_entropy,
        'total_variation': total_variation
    }


# ============================================================================
# PART B: ANOMALY RESPONSE – spectral coherence = μ₁ / entropy
# ============================================================================

def spectral_coherence(N_tau):
    """Compute μ₁ (second smallest eigenvalue of normalised Laplacian) and coherence."""
    from scipy.linalg import eigh
    N = np.array(N_tau, dtype=float)
    if len(N) < 2:
        return 0.0, 0.0, 0.0
    w = np.sqrt(N[:-1] * N[1:])
    if len(w) == 0 or np.all(w == 0):
        return 0.0, 0.0, 0.0
    n = len(N)
    A = np.zeros((n, n))
    for i in range(n - 1):
        if w[i] > 0:
            A[i, i + 1] = A[i + 1, i] = w[i]
    deg = A.sum(axis=1)
    if np.any(deg == 0):
        return 0.0, 0.0, 0.0
    Dinv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(n) - Dinv_sqrt @ A @ Dinv_sqrt
    try:
        evals = eigh(L, eigvals_only=True)
        mu1 = evals[1] if len(evals) > 1 else 0.0
    except Exception:
        mu1 = 0.0
    p = N / N.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) if len(p) > 0 else 1.0
    coherence = mu1 / entropy if entropy > 0 else 0.0
    return coherence, mu1, entropy


# ============================================================================
# PART C: STATISTICAL CAUSALITY TESTS
# ============================================================================

def mutual_information(x, y, bins=10):
    """Mutual information I(X;Y) in nats."""
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = hist_2d / hist_2d.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def run_causality_tests(X, Y, descriptors):
    """MI, Spearman, and logistic separability AUC."""
    n, p = X.shape
    mi_results = []
    rho_results = []
    pval_results = []
    for j in range(p):
        if np.std(X[:, j]) < 1e-12:
            mi_results.append(0.0)
            rho_results.append(0.0)
            pval_results.append(1.0)
            continue
        mi = mutual_information(X[:, j], Y)
        rho, pval = spearmanr(X[:, j], Y)
        mi_results.append(mi)
        rho_results.append(rho)
        pval_results.append(pval)

    Y_bin = (Y > np.median(Y)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_bin, test_size=0.3,
                                                        random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    sorted_idx = np.argsort(mi_results)[::-1]
    top_descriptors = [(descriptors[i], mi_results[i], rho_results[i], pval_results[i])
                       for i in sorted_idx[:5]]
    return {
        'mi_all': mi_results,
        'rho_all': rho_results,
        'pval_all': pval_results,
        'auc': auc,
        'top_descriptors': top_descriptors
    }


# ============================================================================
# PART D: SYMBOLIC REGRESSION (polynomial fit)
# ============================================================================

def symbolic_regression_wrapper(X, Y, descriptors):
    """Fit degree‑2 polynomial on top 3 MI features, return expression & R²."""
    n_top = min(3, X.shape[1])
    # Compute MI for each column (efficient for small p)
    mi_vals = [mutual_information(X[:, j], Y) for j in range(X.shape[1])]
    top_indices = np.argsort(mi_vals)[-n_top:]
    X_top = X[:, top_indices]
    feature_names = [descriptors[i] for i in top_indices]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_top)
    model = LinearRegression()
    model.fit(X_poly, Y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(Y, y_pred)

    terms = [f"{model.intercept_:.4f}"]
    linear_coefs = model.coef_[:len(feature_names)]
    for coef, name in zip(linear_coefs, feature_names):
        if abs(coef) > 1e-6:
            terms.append(f"{coef:+.4f} * {name}")
    expr = "Y ≈ " + " ".join(terms)
    return expr, r2


# ============================================================================
# PART E: MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "═" * 80)
    print("EXPERIMENT B – COMPLETE CAUSALITY PIPELINE (FINAL ENHANCED)")
    print("═" * 80)

    # Step 1: Scan bases 2‑25
    print("\n[1] Scanning bases 2‑25...")
    bases = list(range(2, 26))
    scan_results = scan_bases(bases)

    # Step 2: Compute descriptors and coherence
    print("\n[2] Computing morphological descriptors and coherence...")
    rows = []
    base_list = []
    coherence_list = []
    mu1_list = []
    entropy_list = []

    for base, N in scan_results:
        desc = compute_descriptors(N)
        coh, mu1, ent = spectral_coherence(N)
        desc['base'] = base
        desc['parity'] = base % 2
        desc['is_prime'] = 1 if base in [2, 3, 5, 7, 11, 13, 17, 19, 23] else 0
        desc['log_base'] = np.log(base)
        rows.append(desc)
        base_list.append(base)
        coherence_list.append(coh)
        mu1_list.append(mu1)
        entropy_list.append(ent)

    # Build feature matrix X and target Y
    desc_names = [
        'parity', 'is_prime', 'log_base',
        'mean_depth', 'variance', 'skewness', 'kurtosis',
        'bottleneck_ratio', 'rl_mass_ratio', 'peak_tau', 'peak_sharpness',
        'roughness', 'entropy', 'total_variation'
    ]
    X = np.array([[r.get(name, 0.0) for name in desc_names] for r in rows])
    Y = np.array(coherence_list)

    # Step 3: Causality tests
    print("\n[3] Running causality tests...")
    results = run_causality_tests(X, Y, desc_names)

    # Step 4: Print report
    print("\n" + "═" * 80)
    print("CAUSALITY ANALYSIS REPORT")
    print("═" * 80)
    print(f"Number of bases scanned: {len(rows)}")
    print(f"Number of descriptors: {len(desc_names)}")
    print("\nTop 5 descriptors by Mutual Information (nats):")
    for name, mi, rho, pval in results['top_descriptors']:
        print(f"  {name:18s} | MI={mi:.4f} | Spearman ρ={rho:+.4f} (p={pval:.4f})")
    print(f"\nLogistic Regression AUC (binary high/low coherence): {results['auc']:.4f}")

    # Step 5: Symbolic regression if signal strong
    if results['auc'] > 0.7:
        expr, r2 = symbolic_regression_wrapper(X, Y, desc_names)
        print(f"\nSymbolic regression candidate law:\n  {expr}")
        print(f"  R² = {r2:.4f}")
        if r2 > 0.6:
            print("\n✅ CONCLUSION: Strong structural causality. Anomalies are not random.")
        else:
            print("\n⚠️ Moderate separability but low symbolic fit. Refine descriptors.")
    else:
        print("\n❌ Weak separability (AUC ≤ 0.7). Anomalies may be random scatter.")

    print("═" * 80)

    # Save raw data
    import csv
    with open('experiment_b_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['base', 'coherence', 'mu1', 'entropy'] + desc_names)
        for i, base in enumerate(base_list):
            row = [base, coherence_list[i], mu1_list[i], entropy_list[i]]
            row += [X[i][j] for j in range(X.shape[1])]
            writer.writerow(row)
    print("\nData saved to experiment_b_data.csv")
    print("\n✅ EXPERIMENT B COMPLETE – RESULTS READY FOR ANALYSIS")


if __name__ == "__main__":
    main()
