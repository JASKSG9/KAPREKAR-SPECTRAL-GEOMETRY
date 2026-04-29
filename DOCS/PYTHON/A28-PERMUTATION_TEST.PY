"""
scripts/permutation_test_primality.py
Permutation test: do primes separate from composites in {Phi_*, M1, h}?

Uses graph_native_invariants.csv if present, else recomputes b=2-20.
Shuffles prime/composite labels n_trials times, computes separation statistic.
p_perm = fraction of shuffles where separation >= observed.

NO assumed p-values. Data produces them.

Outputs:
  results/permutation_test_results.txt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv


def load_invariants(path='results/graph_native_invariants.csv'):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) if k not in ('cyc_lens',) else v
                         for k, v in r.items()})
    return rows


def separation_statistic(vals_a, vals_b):
    """|mean_a - mean_b| / pooled_std — Welch-like effect size."""
    ma, mb = np.mean(vals_a), np.mean(vals_b)
    sa, sb = np.std(vals_a, ddof=1), np.std(vals_b, ddof=1)
    na, nb = len(vals_a), len(vals_b)
    pooled = np.sqrt((sa**2/na + sb**2/nb) + 1e-12)
    return abs(ma - mb) / pooled


def run(n_trials=5000, seed=42):
    os.makedirs('results', exist_ok=True)
    np.random.seed(seed)

    # Load or compute
    inv_path = 'results/graph_native_invariants.csv'
    if not os.path.exists(inv_path):
        print("graph_native_invariants.csv not found — running sweep first (bases 2-20)...")
        from scripts.graph_native_invariants import run as sweep
        sweep(bases=list(range(2, 21)))

    rows = load_invariants(inv_path)
    primes  = {2, 3, 5, 7, 11, 13, 17, 19, 23}
    squares = {4, 9, 16, 25}

    bases    = [int(r['b']) for r in rows]
    prime_mask = np.array([b in primes for b in bases])
    comp_mask  = np.array([b not in primes for b in bases])

    report_lines = [
        "PERMUTATION TEST — PRIME vs NON-PRIME SEPARATION",
        "=" * 50,
        f"n_trials = {n_trials}",
        f"Bases tested: {bases}",
        f"Primes: {[b for b in bases if b in primes]}",
        f"Non-primes: {[b for b in bases if b not in primes]}",
        "",
    ]

    metrics = ['Phi_star', 'M1', 'h']
    p_vals  = {}

    for metric in metrics:
        vals = np.array([r[metric] for r in rows])
        obs_stat = separation_statistic(vals[prime_mask], vals[comp_mask])

        # Permutation
        null_stats = []
        for _ in range(n_trials):
            shuffled = prime_mask.copy()
            np.random.shuffle(shuffled)
            s = separation_statistic(vals[shuffled], vals[~shuffled])
            null_stats.append(s)
        null_stats = np.array(null_stats)

        p = float((null_stats >= obs_stat).mean())
        p_vals[metric] = p

        mean_p = float(vals[prime_mask].mean())
        mean_c = float(vals[comp_mask].mean())
        report_lines += [
            f"METRIC: {metric}",
            f"  Prime mean    = {mean_p:.4f}",
            f"  Non-prime mean= {mean_c:.4f}",
            f"  Observed sep  = {obs_stat:.4f}",
            f"  Null mean sep = {null_stats.mean():.4f}",
            f"  p_perm        = {p:.4f}",
            f"  Significant (p<0.05): {p < 0.05}",
            "",
        ]

    # Combined 3D statistic
    Phi = np.array([r['Phi_star'] for r in rows])
    M1  = np.array([r['M1']       for r in rows])
    h   = np.array([r['h']        for r in rows])

    # Normalize each to [0,1]
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)
    X = np.column_stack([norm(Phi), norm(M1), norm(h)])
    # 3D separation: mean pairwise distance prime vs nonprime
    Xp = X[prime_mask]; Xc = X[comp_mask]
    obs_3d = float(np.mean([np.linalg.norm(p - c) for p in Xp for c in Xc]) -
                   np.mean([np.linalg.norm(p1 - p2)
                            for i, p1 in enumerate(Xp)
                            for j, p2 in enumerate(Xp) if i != j] or [0]))

    null_3d = []
    for _ in range(n_trials):
        sh = prime_mask.copy(); np.random.shuffle(sh)
        Xps = X[sh]; Xcs = X[~sh]
        s = float(np.mean([np.linalg.norm(p - c) for p in Xps for c in Xcs]) -
                  np.mean([np.linalg.norm(p1 - p2)
                           for i, p1 in enumerate(Xps)
                           for j, p2 in enumerate(Xps) if i != j] or [0]))
        null_3d.append(s)
    null_3d = np.array(null_3d)
    p_3d = float((null_3d >= obs_3d).mean())
    p_vals['combined_3D'] = p_3d

    report_lines += [
        "COMBINED 3D TEST (Phi_*, M1, h jointly):",
        f"  Observed separation = {obs_3d:.4f}",
        f"  p_perm (3D)         = {p_3d:.4f}",
        f"  Significant (p<0.05): {p_3d < 0.05}",
        "",
        "SUMMARY TABLE:",
        f"  Phi_*:    p = {p_vals['Phi_star']:.4f}",
        f"  M1:       p = {p_vals['M1']:.4f}",
        f"  h:        p = {p_vals['h']:.4f}",
        f"  3D joint: p = {p_vals['combined_3D']:.4f}",
        "",
        "NOTE: All p-values computed from data. No assumed distributions.",
        "STATUS: Empirical permutation test — results are what they are.",
    ]

    report = '\n'.join(report_lines)
    print(report)
    with open('results/permutation_test_results.txt', 'w') as f:
        f.write(report)
    print("\nSaved: results/permutation_test_results.txt")
    return p_vals


if __name__ == '__main__':
    run()
