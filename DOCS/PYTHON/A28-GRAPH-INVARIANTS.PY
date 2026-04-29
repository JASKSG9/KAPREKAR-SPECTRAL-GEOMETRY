"""
scripts/graph_native_invariants.py
OP13: {Phi_*, mu1, M1, h} for bases 2-25, d=4.
Shell-path Cheeger (exact) + simulation for M1 and h.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import csv
from core.kaprekar import build_map, compute_tau, find_attractors, shell_sizes


def shell_cheeger(sz):
    n = len(sz)
    if n < 2: return 0.0
    N = np.array(sz, dtype=float)
    w = np.sqrt(N[:-1] * N[1:])
    deg = np.zeros(n)
    for j in range(n):
        if j > 0:   deg[j] += w[j-1]
        if j < n-1: deg[j] += w[j]
    best, vol_l = np.inf, 0.0
    vol_tot = deg.sum()
    for k in range(n - 1):
        vol_l += deg[k]
        vol_r = vol_tot - vol_l
        if vol_l > 0 and vol_r > 0:
            best = min(best, w[k] / min(vol_l, vol_r))
    return float(best) if best < np.inf else 0.0


def shell_mu1(sz):
    n = len(sz)
    if n < 2: return 0.0
    N = np.array(sz, dtype=float)
    w = np.sqrt(N[:-1] * N[1:])
    A = np.zeros((n, n))
    for j in range(n - 1):
        A[j, j+1] = A[j+1, j] = w[j]
    d = A.sum(axis=1); d[d < 1e-12] = 1.0
    L = np.eye(n) - np.diag(1/np.sqrt(d)) @ A @ np.diag(1/np.sqrt(d))
    ev = np.sort(np.linalg.eigvalsh(L))
    return float(ev[1]) if len(ev) > 1 else 0.0


def mean_first_passage(kmap, tau, n_sample=3000, seed=42):
    np.random.seed(seed)
    trans = np.where(tau > 0)[0]
    if len(trans) == 0: return 0.0
    state = np.random.choice(trans, size=min(n_sample, len(trans)), replace=True)
    times = np.full(len(state), int(tau.max()) + 1, dtype=np.int32)
    alive = np.ones(len(state), dtype=bool)
    for step in range(1, int(tau.max()) + 2):
        state[alive] = kmap[state[alive]]
        absorbed = alive & (tau[state] == 0)
        times[absorbed] = step
        alive[absorbed] = False
        if not alive.any(): break
    return float(times.mean())


def entropy_rate_sim(kmap, tau, n_sample=3000, n_steps=50, seed=42):
    N = len(kmap)
    in_d = np.bincount(kmap.astype(np.int64), minlength=N)
    trans = np.where(tau > 0)[0]
    if len(trans) == 0: return 0.0
    np.random.seed(seed)
    state = np.random.choice(trans, size=min(n_sample, len(trans)), replace=True)
    total_h, total_steps = 0.0, 0
    for _ in range(n_steps):
        state = kmap[state]
        alive = tau[state] > 0
        if not alive.any(): break
        ids = in_d[state[alive]]
        ids = ids[ids > 0]
        if len(ids) > 0:
            total_h += np.log(ids.astype(float)).mean()
            total_steps += 1
    return float(total_h / total_steps) if total_steps > 0 else 0.0


def run(bases=None, d=4):
    if bases is None:
        bases = list(range(2, 26))
    os.makedirs('results', exist_ok=True)
    rows = []
    print(f"{'b':>4} {'N':>8} {'Phi*':>8} {'mu1':>8} {'M1':>7} {'h':>8} {'|A|':>5} {'tau_max':>8}")
    print('-' * 62)
    for base in bases:
        N    = base ** d
        kmap = build_map(d, base)
        tau  = compute_tau(kmap)
        cycs = find_attractors(kmap)
        sz   = shell_sizes(kmap, tau)
        sz_t = sz[1:]
        phi  = shell_cheeger(sz_t) if len(sz_t) >= 2 else 0.0
        mu1  = shell_mu1(sz_t)    if len(sz_t) >= 2 else 0.0
        m1   = mean_first_passage(kmap, tau)
        h    = entropy_rate_sim(kmap, tau)
        n_att    = sum(len(c) for c in cycs)
        n_fixed  = sum(1 for c in cycs if len(c) == 1)
        n_cycles = sum(1 for c in cycs if len(c) > 1)
        cyc_lens = sorted(set(len(c) for c in cycs if len(c) > 1))
        tau_max  = int(tau.max())
        print(f"{base:>4} {N:>8} {phi:>8.4f} {mu1:>8.4f} {m1:>7.2f} {h:>8.4f} {n_att:>5} {tau_max:>8}")
        rows.append({'b': base, 'N': N,
            'n_transient': int((tau > 0).sum()),
            'Phi_star': round(phi, 6), 'mu1': round(mu1, 6),
            'M1': round(m1, 4), 'h': round(h, 6),
            'n_attractors': n_att, 'n_fixed': n_fixed,
            'n_cycles': n_cycles, 'cyc_lens': str(cyc_lens),
            'tau_max': tau_max})
    with open('results/graph_native_invariants.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    primes = {2,3,5,7,11,13,17,19,23}
    squares = {4,9,16,25}
    comps = {r['b'] for r in rows} - primes - squares
    def gs(key, grp):
        v = [r[key] for r in rows if r['b'] in grp]
        return (np.mean(v), np.std(v)) if v else (0,0)
    report = ["OP13 GRAPH-NATIVE INVARIANTS", "="*50,
        f"{'Group':<12} {'Phi*':>10} {'M1':>10} {'h':>10}", '-'*44]
    for name, grp in [('Prime', primes), ('Square', squares), ('Composite', comps)]:
        pm,ps = gs('Phi_star',grp); mm,ms = gs('M1',grp); hm,hs = gs('h',grp)
        report.append(f"{name:<12} {pm:.4f}±{ps:.4f}  {mm:.3f}±{ms:.3f}  {hm:.4f}±{hs:.4f}")
    b10 = next(r for r in rows if r['b']==10)
    all_h = [r['h'] for r in rows]
    z_b10 = (b10['h']-np.mean(all_h))/(np.std(all_h)+1e-12)
    report += ["", f"BASE-10: h={b10['h']:.4f} z={z_b10:+.3f}  M1={b10['M1']:.4f}  Phi*={b10['Phi_star']:.4f}",
        "STATUS: No thresholds assumed. Run permutation_test_primality.py for p-values."]
    txt = '\n'.join(report)
    print('\n'+txt)
    with open('results/op13_report.txt','w') as f: f.write(txt)
    print("\nSaved: results/graph_native_invariants.csv, results/op13_report.txt")
    return rows

if __name__ == '__main__':
    run()
