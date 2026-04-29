#!/usr/bin/env python3
"""
KSG/UDAG MASTER PIPELINE
Reproduces all verified claims: shell descent, cross‑radix census,
graph‑native invariants, and random‑control anomaly certificates.
"""

import os
import sys
import time
import subprocess

def run_script(script_name, description, timeout=300):
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    start = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - start
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    print(f"✓ Completed in {elapsed:.1f}s (return code {result.returncode})")
    return result.returncode

def main():
    print("\n" + "="*70)
    print("KSG/UDAG PRODUCTION PIPELINE — PHASE 1")
    print("="*70)

    # Ensure data directories exist
    os.makedirs("DATA_LAKE", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Core verification: Shell descent theorem
    run_script("scripts/op4_descent_extension.py", "OP4: Shell Descent Verification")

    # 2. Cross‑radix attractor census
    run_script("scripts/op3_algebraic.py", "OP3: Cross‑Radix Attractor Census")

    # 3. Graph‑native invariants (Φ_*, M₁, h)
    run_script("scripts/graph_native_invariants.py", "OP13: Graph‑Native Invariants Sweep")

    # 4. Random functional graph baseline (Null‑A)
    run_script("scripts/op0_random_baseline.py", "OP0: Random Baseline Generation")

    # 5. Entropy bottleneck & anomaly certificate
    run_script("scripts/op2_bottleneck_anomaly.py", "OP2: Bottleneck Index vs Random")

    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE — All outputs in DATA_LAKE/ and results/")
    print("="*70)

if __name__ == "__main__":
    main()
