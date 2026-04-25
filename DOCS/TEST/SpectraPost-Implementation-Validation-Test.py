def full_spectral_validation():
    """Audit v3.1 compliance test."""
    L, N = kaprekar_tau_graph_laplacian()
    
    # Core invariants
    assert abs(eigh(L, eigvals_only=True)[1] - 0.1624262417339861) < 1e-12
    assert abs(np.sum(N) - 9990) < 1e-6
    assert abs(np.trace(L) - 7) < 1e-10
    
    print("✅ Full KSG spectral validation PASSED")
    return True

full_spectral_validation()
