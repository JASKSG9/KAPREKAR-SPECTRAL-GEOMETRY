"""
Spectral Graph Conductance Analysis for d=5 Kaprekar Functional Graph
Adapted for directed / functional graphs (out-degree=1).
"""

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

def build_kaprekar_graph(N=10000):
    def kaprekar_step(n, d=5):
        s = str(n).zfill(d)
        return int(''.join(sorted(s, reverse=True))) - int(''.join(sorted(s)))
    
    G = nx.DiGraph()
    for i in range(N):
        nxt = kaprekar_step(i)
        G.add_edge(i, nxt)
    return G


def compute_conductance_metrics(G, basin_arr=None):
    """Adapted conductance for functional graph + normalized Laplacian."""
    print("Computing spectral conductance metrics...")
    
    # Undirected projection for standard Cheeger
    G_und = G.to_undirected()
    L = nx.normalized_laplacian_matrix(G_und)
    L = L.tocsr()
    
    try:
        eigenvalues = eigsh(L, k=5, which='SM', return_eigenvectors=False)
        spectral_gap = eigenvalues[1]  # λ2
        print(f"Spectral gap (λ2) ≈ {spectral_gap:.6f}")
    except:
        spectral_gap = np.nan
        print("Eigen decomposition failed (graph too large)")
    
    # Simple edge conductance heuristic (cut size)
    if basin_arr is not None:
        # Inter-basin conductance
        cross = 0
        for u, v in G.edges():
            if basin_arr[u] != basin_arr[v]:
                cross += 1
        phi = cross / G.number_of_edges() if G.number_of_edges() > 0 else 0
        print(f"Inter-basin conductance (heuristic) ≈ {phi:.6f}")
    
    return {"spectral_gap": spectral_gap}


if __name__ == "__main__":
    # Small subgraph for demo
    G = build_kaprekar_graph(5000)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    metrics = compute_conductance_metrics(G)
