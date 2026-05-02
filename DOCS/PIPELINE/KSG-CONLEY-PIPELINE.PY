# Pseudocode for Kaprekar Conley index
for each quotient cell C_i:
    # Build IIN around all x with Q(x) = C_i
    N_i = {x : Q(x) ∈ ball(C_i, ε)}
    # Exit set (points leaving under K)
    N_i^- = {x ∈ N_i : K(x) ∉ int(N_i)}
    # Entrance set  
    N_i^+ = N_i  N_i^-
    # Index pair homology → symbol s_i
for each cell C_i:
    N_i = {x : d(Q(x), (g1,g2)) < ε}      # Fiber nbhd
    L_i = {x ∈ N_i : K(x) ∉ N_i}          # Exit set  
    index_i = homology(N_i/L_i)           # ℤ-graded vector space
    
assert all(index_i ≠ index_j for i ≠ j)  # 55 distinct indices
# Verify: H_*(I(C_i)) distinguishes all 55 cells
# Transition: s_i → s_j iff tilde{K}(C_i) = C_j
