def compute_tau_histogram(d=4):
    """Exhaustive: returns [383, 576, 2400, 1272, 1518, 1656, 2184]"""
    N = 10**d
    cycle_set = {6174} if d == 4 else set()
    
    def kaprekar_step(n):
        s = f"{n:0{d}d}"
        return int("".join(sorted(s, reverse=True))) - int("".join(sorted(s)))
    
    def tau_depth(start):
        seen, n, depth = set(), start, 0
        while n not in cycle_set and n not in seen:
            seen.add(n); n = kaprekar_step(n); depth += 1
        return depth
    
    # Non-repdigits only
    states = [i for i in range(N) if len(set(f"{i:0{d}d}")) > 1]
    histogram = [0] * 8
    
    for state in states:
        tau = tau_depth(state)
        histogram[tau] += 1
    
    return histogram[1:8]  # τ=1..7

tau_dist = compute_tau_histogram()
target = [383, 576, 2400, 1272, 1518, 1656, 2184]
assert tau_dist == target
print(f"✅ τ-histogram: {tau_dist}")
