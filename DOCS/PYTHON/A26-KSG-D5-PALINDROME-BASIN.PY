# d5_palindrome_basin_final.py
# Requires basin_id dict from your d=5 enumeration
basin_id = {}  # load your mapping here

pals = [a*10000 + b*1000 + c*100 + b*10 + a 
        for a in range(1,10) for b in range(10) for c in range(10) 
        if not (a == b == c)]

counts = {1:0, 2:0, 3:0, 4:0}
for p in pals:
    b = basin_id.get(p)
    if b in counts: counts[b] += 1

print(counts)  # {1:0, 2:0, 3:36, 4:40}
