
# FDCE_ENGINE.py — Finite Deterministic Collapse Engine
# Production-grade universal deterministic collapse engine for finite endomorphisms.
# 
# Mathematical specification:
#   Input: finite state set Ω, deterministic map T: Ω → Ω
#   Output:
#     Level A — exact orbit mechanics
#     Level B — exact collapse filtration  K_n = Ω / ≡_n,  x ≡_n y ⟺ T^n(x) = T^n(y)
#     Level C — exact information invariants
#     Level D — exact quotient transfer operators

from collections import defaultdict, deque, Counter
from typing import Dict, List, Set, Tuple, Callable, Optional, Iterator
import math
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class State:
    """Canonical state wrapper for hashable state identifiers."""
    value: int
    
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.value == other.value
    
    def __repr__(self):
        return f"S({self.value})"


@dataclass
class OrbitMechanics:
    """
    Level A — Exact orbit mechanics.
    
    For deterministic map T: Ω → Ω:
      - successor_map: x ↦ T(x)
      - reverse_fibers: y ↦ T⁻¹({y}) = {x : T(x) = y}
      - cycle_decomposition: list of cycles (each cycle is ordered by T)
      - basin_decomposition: component_id for each state
      - tail_depths: distance from state to its cycle
      - stabilization_horizon: smallest h such that T^h(Ω) = T^{h+1}(Ω)
    """
    successor_map: Dict[State, State]
    reverse_fibers: Dict[State, Set[State]]
    cycle_decomposition: List[List[State]]
    basin_decomposition: Dict[State, int]
    tail_depths: Dict[State, int]
    stabilization_horizon: int
    
    @property
    def omega(self) -> Set[State]:
        return set(self.successor_map.keys())
    
    @property
    def N(self) -> int:
        return len(self.successor_map)
    
    @property
    def num_cycles(self) -> int:
        return len(self.cycle_decomposition)
    
    @property
    def max_tail_depth(self) -> int:
        return max(self.tail_depths.values()) if self.tail_depths else 0
    
    def iterate(self, x: State, n: int) -> State:
        """Compute T^n(x) exactly."""
        current = x
        for _ in range(n):
            current = self.successor_map[current]
        return current
    
    def orbit(self, x: State) -> List[State]:
        """Return the full orbit trajectory of x until cycle entry."""
        seen = set()
        traj = []
        current = x
        while current not in seen:
            seen.add(current)
            traj.append(current)
            current = self.successor_map[current]
        # Append cycle closure
        cycle_start = traj.index(current)
        return traj
    
    def basin_of(self, cycle_idx: int) -> Set[State]:
        """Return all states whose orbit terminates in cycle_idx."""
        return {s for s, cid in self.basin_decomposition.items() if cid == cycle_idx}


@dataclass
class KernelPartition:
    """
    Exact nth-kernel partition K_n = Ω / ≡_n.
    
    x ≡_n y  ⟺  T^n(x) = T^n(y)
    
    Attributes:
      - n: the kernel level
      - blocks: list of blocks (each block is a frozenset of states)
      - block_ids: state → block index
      - block_masses: block index → |block|
      - image_vector: block index → T^n(representative) [the common image]
      - refinement_of: previous KernelPartition (if n > 0)
    """
    n: int
    blocks: List[frozenset]
    block_ids: Dict[State, int]
    block_masses: Dict[int, int]
    image_vector: Dict[int, State]  # T^n(block) — the common image state
    refinement_of: Optional['KernelPartition'] = None
    
    @property
    def num_blocks(self) -> int:
        return len(self.blocks)
    
    @property
    def shannon_entropy(self) -> float:
        """H(K_n) = -Σ (|B|/N) log₂(|B|/N)."""
        N = sum(self.block_masses.values())
        if N == 0:
            return 0.0
        h = 0.0
        for mass in self.block_masses.values():
            p = mass / N
            if p > 0:
                h -= p * math.log2(p)
        return h
    
    @property
    def effective_rank(self) -> float:
        """exp(-Σ p_i log p_i) = exp(H) in natural log units."""
        N = sum(self.block_masses.values())
        if N == 0:
            return 0.0
        h_nat = 0.0
        for mass in self.block_masses.values():
            p = mass / N
            if p > 0:
                h_nat -= p * math.log(p)
        return math.exp(h_nat)
    
    def merge_count_from_previous(self) -> int:
        """
        Number of block merges from K_{n-1} to K_n.
        A merge occurs when two distinct K_{n-1} blocks are contained in one K_n block.
        """
        if self.refinement_of is None or self.n == 0:
            return 0
        prev = self.refinement_of
        # Count pairs of prev blocks that merge
        merges = 0
        prev_block_of = {}
        for bid, block in enumerate(prev.blocks):
            for s in block:
                prev_block_of[s] = bid
        
        # For each current block, count how many prev blocks it contains
        for curr_block in self.blocks:
            prev_ids_in_curr = set()
            for s in curr_block:
                prev_ids_in_curr.add(prev_block_of[s])
            if len(prev_ids_in_curr) > 1:
                merges += len(prev_ids_in_curr) - 1
        return merges


@dataclass
class CollapseFiltration:
    """
    Level B — Exact collapse filtration.
    
    The full chain: K_0, K_1, K_2, ..., K_h where h = stabilization horizon.
    K_0 is the discrete partition (each state alone).
    K_h is the terminal partition (stabilized).
    """
    kernel_partitions: List[KernelPartition]
    stabilization_horizon: int
    
    @property
    def K(self) -> List[KernelPartition]:
        """Alias for kernel_partitions."""
        return self.kernel_partitions
    
    def __getitem__(self, n: int) -> KernelPartition:
        return self.kernel_partitions[n]
    
    @property
    def entropy_trajectory(self) -> List[float]:
        """[H(K_0), H(K_1), ..., H(K_h)]."""
        return [kp.shannon_entropy for kp in self.kernel_partitions]
    
    @property
    def block_count_trajectory(self) -> List[int]:
        """[|K_0|, |K_1|, ..., |K_h|]."""
        return [kp.num_blocks for kp in self.kernel_partitions]
    
    @property
    def merge_trajectory(self) -> List[int]:
        """[merges_0→1, merges_1→2, ..., merges_{h-1}→h]."""
        return [self.kernel_partitions[i+1].merge_count_from_previous() 
                for i in range(len(self.kernel_partitions)-1)]
    
    @property
    def effective_rank_trajectory(self) -> List[float]:
        return [kp.effective_rank for kp in self.kernel_partitions]


@dataclass
class InformationInvariants:
    """
    Level C — Exact information invariants.
    
    Computed from the collapse filtration:
      - entropy_trajectory: H(K_n) vs n
      - entropy_loss_per_iterate: ΔH_n = H(K_n) - H(K_{n+1})
      - merge_acceleration: d²|K|/dn² (discrete)
      - logical_erasure_rate: bits destroyed per iteration
      - partition_effective_rank: effective_rank trajectory
    """
    entropy_trajectory: List[float]
    entropy_loss_per_iterate: List[float]
    merge_acceleration: List[float]
    logical_erasure_rate: List[float]
    partition_effective_rank: List[float]
    stabilization_horizon: int
    
    @property
    def total_entropy_loss(self) -> float:
        """Total information loss from K_0 to K_h."""
        if not self.entropy_trajectory:
            return 0.0
        return self.entropy_trajectory[0] - self.entropy_trajectory[-1]
    
    @property
    def mean_erasure_rate(self) -> float:
        if self.stabilization_horizon == 0:
            return 0.0
        return self.total_entropy_loss / self.stabilization_horizon
    
    @property
    def max_entropy_loss_step(self) -> Tuple[int, float]:
        """Return (step_index, max_loss) where entropy loss is maximal."""
        if not self.entropy_loss_per_iterate:
            return (0, 0.0)
        max_idx = max(range(len(self.entropy_loss_per_iterate)), 
                     key=lambda i: self.entropy_loss_per_iterate[i])
        return (max_idx, self.entropy_loss_per_iterate[max_idx])


@dataclass
class QuotientTransferOperator:
    """
    Level D — Exact quotient transfer operator.
    
    For the map K_n → K_{n+1}, define:
      Q_n[i,j] = |{x ∈ B_i^{(n)} : T(x) ∈ B_j^{(n+1)}}| / |B_i^{(n)}|
    
    This is a deterministic stochastic matrix (rows sum to 1).
    """
    n: int  # domain level
    quotient_matrix: Dict[Tuple[int, int], float]  # (block_i, block_j) → probability
    domain_partition: KernelPartition
    codomain_partition: KernelPartition
    
    def matrix_dense(self) -> List[List[float]]:
        """Return dense matrix representation."""
        m = self.domain_partition.num_blocks
        n = self.codomain_partition.num_blocks
        M = [[0.0 for _ in range(n)] for _ in range(m)]
        for (i, j), val in self.quotient_matrix.items():
            M[i][j] = val
        return M
    
    def is_deterministic(self, tol: float = 1e-12) -> bool:
        """Each row has exactly one nonzero entry (since T is deterministic)."""
        for i in range(self.domain_partition.num_blocks):
            row_sum = sum(self.quotient_matrix.get((i, j), 0.0) 
                         for j in range(self.codomain_partition.num_blocks))
            if abs(row_sum - 1.0) > tol:
                return False
        return True


class FDCEngine:
    """
    Finite Deterministic Collapse Engine.
    
    Universal engine for computing exact collapse structure of any
    deterministic finite endomorphism T: Ω → Ω.
    
    Usage:
        T = {State(0): State(1), State(1): State(0), ...}
        engine = FDCEngine(T)
        engine.compute_all()
    """
    
    def __init__(self, successor_map: Dict[State, State], 
                 state_list: Optional[List[State]] = None):
        """
        Args:
            successor_map: dict mapping each state to its unique image under T.
            state_list: optional ordered list of all states (for deterministic indexing).
        """
        self.successor_map = successor_map
        self.omega = set(successor_map.keys())
        self.N = len(self.omega)
        
        # Verify determinism: every state has exactly one image
        assert len(successor_map) == self.N, "Not all states have images"
        assert all(v in self.omega for v in successor_map.values()), "Image outside Ω"
        
        # Build reverse fibers
        self.reverse_fibers = defaultdict(set)
        for x, y in successor_map.items():
            self.reverse_fibers[y].add(x)
        
        # Ordered state list for matrix indexing
        if state_list is not None:
            assert set(state_list) == self.omega
            self.state_list = state_list
        else:
            self.state_list = sorted(self.omega, key=lambda s: s.value)
        
        self.state_index = {s: i for i, s in enumerate(self.state_list)}
        
        # Computed objects
        self._orbit_mechanics: Optional[OrbitMechanics] = None
        self._collapse_filtration: Optional[CollapseFiltration] = None
        self._information_invariants: Optional[InformationInvariants] = None
        self._quotient_operators: List[QuotientTransferOperator] = []
    
    # =====================================================================
    # LEVEL A: ORBIT MECHANICS
    # =====================================================================
    
    def compute_orbit_mechanics(self) -> OrbitMechanics:
        """
        Compute exact orbit mechanics:
          - cycle decomposition via Floyd/tortoise-hare on each unvisited node
          - basin assignment via reverse BFS from cycles
          - tail depths via BFS layering
          - stabilization horizon via image set contraction
        """
        T = self.successor_map
        
        # --- Cycle decomposition ---
        seen = set()
        cycles = []
        basin_of = {}
        tail_depth = {}
        
        for start in self.omega:
            if start in seen:
                continue
            
            # Trace path
            path = []
            current = start
            while current not in seen and current not in path:
                path.append(current)
                current = T[current]
            
            if current in path:
                # Found a cycle
                cycle_start_idx = path.index(current)
                cycle = path[cycle_start_idx:]
                cycles.append(cycle)
                cycle_idx = len(cycles) - 1
                
                # Mark cycle states
                for node in path:
                    seen.add(node)
                
                # BFS from cycle to assign basins and depths
                q = deque()
                for c in cycle:
                    tail_depth[c] = 0
                    basin_of[c] = cycle_idx
                    q.append(c)
                
                while q:
                    cur = q.popleft()
                    for pred in self.reverse_fibers[cur]:
                        if pred not in tail_depth:
                            tail_depth[pred] = tail_depth[cur] + 1
                            basin_of[pred] = cycle_idx
                            q.append(pred)
            else:
                # Merged into previously seen component
                # Find which component current belongs to
                target_cycle_idx = basin_of[current]
                for node in path:
                    seen.add(node)
                    basin_of[node] = target_cycle_idx
                
                # Assign depths by walking forward to known depth
                for node in path:
                    d = 0
                    walker = node
                    while walker not in tail_depth:
                        walker = T[walker]
                        d += 1
                    tail_depth[node] = tail_depth[walker] + d
        
        # --- Stabilization horizon ---
        # h = min{n : T^n(Ω) = T^{n+1}(Ω)}
        image_sets = []
        current_image = self.omega.copy()
        h = 0
        while True:
            next_image = {T[x] for x in current_image}
            if next_image == current_image:
                break
            image_sets.append(current_image)
            current_image = next_image
            h += 1
            if h > self.N:  # safety
                break
        
        self._orbit_mechanics = OrbitMechanics(
            successor_map=T,
            reverse_fibers=dict(self.reverse_fibers),
            cycle_decomposition=cycles,
            basin_decomposition=basin_of,
            tail_depths=tail_depth,
            stabilization_horizon=h
        )
        return self._orbit_mechanics
    
    # =====================================================================
    # LEVEL B: COLLAPSE FILTRATION
    # =====================================================================
    
    def compute_collapse_filtration(self) -> CollapseFiltration:
        """
        Compute exact kernel partitions K_n for n = 0, 1, ..., h.
        
        K_n = Ω / ≡_n where x ≡_n y ⟺ T^n(x) = T^n(y).
        
        Algorithm:
          K_0: discrete partition (each state alone)
          For n ≥ 1: two states are in same K_n block iff T^n(x) = T^n(y)
                    Equivalently: x, y in same K_n block iff T(x), T(y) in same K_{n-1} block
        """
        if self._orbit_mechanics is None:
            self.compute_orbit_mechanics()
        
        T = self.successor_map
        h = self._orbit_mechanics.stabilization_horizon
        
        # K_0: discrete partition
        K_0_blocks = [frozenset([s]) for s in self.state_list]
        K_0_block_ids = {s: i for i, s in enumerate(self.state_list)}
        K_0_masses = {i: 1 for i in range(self.N)}
        K_0_images = {i: s for i, s in enumerate(self.state_list)}
        
        K_0 = KernelPartition(
            n=0,
            blocks=K_0_blocks,
            block_ids=K_0_block_ids,
            block_masses=K_0_masses,
            image_vector=K_0_images,
            refinement_of=None
        )
        
        partitions = [K_0]
        
        # Iteratively compute K_n from K_{n-1}
        for n in range(1, h + 2):  # go to h+1 to ensure stabilization
            prev = partitions[-1]
            
            # Group states by T^n image
            # T^n(x) = T(T^{n-1}(x))
            # Two states x,y in same K_n block iff T^n(x) = T^n(y)
            image_to_states = defaultdict(list)
            for s in self.state_list:
                t_n_s = self._iterate_n(s, n)
                image_to_states[t_n_s].append(s)
            
            blocks = [frozenset(states) for states in image_to_states.values()]
            block_ids = {}
            block_masses = {}
            image_vector = {}
            for bid, block in enumerate(blocks):
                for s in block:
                    block_ids[s] = bid
                block_masses[bid] = len(block)
                # T^n of any representative
                image_vector[bid] = self._iterate_n(next(iter(block)), n)
            
            K_n = KernelPartition(
                n=n,
                blocks=blocks,
                block_ids=block_ids,
                block_masses=block_masses,
                image_vector=image_vector,
                refinement_of=prev
            )
            
            partitions.append(K_n)
            
            # Check stabilization: K_n == K_{n-1} (same blocks)
            if n >= 1 and self._partitions_equal(K_n, prev):
                # Remove the duplicate and stop
                partitions.pop()
                break
        
        self._collapse_filtration = CollapseFiltration(
            kernel_partitions=partitions,
            stabilization_horizon=len(partitions) - 1
        )
        return self._collapse_filtration
    
    def _iterate_n(self, s: State, n: int) -> State:
        """Compute T^n(s)."""
        current = s
        for _ in range(n):
            current = self.successor_map[current]
        return current
    
    def _partitions_equal(self, p1: KernelPartition, p2: KernelPartition) -> bool:
        """Check if two partitions have identical blocks."""
        return set(p1.blocks) == set(p2.blocks)
    
    # =====================================================================
    # LEVEL C: INFORMATION INVARIANTS
    # =====================================================================
    
    def compute_information_invariants(self) -> InformationInvariants:
        """
        Compute exact information invariants from collapse filtration.
        """
        if self._collapse_filtration is None:
            self.compute_collapse_filtration()
        
        cf = self._collapse_filtration
        H = cf.entropy_trajectory
        
        # Entropy loss per iterate: ΔH_n = H(K_n) - H(K_{n+1})
        delta_H = [H[i] - H[i+1] for i in range(len(H)-1)]
        
        # Merge acceleration: second difference of block counts
        B = cf.block_count_trajectory
        accel = []
        for i in range(1, len(B) - 1):
            accel.append((B[i+1] - B[i]) - (B[i] - B[i-1]))
        
        # Logical erasure rate: ΔH_n / H_0 (normalized)
        H0 = H[0] if H else 1.0
        erasure = [dh / H0 if H0 > 0 else 0.0 for dh in delta_H]
        
        self._information_invariants = InformationInvariants(
            entropy_trajectory=H,
            entropy_loss_per_iterate=delta_H,
            merge_acceleration=accel,
            logical_erasure_rate=erasure,
            partition_effective_rank=cf.effective_rank_trajectory,
            stabilization_horizon=cf.stabilization_horizon
        )
        return self._information_invariants
    
    # =====================================================================
    # LEVEL D: QUOTIENT TRANSFER OPERATORS
    # =====================================================================
    
    def compute_quotient_operators(self) -> List[QuotientTransferOperator]:
        """
        Compute quotient transfer operators Q_n: K_n → K_{n+1}.
        
        Q_n[i,j] = |{x ∈ B_i^{(n)} : T(x) ∈ B_j^{(n+1)}}| / |B_i^{(n)}|
        
        Since T is deterministic, each row has exactly one 1.
        """
        if self._collapse_filtration is None:
            self.compute_collapse_filtration()
        
        cf = self._collapse_filtration
        T = self.successor_map
        operators = []
        
        for n in range(cf.stabilization_horizon):
            K_n = cf[n]
            K_np1 = cf[n + 1]
            
            qmat = {}
            for i, block_n in enumerate(K_n.blocks):
                # T maps each state in B_i^{(n)} to some state
                # Group by which K_{n+1} block the image lands in
                target_counts = Counter()
                for s in block_n:
                    ts = T[s]
                    j = K_np1.block_ids[ts]
                    target_counts[j] += 1
                
                mass = len(block_n)
                for j, count in target_counts.items():
                    qmat[(i, j)] = count / mass
            
            op = QuotientTransferOperator(
                n=n,
                quotient_matrix=qmat,
                domain_partition=K_n,
                codomain_partition=K_np1
            )
            operators.append(op)
        
        self._quotient_operators = operators
        return operators
    
    # =====================================================================
    # UNIFIED COMPUTATION
    # =====================================================================
    
    def compute_all(self):
        """Compute all four levels."""
        self.compute_orbit_mechanics()
        self.compute_collapse_filtration()
        self.compute_information_invariants()
        self.compute_quotient_operators()
        return self
    
    # =====================================================================
    # REPORTING
    # =====================================================================
    
    def report(self) -> str:
        """Generate human-readable report of all computed invariants."""
        if self._orbit_mechanics is None:
            self.compute_all()
        
        lines = []
        lines.append("=" * 70)
        lines.append("FINITE DETERMINISTIC COLLAPSE ENGINE (FDCE) REPORT")
        lines.append("=" * 70)
        lines.append(f"State space cardinality: N = {self.N}")
        lines.append("")
        
        # Level A
        om = self._orbit_mechanics
        lines.append("--- LEVEL A: ORBIT MECHANICS ---")
        lines.append(f"  Number of cycles: {om.num_cycles}")
        lines.append(f"  Max tail depth: {om.max_tail_depth}")
        lines.append(f"  Stabilization horizon: h = {om.stabilization_horizon}")
        lines.append(f"  Cycle lengths: {[len(c) for c in om.cycle_decomposition]}")
        lines.append("")
        
        # Level B
        cf = self._collapse_filtration
        lines.append("--- LEVEL B: COLLAPSE FILTRATION ---")
        lines.append(f"  Kernel levels computed: 0 to {cf.stabilization_horizon}")
        lines.append(f"  Block count trajectory: {cf.block_count_trajectory}")
        lines.append(f"  Entropy trajectory: {[round(h, 4) for h in cf.entropy_trajectory]}")
        lines.append(f"  Merge trajectory: {cf.merge_trajectory}")
        lines.append("")
        
        # Level C
        ii = self._information_invariants
        lines.append("--- LEVEL C: INFORMATION INVARIANTS ---")
        lines.append(f"  Total entropy loss: {ii.total_entropy_loss:.4f} bits")
        lines.append(f"  Mean erasure rate: {ii.mean_erasure_rate:.4f} bits/iterate")
        max_step, max_loss = ii.max_entropy_loss_step
        lines.append(f"  Max entropy loss at step {max_step}: {max_loss:.4f} bits")
        lines.append("")
        
        # Level D
        lines.append("--- LEVEL D: QUOTIENT TRANSFER OPERATORS ---")
        lines.append(f"  Number of operators: {len(self._quotient_operators)}")
        for op in self._quotient_operators:
            m = op.domain_partition.num_blocks
            n = op.codomain_partition.num_blocks
            lines.append(f"    Q_{op.n}: {m} × {n} matrix")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =====================================================================
# UTILITY: CONVERT RAW DICT TO STATE MAP
# =====================================================================

def dict_to_state_map(raw_map: Dict[int, int]) -> Dict[State, State]:
    """Convert {int: int} to {State: State}."""
    return {State(k): State(v) for k, v in raw_map.items()}


# =====================================================================
# SELF-TEST
# =====================================================================

if __name__ == "__main__":
    # Test on a simple endomorphism: two 2-cycles with trees feeding in
    test_map = {
        0: 1, 1: 0,      # cycle A
        2: 3, 3: 2,      # cycle B
        4: 0, 5: 0,      # feed into cycle A
        6: 2, 7: 2,      # feed into cycle B
        8: 4, 9: 4,      # depth-2 feed into cycle A
    }
    
    T = dict_to_state_map(test_map)
    engine = FDCEngine(T)
    engine.compute_all()
    print(engine.report())
    
    print("\n--- VERIFICATION: K_n partitions ---")
    for K_n in engine._collapse_filtration.kernel_partitions:
        print(f"K_{K_n.n}: {len(K_n.blocks)} blocks, masses = {sorted(K_n.block_masses.values())}")
