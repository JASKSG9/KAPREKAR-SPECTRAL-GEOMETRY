"""
HTM Temporal Memory Integration for Kaprekar OISG System
Predictive sequence modeling + anomaly detection on state trajectories.
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# Simple lightweight Temporal Memory (TM) implementation
# (Production: replace with nupic.py or htm-community if preferred)
class SimpleTemporalMemory:
    """Minimal HTM Temporal Memory for sequence prediction & anomaly scoring."""
    
    def __init__(self, n_columns=512, cells_per_column=8, activation_threshold=12,
                 permanence_inc=0.1, permanence_dec=0.02, predicted_segment_dec=0.01):
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.n_cells = n_columns * cells_per_column
        
        self.active_cells = np.zeros(self.n_cells, dtype=bool)
        self.predictive_cells = np.zeros(self.n_cells, dtype=bool)
        
        # Simplified segment connections (column → previous active cells)
        self.connections = {}  # column -> set of previous cell indices
        self.permanences = {}  # (col, prev_cell) -> permanence
        
        self.activation_threshold = activation_threshold
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.predicted_segment_dec = predicted_segment_dec
        
        self.history = deque(maxlen=200)
        self.anomaly_scores = deque(maxlen=200)
    
    def _encode(self, state_id, basin, depth):
        """Simple scalar encoder → SDR (Sparse Distributed Representation)"""
        col = (state_id * 37 + basin * 17 + depth * 13) % self.n_columns
        cells = [col * self.cells_per_column + i for i in range(self.cells_per_column)]
        sdr = np.zeros(self.n_cells, dtype=bool)
        sdr[cells] = True
        return sdr, col
    
    def compute(self, state_id, basin=0, depth=0, learn=True):
        sdr, col = self._encode(state_id, basin, depth)
        
        # Predict from previous active cells
        predicted = np.zeros(self.n_cells, dtype=bool)
        if col in self.connections:
            for prev_cell in self.connections[col]:
                if self.permanences.get((col, prev_cell), 0) > 0.5:
                    predicted[prev_cell] = True
        
        # Bursting / active cells
        active = sdr.copy()
        anomaly = np.sum(predicted) == 0  # No prediction → anomaly
        
        # Learning
        if learn:
            if col not in self.connections:
                self.connections[col] = set()
            for prev in np.where(self.active_cells)[0]:
                self.connections[col].add(prev)
                key = (col, prev)
                self.permanences[key] = self.permanences.get(key, 0.0) + self.permanence_inc
        
        self.active_cells = active
        self.predictive_cells = predicted
        self.history.append((state_id, basin, depth))
        self.anomaly_scores.append(1.0 if anomaly else 0.3)
        
        return {
            "anomaly_score": float(np.mean(self.anomaly_scores)),
            "active_columns": int(np.sum(active) / self.cells_per_column),
            "prediction_match": float(np.sum(predicted & sdr) / np.sum(sdr))
        }


class KaprekarHTMMonitor:
    """Full integration: OISG Controller + GMM Drift + HTM TM"""
    
    def __init__(self, N=1024):
        from kaprekar_oisg_drift_monitor import KaprekarOISGMonitor  # assuming previous file
        self.oisg = KaprekarOISGMonitor(N=N)
        self.tm = SimpleTemporalMemory(n_columns=256, cells_per_column=8)
        self.trajectory = deque(maxlen=100)
    
    def update(self, x_prev, x_curr):
        oisg_report = self.oisg.update(x_prev, x_curr)
        
        # Extract context
        basin = 1 if x_curr > 50000 else 3  # simplified lookup
        depth = 3  # replace with real depth lookup
        
        tm_report = self.tm.compute(x_curr, basin=basin, depth=depth)
        
        combined_anomaly = (oisg_report["drift"]["drift"] or 
                           tm_report["anomaly_score"] > 0.6)
        
        self.trajectory.append(x_curr)
        
        return {
            "control": oisg_report["control"],
            "drift": oisg_report["drift"],
            "htm": tm_report,
            "combined_anomaly": combined_anomaly,
            "timestamp": datetime.now().isoformat()
        }


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    monitor = KaprekarHTMMonitor(N=1024)
    print("HTM Temporal Memory + OISG Monitor Demo\n")
    
    x = 500.0
    for t in range(60):
        drift = 25 if t > 30 else 3
        x_next = np.clip(x + drift + np.random.normal(0, 4), 0, 1023)
        report = monitor.update(x, x_next)
        x = x_next
        
        if t % 10 == 0 or report["combined_anomaly"]:
            h = report["htm"]
            print(f"t={t:2d}  Ξ={report['control']['Xi_eff']:.3f}  "
                  f"Bα={report['control']['B_alpha']:.1f}  "
                  f"Anomaly(HTM)={h['anomaly_score']:.3f}  "
                  f"Combined={'DRIFT' if report['combined_anomaly'] else 'ok'}")
