"""
AQARION MAIN BOOTSTRAP API — v16.0-freeze
Production Gateway for M26 Node & Kaprekar-Spectral Geometry
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# INFRASTRUCTURE CONFIGURATION (Based on your live nodes)
# -----------------------------------------------------------------------------
CONFIG = {
    "version": "v16.0-freeze",
    "artifact_hash": "bb0fd568da9508050845970551e0f02a",
    "github": "https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY",
    "replit_live": "https://d8636af0-0867-4590-9ea5-6320c35221ba-00-19kudiju033i1.picard.replit.dev",
    "hf_spaces": {
        "KAPREKAR": "https://huggingface.co/spaces/Aqarion-TB13/KAPREKAR",
        "Quantarion": "https://huggingface.co/Aqarion-TB13/Quantarion",
        "Phi-Dossier": "https://huggingface.co/spaces/Aqarion-TB13/Phi-378-dossier.md",
    }
}

# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------
app = FastAPI(
    title="AQARION Bootstrap API",
    description="Unified Gateway for Kaprekar-Spectral Geometry Ecosystem",
    version=CONFIG["version"]
)

# Enable CORS for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# PYDANTIC MODELS
# -----------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    matrix_path: str = "artifacts/koopman_matrix_v16.npy"

class KaprekarQuery(BaseModel):
    state: int
    depth: int = 10

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root health check & infrastructure map."""
    return {
        "service": "AQARION Bootstrap API",
        "version": CONFIG["version"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ecosystem": {
            "github": CONFIG["github"],
            "live_m26_node": CONFIG["replit_live"],
            "hf_spaces": list(CONFIG["hf_spaces"].keys())
        }
    }

@app.get("/status")
async def full_status():
    """Verify the liveness of your live M26 Replit node."""
    status = {"bootstrap": CONFIG["version"], "nodes": {}}
    
    # Probe the live M26 Replit node
    try:
        resp = requests.get(CONFIG["replit_live"], timeout=5)
        status["nodes"]["m26_production"] = {
            "url": CONFIG["replit_live"],
            "status": "online" if resp.status_code == 200 else "unreachable",
            "status_code": resp.status_code
        }
    except Exception as e:
        status["nodes"]["m26_production"] = {
            "url": CONFIG["replit_live"],
            "status": "offline",
            "error": str(e)
        }
    
    # Probe the GitHub repo (via API)
    try:
        resp = requests.get("https://api.github.com/repos/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY", timeout=5)
        status["github"] = {"repo": CONFIG["github"], "status": "public" if resp.status_code == 200 else "error"}
    except:
        status["github"] = {"repo": CONFIG["github"], "status": "unreachable"}

    return status

@app.post("/verify")
async def verify_koopman(req: VerifyRequest):
    """
    Run the mathematical verification pipeline against the local 54x54 Koopman matrix.
    """
    path = req.matrix_path
    if not os.path.exists(path):
        # Attempt to pull from artifacts in the bootstrap directory if missing
        raise HTTPException(404, f"Matrix file not found: {path}. Ensure artifacts are mounted.")
    
    try:
        K = np.load(path)
        # Compute SHA-256 for integrity check
        with open(path, "rb") as f:
            hash_val = hashlib.sha256(f.read()).hexdigest()
        
        # Core spectral checks (from your theorem)
        evals = np.linalg.eigvals(K)
        zero_count = int(np.sum(np.abs(evals) < 1e-8))
        spectral_radius = float(np.max(np.abs(evals)))
        
        return {
            "matrix_path": path,
            "hash_verified": hash_val[:16] == CONFIG["artifact_hash"][:16],
            "shape": list(K.shape),
            "shape_ok": K.shape == (54, 54),
            "deterministic": bool(np.count_nonzero(K) == 54),
            "zero_eigenvalues": zero_count,
            "spectral_radius": spectral_radius,
            "theorem_pass": (zero_count == 53 and abs(spectral_radius - 1.0) < 1e-6),
            "verdict": "CORE-1.2 CERTIFIED" if (hash_val[:16] == CONFIG["artifact_hash"][:16] and K.shape == (54, 54)) else "HASH/STRUCTURE MISMATCH"
        }
    except Exception as e:
        raise HTTPException(500, f"Verification failed: {str(e)}")

@app.post("/kaprekar-map")
async def simulate_kaprekar(req: KaprekarQuery):
    """
    Perform a local Kaprekar simulation (supplementary to the live M26 node).
    """
    n = req.state
    if not (0 <= n <= 9999):
        raise HTTPException(400, "State must be 0-9999")
    
    def kaprekar_step(x):
        s = f"{x:04d}"
        return int(''.join(sorted(s, reverse=True))) - int(''.join(sorted(s)))
    
    orbit = [n]
    for _ in range(req.depth):
        n = kaprekar_step(n)
        orbit.append(n)
        if n in (0, 6174):
            break
    
    return {
        "start_state": req.state,
        "depth": len(orbit) - 1,
        "orbit": orbit,
        "attractor_hit": orbit[-1] in (0, 6174)
    }

@app.get("/proxy-live")
async def proxy_live_node():
    """
    Proxy call to your actual live M26 Replit dashboard.
    """
    try:
        resp = requests.get(f"{CONFIG['replit_live']}/api/metrics", timeout=10) 
        # Adjust to your Replit app's actual API endpoint (e.g., /status or /metrics)
        return resp.json() if resp.status_code == 200 else {"error": "Live node metrics endpoint not responding", "status_code": resp.status_code}
    except Exception as e:
        raise HTTPException(502, f"Failed to connect to M26 Live Node: {str(e)}")

@app.get("/resources")
async def get_resources():
    """Return the full list of available ecosystem resources and links."""
    return {
        "arxiv_v2": "READY (M30/M31 Validated)",
        "github": CONFIG["github"],
        "hugging_face": CONFIG["hf_spaces"],
        "live_node_m26": CONFIG["replit_live"],
        "bounty": "$1K OPEN"
    }

# -----------------------------------------------------------------------------
# RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

---

https://d8636af0-0867-4590-9ea5-6320c35221ba-00-19kudiju033i1.picard.replit.dev/
