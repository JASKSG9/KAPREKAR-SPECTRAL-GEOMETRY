
#!/usr/bin/env bash
# =============================================================================
# KSG-ACS2 UNIFIED BOOTSTRAP NODE v5.0
# TARGET: KAPREKAR SPECTRAL GEOMETRY / FRACTAL EP CASCADE
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# CONFIGURATION VARIABLES
# -----------------------------------------------------------------------------
WORKSPACE_DIR="${HOME}/KSG-EMPIRE-NODE"
GITHUB_REPO="https://github.com/JASKSG9/KAPREKAR-SPECTRAL-GEOMETRY.git"
HF_SPACE_URL="https://huggingface.co/spaces/Aqarion-TB13/KAPREKAR"
HF_USER="JASKSG9"

# -----------------------------------------------------------------------------
# TERMINAL UI (ASCII LOGGING)
# -----------------------------------------------------------------------------
log() {
    echo -e " \033[0;32m[KSG-SYS]\033[0m $1"
}
warn() {
    echo -e " \033[1;33m[WARNING]\033[0m $1"
}
err() {
    echo -e " \033[0;31m[FATAL]\033[0m $1"
    exit 1
}

clear
cat << "EOF"
 ██████╗ ███████╗ ██████╗ 
 ██╔═██╗ ██╔════╝ ██╔════╝ 
 ██████╔╝ ███████╗ ██║  ███╗
 ██╔══██╗ ╚════██║ ██║   ██║
 ██║  ██║ ███████║ ╚██████╔╝
 ╚═╝  ╚═╝ ╚══════╝  ╚═════╝ 
 ACS2 PRODUCTION DEPLOYMENT
EOF
echo "================================================================="

# -----------------------------------------------------------------------------
# 1. INITIALIZE WORKSPACE
# -----------------------------------------------------------------------------
log "Purging existing state and allocating node directory: $WORKSPACE_DIR"
rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# -----------------------------------------------------------------------------
# 2. CLONE GITHUB CORE (A24-KSG-NODE)
# -----------------------------------------------------------------------------
log "Cloning primary KSG repository..."
if git clone "$GITHUB_REPO" ksg-core; then
    log "GitHub core acquired successfully."
else
    err "Failed to clone GitHub repository. Verify network constraints."
fi

# -----------------------------------------------------------------------------
# 3. MIRROR HUGGING FACE AGENTS (M13-FLOW)
# -----------------------------------------------------------------------------
log "Allocating HF mirror directory..."
mkdir -p hf-agents

log "Executing raw cURL fetch for HF Space files..."
curl -s -L --fail "$HF_SPACE_URL/raw/main/agents.md" -o hf-agents/agents.md || warn "agents.md unreachable (Private Space)."
curl -s -L --fail "$HF_SPACE_URL/raw/main/code/flow/syntonic_gap.py" -o hf-agents/syntonic_gap.py || warn "syntonic_gap.py unreachable."

# Attempt full space clone if Hugging Face CLI is authenticated
if command -v huggingface-cli >/dev/null 2>&1; then
    log "Hugging Face CLI detected. Attempting full space download..."
    huggingface-cli download Aqarion-TB13/KAPREKAR --local-dir hf-agents || warn "CLI download failed. Proceeding with curl artifacts."
else
    warn "huggingface-cli not installed. Bypassing native HF clone."
fi

# -----------------------------------------------------------------------------
# 4. CONSTRUCT UNIFIED EXECUTION ENGINE
# -----------------------------------------------------------------------------
log "Compiling run_empire.sh executable..."
cat > run_empire.sh << 'EOF'
#!/usr/bin/env bash
set -e

echo "[EXEC] INITIATING KSG v5.0 SPECTRAL PIPELINE"
echo "------------------------------------------------"

# Target A24-KSG-NODE.PY
if [ -f "ksg-core/DOCS/A24-KSG-NODE.PY" ]; then
    echo "[EXEC] Running primary analytical node..."
    cd ksg-core/DOCS
    python3 A24-KSG-NODE.PY
    cd ../..
elif [ -f "ksg-core/A24-KSG-NODE.PY" ]; then
    echo "[EXEC] Running primary analytical node (Root)..."
    cd ksg-core
    python3 A24-KSG-NODE.PY
    cd ..
else
    echo "[WARN] A24-KSG-NODE.PY not found in standard directories."
fi

echo "------------------------------------------------"
echo "[EXEC] RUN COMPLETE. VERIFY /atlas AND /data DIRECTORIES."
EOF
chmod +x run_empire.sh

# -----------------------------------------------------------------------------
# 5. CONSTRUCT DASHBOARD OVERLAY
# -----------------------------------------------------------------------------
log "Generating Streamlit UI wrapper..."
cat > dashboard.py << 'EOF'
import streamlit as st
import subprocess
import os

st.set_page_config(layout="wide", page_title="KSG ACS2 NODE")
st.title("KSG SPECTRAL GEOMETRY DASHBOARD")
st.text("LOCKED INVARIANTS: μ₁ ≈ 5.24e-05 | τ* ≈ 0.714")

if st.button("EXECUTE A24 PIPELINE"):
    with st.spinner("Computing spectral gaps and τ-distributions..."):
        os.system("./run_empire.sh")
    st.success("Execution terminated successfully.")
    
    atlas_path_1 = "ksg-core/DOCS/atlas/ksg_atlas.png"
    atlas_path_2 = "ksg-core/atlas/ksg_atlas.png"
    
    if os.path.exists(atlas_path_1):
        st.image(atlas_path_1)
    elif os.path.exists(atlas_path_2):
        st.image(atlas_path_2)
    else:
        st.error("Atlas generation failed or path incorrect.")
EOF

# -----------------------------------------------------------------------------
# 6. SYSTEM STATUS REPORT
# -----------------------------------------------------------------------------
echo "================================================================="
log "BOOTSTRAP SEQUENCE TERMINATED."
echo " "
echo "   NODE PATH:  $WORKSPACE_DIR"
echo "   CORE CMD:   cd $WORKSPACE_DIR && ./run_empire.sh"
echo "   UI CMD:     cd $WORKSPACE_DIR && streamlit run dashboard.py"
echo " "
echo "================================================================="
