#!/usr/bin/env bash
# =============================================================================
# RunPod A100 Setup Script — ECS_MHD_2D
# Exact Coherent Structures in 2D MHD (JAX / CUDA 12)
# =============================================================================
# Usage:
#   chmod +x runpod_setup.sh && bash runpod_setup.sh
#
# This script pins all versions to match your local environment.yml so that
# the RunPod environment is byte-for-byte consistent with what you tested on.
# =============================================================================

set -e  # exit on any error

echo "============================================="
echo " ECS_MHD_2D  —  RunPod A100 Setup"
echo "============================================="

# ── 0. System deps ─────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq git curl wget htop nvtop 2>/dev/null || true

# ── 1. Core scientific stack (pinned to environment.yml) ───────────────────
echo "[2/6] Installing pinned core packages..."
pip install --quiet \
    "numpy==2.3.2" \
    "scipy==1.16.1" \
    "matplotlib==3.10.5" \
    "pillow==11.3.0" \
    "imageio==2.37.0" \
    "imageio-ffmpeg==0.6.0" \
    "opt-einsum==3.4.0" \
    "ml-dtypes==0.5.3" \
    "packaging==25.0"

# ── 2. JAX + CUDA 12 (pinned to environment.yml) ──────────────────────────
echo "[3/6] Installing JAX 0.7.0 with CUDA 12 support..."
pip install --quiet \
    "jax==0.7.0" \
    "jaxlib==0.7.0" \
    "jax-cuda12-plugin==0.7.0" \
    "jax-cuda12-pjrt==0.7.0"

# ── 3. NVIDIA CUDA wheels (pinned to environment.yml) ─────────────────────
echo "[4/6] Installing pinned NVIDIA CUDA wheels..."
pip install --quiet \
    "nvidia-cublas-cu12==12.9.1.4" \
    "nvidia-cuda-cupti-cu12==12.9.79" \
    "nvidia-cuda-nvcc-cu12==12.9.86" \
    "nvidia-cuda-nvrtc-cu12==12.9.86" \
    "nvidia-cuda-runtime-cu12==12.9.79" \
    "nvidia-cudnn-cu12==9.12.0.46" \
    "nvidia-cufft-cu12==11.4.1.4" \
    "nvidia-cusolver-cu12==11.7.5.82" \
    "nvidia-cusparse-cu12==12.5.10.65" \
    "nvidia-nccl-cu12==2.27.7" \
    "nvidia-nvjitlink-cu12==12.9.86" \
    "nvidia-nvshmem-cu12==3.3.20"

# ── 4. Utilities for long continuation runs & data I/O ────────────────────
echo "[5/6] Installing utility packages..."
pip install --quiet \
    "tqdm" \
    "h5py" \
    "pandas" \
    "psutil==7.0.0" \
    "networkx" \
    "plotly" \
    "scikit-sparse" \
    "jupyter" \
    "notebook" \
    "ipykernel"

# ── 5. Optional heavy solvers (comment out if not needed) ─────────────────
# PETSc is large; skip unless you're using a PETSc backend
# pip install petsc4py

# FEniCS / Dedalus require special system-level installs; use Docker images:
#   fenics:  dolfinx/dolfinx:stable
#   dedalus: dedalus-project/dedalus:latest

# ── 6. Clone repo (edit URL if private / SSH) ─────────────────────────────
echo "[6/6] Cloning ECS_MHD_2D repo..."
if [ ! -d "ECS_MHD_2D-MAI1" ]; then
    git clone https://github.com/michaelincorvaia/ECS_MHD_2D-MAI1.git   # update URL as needed
    cd ECS_MHD_2D-MAI1
    git submodule update --init --recursive
    cd ..
else
    echo "  Repo already present, skipping clone."
fi

# ── Verify JAX sees the A100 ───────────────────────────────────────────────
echo ""
echo "============================================="
echo " Verifying JAX GPU access..."
echo "============================================="
python - <<'EOF'
import jax
print(f"JAX version   : {jax.__version__}")
print(f"Devices found : {jax.devices()}")
print(f"Backend       : {jax.default_backend()}")

# Quick sanity — small matmul on GPU
import jax.numpy as jnp
x = jnp.ones((1024, 1024))
y = jnp.dot(x, x).block_until_ready()
print(f"1024x1024 matmul passed on: {y.device()}")
EOF

echo ""
echo "============================================="
echo " Setup complete."
echo ""
echo " A100 memory tips (add to your script or ~/.bashrc):"
echo "   export XLA_PYTHON_CLIENT_PREALLOCATE=false"
echo "   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90"
echo "   # ↑ lets JAX use 90% of the 80 GB HBM2e"
echo ""
echo " To launch Jupyter remotely:"
echo "   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
echo "============================================="
