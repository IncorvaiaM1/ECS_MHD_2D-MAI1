"""
A100 JAX configuration — import this at the top of any ECS_MHD script
before any other jax imports to squeeze the most out of the 80 GB HBM2e.

Usage:
    import a100_config  # must be FIRST import
    import jax
    import jax.numpy as jnp
    ...
"""

import os

# ── Memory allocation ───────────────────────────────────────────────────────
# Don't preallocate the full GPU memory on startup; grow as needed.
# This lets you run multiple processes without OOM on the first one.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Allow JAX to use up to 90% of the 80 GB HBM2e (leaves ~8 GB for CUDA overhead).
# Raise to 0.95 if you're running a single large continuation and need max memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

# Enable TF32 matmuls on A100 (huge throughput boost for float32 work, ~10x).
# Safe for MHD Newton-Krylov unless you need strict float32 bit-exactness.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")

# Use the CUDA allocator (faster for JAX than the default).
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# ── JAX flags (set before jax is imported) ─────────────────────────────────
# x64 mode — needed for Newton-Krylov / GMRES convergence at tight tolerances.
# Comment out if your code is explicitly float32-only.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# ── Confirm at import time ──────────────────────────────────────────────────
import jax
print(f"[a100_config] JAX {jax.__version__} | backend: {jax.default_backend()} | devices: {jax.devices()}")
