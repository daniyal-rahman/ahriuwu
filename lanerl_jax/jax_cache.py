"""Persistent XLA compilation cache shared by tests, benchmarks and trainers.

A full modern-world tick takes 2-4 min to compile; the cache makes every later
process with the same program, backend and flags load it in seconds.
Location: ``$LANERL_JAX_CACHE``, else the desktop's ``/scratch`` (fast local
NVMe, same directory as ``train/wave_scenario_train.py``), else ``~/.cache``.
"""
from __future__ import annotations

import os

SCRATCH_CACHE = "/scratch/lanerl-jax-compilation-cache"


def enable_compile_cache(path: str | None = None) -> str:
    """Point JAX's persistent compilation cache at ``path`` (see module doc); returns it."""
    import jax
    if path is None:
        path = os.environ.get("LANERL_JAX_CACHE") or (
            SCRATCH_CACHE if os.access("/scratch", os.W_OK)
            else os.path.expanduser("~/.cache/lanerl-jax-compilation-cache"))
    os.makedirs(path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    return path
