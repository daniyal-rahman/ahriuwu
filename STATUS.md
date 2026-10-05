# Status

**2026-10-05 — repository cut to the modern world (MODERN-025).** This branch now holds only the 26.19 JAX
world (`lanerl_jax/modern/`), its tools (`ops/modern/`) and specs (`docs/modern/`). The legacy C#-server
project is at tag `pre-modern-cleanup-2026-10-05`. The package is self-contained: the trainer's policy,
PPO and learner live in `modern/rl/`; the camera/screen model in `modern/screen.py`; sight rays in
`modern/rays.py`.

**State of the world.** Full map verified bit for bit across the MODERN-024 restructure
(`ops/modern/golden.py`); the 88-unit top-lane layout is game-identical to the 216-unit one. Fidelity
checks against 26.9 replays and Riot timelines: docs/modern/REPLAY_FIDELITY.md, MECHANICS_AUDIT.md.

**Open.**
- GPU timing of the full map vs the top-lane layout and the legacy 4.20 lane: queued (desktop was in
  Windows), job `THROWAWAY-m024-gpu-bench`.
- No training experiment on the modern world yet; decisions open: reward weights, input latency
  (`--action-delay-ticks`), start state, shop handling.
