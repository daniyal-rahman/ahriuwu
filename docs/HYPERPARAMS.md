# Hyper-parameters: known defaults, and where ours deviate

Adopted 2026-09-26 (Dani): every arm starts from published defaults so a
result never depends on a value we invented. `PPOConfig.standard()` in
`lanerl_jax/train/ppo.py` encodes the table; `--preset standard` selects it.

| Parameter | Standard | Source | E01-E04 (legacy) | Note |
|---|---|---|---|---|
| Optimiser | Adam, eps 1e-5 | CleanRL PPO detail 3 | same | |
| lr | 2.5e-4, linear anneal to 0 | CleanRL `ppo`/`ppo_lstm` | 3e-4 constant | `--lr-anneal` |
| Critic lr | same as actor (shared trunk) | CleanRL | same as actor | |
| gamma | horizon-based: 120 s at 10 Hz (0.99917) | OpenAI Five used 180-360 s horizons; Atari 0.99 | same | kept |
| GAE lambda | 0.95 | Schulman 2017, CleanRL | 0.97 | |
| Clip eps | 0.2 | PPO paper | same | dual clip 3.0 kept (Ye et al. 2020) |
| Value coef | 0.5, clipped value loss | CleanRL | same | |
| Entropy coef | 0.01 per head = 0.01/3 on our 3-head sum | CleanRL (single head) | 0.001 | E05 used 0.01 on the sum: entropy pinned at 9.0, no learning (frozen 6.5/10.0 at 2.3M decisions). E06 uses 0.0033 |
| Max grad norm | 0.5 | CleanRL | 1.0 | |
| Epochs x minibatches | 4 x 4 | CleanRL Atari | same | gru: minibatches split sequences |
| Advantage normalisation | on, per minibatch | CleanRL | off (E01-E04) | off was a workaround for PPO-15 |
| KL early stop | none | CleanRL default | 0.02 | |
| Rollout | 128 steps | CleanRL | same | |
| Decision rate | 10 Hz | Atari 15 Hz (skip 4), Five 7.5 Hz | same | |
| Core | GRU (LSTM in CleanRL/Five), BPTT over the rollout, reset at terminals | CleanRL `ppo_lstm`, OpenAI Five | MLP, no memory | `--core gru` |
| Init | orthogonal sqrt(2) trunk, 0.01 policy heads, 1.0 value | CleanRL detail 2 | same | |
| Envs | 10 servers x 2 agents (server-CPU bound) | CleanRL 8 envs | same | |

Reward (task-specific, not a published default): +1 CS, -2 death,
5/10000 u lane-approach potential (undiscounted difference, REW-11),
+0.002 per xp (enemy minion deaths in range; SIDE-001). Every reward term must
name its team relative to the actor and have a zero-for-the-other-side test.
