# Hyper-parameters: known defaults, and where ours deviate

Adopted 2026-09-26 (Dani): every arm starts from published defaults so a
result never depends on a value we invented. `PPOConfig.standard()` in
`lanerl_jax/train/ppo.py` encodes the table; `--preset standard` selects it.

| Parameter | Standard | Source | E01-E04 (legacy) | Note |
|---|---|---|---|---|
| Optimiser | Adam, eps 1e-5 | PureJaxRL `ppo_rnn.py` | same | |
| lr | 2.5e-4, linear anneal to 0 | PureJaxRL `ppo_rnn.py` | 3e-4 constant | `--lr-anneal` |
| Critic lr | same as actor, one Adam | PureJaxRL | same as actor | unequal `--critic-lr` rejected |
| gamma | 0.99 | PureJaxRL `ppo_rnn.py` | horizon-based: 120 s | changed by PPO-17; historical standard runs used 0.99917 at 10 Hz |
| GAE lambda | 0.95 | Schulman 2017, CleanRL | 0.97 | |
| Clip eps | 0.2 | PPO paper | same | no dual clip (PPO-17) |
| Value coef | 0.5, clipped value loss | CleanRL | same | |
| Entropy coef | 0.01/3 multiplying the unconditional 3-head sum | CleanRL (single head) | 0.001 | E05 used 0.01 on the sum: entropy pinned at 9.0, no learning (frozen 6.5/10.0 at 2.3M decisions). E06 uses 0.0033 |
| Max grad norm | 0.5 | CleanRL | 1.0 | |
| Epochs x minibatches | 4 x 4 | CleanRL Atari | same | gru: minibatches split sequences |
| Advantage normalisation | on, per minibatch | CleanRL | off (E01-E04) | off was a workaround for PPO-15 |
| KL early stop | none | PureJaxRL | 0.02 | `--target-kl` accepted but ignored for old launch commands |
| Rollout | 128 steps | CleanRL | same | |
| Decision rate | 10 Hz | Atari 15 Hz (skip 4), Five 7.5 Hz | same | |
| Core | GRU (LSTM in CleanRL/Five), BPTT over the rollout, reset at terminals | CleanRL `ppo_lstm`, OpenAI Five | MLP, no memory | `--core gru` |
| Init | orthogonal sqrt(2) trunk, 0.01 policy heads, 1.0 value | CleanRL detail 2 | same | |
| Envs | 10 servers x 2 agents (server-CPU bound) | CleanRL 8 envs | same | |

Historical farm reward (task-specific, used by E14/E21): +1 CS, -2 death,
5/10000 u lane-approach potential (undiscounted difference, REW-11),
+0.002 per xp (enemy minion deaths in range; SIDE-001). Every reward term must
name its team relative to the actor and have a zero-for-the-other-side test.

PPO-17 (reference PPO port): the live loss, GAE, trajectory-shuffled nested
update scans and clipped Adam (eps 1e-5) transcribe Chris Lu's
[PureJaxRL `ppo_rnn.py`](https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/purejaxrl/ppo_rnn.py),
with the existing LanePolicy network, factored heads, click mask, optional
KL-to-prior and detached critic. Agent-major batches and post-action dones
are layout adapters; carries reset before the next observation. Advantages
are normalised per minibatch; value clipping uses the actor clip epsilon;
all epochs/minibatches apply, and diagnostics average every step. Linear
annealing holds the rate constant within an update, exactly as the reference.
Legacy horizon/normalisation settings remain explicitly configurable, while
`standard()` uses gamma .99 and normalisation on. Historical split-optimizer
states cannot resume into this optimizer: use `--init-from` for the new arm.
The launch owner will make a new experiment ID from E14 plus `--detach-critic`
(from E12a; compare with E21 frozen 45.8 / 43.0). E14's JSON is unchanged;
it omits `--reward`, so explicitly select `--reward farm` to match the old
reward rather than today's `relative` default. Gamma .99 and the removal
of KL stopping are intentional differences from E21.
