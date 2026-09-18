# What stands between this and a "good" agent — gaps + fixes (2026-07-22)

Grounded in the test battery just run. "Good" = an **imagination-RL** policy (DreamerV4 Phase 3), *not* behavior cloning. BC can only imitate humans, on a lossy view — it's a **floor**, not the goal. The goal is a policy that improves by trial-and-error inside its own world-model dream, shaped by reward. That one distinction is why the whole analysis points at one link.

## The chain, and where each link measures out

| Link | Role toward "good" | Measured state (this session) | Gap | Priority |
|---|---|---|---|---|
| **Tokenizer** (206M) | compress frames → latents the policy sees | recon **~26–29 dB** (soft/blurry) | fine detail (HP, cooldown #s, minion aggro) may be lost | MED |
| **World model** (114M) | **dream action-conditioned futures** (imagination trains *in* the dream) | best case (135, real actions): dream **beats persistence by only ~2 dB**, drifts h1=22.5→h32=9.0 latent, ~9 dB under the tokenizer ceiling. Unconditioned (179): *below* persistence. | **dream not accurate enough for RL — THE gap** | **HIGHEST** |
| **Reward** | define "good play" | solo Δgold, `gold_scale=1e-3` placeholder | gold ≠ winning; untuned | MED (post-WM) |
| **Imagination RL** (Phase 3) | **the thing that makes it good** | code exists, synthetic-validated; **BLOCKED** (WM can't dream) | gated entirely on the world model | HIGHEST (gated) |
| **BC policy** (current agent) | warm-start / demo floor | movement imitation works (28–71% bin-acc, window-dep, reactive: ctx8>ctx16); casting = **calibration** (probe AUC 0.77–0.89) | it's imitation — caps at human-mimicry | LOW for "good" |

## THE gap: the world model is a marginal dreamer

Imagination RL trains the policy on **imagined** trajectories. If the dream is only ~2 dB better than *freezing the frame* and has drifted badly by 8 frames, the policy is learning inside an unreliable fantasy. That is exactly why Phase 3 is blocked and why BC is all we have.

Two things the tests nailed down:
1. **Action-conditioning matters and works** — 135 (with actions) beats persistence; 179 (no actions) fell below it. So conditioning on actions is necessary and helps.
2. **But even the conditioned model is only marginal** — ~2 dB over persistence, ~9 dB under the ceiling, drifting. LoL is a hard domain (fast, high-entropy, camera-driven) and this is a 114M model trained on 125 replays + YT on consumer hardware.

## Fixes, ranked by leverage toward "good"

1. **Action-conditioned world-model retrain (the paper recipe) — the critical path.** Train the WM with `use_actions=True` on the *mixed* corpus (replays w/ real actions + YT with the `no_action_embed` placeholder). We never actually did this — 179 ran fully unconditioned, and 135 was replays-only. This is the single highest-leverage move. **Gate:** at imagination horizon H=8, does the dream beat persistence by a *useful* margin and stay coherent? (At H=8, 135 is ~12.5 dB — borderline; the retrain needs to push this up.)
2. **Use a short imagination horizon (H≈4–8), not 32.** The dream is decent for the first ~4–8 frames (16–22 dB) and only collapses later. Standard Dreamer imagination is H=8–16; we don't need 32-frame dreams. This makes the *current* WM closer to usable.
3. **Scale the world model (capacity + data, on cloud).** The paper says WMs need high capacity; ours is "medium." More params + more action-labeled data (see IDM below) + the alternating-length / τ_ctx-curriculum / Mamba-temporal items from the roadmap all target rollout accuracy.
4. **More action-labeled data via IDM (roadmap 1.C).** Pseudo-label the 906 YT games → 10–50× more action-conditioned training. Helps both the WM's action-conditioning and the rare abilities (ults/summoners have only 300–600 examples; Q/E/AA already have 10–18k).
5. **Reward tuning + enrichment.** Tune `gold_scale` on real return magnitudes; consider adding kills/XP/objective terms — but only once imagination runs.
6. **Cheap BC/demo wins in parallel** (don't block on the WM): per-ability calibrated cast thresholds (Q/E/AA are just miscalibrated, not unlearned), focal loss, and **feeding the HUD to the policy** (cooldowns/mana — the agent currently casts blind).
7. **Tokenizer fidelity (only if it becomes the ceiling).** Bigger bottleneck or multi-scale (roadmap 2.F/3.I). Movement already works on the current latents, so this is not the bottleneck yet.

## Honest expectations
DreamerV4 reached ~diamond in *Minecraft* — slower, blockier, more forgiving than LoL. At this scale (114M WM, 125 replays, consumer GPUs), "good" realistically means **a coherent laning phase** — sensible movement, last-hitting, occasional correct casts — not winning ranked games. And the world-model dreaming is the genuine research risk: it may not get accurate enough at this scale, in which case the ceiling is "good BC + calibrated casting," which is still a real, demoable agent.

## The concrete path
1. **Action-conditioned WM retrain** (mixed labeled+unlabeled) on the 5080 → re-run the rollout gate at H=8.
2. If it clears the gate → **Phase-3 imagination** (PMPO policy + value in dreams) → the good policy.
3. If it doesn't → scale on cloud, or accept the BC-tier ceiling.
4. **In parallel, cheap:** calibrate cast thresholds + HUD-to-policy → make the demo agent actually cast.

**One line:** *everything downstream is gated on the world model learning to dream action-conditioned LoL futures accurately — it currently dreams only marginally-better-than-nothing, action-conditioning is the proven lever, and the honest next move is the action-conditioned mixed retrain, gated on an H=8 rollout that clears persistence by a real margin.*
