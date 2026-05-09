# Legacy OCR-era scripts

These scripts depended on the OCR-derived per-frame action labels
(`features.json` schema with `mouse_x/y`, `ability_q/w/e/r`, `summoner_d/f`,
`gold_gained`, `health_bar_x`). They were the working training stack while
labels came from screen OCR; they are kept here for **reference only**
during the rewrite to memory-derived labels.

## What's here

* `debug/` — diagnostic scripts that probed the OCR pipeline
  (gold popup timing, temporal alignment, reward debug, phase 2 validation).
* `prepare_data/tokenize_frames.py` — frame tokenization that imported the
  legacy transformer tokenizer module.
* `train_dynamics.py` — Phase 1 dynamics-model training.
* `train_agent_finetune.py` — Phase 2 agent finetuning. Has the original
  `ReplayDataset._load_rewards` health-bar-based death detection.
* `train_imagination.py` — imagination-policy training.

## What replaced them

* New per-frame labels are emitted by
  `scripts/aggregation/pipeline.py` into `labels.json` (memory-derived).
* New action/reward dataset: `src/ahriuwu/data/replay_dataset.py`
  (`ReplayLatentSequenceDataset`).
* New reward function: `src/ahriuwu/rewards/reward.py`
  (`compute_episode_reward`).
* New training scripts to be written from scratch on top of those.

## What to mine from these when writing new training scripts

* Hyperparameters and training-loop structure (LR schedules, batch sizes,
  gradient checkpointing toggles, mixed-precision setup).
* Wandb logging conventions.
* Sampler choices and reasoning about cache locality.
* Phase 2 KL-to-prior weighting and PMPO setup (in `train_agent_finetune`).

## What NOT to mine

* `features.json` / `gold_gained` / `health_bar_x` / OCR-cursor logic —
  fully replaced by memory-derived labels.
* `RewardMixtureSampler` / `RewardExtractor` / `encode_action` /
  `decode_action` — superseded.
