# REPLAY_FIDELITY.md — 26.19 modern sim vs real 16.9 replays

**Status (2026-10-03).** Report-only fidelity pass of the modern sim against client-memory recordings of
**145 NA games** (`/mnt/nfs/datasets/lol_replays_16_9_772`, two of 147 games have no `raw_mem.json`). Client
16.9 = patch **26.9**; the sim targets **26.19**. Every disagreement was checked against the 26.9→26.19 notes
(the wiki's Homeguard history has no change after 26.1; Recall timing has no SR change), so the
mismatches below are sim bugs, not patch differences.

This adds to the economy replay oracle (`ops/modern/replay_oracle_extract.py`, `test_economy_oracle.py`,
[ECONOMY_IMPLEMENTATION.md](ECONOMY_IMPLEMENTATION.md)). That oracle already checks starting gold,
the ambient-gold phase, the death-timer table and scaling, kill/assist gold, fountain regen, level-up HP and
HP growth. Its tests and the Riot match-v5 stat oracle still pass on the current code (25 passed).

## Tool

`ops/modern/replay_fidelity.py`:
- `extract`: standard library only. Reads `raw_mem.json` and the `labels.json` recall actions, never
  `frames/`. Writes one `<match>.json.gz` per game.
- `analyze`: imports `economy`, `core.stat_pipeline` and `world.config` constants (no tick compile)
  plus the Riot extract, and prints the table below (`--json` writes it).

```
ops/login_capped.sh 8G 3 .venv-jax/bin/python ops/modern/replay_fidelity.py extract \
    /mnt/nfs/datasets/lol_replays_16_9_772/NA1_* --out /mnt/nfs/shared/TMP_fid --jobs 3   # ~25 min
ops/login_capped.sh 6G 1 .venv-jax/bin/python ops/modern/replay_fidelity.py analyze --out /mnt/nfs/shared/TMP_fid
```

`lanerl_jax/modern/tests/test_replay_fidelity_oracle.py` runs the same checks on one smoke-test game
(about 10 s) and is skipped when the dataset is absent. Its Homeguard check tests the client rule (bonus % MS before the soft caps); the strict xfail that
recorded the sim bug was removed when the bug was fixed (MODERN-022).

## Results

| # | Check | Method | n | Sim | Replay | Verdict |
|---|---|---|---|---|---|---|
| 1 | Death duration | hp = 0 run (≥5 s) → first hp > 0, vs `E.death_time(level, t)` | 6,025 | BRW × (1+TIF) | 97.8% within 0.15 s, median −0.02 s (one sample) | **OK** |
| 2 | Ambient gold, whole game | `gold_total` slope while dead (death +2.5 s → respawn −0.3 s, windows with no step > 3 g) | 4,515 | 2.04 g/s (`ambient_payments`) | median 2.042 g/s; 2.03 / 2.043 / 2.043 / 2.042 in the 0–10 / 10–20 / 20–30 / 30+ min buckets | **OK** (rate holds all game) |
| 3a | Level-1 max HP (Garen) | `hp_max` at load vs 26.19 record + shards {0, 10, 20, 65, 75} | 145 | base 690 | 690 / 700 / 710 / 755; 99.3% explained | **OK** |
| 3b | Level-1 max HP (Jax) | same | 4 | base 650 | 660, 715 ×3 | **OK** (small n) |
| 3c | Garen/Jax records 26.19 vs 16.9 bins | all `ChampionBase` fields | — | — | identical | **OK** (growth already oracle-checked) |
| 4a | Respawn state | first alive sample after death | 6,025 | full HP at `FOUNTAINS` | hp = hp_max 99.1%; ≤1 u from (394, 461) / (14340, 14391) 92.5% (rest: first sample already moving) | **OK** |
| 4b | Recall duration | recorded champion: `labels.json` recall cast → teleport | 830 | channel **8.0 s** (4.0 empowered) | **8.50 s** (n = 748), **4.50 s** empowered (n = 73) | **MISMATCH** |
| 4b′ | (all 10 heroes) | stand-still before a fountain teleport (no damage) | 10,423 | 8.0 | mode 8.56 s, p1 8.46 s; 4.58 s bucket | confirms 4b |
| 4c | Homeguard floor speed, before 14:00 (recall exits) | speed 4.5–8 s after leaving the 1100 radius vs post-Homeguard plateau v0 | 2,756 | `v0 + 0.40·base_ms` (post-cap) | `softcap(v0·1.40)`: \|err\| 0.5 u (90% within 10 u); sim form 13.5 u (24%) | **MISMATCH** |
| 4d | Homeguard floor, after 14:00 | same | 2,661 | `v0 + 0.65·base_ms` | `softcap(v0·1.65)`: \|err\| 6.5 u; sim form **61.5 u** (1% within 10 u); implied bonus 0.647 | **MISMATCH** (magnitude OK) |
| 4e | Respawn exits (Deathguard) | same, after a respawn | 1,210 / 1,425 | Homeguard 0.40 / 0.65 (`deathguard_ms` 0.75 is not wired) | implied 0.400 / 0.647; same as recall | **OK** for the magnitude (no separate 75% buff in 26.9); same composition bug |
| 5 | Base movement speed | modal 1 s straight-line speed in lane, 90 s → first recall/death | 1,142 heroes | Garen 340, Jax 350 (26.19 = 16.9) | Garen modes 340 (43), 343 (26, Celerity 1%); Jax 350; 79% of all heroes within 2 u of Riot `movementSpeed` at 2:00 | **OK** |

**Confounders.**
- **2:** 75% of windows are within ±1 payment, which is sample phase. About 20% carry extra small steps of 3 g or
  less from other income, so the median is the statistic to use.
- **4c–4e:** v0 is measured in the same trip, with the same items and runes. The pre-cap model uses
  `raw ≈ v0·(1+h)`, which ignores the 1% Celerity and 2% shard additive terms; this error is under 3 u. After
  14:00, v0 includes higher MS and the residual is larger, but the sim form is about 10× worse.
- **4b:** the measured value is a 0.5 s Recall **cast time** plus the 8 s channel (the wiki lists 0.5 s cast +
  8 s channel; empowered 0.5 + 4). The 16.9 data contain no mid-quest 4 s recalls; that reward was removed in
  26.9.

## Likely sim bugs

**Status (MODERN-022): bugs 1–3 are fixed.** Homeguard, Ghost/Heal and Gustwalker now enter the STAT pipeline as bonus % MS before the soft caps (`world.tick._move`); Recall has `RECALL_CAST = 0.5` before the channel and the damage grace is measured on the whole recall (`economy.recall_step`).

1. **Homeguard MS skips the soft caps** (`lanerl_jax/modern/world/tick.py:1007-1009`). The code adds
   `(homeguard_ms + hg_bonus) * base_ms` after `st.move_speed`. The client treats it as **bonus % MS inside
   the raw sum, before the soft caps**: `softcap((base+flat)·(1+add%+hg))`. After 14:00 the sim is about 60 u
   too fast (for example 640 vs 576 at v0 = 420). The same line also applies `s_out.bonus_ms_pct` (Ghost, Heal
   and similar) after the caps. That was not measured here, but it is likely the same issue.
2. **Recall takes 8.5 s, not 8.0 s** (`economy.py:45` `RECALL_CHANNEL`, and `world/tick.py:1378`
   empowered 4.0). There is a 0.5 s cast before the channel (movement locked, 4.5 s empowered).
3. **Code reading, not measured:** `recall_step`'s damage grace uses `RECALL_CHANNEL - 0.1`
   (`economy.py:533`) even when `channel = 4.0`. For an empowered recall, the last-0.1 s grace therefore
   never applies.
4. **Spec note:** `deathguard_ms` (75%, `economy.py:505`) is defined but unused. The replays show respawn
   exits get exactly Homeguard's 80%→40% / 150%→65%, so not wiring it is correct for 26.9. ECONOMY §10.3 should
   say so.
