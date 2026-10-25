# Behaviour cloning from decoded replays (modern world)

**2026-10-22: v4 replay-only pilot passed conversion validation; learning validation is next.**
Historical v1-v3 observation datasets remain rejected. Sections10–13 describe the
audit and interim holds; section14 describes the replacement and current evidence.

Written 2026-10-17. Code: `lanerl_jax/modern/bc/` (`replay.py` reads a game, `labels.py` labels actions,
`convert.py` builds the arrays, `dataset.py` builds the dataset, `targets.py` the click loss, `train.py` the
supervised trainer). Data: the native-event exports of the rofl decoder. The first dataset is in section 8.

**The world is used only as the observation formatter.** No simulation step runs in producing training
data: each decision's replay state is written into a `ModernState` and the world's own fog
(`views.visibility`) and observation (`obs.build_modern_observation`) read it, so the arrays have exactly
the layout, normalisation, selection and fog rules the policy sees in RL (`obs` profile `modern-world-v1`,
actions `modern-world-v2`). The one other use is a check: `modern_orders_from` reads each label back on that
state.

The goal: a first Garen policy cloned from real Garen top laners in exactly the network and action space
`modern/train.py` trains, so that it can start (`--init-from`) and tether (`--kl-prior`) the 1v1 lane RL.

## 1. The two sides

**The policy** (`rl/policy.LanePolicy`, 10 decisions a second): entities (32, 20), a pad mask, self (56),
global (6); 20 buttons and a 96 x 54 screen cell. The trainer masks `buy`, `sell`, `use_item` and
`level_q..r` (no choice head; loadouts auto-level). The world's episode runs from a 60 s start at a hold
point on the top lane to 600 s, with Garen and Jax, Doran's Blade and a potion, Flash and Teleport.

**The data**: 353 Garen TOP games against 78 opponents and 13 Garen-vs-Jax games, patch 26.19 (client
16.19), the same patch the world implements (`rofl_bc_2619_garen_top_native_20261015`, decoder `b2614e72`;
`rofl_bc_2619_garen_vs_jax_native_20261014`, `f706f5ce`). Both predate the decoder tag
`2619-received-countdowns-v2`: their cooldown fields are known-wrong and are not read. The window is game
clock 0:30 to 14:00 (8100 decisions a game).

## 2. Observation: how each feature is filled

| feature | source | quality |
|---|---|---|
| unit rows: position relative to me | decoder position model (`waypoint_integrated_2619_v2`); structures at the world's placements, checked against the replay's turret placements (within 10 units) | 8-40 units median error |
| unit rows: health fraction | `stat.mHP / stat.mMaxHP`, zero-order hold | exact |
| unit rows: type, team, minion subtype | family, received `identity.type` (`ranged` = caster), minion team from the side of the river its first path starts on | exact |
| which units get a row (fog) | the world's `visibility` over the rebuilt state: sight radii, walls and brush of the 26.19 map (`fog="rays"`) | the world's rule; no wards (not decoded), no jungle monsters, no other champions |
| which units get a row (screen) | `target_on_screen` in the lane frame, the locked camera | exact under the policy's camera (not the player's) |
| self: lane s, n | position model in the world's lane frame | approximate |
| self: health, dead, level, current gold, cs, xp (level progress) | replicated stats; last hits by the damage-notice rule | exact up to last-hit attribution |
| self: Q/W/E/R cooldown fraction | **rebuilt from received casts** with Garen's cooldown table, the replay's ability haste and the kit's rules (`convert.spell_state`); premature presses counted (`cooldown_conflicts`) | sim-consistent; no refunds |
| self: AD, AP, armour, MR, move speed, attack range | the world composes a Garen with starting items, then the replay's replicated totals **replace** them (`patched_from_replay`) | exact; items and runes included |
| self: own and enemy champion one-hot | the world config built for the (blue, red) pair: the opponent's kit where it is one of the 14 registered, else zeros ("unknown opponent") | exact |
| self: mana, shield | replicated (Garen: 0) | exact |
| self: summoner D/F cooldowns | rebuilt from summoner casts, unhasted; Flash is always slot D (`replay.canonical_summoners`) | approximate |
| self: in_combat | damage notices between the champion and an enemy champion within 5 s | exact |
| self: recalling | recall cast to 8.5 s or the next action | approximate |
| self: quest, trinket charges, unspent skill points | quest 0; trinket the world's starting charge; skill points from level and ranks | partial |
| global: clock | t / 600 s (the trainer's default episode) | exact; above 1 after 10:00 |
| global: enemy visible, time since the enemy cast Q/W/E/R | from the rows; witnessed casts normalised by the kit's rank-1 cooldowns (a default row for an unknown kit) | exact timing |
| `champion_stats` (side output, not a policy input) | raw replicated stats of both laners / `STAT_SCALE`, the enemy's zero while it has no row | exact |

## 3. Action labelling

| received | label |
|---|---|
| a path with 2+ points | `move`, click on the path's end (a click on a hostile unit decodes to an attack) |
| an executed basic attack on a new target | `move`, click on the unit (`click_on_unit`) |
| repeats on the same target | `noop` |
| cast slot 0/1/2 | `q`/`w`/`e`: self casts, no click (`click_valid` False) |
| cast slot 3 | `r`, click on the target |
| cast slot 4/5 | `summoner_d`/`summoner_f` (Flash = D), click at the cast record's cursor vector |
| cast slot 12 | `ward`, click at the cursor vector |
| cast slot 13 | `recall` |
| item slots 6-11, other slots | not representable (masked out of the loss) |
| a 1-point path at the champion while walking, no attack within 0.6 s | `stop` (heuristic) |
| nothing | `noop` |

Several actions in one decision: cast > attack > stop > path, latest among equals. Off-screen clicks are
pulled back to 97% of the way to the edge. `attack_move` is never labelled. `uses_screen` is the trainer's
definition (`actions.screen_usage`); the click loss applies where `click_valid`, with a plain target on a
unit and a sloped one on the ground (`targets.py`).

**Timing.** Labels use the server's receipt time; the observation is lagged (`--lag`, default 1 decision =
100 ms) so the policy is a decision-maker with a reaction time. RL must delay its observations the same way
(not built yet).

**Scope.** `label_valid = in_scope & representable`: the champion alive, within the trainer's 1400-unit
corridor around the top lane extended to the fountains, no champion but the two laners within 2000 units.

**Held labels (dataset v2 onwards).** Labelling only the instant of a click leaves 72% of decisions as
noop while the champion is walking or auto-attacking, and the instant a human re-clicks is not predictable
from the observation: run v1 (events only) reached val button NLL 0.631 against a prior of about 0.69 and
predicted noop 99.6% of the time. So between events the labeller now fills the order still active: a move to
the destination the champion is walking to (the screen cell is recomputed each decision as it approaches),
or a click on the unit it is auto-attacking. A held move ends on arrival (80 units), after 0.3 s standing
still, or at the next order; a held attack when the target is lost or further than 1500 units; both at a
stop, recall or R. `held` marks them in the dataset and the trainer weights them `--hold-weight` (0.3) so the
decision moments still dominate. In the sim, re-issuing the same order each decision is harmless. On one
game this lifts click supervision from 21% to 87% of valid decisions and noop to 11%; the price is more
clamped clicks (12%, far destinations pulled to the screen edge) and 6% of ground clicks that the world's
hit-test reads as an attack because a unit sits under the destination.

## 4. Checks built into the converter

Per game (`label_check` in the `.json`): attacks and R casts resolving through `modern_orders_from` to an
order on the intended unit, that unit having an observation row, move clicks' distance from the received
path's end, stops decoding as stops.

## 5. Output format

`<game>.npz` (leading axis = decision): `entities` (T, 32, 20), `entity_pad_mask` (T, 32), `self_vec`
(T, 56), `global_vec` (T, 6), `champion_stats` (T, 2, 11), `button`, `screen_x`, `screen_y`, `uses_screen`,
`click_valid`, `click_on_unit`, `label_valid`, `in_scope`, `obs_index`, `t`; observation arrays stored
already lagged. `<game>.json`: provenance (dataset, decoder commit, converter commit, world and action
profiles), counts, label checks, what was patched or not from the replay.

    python -m lanerl_jax.modern.bc.convert OUT GAME_DIR ... [--lag 1]
    python -m lanerl_jax.modern.bc.dataset plan|convert|finalize OUT      (slurm/modern_bc_dataset.sbatch)

## 6. The domain gap

| gap | world | replays | handling |
|---|---|---|---|
| matchup | Garen vs Jax by default; 14 kits | 78 opponents, 142 games vs a registered kit, 13 vs Jax | kit one-hot where registered, zeros otherwise; dropout on the enemy one-hot in training |
| items | starting items only; `buy` masked | humans buy | stats are patched from the replay, so the clone sees item-grown stats it will not get in the world |
| third champions, wards, monsters | none | present | scope filter; wards absent from the rebuilt state |
| camera, clicks | locked, lane-aligned | free | labels in the policy's camera |
| cooldowns | the world's | known-wrong fields | rebuilt from casts |

## 7. From BC to RL

`modern/train.py --init-from <bc checkpoint> --kl-prior <coef> --observation-lag 1` with `enemy_scale .5`
(own income plus the lead at 2:1), `xp_scale` a fifth of gold per unit, approach shaping 0, and the converter's
world (`--lanes 2 --no-jungle --no-objectives --lane-structures`).

`--observation-lag N` makes the trainer act on the world as it was N decisions earlier: `ModernEnvState`
carries a ring of N observations (oldest first); each decision reads the oldest entry, pushes the current
state's observation, and the bootstrap value uses the same lagged entry. A fresh episode's ring holds the start
state N times, so its first N decisions see the start state, which matches how the datasets are lagged
(within the game, never across the window start). The datasets use lag 1 (100 ms), so BC checkpoints must be
run with `--observation-lag 1`; lag 0 shows them a world 100 ms newer than anything they were trained on.

The gate, `python -m lanerl_jax.modern.bc.gate CKPT --opponent afk|frozen|self` (`slurm/modern_bc_gate.sbatch`),
runs the checkpoint frozen in the converter's world, sampling actions as the trainer does, for one full
episode per env (64 envs, 1:00 to 10:00 by default) and writes next to the checkpoint the per-episode blue CS,
gold, XP, deaths, kills, mean lane distance and button shares plus red's CS, gold and XP. What to read: CS and
deaths against an afk Jax (can it farm at all, does it walk into the turret), then against a frozen Jax policy
from the sim thread's runs (does it trade), compared with the humans in the dataset: at 10:00 the 357 Garens hold median 75 CS (p10 57, p90 94), level 8,
and have died 1.4 times on average (28% not at all, 41% twice or more; counted from the self vector's dead
flag, so a flickering flag would inflate it).

## 8. The dataset

`/mnt/nfs/shared/bc_modern_v1` (converter 86f11199, decoder exports b2614e72 for the 344 top-lane games and
f706f5ce for the 13 Garen-vs-Jax games; built 2026-10-17 by Slurm jobs 7808-7810, about 3 min per 15 games on
2 CPUs). Observation lag 1 decision, window 0:30-14:00, 8100 decisions per game.

| | all | train | val |
|---|---|---|---|
| games | 357 | 321 | 36 |
| decisions in window | 2,891,700 | 2,600,100 | 291,600 |
| label-valid decisions | 2,326,688 (80.5%) | 2,089,094 | 237,594 |
| decisions with a click | 615,341 | 552,597 | 62,744 |

Nine games were skipped for ending before 14:00 (surrenders and remakes, 1-14 min long). 19.5% of
decisions carry no label; the reasons overlap, and as shares of all decisions they are: a third champion within 2000 units 12.2%,
Garen outside the lane corridor 8.1%, dead 5.1%, cast with no representable target 0.14%. Per game the
label-valid count runs 4,628 to 8,003 (median 6,571).

Button shares over the label-valid decisions (identical to three decimals between train and val):
noop 71.7%, move 26.4%, stop 0.81%, e 0.46%, q 0.42%, w 0.13%, recall 0.07%, ward 0.05%, summoner_f 0.02%,
summoner_d 0.01%, r 0.012%. `attack_move`, `level_*`, `buy`, `sell`, `use_item` and `control_ward` never
occur (attacks are labelled as unit clicks, item use is not decoded). A majority-class clone scores 71.7%
button accuracy, so accuracy is not the metric; per-button recall and NLL are.

Label checks aggregated over all games: attacks resolved through `modern_orders_from` 32,336 of 32,586
(99.2%), stops 18,937 of 18,937, R casts 274 of 289, 48,559 clicks clamped back on to the screen (1.7%),
64,718 orders superseded within a decision (2.2%), 1,104 casts recorded while the rebuilt cooldown said the
spell was down (0.04%, the haste and rank reconstruction is not exact).

Opponents: 78 distinct champions, 300 melee and 57 ranged games. 142 games face a registered kit
(Camille 28, Fiora 21, KSante 15, Aatrox 14, Jax 13, Renekton 12, Darius 9, Ambessa 9, Yone 7, Nasus 5,
Jayce 5, Malphite 4); the other 215 get a zeroed enemy one-hot. The val split is drawn per game at 10%, so a
few opponents are absent from val (Camille, Ambessa, Irelia, Gnar) and Shen is over-represented (4 of 11).
The same players recur across games, so val measures held-out games, not held-out players.

`/mnt/nfs/shared/bc_modern_v2` (converter 73a9dd2d, jobs 7846-7848, 2026-10-17) is the same 357 games with held
labels: of the 2,326,688 label-valid decisions 1,290,581 (55%) are held (92,445 of them unit clicks), click
supervision covers 1,905,922 decisions (82%), and the button shares become move 81.8%, noop 16.2%, stop 0.81%,
with the cast and summoner shares unchanged. The event-label checks are identical to v1.

## 9. Runs

| run | dataset | epochs | val button NLL | event NLL | click NLL / dist | gate vs afk Jax at 10:00 |
|---|---|---|---|---|---|---|
| v1 `modern-bc-v1-e1` (job 7836, 51 min) | v1, events only | 30 | 0.631 (prior ~0.69, majority acc 0.717) | n/a | 7.19 / 16.1 cells | CS 35 (humans 75), deaths 1.9, kills 1.0, lane distance 1100 |
| v2 `modern-bc-v2-hold` (job 7849, 18 min) | v2, held labels, hold weight 0.3 | 30 (best 20) | 0.347 (majority acc 0.816, acc 0.895) | 0.594 | 6.59 / 13.1 cells (held clicks 6.31 / 11.8) | CS 41, deaths 3.6, kills 1.0, lane distance 1250 (job 7933) |
| v3 `modern-bc-v3-castw` (job 7934, 16 min) | v2, class weights x5 casts/summoners/ward, x3 stop | 24 (best 23) | 0.580 weighted (acc 0.882); unweighted q 2.37, e 1.78, stop 3.0 | 0.987 weighted | 6.63 / 13.4 cells | CS 37, deaths 2.5, kills 1.0, lane distance 1680 (job 7965) |
| v4 `modern-bc-v4-garen-v2obs` (job 8297, 15 min) | `bc_modern_garen_v2obs` (the v2 games under observation profile `modern-world-v2`: identity embeddings + kit descriptors, docs/modern/MULTICHAMP.md), v3 recipe | 24 (best 19) | 0.596 weighted (acc 0.876); recall q 0.04, w 0.09, e 0.30, recall 0.06 (v3: 0.06, 0.05, 0.28, 0.13) | 1.032 weighted | 6.68 / 13.7 cells | seed 0: CS 46.0, deaths 2.9, lane distance 1110 (job 8316); seed 1: CS 44.7, deaths 3.1, 1160 (8327). v3 seed 1: CS 38.1, deaths 2.3, 1620 (8328) |
| v4 `modern-bc-v4-jax-v2obs` (job 8444, 15 min) | `bc_modern_jax_v2obs`: 367 Jax-top games (decoder v7, received countdowns), v2 profile, v3 recipe | 24 (best 24) | 0.621 weighted (acc 0.869); recall q 0.06, w 0.21, e 0.14, recall 0.17 | | 6.46 / 12.4 cells | Jax vs afk Garen: CS 30.6, deaths 2.9, lane 1280, buttons w .035 e .009 q .007 (job 8479). Garen v4 vs this Jax (job 8478): Garen CS 39.7 / deaths 2.2 / kills 2.8 / gold 3027, Jax CS 23.9 / gold 2464 |
| v4 `modern-bc-v4-garen-noenemyid` (job 8515) | as v4 Garen, `--enemy-id-dropout 1.0` (ablation, MULTICHAMP §7) | 24 (best 20) | 0.594 weighted (acc 0.878); recall q 0.01, w 0.06, e 0.28 | | | not gated |

v1: the button head predicts noop 99.6% of the time (move recall 0.8%); the one-per-click labels do not
carry enough signal, see section 3. In the sim it samples noop 77% and move 21%, farms half the human CS
against an afk opponent and dies twice in nine minutes, with deaths uncorrelated with its lane distance
(turret and minion deaths from aimless walking rather than wandering off). The gate itself runs 64 full
episodes in 4 minutes on the 5080 (`gate-afk-*.json` next to the checkpoint).

v2: held labels change the picture: move recall 0.97, noop recall 0.64, event-only button NLL 0.594. Casts
are still never the argmax (recall 0 for q/w/e/r, summoners, ward, stop), but their conditional NLL moves
well below the prior (q 4.23 vs 5.5, e 3.49 vs 5.4, stop 3.95 vs 4.8, summoner_d 7.2 vs 9.2), so a sampled
policy casts at roughly the right moments more often than chance. Epochs ran at 32 s throughout (v1
alternated 32 s and 130 s with the same code). Kit-known games score 0.340, unknown 0.350; melee and ranged
are equal. In the sim v2 samples move 71%, noop 25%, stop 2%, q and e under 1% each: CS 41 against an afk
Jax (v1 35, humans 75) but 3.6 deaths (v1 1.9). It walks more and farther than v1 and pays for it under the
turret and in the minion wave; it still does not trade.

v3 repeats v2 with class weights x5 on casts, summoners and ward and x3 on stop (the reported NLLs are
weighted, so only the per-button values divided by their weight compare with v2). Casts now reach the argmax:
recall e 0.28, recall(button) 0.13, q 0.06, w 0.05, ward 0.05; r, summoners and stop still 0. The move head
pays a little: move recall 0.952 (v2 0.968), noop 0.634 (0.643), click distance 13.4 (13.1). Weighted
conditional NLLs fall further (q 2.37 vs 4.23, e 1.78 vs 3.49, summoner_d 4.96 vs 7.18). In the sim v3
samples q 4.5%, stop 3.7%, e 1.2% (v2: under 1%), dies less than v2 (2.5 vs 3.6) but farms a little less
(CS 37 vs 41) and strays farther from the lane (1680 vs 1250). None of the three clones is near the human
CS 75 / 1.4 deaths; the gap is the sim's un-cloned skills (last-hit timing under a turret, trading) that
the RL phase has to supply. `record.py` + `render.py` turn one episode into a video for eyeballing this.



## 10. Replay-only source audit (2026-10-22)

Dani requires all BC to come from replay evidence, including fields that were
previously filled by the simulator without advancing it. No simulator tick is
necessary for simulator defaults to contaminate an observation. Existing datasets
and v1-v5 checkpoints are mixed-source artifacts, not replay-only artifacts.
They remain preserved; v6 has not been started.

| Consumer / feature | Current source and audit result | Required replacement |
|---|---|---|
| Entity inclusion, enemy visibility, witnessed-cast memory | `convert.state_of` calls `world.views.visibility`, using sim sight radii, brush/walls and an incomplete reconstructed world. Replay positions are spectator information, not team visibility. | Decoded team visibility, or an explicitly different masked spectator profile. Do not treat all received entities as player-visible. |
| Structures | `_unit_fields` copies config positions/kinds, marks every structure alive with full health (1/1), then replaces only received turrets. Missing turret HP becomes 1. Inhibitors/Nexuses and missing turrets remain invented. | Received identities, positions, health and life; mask unknown structures/fields. |
| Own Q/W/E/R timers | Previously `received_spell_state` filled unmatched casts from sim/catalog tables and `spell_state` supplied Garen kit windows/cooldowns for old exports. | Both fallback paths are removed. Received timers are held/decayed only after receipt. A new cast invalidates the prior timer until the next same-slot message. Missing/invalid timers are NaN, never zero. This is a display countdown, not proof of activation eligibility for charges/recasts. Slot/hash alignment remains unverified. |
| Timer normalization, enemy cast-age normalization | v1 reads kit cooldowns. v2/v3 call `champion_table.cooldowns`; `_cooldown_ranks` replaces catalog values with sim-kit values for registered champions, and otherwise uses a default if absent. | Replay duration with availability, or a revised fixed numeric scaling shared with RL. A catalog name does not certify replay provenance. |
| Champion descriptors | `champion_table.descriptors` mixes Data Dragon base stats, decoder asset catalogs and sim-kit cooldown overrides. | Separately sourced replay/build metadata with explicit availability; remove all kit overrides and invented defaults from the BC path. Identity/role/tier must retain their declared metadata source. |
| Own combat stats | The formatter first composes starting inventory, runes and kit stats, then patches the consumed AD/AP/armor/MR/move speed/range/mana/shield columns. Replicated fields still use unmarked default zeros; AD/AP arithmetic is a converter interpretation. | Build these inputs directly from qualified replicated fields and retain availability; no temporary game-state composition. |
| Summoner timers | `_own_fields` uses simulator `SUMMONER_COOLDOWN`, including a 300s default, from cast timestamps rather than received summoner countdowns. | Replay slot 4/5 countdowns, preserving canonical key mapping and unknowns. |
| Recall | Simulator `RECALL_CAST + RECALL_CHANNEL` plus next-action heuristic. | Received channel state/cancel/end evidence; unknown otherwise. |
| XP progress, unspent skill points | `obs` applies simulator XP thresholds, level caps, skill-point and kit starting-rank/bonus rules. | Replicated progress/skill-point fields or explicit masks. Ranks alone do not prove unspent points. |
| Quest, trinket charges | Initial-state quest zero/incomplete and starting trinket charge. | Replicated state with availability; unknown is not zero/empty/incomplete. |
| Labels | Casts previously selected targeting rules from sim kits, otherwise catalog rules. | Removed: use each received cast's targeting mode, target or endpoint. Missing unit target/unknown targeting remains unrepresentable; a unit mode without a target no longer implies self-cast. Received executions, inferred stop/held orders and canonical-camera cells remain proxies for decisions, not original key/click recordings. |
| Position, HP, rank, gold, mana and other scalar gaps | Replay reader uses as-of stats; `_fill` also backfills the first future position. No availability masks reach the current policy. | Preserve field availability and causal as-of positions; never backfill from future observations. The decoder position integrator itself uses received paths/speeds, not sim motion. |
| Geometry and numeric formatting | Lane frame/corridor uses sim geometry; camera projection, quantization, scaling, button order and slot layout are shared representation choices. | Obtain map placements from replay/build evidence. Shared formatting is permitted as a representation, but cannot supply hidden game state or mechanics. |
| Supervised learner | `load_games` loads existing arrays/cache without a source contract; timing-spread targets inspect ready flags, so fake zero cooldowns also affect labels. | A verified replay-only dataset contract plus masks consumed by the policy/target code. Existing caches and relabel sidecars must not certify old observations. |

`bc.sources.require_replay_only` now refuses conversion, relabeling, dataset
finalization and training data loads before bulk IO or writes. The dataset guard
runs before skip-existing and per-game exception handling so it cannot silently
reuse mixed-source games or report a successful empty conversion. There is no
CLI override. Planning and read-only inspection remain available. This is a
hold on the old formatter, not a completed replacement for it; do not remove the
gate merely by adding a metadata flag.

The decoder checkout provides separate evidence that team fog is unavailable:
`scene_policy_adapter.py` reports `player_fog_available=False`,
`ready_for_unmodified_PPO_loader=False`, and requires explicit masks;
`minion_memory.py` distinguishes spectator presentation from player visibility.
That adapter also contains a pinned sim Nexus pad for its frame orientation,
so it is not a drop-in replay-only replacement. No decoder changes were made.

Next implementation needs an explicit observation contract: masked spectator
pretraining (different information from human vision), or human-view BC once
team visibility is qualified. In either case the policy must consume availability
masks, and RL/init-from compatibility must be tested before a reconvert/training
run. Unknown state cannot be relabeled as absence, full health, ready or noop.

Validation: 17 focused local CPU checks passed for the gate, causal countdowns,
unknown/silent slots, cast invalidation, timer corrections, labels and lag. No
GPU work, reconversion or training was launched. Historical probe job 12793
failed with `KeyError: values` on a non-scalar field; it did not establish the
proposed cooldown hash/timing repair. Audit job 12792's result remains evidence
of fallback prevalence, not proof of a matching fix.


## 11. Approved heuristic fog (2026-10-22)

Dani explicitly allows rough heuristic fog as an exception to replay-only BC.
There is no need to wait for decoded player-team visibility or to switch to
unfogged spectator pretraining. This supersedes section 10's unresolved fog
choice only; it does not authorize other simulator-filled features.

`bc.fog.team_visibility` uses live allied replay champions, lane minions and
turrets with finite positions as circular sight sources. The deliberately
approximate radii are 1350/1200/1350 units respectively. They are heuristic
parameters, not claimed decoded values, and are defined in the BC module without
importing sim vision/config. Every available allied champion supplies sight,
even when it is outside the policy slots. Targets must have a live replay unit
and a finite current position; template-only slots never become visible. Frames
are independent, without future backfill or retained enemy sightings. Wall/brush
occlusion, wards, stealth, temporary reveals and units omitted by the replay
reader are not modeled. These omissions can both over-reveal and under-reveal.

The converter now supplies that visibility to entity selection and witnessed
cast memory instead of calling `world.views.visibility`. Its per-game provenance
records `fog.profile=replay-team-circles-v1`, `decoded=False`, the radii and omitted
mechanics. The conversion hold remains for structures, descriptor/normalizer
sources, economy/summoner mechanics and unmarked missing values. This helper is
wired into the held formatter; no full replay conversion is claimed yet.

Validation: 21 focused CPU checks pass (fog, source holds, countdowns, labels and
lag). Fog checks cover team isolation, allied sources outside policy slots,
dead/unknown sources and targets, radius boundaries, absent template slots and
no future/stale visibility. No simulator behavior changes. CPU diagnostic job
12794 was submitted through `t3-sbatch` to inspect six champion-slot countdown
examples and relevant field names using storage-aware scalar/structured access;
log `/mnt/nfs/shared/modern-bc/modern-bc-cooldown-probe-v2-12794.out`. Its automatic
completion receipt will drive follow-up; no polling loop or training was started.


## 12. Cooldown probe result and packet qualification (2026-10-22)

Probe12794 completed (exit0). All six sampled casts per requested champion-slot
(Sylas Q, Viktor E, Renekton Q, Lulu Q, Kai'Sa Q and Garen Q) contain a separate
`received_cast_countdown` / `cooldown.cast_start` payload. The BC reader previously
ignored this field and read only broadcast scalar aliases. This explains a missing
source in the reader; it does not establish the accuracy of every scheduled timer
or explain every fallback in the 400-game audit. Every sampled cast schedule has
`clock_alignment_qualified=False`. Nearby broadcast operands explicitly have
`timer_semantics_validated=False` and
`slot_array_index_semantics_validated=False`. Garen Q examples include a broadcast
zero at the cast and an 8s value later; these cannot be repaired by assuming every
same-slot numeric value is a timer with validated semantics.

`replay._cooldown_events` no longer certifies these aliases based on descendant
status from commit19cbb838. It joins scalar aliases to operand evidence using
source packet event index plus receipt time and actor. Both timer and slot
qualification must be explicitly true, the slot must agree, and values must be
finite/nonnegative. Missing proof, ambiguous duplicates and a different packet
at the same clock do not qualify. This is intentionally stricter than accepting
a plausible duration. Per-game `cooldown_source_audit` records accepted/rejected
broadcast counts, cast-schedule counts and how many schedules qualify their
clock alignment. Cast schedules remain unconsumed until their display-estimate
and availability contract is represented; they are not sim cooldown fallbacks.
`cooldown_fields_read` now means qualified broadcast evidence was accepted, not
that a particular decoder ancestor exists.

The initial400-game fallback percentages are historical converter-coverage
statistics; they do not prove the accepted broadcasts were authoritative timers.
There is now replay-derived cast-schedule evidence to investigate for the gaps,
so no timing/hash repair is claimed from this probe alone. A display countdown
must also remain distinct from ability activation eligibility/charges/recasts.

Validation: 25 focused CPU checks pass, including packet identity, missing/false
qualification, duplicate rows, slot disagreement and nonfinite values. Job12796
was submitted on desktop CPU to exercise the revised reader on all ten actors in
one actual probe export; log
`/mnt/nfs/shared/modern-bc/modern-bc-countdown-source-check-12796.out`.
No reconversion or training was started. Next: inspect that automatic receipt,
then build the remaining replay-field availability handling; the fog exception
remains limited to section11.


## 13. Real-source check and removal of structure/summoner defaults (2026-10-22)

Job12796 completed/0. On gameNA1-5650865612, all ten actors were inspected:
3399 broadcast packets, zero with qualified timer/slot evidence; 2377 separate
cast schedules, zero with qualified clock alignment. This checks the reader's
rejection and source-inventory path on one real export, not the correctness of
schedules or coverage of the archive. Output:
`/mnt/nfs/shared/modern-bc/modern-bc-countdown-source-check-12796.out`.

Further removals in the held formatter:

- `_unit_fields` uses template layout/type/team/lane only to address slots. It
  no longer copies any template positions or creates live/full-health structures.
  A matched turret's position, life and HP come from its replay unit. Missing HP
  remains NaN; absent turrets, inhibitors and Nexuses have no replay entity and
  remain empty slots. Empty padding is not evidence a missing structure is dead.
- Summoner slots4/5 use the same qualified received-timer reconstruction as
  abilities, then map to canonical D/F. No summoner cooldown constants, 300s
  fallback or haste rule is read. Unknown timers remain NaN rather than ready.
- A recall press supplies an action label, not a timed channel state. The sim
  8.5s duration and next-action completion heuristic are removed; the observation
  channel field remains unknown without received state evidence.

27 focused CPU checks pass. New fixtures need no template positions, verify a
received turret's coordinates and missing HP, exclude template-only structures,
and verify summoner key mapping, unknown timers and unknown recall state. No
simulation step, dataset reconversion or training was run. Remaining source
holds cover cooldown/descriptor normalization, progression/quest/trinket fields,
unmarked reader defaults, causal positions and policy availability handling.
The accepted heuristic fog is unchanged. Source audit section10 describes the
starting implementation; sections11–13 record the approved exception and repairs.


## 14. Direct replay inputs, v4 profile and availability (2026-10-22)

Dani approved proceeding through cleanup, representative conversion, learning
check and rollouts before full v6. `modern-world-v4` keeps the v3 tensor widths,
role/identity columns and action ordering, but versions the changed semantics:
Q/W/E/R timers are seconds/60, summoner timers seconds/300, enemy witnessed cast
age clipped to60s/60, and unknown cast history is unavailable. It must be selected
explicitly; the simulator default stays v2 and old width-only checkpoints retain
their historical interpretation. Per-player tier defaults to0 (unknown), not an
assumed Master classification.

`bc.observation` constructs policy inputs directly from replay units and stats.
No world configuration, initial state, starting items/runes or kit composition is
used. Frames use replay outer-turret placements and pinned client geometry spawn
anchors for orientation; camera projection, slot selection, corridor filtering
and numeric scaling are shared representation choices. The approved circular
fog remains approximate. Every alive/positioned replay candidate can supply sight
according to its team; only visible on-screen replay units fill policy rows.
There are no template inhibitors/Nexuses or full-health placeholder structures.
Entity HP is unavailable if its received HP/maxHP is unavailable.

Replicated scalar/rank gaps now remain NaN, including level, gold, mana, shield
and combat stats. Causal position filling never backfills the leading unknown
period from a future position. Recall channel state, XP level fraction, unspent
skill points, quest state, trinket charges and both static descriptor blocks are
unavailable rather than derived from simulator rules or catalogs. Own spell and
summoner timers use only packet-qualified evidence; current pilot timers remain
unknown. Zero resource capacity is distinguished from missing resource evidence.
CS, stop/held orders and executions remain the documented replay-derived proxies.
These are not original human key/click recordings.

`LanePolicy` v4 requires explicit missingness: finite values are zero-filled for
arithmetic and their missing masks get separate learned projections. NaN is not
silently treated as a real zero. Each checkpoint also records columns never
observed during training and masks them in simulation, preventing never-trained
input weights from receiving newly available sim values. The sim v4 formatter
uses the same timer scaling and masks unsupported auxiliary/descriptor fields;
older profiles and their parameter layouts are preserved. BC and evaluation keep
an explicit100ms observation lag. Future full v6 evaluation must preserve that
setting and inspect the independent buying/gold mismatch.

New arrays/plans/manifests carry
`source_contract=replay-fields-circles-missing-v1`; the loader checks it together
with the v4 profile for every game/cache. Legacy arrays cannot be made compatible
by a relabel; standalone relabeling remains disabled. Cache identity includes the
source file path, nanosecond modification time, size and metadata/sidecar stamp.
Finalization refuses missing planned games, and training requires a finalized
manifest. Targeted labels without a target in the reconstructed observation are
excluded rather than turned into noop or impossible edge clicks. R target
coverage is reported for unit-targeted R separately from self casts. Evaluation
adds champion/role groups and joint button-plus-click accuracy per button.

Validation job12851 COMPLETED/0:41 BC checks passed on desktop CPU, followed by
16 converted early/late windows for eight champion-games (Garen/Jax top, Viktor
mid, Lulu/Lux support, Ezreal bot, plus held-out Jax/Viktor games). The original
base-game split was preserved. Valid decisions:6285 train /2330 val. Conversion,
finalization and the local-NVMe cached loader passed. Artifacts:
`/mnt/nfs/shared/modern-bc/v6-pilot-20261022/dataset`; log
`/mnt/nfs/shared/modern-bc/v6-pilot-20261022/modern-bc-v6-pilot-12851.out`.
The job's inherited /srv cwd did not exist on desktop; its script explicitly
changed to /mnt and all checks passed. Subsequent submissions set --chdir.
Pilot metadata records the then-uncommitted converter as1eac9d31-dirty; the
implementation is retained in this commit. Follow-up finalization/cache checks
bring the focused lightweight subset to30 passing checks.

Next is a full-architecture GPU memorization diagnostic on a fixed set of
spell-containing windows, with no dropout or timing smoothing. It reports cast
accuracy and click loss as well as total loss, saves a diagnostic checkpoint,
and stages data/cache under desktop /scratch. This is not a generalization
result or the full v6 run. Broad training and rollout acceptance remain pending;
no claim of improved sim behavior follows from CPU conversion checks alone.


GPU diagnostic12858 stopped before optimizer start because one old test asserted
the prior evaluation group set (42 other checks passed). That assertion now
checks champion/role buckets and joint click accuracy bounds. Replacement12859
is registered to the chat on gpup,30min limit, explicit/mnt cwd. Its preflight
reruns43 BC checks before512 fixed-window updates using the full architecture.
Outputs live under the pilot directory; the result pointer is
`overfit-result.json`. No full v6 training is submitted before its result is
inspected, and a memorization pass alone does not establish rollout quality.


GPU12859 COMPLETED/0:43 BC tests passed; the512-update memorization diagnostic
passed its predefined loss/cast/click checks. Fixed-window loss10.399→1.405,
button accuracy92.1%, cast accuracy34/35 (97.1%), click NLL8.556→1.379,
exact clicks60.5%, mean cell distance54.85→3.03. These are training-window
metrics, not evidence of generalization or rollout quality. Saved checkpoint:
`/mnt/nfs/checkpoints/lanerl-jax/modern-bc/modern-bc-v6-overfit-20261022-113216-2e9dd8f6/ckpt_000000512.msgpack`.

Before rollouts, corrected a default-tier mismatch: v4 simulation now defaults
to unknown tier0, matching BC, while older profiles retain their historical
Master default. An explicit tier argument still overrides the default. The v4
test checks that value. The gate now records alive time within2500 units of
fountain, reconstructed from the lagged observation pose (100ms delay), alongside
CS/deaths/buttons/overflow. Buying remains masked in the initial baseline; it
must not be treated as a resolved economy gap. Next job evaluates joint targeted
cast accuracy on fixed windows, true held-out pilot games, and8 Garen plus8 Jax
full lane episodes. No full-dataset conversion/training is launched yet.


Rollout attempt12866 passed43 checks but stopped before episodes when writing
an offline metric with NumPy float32 to JSON. `evaluate` now returns native
numeric leaves, and its test asserts JSON serialization. Replacement12869 is
registered on gpup (50min, explicit/mnt cwd); inspect its receipt and actual
joint-target/held-out/episode metrics before proceeding.

Full-cache capacity preflight:165GB free desktop NVMe,718GB free NFS. The source
plan contains58470 champion-games (~473.6M decisions). Extrapolating the16 pilot
windows gives502.6GiB for the current cache, so it must not be staged as-is.
Pilot self vectors have24–27 ever-observed columns; entity occupancy ranges
4.78–11.73 rows/decision. Omitting always-missing self columns and encoding exact
entity one-hots categorically (retaining the current fp16 ds/dn/HP values) gives
an estimated105.1GiB. This compact cache is not implemented yet; it must preserve
reconstructed arrays and be tested before full training. Estimates may change
with full-corpus occupancy. No existing cache/data was deleted.


Job12869:43 checks passed; fixed targeted-cast joint exact9/19 (47.4%),
mean target distance8.63 cells. Held-out button accuracy74.6% versus82.5%
majority; no observed held-out cast recall. These diagnose the intentionally
memorized tiny checkpoint, not broad training. Episode reporting failed because
the base metric used `self_vec` instead of `Transition.obs_self`; corrected.
Dani authorized scratch reclamation: obsolete474GB BC cache removed after
checking no active jobs and cache-only contents; free NVMe638GB. The earlier
capacity blocker is superseded;~502.6GiB is still an estimate and requires a
20GiB reserve plus staging duplication. Retry uses a private source snapshot,
private compilation cache, and EXIT cleanup; checkpoint remains durable on NFS.


Rollout12873 COMPLETED/0:43 checks;8 Garen and8 Jax episodes,540s each,
100ms observation lag, AFK opponents. Mean CS2.25/1.25, deaths0.25/1.0,
kills0.875/0.375, base fraction0.31%/1.40%, overflow0. The tiny memorization
checkpoint integrates successfully but farms poorly and does not generalize;
these results do not resolve buying or establish a useful BC policy. Outputs:
`v6-pilot-20261022/gate-Garen.json` and `gate-Jax.json`. Isolated scratch cleanup
verified after completion,638GB free remains. CPU12879 now measures full-length
conversion and current cache size on the same eight champion-games, with the
original split and30..840s window. Independent source/decoder/data copies,
local cache, durable NFS outputs and EXIT cleanup; no full-corpus training yet.


Full-length attempt12879 failed during staging before conversion: preserving
source permissions with `cp -a` is unsupported on this scratch path. Failed-job
staging was cleaned. Changed shell staging to `cp -rL --reflink=never
--no-preserve=all`; all544 decoder files match recorded SHA256 hashes in a
lightweight desktop copy check. Replacement CPU12880 registered with the same
40min/16GB limits and checks staged decoder hashes before conversion. No BC
input/label change; no full-length sizing result yet. Host guidance updated to
use independent content copies without preserving unsupported metadata.


Full-length12880 verified decoder staging544/544 files but failed at converter
provenance before saving the first game: the code archive has no `.git` file.
Converter now supports a pinned `.bc-source-commit` containing a full40-character
SHA, rejecting absent/malformed records; ordinary repositories/worktrees retain
Git dirty-state reporting. Ten focused converter checks passed, including the
archive provenance regression. Retry will use a fresh archive with its actual
commit record and rerun the complete BC suite before full-length conversion.
No finalized full-length dataset or sizing result exists yet.


12882 passed44 BC tests and converted all eight full-length games (8100 rows
per game,2.9–5.7s conversion+save each). Finalization failed because the temporary
sizing worker omitted `vocabulary` from its hand-built plan. The worker now
preserves all original source-plan metadata and overrides only the approved
profile/tier/lag/window/game subset. Preflight checks every literal plan key
consumed by finalization, champion vocabulary coverage and both data splits.
Replacement CPU12883 registered with the same source snapshot6b20bdcf and
40min/16GB limits. No BC code/label change; cache sizing and durable finalized
dataset still pending. Failed-job private scratch cleanup verified.


Full-length12883 COMPLETED/0:44 checks,8 full games (64800 decisions),
40799 valid train/14414 valid validation decisions. Finalization and both split
cache loads passed. Conversion+save2.9–5.7s/game, max RSS2.34GiB; dataset6350884
bytes, cache69913944 bytes, extrapolated475.9GiB over473607000 decisions.
Current638GB free NVMe leaves reserve, but full-corpus occupancy remains unknown.
Private scratch cleanup verified. Durable outputs: pilot `full-length-dataset`
and `full-length-result.json`.

Full replay-only conversion array12886 registered:8 shards,2 concurrent,
CPU desktop10GB/2cores each,12h limit. Plan58470 champion-games/7309 bases,
original train/validation split, v4/tier0/100ms lag and30..840s. Frozen converter
6b20bdcf/decoder copies; private scratch with per-base deep copies and EXIT
cleanup. Completed per-game arrays and metadata persist to
`/mnt/nfs/shared/modern-bc/v6-full-20261022/games` with metadata committed last;
failures remain explicit and must be reconciled before finalization. No broad
training/PPO has started. Only replay inputs plus approved heuristic fog enter
BC; the simulator is used for the separate rollout diagnostic.


2026-10-23: full conversion12886 COMPLETED/0,8/8 shard receipts,58470 NPZ
and58470 metadata files, no failed games. Audit13005 now finalizes and checks
all arrays/source/split contracts. Dani says proceed without waiting on decoder;
heuristic fog remains explicitly approved. Standalone BC and current-patch data
scaling are the objective; no RL/PPO or simulator economy repair in this plan.
Decoder v18 is a released partial player-view interface, with complete rendered
and terrain visibility still unqualified; it is reserved for later integration.

Trainer adds opt-in step evaluation/checkpoints, deterministic champion/role
held-out sampling, a baseline evaluation, event-button+click NLL plateau stopping,
and immediate nonfinite loss/gradient stopping. Sampled improvement checkpoints
are retained as `best_partial.msgpack` independently of checkpoint rotation;
full validation still runs at completed epochs. Defaults keep epoch-only behavior.
A real short training smoke must verify checkpoint/early-stop paths before full BC.


Source137e30b3: focused group-sampling regression passes. Isolated GPU13007
runs all45 checks and two real short training runs to verify within-epoch
checkpoints/evaluation and forced plateau stopping. Full BC13009 is queued
with afterok dependencies on audit13005 and smoke13007, plus receipt assertions.
Recipe:3 epochs maximum, window256/burn32/batch128,500-step evaluation on512
fixed held-out champion/role-spread games; patience4/min_delta0.001 on event-button
plus click NLL. Full validation remains at epoch boundaries. Cast weight5,
summoners3, ward/recall/stop2; enemy/stat dropout0 for standalone BC, no timing
smoothing on unknown cooldowns. Checkpoint selection uses sampled scores only
for interim decisions and records full-epoch metrics separately.16h/24GB/4CPU.
Private independent source/data/cache; cache capacity preflight15% headroom and
20GiB reserve, reserve guarded during build. Full training cannot start before
both prerequisites succeed. No native-FoW wait, no RL/PPO. Actual results pending.


GPU smoke13007 COMPLETED/0:45 tests; normal short run64 steps, four step
evaluations, saved best sampled checkpoint, forced plateau run stopped at4steps
after two failed improvement checks. Workflow verified, not generalization.
Audit13005 finalized metadata:52590 train/5880 val player perspectives,
340720945/38055900 valid labels,7309 unique26.19 matches. Full-array checks
still running;2000 files/341s measured, time limit raised2h→4h. Full training
13009 remains dependency-blocked until audit succeeds; no optimization has begun.


Full audit13005 COMPLETED/0 in5629s (~94min): all58470 player perspectives
from7309 unique26.19 matches passed source/fog/converter/split/shape/nonfinite
and action/cell bounds.473607000 total decisions,340720945 train/38055900
held-out valid labels; NPZ storage44237490316 bytes. `audit-result.json` passed.
GPU13009 released after both prerequisites and is running; initial capacity/data
staging/cache phase has no optimizer output yet. Full BC results remain pending.


13009 trained10000 steps before preemption/requeue. Durable sampled metrics:
event-button+click NLL7.398, button accuracy91.23%, click distance12.71;
Q/E/W/R recall3.45%/2.15%/0.57%/0.09%. Overall improvement is not adequate cast
quality. The automatic restart failed its capacity guard because first-attempt
private staging/cache (503.8GB) survived while the wrapper attempted a second
copy. Checkpoint10000 is safe on NFS; no optimizer divergence established.
Trainer resume now skips previously completed batches, replaying RNG draws so
the configured epoch ceiling remains cumulative. Wrapper recovery/resume check
must precede retry; no reconversion is needed.


Resume smoke13011 (GPU) verifies restoring the old4-step checkpoint completes
at64total steps, and logs four skipped batches. Full retry13012 depends on
its success, source104b05b8. Same training seed/recipe/3epoch total ceiling,
latest owned NFS checkpoint selected. Recovery copies old terminal13009 private
staging/cache into a new exclusive directory with independent files, progressively
reclaiming copied originals to avoid doubling503.8GB. Cache relocation rewrites
paths only with matching source size/mtime; incompatible entries rebuild.
Wrapper appends logs across requeues, cleans its own orphan staging at startup,
forwards termination to the child and waits before cleanup. No shared mutable
cache or hardlink/symlink staging. Resume results remain pending.


13011 COMPLETED/0:45 tests; old4step checkpoint resumes to64total rather
than68steps, verifying cursor/budget behavior.13012 failed before donor recovery
because desktop sacct could not connect to localhost6819. No data/checkpoint
changes. Recovery now uses a terminal-state receipt recorded by danilogin sacct,
plus live scheduler absence and private donor ownership checks. The exact check
passed on desktop before submitting13013 (gpup24GB/4cores/16h). Source104b05b8
and the training recipe remain unchanged; checkpoint10000 and original cache
were intact at submission. Retry uses append logs and existing recovery/cleanup.


13013 COMPLETED/0, early stop38500total steps; best sampled36500 retained
in `best_partial.msgpack`. Sampled event+click NLL7.1787, accuracy91.32%
(majority82.15%), click exact6.00%, distance12.10. Q/W/E/R argmax recall
7.54%/1.43%/4.92%/0.40%, joint exact Q0.17%/W0.02%/E0.52%/R0%.
Full epoch2 held-out accuracy91.17% (majority82.22%), event NLL0.9158,
click NLL6.3211, exact5.14%, distance12.56. The pipeline and broad learning work,
but this is not adequate standalone action/target quality. No improvement claim
based solely on overall accuracy. Scratch cleanup verified (638GB free).

GPU13014 evaluates best36500 across all5880 held-out perspectives and adds
probability calibration/expected sampled rates, true-event probabilities,
threshold precision/recall, and expected independent categorical button+click
sampling within2cells of each axis. This diagnoses weak strict argmax recall
versus weak assigned cast probabilities and targeting before another long run.
Current decoder/native FoW remains deferred per Dani; no RL/PPO. Results pending.


13014 COMPLETED/0, best36500 on5880 held-out perspectives/33233804 scored
frames: accuracy91.34% vs82.22% majority, click exact5.88%/distance12.37.
Q/W/E/R argmax recall10.83/2.05/3.95/0.57%, but expected sampled shares are
4.19/2.04/2.20/0.355% vs labels0.679/0.311/0.320/0.0565% (~6–7x over-casting).
Mean true-label probability19.55/10.07/12.18/6.10%; expected joint button+click
within2cells peraxis0.87/0.33/0.67/0.19%. Low argmax recall is not evidence of
low total sampled cast rate. More cast weighting would worsen calibration.
The current click distribution is independent of selected button, mixing movement
and all spell aims. Next is a small frozen-trunk shared-vs-action-conditioned
readout ablation and button calibration using separate validation bases for fit
and assessment. No production action contract change or long retraining yet.


Controlled GPU probe13015 registered (20GB/4cores/1h), sourcef126f4e5.
All46 BC checks precede extraction; the new check requires optional feature
readout to preserve normal outputs bit-for-bit. Best36500 backbone frozen;
shared and action-conditioned residual aiming heads receive identical1000
balanced-button minibatches from128 train perspectives.64 calibration and64
assessment perspectives have disjoint validation replay bases. Pointing results
are conditional on the known action label, not end-to-end action success.
Calibration compares original logits, inverse class weights and fitted
bias/temperature, assessed on other replay bases. No production action contract
or full training changes yet. Note13014's NLLs are unweighted; training-monitor
event NLL includes class weights and is not directly comparable in magnitude.


Probe13015 COMPLETED/0,46 tests. On64 assessment perspectives separate from
64 calibration-fit perspectives/bases, fitted logit bias/temperature lowers
unweighted NLL0.3731→0.2912, accuracy91.38→92.08%, Q sampling4.19%→0.686%
(human0.696%); other cast rates closer too. Argmax cast recall remains nearzero:
rate calibration does not recover intent. Frozen action-conditioned aiming is
mixed: Q/E distance22.89→15.98 /17.17→14.20; within2cells Q7.43→10.59%,
E19.16→17.06%; W/R NLL worsen. Summoner/ward100% within2 on65/97/130 labels
may reflect caster-centered target labels, not improved general aiming.
CPU13017 audits raw slot4/5/12 targets/vectors against own position and converted
cell distributions on26 perspectives.13016 failed on wrong ROLE_LANE import,
fixed to bc.convert before submitting13017. No new broad training or production
head/calibration integration until target-source results establish what is real.


Target audit13017 COMPLETED/0:26 perspectives/228 slot4/5/12 casts,
224 selected0xd0 vectors,217 within50units of caster; converted clicks cluster
near screen center. All228 lacked resolved points.0x2c often differs but is not
qualified here. Thus summoner/ward readout gains were learning bad caster-origin
labels and cannot be promoted as aiming improvements.

Reader removes slot4+0xd0 fallback. New conversions explicitly record
`label_target_contract=received-targets-no-d0-v2`; resolved/unit targets remain
usable through existing representability checks. Existing v4 arrays cannot
separate those from fallback targets, so the loader conservatively excludes
summoner_d/summoner_f/ward labels in unmarked games. They retain their original
buttons and observations for audit; validity/click masks exclude supervision,
not relabel to noop. Cached observations remain reusable; manifests record
excluded train/val target counts. No full observation reconversion required.
Next is the identical controlled readout probe with these labels excluded.


Corrected probe13018 registered (gpup20GB/4cores/1h), source63c8d575.
Same256 selected perspectives, frozen best36500 backbone, two residual heads
and independent calibration/assessment split. Loader excludes legacy unqualified
summoner/trinket target labels; observation arrays stay unchanged. All48 BC
checks precede the experiment, including a cached-dataset integration test for
masking without modifying buttons/observations and a resolved-target reader
regression. No full conversion or broad training is repeated. Results pending.


Corrected13018 COMPLETED/0,48 checks. Calibration on separate assessment bases
NLL0.3698→0.2859; Q sampled4.19→0.672% (human0.696%). Conditional readout still
mixed: Q distance22.89→16.06/within2 7.47→9.76%; E within2 19.25→16.61%,
W/R NLL worse; no promotion. Next diagnostic retains actual replay target-unit
IDs, maps them to lagged visible observation slots, and tests a direct entity
pointer. This preserves identity lost by screen-grid regression; source IDs,
not nearest-coordinate matching, provide supervision. Existing Labels gains an
optional target_id output only; original button/click calculations are unchanged.
Only the same small subset is reconstructed, with observation equality checks;
no full corpus rebuild or production action-contract change yet.

CPU13019 registered for exact target-ID reconstruction on192 perspectives
(128 train/64 assessment, sourcebc6b8e14,2 CPUs/8GB/1h). Requires all BC tests,
frozen decoder SHA256 checks, original entity/padding equality at stored lag, and
unchanged guarded button/click labels. Sidecars retain actual target ID, visible
row and eligibility; report coverage lost when targets leave the lagged view.
No policy promotion or full-corpus rebuild. Learned pointer comparison follows
the reconstruction receipt.

CPU13019 completed:48 tests,544 decoder hashes, all192 perspectives pass exact
input and guarded-label reproduction.85,795/85,872 eligible unit targets map into
lagged visible rows;77 excluded, no inferred target IDs.58,908 train/26,887 heldout
labels before recurrent burn-in. Reconstruction took952 seconds after tests.

GPU13020 registered (gpup,4 CPUs/20GB,1h limit, sourcebc6b8e14, immutable probe
worker79ecc87a4643). Frozen best36500 context, same128train/64assessment bases;
entity-only and contextual pointer heads,1000 balanced-button batches each.
Compare with nearest/first-visible and screen-snap baselines. Reports unit ID
accuracy, top3/NLL, projected screen errors, held/event and per-button counts;
oracle target-row projection exposes lag/quantization limits. Oracle button and
unit-click type are supplied, so this is not end-to-end BC evidence. Only small
heads train, not the base checkpoint; no production action contract change.

GPU13020 completed in55.2s worker time. Context pointer exact unit accuracy73.67%
versus66.75% screen-snap,65.14% entity-only and26.69% nearest-visible across23724
heldout clicks; event-only70.78 versus60.70% screen-snap (7066). Within2 cells
74.59 versus68.46%; mean screen distance3.72 versus5.08. Q/W/E near rates improve,
R drops67.7→38.7% on31 cases. Train accuracy76.0%; spell-specific overfit remains.
Oracle target identity projects within2 in97.53%, so lag is not the main error.
No production promotion: oracle action button and unit-click type were supplied.

GPU13021 registered: sourcebc6b8e14/workerb47a0f18024a,20GB/4cores/1h. Frozen
base and context pointer; learn unit/ground routing on128 training perspectives,
1000 balanced-button steps, fixed0.5 cutoff.512 fresh assessment perspectives,
excluding prior calibration/assessment replay bases, uniform1024-click cap per
perspective, seed419. Reports ground-click damage and unit gains, per-button,
oracle-button and actual-button routing, and joint button/near-click accuracy.
No oracle target type in inference; empty entity sets and predicted ground clicks
retain original screen predictions. This tests the missing routing decision before
considering any architecture/action-contract integration or broad retraining.

GPU13021 completed,71.6s worker. Hard routing failed on524288 sampled clicks/512
fresh perspectives: actual-button within2 24.509→24.429%, joint23.718→23.638%,
distance12.235→12.332. Unit near61.20→61.96%, ground21.213→21.057%; ground
comprises91.76% of clicks. Unit recall19.38%, ground false-unit2.71%. Oracle
button does not rescue overall performance. Reject frozen router/pointer hybrid.
A correct unit-target diagnostic did not establish an end-to-end improvement.

GPU13022 registered: matched factorized and joint96x54 residual click heads on
frozen best36500 context. Both action-conditioned, same128 train/64 selection,
identical2000-step ceiling/minibatches, Gaussian sigma1 and held weight0.3.
Half natural-frequency and half balanced-button examples.2048 train and1024
selection/test uniformly sampled clicks per perspective, seed419. Every500 steps
select by held-weighted soft click CE; unchanged baseline eligible at step0,
patience3. Another512 heldout perspectives exclude all earlier assessment and
calibration replay bases. Reports actual/oracle button NLL, click distances and
joint accuracy, including ground/unit/event/per-button slices. No target-type
routing or pointer. Tests the independent-x/y bottleneck directly, without
production model changes. Sourcebc6b8e14, workerbab0037e5cf9; gpup20GB/4cores/1h.

Architecture reference consulted before the next experiment: [DeepMind's AlphaStar
paper](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf),
extended data architecture, separates target-unit attention and spatial target
prediction with action-conditioned heads. This supports inspecting the action
representation; it does not establish that our joint click experiment will work.

GPU13022 completed,99.5s worker: both readouts select unchanged baseline at step0
and stop at1500. Initial soft CE6.56737; best trained factorized6.58335, joint6.56862.
Joint exact selection NLL falls6.17864→6.15698, but fails prespecified soft CE.
Consequently524288 assessment clicks reproduce baseline NLL6.17073/near24.174%/
joint23.348%. The frozen probe used sigma1 also for unit clicks, while production
uses sigma0 there; result is limited to its recipe. No architecture-level claim
that joint heads cannot help. Frozen-feature probes end without promotion.

GPU13023 registered: bounded full-gradient BC continuation control versus joint
residual head + full base fine-tuning.1024 training perspectives stratified over
champion/role;64 fixed selection; remaining240 assessment perspectives/30 bases
outside preceding head-probe assessment/calibration sets. Main model has already
had full-val readout, so this is validation, not pristine test evidence. Native
unit sigma0/ground sigma1, existing class weights, held0.3; matched batch8/window256/
burn32 and1000-step cap, base Adam3e-5/new head Adam1e-3, clip0.5. Selection every200
by unweighted event-button NLL + soft click CE; patience3/min_delta0.001 including
initial baseline. Initial losses must match native trainer; gradient norm and
actual base parameter changes must be nonzero. Save best parameter trees and
partial arm results durably. Sourcebc6b8e14/workerd841bb67ce9e; gpup24GB/4cores/1h.
No production architecture change, full-corpus retrain, simulator data, or RL.

GPU13023 failed after4m10s at JSON reporting after the continuation control,
before joint training. Native loss check6.255844 vs6.255849 passed; base gradient
norm5.404 and parameter delta0.0620 nonzero. Control scores6.85434/6.85305/6.87758
at200/400/600 versus6.84000 initial; early-stopped, retained baseline0. Assessment
1365775 valid frames/1127853 clicks: NLL6.16265, near24.354%, joint23.634%.
Checkpoint survived on NFS; `endtoend-control-13023-recovered.json` captures log
scores/gradient/assessment and checkpoint provenance. Failure is float32 JSON
serialization, not a numerical/training failure or candidate result.

GPU13024 registered for unfinished joint arm only. Sourcebc6b8e14/worker5fb84526f151,
same data/recipe. Recompute control assessment against saved checkpoint, assert
parameter identity and agreement with logged metrics; do not repeat its600 updates.
Normalize evaluation scalars to Python numbers and serialize NumPy scalars via
`.item()`; real-NumPy nested float32/int64/bool test passes locally. Validate entire
metric serialization before optimization. gpup24GB/4cores/1h, private staging and
cleanup, durable best/partial outputs. No full retraining, production change or RL.

GPU13024 stopped after1m09s, still before joint training. Parameter identity and
continuous loss checks passed; within2 count differed by16/1127853 (274675 versus
274691), rate delta0.000014186 beyond the original strict metric tolerance.
The exact cause of hard argmax drift is not established. This is neither a
candidate gain nor training failure. Reproducibility guard needs metric-specific
limits rather than one tolerance for losses and discrete output rates.

GPU13025 registered: same unfinished joint arm/sourcebc6b8e14, worker000bd4769814.
Exact parameter-tree/dtype/value and observation counts; losses rtol/atol1e-5;
hard accuracy rates absolute1e-4 and mean distance0.01 cell. Persist all prior/current
metric deltas to `control-reproduction.json` before asserting, and require exact
repeat of fixed-batch outputs in this run. No repeated control optimization.
Same data/recipe/early-stop/checkpoints as13024; gpup24GB/4cores/1h.

GPU13025 COMPLETED/0,316s. Native loss and full-gradient checks pass. Best joint
step800, selection6.84000→6.74844. Actual-button heldout click NLL6.16265→6.07386,
joint near23.636→23.916%; Q/W/E button recall7.568/2.624/3.657→5.692/1.858/2.026%.
Per-class NLL Q2.119→2.196/W2.653→2.753/E2.605→2.902. Reject promotion despite
aggregate gain. All selection checkpoints worsen E NLL>0.05 versus initial.

Cast guard being added to sampled selection: unweighted per-QWER NLL at fixed
support, minimum100 events, maximum0.05 increase from baseline. Violations block
best replacement and count toward existing patience. Epoch winners also require
same sampled guard; retain baseline best_partial at step0. Guard unavailable when
within-epoch evaluation is disabled. Tests pending.

All192 pointer-subset metadata files show no available own QWERDF cooldowns.
Next small input diagnostic records elapsed time since received own casts from raw
replays, preserving canonical summoners and original lag. Unknown before first
receipt; never infers readiness, duration, kit constants or future information.
This source-only helper/sidecar does not alter v4 width/production inputs. Audit
will measure history outside256-decision recurrent windows before another model
experiment; no broad reconversion or simulator data.

CPU13026 registered (source5450052a/worker98bf4406aa4e,2CPUs/8GB/1h): full BC tests,
real13025 regression-gate replay,544 decoder hashes,192-perspective own-cast age
sidecars. Reports per-slot known and outside25.6s recurrent-window support; canonical
D/F identity, NaN before first receipt, exact stored observation lag. Sidecars and
source audit only, with no production observation/cache mutation. NFS outputs
`cast-history-13026/results.json`; staging cleaned on EXIT. No GPU training running.

CPU13026 COMPLETED/0:53 BC tests pass, real13025 cast-guard replay rejects every
selection candidate plus assessment Q/W/E regressions.544 decoder hashes pass;
192 received-age sidecars take771s. Heldout known QWERDF histories374832/331986/
341601/167489/261368/306740 over410529 valid frames. Beyond25.6s among known:
25.7/44.0/40.6/74.5/93.3/90.9%. Added history is available, predictive value untested.

GPU13027 registered, source5450052a/worker3e43a24fe04d. Three paired frozen-base
button readouts: bias calibration, learned existing context, and identical context
readout with causal cast ages (log1p seconds/log1p300 + known bits).128 training
perspectives split by replay base into fit/selection;64 assessment perspectives
previously used by head probes. Same natural-frequency examples, inverse-weight
initial bias, Adam3e-4,2000-step cap; selection event NLL every250 with cast guard
and patience4. Heldout shuffled-age falsification. No new architecture until gains
beat both controls without cast regression. gpup20GB/4cores/1h, private copies,
EXIT cleanup. Production observation widths/checkpoint remain unchanged.

GPU13027 COMPLETED/0,82s. Split99 fit/29 selection/64 assessment. All bias/context/
history heads select step0, stopping at1000 after every trained checkpoint worsens
selection event NLL and QWER NLL limits. Assessment NLL0.300799 is only inverse
class-weight baseline; no learned history improvement. Shuffling is uninformative
when selected offsets remain zero. Training optimized natural all-frame CE while
selection optimized event-only CE, limiting the experiment's interpretation.

GPU13028 registered, source5450052a/workerea5ba252f569: paired low-capacity bias
versus additive per-class received-age-bin curves.20 bias +6x12 offsets, unknown
bin and edges0.5/1/2/4/8/16/32/64/128/256 seconds; no cooldown estimates. Same
99/29/64 and natural-frequency batches, but training/selection now both use
class-unweighted CE with held0.3; age offsets L2 coefficient0.001.2000-step cap,
selection250/patience4 with cast guard and unchanged baseline eligible. Report
age-bin human and original/inverse-weight expected QWER counts to inspect raw
signal, plus shuffled heldout ages. Existing base model and production input
profile unchanged. gpup20GB/4cores/1h, private stage/EXIT cleanup, no broad retrain/RL.

GPU13028 COMPLETED/0,65.8s. Selected history1000/bias0. Assessment weighted CE
0.403222→0.402078, all-frame NLL0.300801→0.298085; event NLL worsens0.502648→
0.503030. R class NLL4.7248→4.7759 violates0.05 heldout regression limit.
Shuffled ages weighted CE0.402559, retaining most of the overall gain; direct
age-specific gain~0.00048. No production promotion or larger age-model run.
Read-only decoder recheck remains043b9a83/v18 with unchanged production exports
and partial display estimates; no new qualified readiness source imported.

Next evidence is a replay observation/prediction audit, not another training run.
Eight deterministic8.1s clips: Garen Q/R, Jax Q/E, Ezreal Q/R, Lux Q/E, from first
SHA-sorted heldout perspective having valid non-held events after180s outside
burn-in. Include all local frames, entity rows and human/model targets/buttons,
relative click heatmap, raw and previously fitted calibrated probabilities.
Fixed replay state is never advanced by predictions; heuristic fog/canonical
camera remain explicit. No error-based clip selection or closed-loop success claim.
GPU13029 failed from scalar indexing of packed entity wrapper; fixed slice read
and actual loader shape check. GPU13030 registered, source5450052a/workerce8a2b20dcd3,
gpup8GB/4cores/20min, NFS `replay-behavior-13030/scenes.json`. Visual comparison
follows receipt. No simulator, full retrain, production policy change or RL.

GPU13030 completed:8 clips/622451 bytes. `replay-behavior-13030/replay-comparison.html`
passes desktop-dark/mobile-light previews and actual scene/calibration/time-input
checks. At center presses all8 raw argmax buttons are Move; Garen R/Jax Q targets
are near, Ezreal Q/R/Lux Q far. Deterministic illustrations, no standalone or
fresh-test claim. Fixed replay states, heuristic vision and canonical camera.

Clean retrain preparation: the current36500 checkpoint predates target-origin
repair. Most head probes reused that checkpoint; a clean baseline should precede
further architecture experiments. Fresh training reuses finalized v4 arrays with
conservative slot4/5/12 target masks and disables those unsupported output buttons.
No expensive reconversion. Add `--cast-guard-start-step` (default0, resume checks
immediate) to bootstrap fresh class probabilities before fixing a reference;
otherwise a learned rare-class posterior can fail against uniform initialization.
Planned fresh reference step2000; preserve existing within-epoch early stopping
and baseline checkpoint. Tests and real training smoke gate the full job.

Clean run queued, source1c5fab88. GPU13031 gates full BC tests plus real monitoring/
forced-stop smoke: verify initial best_partial preservation, guard activation and
D/F/ward output masks. GPU13032 has afterok:13031 dependency plus durable passed
smoke check before staging. Dataset remains58470 perspectives/7309 matches,26.19;
same finalized v4 inputs, corrected legacy target masks, no reconversion. Fresh
model with unsupported D/F/ward outputs disabled, no inherited poisoned parameters.
Same batch128/window256/burn32/class weights/no dropout;3-epoch ceiling,500-step
512-perspective validation, patience4/min_delta0.001. Cast guard reference begins
at2000 with0.05/min100 constraints. Full job gpup24GB/4cores/16h cap, --no-requeue;
forward TERM, save NFS checkpoints and clean private staging. Source snapshot and
all scripts/logs at `/mnt/nfs/shared/modern-bc/v7-clean-20261024`. This is training,
not a new23h conversion. No claim of quality until clean heldout results arrive.
Interactive replay comparison published inline after desktop/mobile/theme and
actual control-event checks; original model remains until clean validation.

**2026-10-24 clean smoke13031 COMPLETED/0:**54 BC tests passed. Actual training
completed64 steps/four interim evaluations with saved checkpoints and cast-guard
activation; forced plateau stopped at4 steps/two checks and retained step0 best.
D/F/ward output-mask assertions passed. Durable `v7-clean-20261024/smoke-result.json`
is passed. Full clean13032 released and RUNNING; no optimizer metrics yet.
Existing converted replay arrays are reused; no reconversion, simulator or RL.

**2026-10-24 clean13032 CANCELLED by UID1000 after3h10m45s.** Slurm records
explicit cancellation, while trainer labels its TERM handler `preempted`; this
does not establish scheduler pre-emption. No automatic restart. Cache build
completed3250s/503.8GB. Training reached6453 steps and saved NFS latest plus
`ckpt_000006453.msgpack`; sampled best5500 preserved separately (score7.5057,
button accuracy91.01%, click distance13.22). Cast recall remains weak; no
standalone-quality claim. Last evaluation6000 rejected Q regression despite
improved aggregate7.480. Run `modern-bc-v7-clean-13032-20261024-072516-?` under
`/mnt/nfs/checkpoints/lanerl-jax/modern-bc`. Scratch cleanup not independently
verified (desktop SSH check pending); sole checkpoints are durable on NFS.

**Resume authorized by Dani.** Prepare6453-step continuation with original
3-epoch ceiling and unchanged recipe. Resume now optionally restores donor
monitoring: same validation sample, step2000 cast reference, best5500 checkpoint
and stale1 at6000, instead of resetting them. Add recovery test plus real
resume/forced-stop smoke before releasing the full job. No new conversion/RL.

**Resume13034 queued afterok:13033**, source2568861a, user-authorized. Smoke13033
runs full BC suite and actual2→4-step resume: preserve best0 bytes, reference2,
stale1 and stop at inherited patience2. Full continuation restores checkpoint6453,
best5500, reference2000 and stale1; recipe and3-epoch total ceiling unchanged.
Both registered with this thread. GPU full24GB/4CPUs/16h cap/no-requeue; immutable
source/private deep copies, NFS checkpointing and EXIT cleanup. Desktop reachable
again; no13032 scratch directory remains,684.7GB free. Cache must be rebuilt
(~54min measured) and arrays staged (~46min previously); no replay reconversion.
Rough resumed wall time2–4h if plateau soon,10–12h if reaching remaining budget,
excluding queue/pre-emption. First resumed evaluation expected~2h after launch.

Observed clean progress500→6000: event+click8.364→7.480, click distance17.96→13.31;
accepted best5500 score7.5057. At5500 Q/W/E/R exact recall1.20/0.0108/0.911/0%;
overall91.01% accuracy is not useful cast evidence. Next24h: continue while guarded
validation improves, score best checkpoint's rate/timing/target behavior by
champion/role and replay clips (~1–3h after run). Next24–48h conditional on results:
if spells remain weak, choose one targeted timing/class-imbalance or aiming
experiment with matched baseline, instead of more unfocused frozen-head probes.
Calibration may repair probability rates but is not a substitute for cast intent.
New-patch data scaling follows a defensible clean baseline; no RL or guessed CD.

**2026-10-24 resume smoke13033 COMPLETED/0:**55 tests passed. Actual2→4-step
resume preserved best0 checkpoint bytes, stale1 and cast reference2, then stopped
at inherited patience2. Durable `resume-smoke-result.json` passed. Full13034
released and RUNNING (input staging, no resumed optimizer metrics yet). Startup
Slurm cwd warning uses /tmp fallback because /srv alias is absent on desktop;
both scripts explicitly cd into private extracted code before execution, and
the actual smoke passed. Future submissions should specify --chdir=/mnt/nfs/projects/ahriuwu-modern-bc.

**Predictive failure audit13035 registered**, responding to Dani's request to
work ahead while13034 trains. CPU partition explicitly danilogin2CPUs/6GB/45min,
no GPU contention and no optimizer or simulator execution. Source2568861a plus
`failure-audit-TEMP.py` frozen at submission; private /tmp code/data/checkpoint
copies removed on EXIT, NFS results.48 deterministic champion/role-spread
perspectives (32train16val): valid/event/held and weighted button/click mass,
spell-context primary/anonymous enemy visibility, known mana/rank/CD coverage.
Accepted clean5500 checkpoint:8 natural+8 cast-centered256step training windows
measure trunk gradient norms/cosines for button/click/spell-button/spell-click
native loss contributions. Diagnostic strata only, no heldout quality claim.
Results gate task-balancing/sampling experiment versus input coverage work.

Code confirms no previous-action input, GRU reset each25.6s and missing qualified
own cooldowns; anonymous extra champions lack identity embeddings. Shared
factorized screen head is independent of chosen button. These predict repeat-cast
timing, teamfight specificity and multi-target aiming limits; their actual impact
is unmeasured. Earlier age readout failed and oracle pointer gains failed routing,
so neither is promoted or repeated blindly. Fixed-reference cast NLL guard may
reject aggregate gains; evaluate rate/precision/tolerant recall jointly before
interpreting early stop as architectural evidence. No active-training recipe change.

**Failure audit13035 FAILED before metrics:** packed entity wrapper requires a
time slice, not NumPy tuple indexing. Correct v2 materializes one game's dense
rows via `entities[:]` and asserts shape before indexing; CPU13036 registered
with the same bounded/no-update diagnostic. Original worker retained for
evidence; active13034 recipe/checkpoints unaffected. No audit findings yet.

**CPU13036 COMPLETED/0,109.9s.**48 stratified perspectives:197587 valid train/
103350 validation decisions,4352 QWER events total. Movement/basic-attack button
accounts for97.964%/97.468% of weighted click supervision; QWER only2.036%/2.532%.
Even after5x button weights, movement+noop comprise87.26%/85.35% button mass.
Own cooldowns known at0 sampled spell events; mana known atall4352. Extra anonymous
enemy champions visible at24.27%/18.72% of spells, and are the only visible
enemies at7.71%/6.45%. Paired enemy absent44–45% includes farming/self casts;
not evidence those labels are wrong. Gradient audit:8 natural/8 cast-centered
windows, click-versus-spell-button cosines approximately-0.113..0.076; no
consistent destructive alignment. Natural click norms1.51–2.73x button norms.
Small diagnostic, not causal proof or full-corpus statistics.

**CPU13037 registered:** paired native control versus spell-aim x5 weighting
(~9.4% expected click mass instead of2.0%, global normalization retains mean
click coefficient). Same clean5500 parameters, button weights, natural sampled
windows, fresh Adam3e-5/clip0.5,batch4/window256/burn32;200 updates/arm max.
Selection every50, patience3 including baseline.32train perspectives,16 previously
audited validation perspectives split by replay base into selection/assessment,
4 fixed windows/game. Select lower spell-click CE only with movement CE+0.03,
distance+0.25, event-button NLL+0.01 and supported per-QWER NLL+0.05 gates.
Original loss equivalence asserted before updates. Assessment logs expected cast
counts, exact recall and joint targeting. Existing validation reuse, small subset,
no fresh-test or promotion claim. CPUdanilogin2cores/6GB/90min cap, independent
private /tmp staging and NFS outputs; GPU13034 unchanged, still staging at check.

**CPU13037 COMPLETED/0,2175s (36min).** Both arms select baseline0; all50/100/150
updates fail prespecified event-button NLL+0.01 guard, so no promotion and saved
assessment remains baseline. Selection spell-click CE baseline7.98077: native
50/100/150=7.82395/7.89457/7.87294; weighted=7.74550/7.76365/7.74116. Treatment
consistently better aiming than matched control, but event NLL rises0.013–0.023
(vs native0.012–0.021). QWER conditional NLL improves in both, movement CE within
gates. This is not proof weighting cannot help; small-batch fresh-Adam native
control also fails. Only58 selection/101 assessment spell clicks; R assessment5.
Baseline assessment expected Q315 vs59 human, W103vs27/E110vs27/R19vs5: same
class-weight calibration concern as earlier v6, despite low argmax recall.

**CPU13038 registered, no optimizer:** fixed clean5500 checkpoint,64 existing
assessment perspectives/full valid frames. Measure per-QWER average precision
(exact and0.5s-near-press positives), expected counts and argmax recall, fixed
inverse-button-weight correction, and oracle human-count ranking diagnostic.
Separates ranking/discrimination from biased probabilities before further
training changes. Existing validation reused, no fresh-test claim; human-count
threshold not deployment policy, inverse weights cannot exactly undo held-label
weighting.2CPUs/6GB/45min danilogin, private code/data/checkpoint copies and EXIT
cleanup, NFS outputs. Main13034 unchanged, cache22000 at receipt check.

**CPU13038 COMPLETED/0,675.5s:**64 existing heldout perspectives,358473 valid
decisions at clean5500. Q/W/E/R exact AP3.732/1.845/3.062/0.780% versus class
prevalence0.696/0.312/0.372/0.0787%; signal above prior, weak exact localization.
Within0.5s-near labels AP29.23/16.03/25.09/8.20% versus near prevalence
7.12/3.23/3.80/0.803%. These are frame rankings, not event-match recall or a
deployable policy. At oracle human-count budget exact precision7.69/5.36/6.75/
0.71%. No inference that unmatched frames are necessarily invalid human choices.

Raw expected counts Q/W/E/R14454/5863/7376/1325 versus2496/1120/1334/282human.
Fixed inverse-button-weight correction gives3591/1394/1830/321, but exact AP
3.782/1.815/2.986/0.780% barely changes and argmax cast recall goeszero. Counts
correction is not timing recovery; held-label weights prevent exact analytic
calibration. Casting-context probabilities are elevated around human casts,
but calibration alone cannot recover a strong policy at this checkpoint.

Earlier history readout already had a64-wide nonlinear context+age head, so do
not claim it lacked interactions or blindly rerun the same approach. It used the
old pre-target-repair checkpoint and mismatched all-frame training/event selection;
negative result is limited, but no history improvement is established. Do not
launch another broad retrain or frozen head on this early checkpoint now. Let
main13034 produce later clean parameters; repeat the same fixed ranking/rate/
timing evaluation on the selected checkpoint before deciding whether causal
own-action history requires a full-gradient input ablation. No deployment or
production change promoted. Main13034 still active in cache construction at check.

**2026-10-24 scope correction / expert review:** actual finalized plan uses
30–840s (00:30–14:00) at10Hz, not full matches. Valid labels additionally require
alive and inside the lane corridor, even with extra_champions=True.7309 matches
provide58470 early-game player perspectives;172 champions; match-base split
6574train/735val. Earlier "full" meant full configured8100-step window and
corpus, not full-match or unrestricted-map coverage. Tier is constant0. This
limits standalone full-game claims and requires future data-scope expansion.

Current target-type supervision: received self/no-cursor modes mask aim loss;
received unit targets use one-hot grid cells, other received points Gaussian
sigma1. No explicit skillshot metadata input or targeting-mode head, no
action-conditioned aim; shared independent button/X/Y heads. Actual185-wide
self input has158 never-observed columns masked, leaving27 potentially observed
(including role/IDs/tier). No kit descriptions, own qualified cooldowns or prior
action inputs. Reviewer brief uses exact run manifest and current source, not
PPO defaults. Within-epoch monitor's event CE retains class weights; per-class
NLL diagnostic is unweighted. Burn-in loss mask does not detach BPTT through
the32-step prefix. Model6,243,435 parameters, entity2x4head128/FFN256, context256,
MLP4x1024→512, residual LayerNorm-input GRU512.

**Resume13034 CANCELLED by UID1000 after1h57m50s.** Completed staging/cache,
restored best5500/stale1/reference2000, then validated6500 score7.46719 and
accepted it. E recall1.99%,Q1.10%,W0.194%,R0% at6500. Saved step7000 checkpoint
before next evaluation; no completed7000 metrics. Slurm batch endedSIGKILL,
manifest has no terminal status; do not infer graceful completion or cleanup.
Best6500 and latest7000 checkpoint payloads readable on NFS. No auto-restart.
Scratch cleanup unverified. Target-supervision explanation passes728dark/360light
previews and all3 target-mode checks; published inline. No training changes.

## 15. Lane BC v2: design pivot (2026-10-24)

This section supersedes the v6/v7 training plan above. Dani adopted the supplied
“Lane BC v2 — design spec and build plan” and authorized changing the design without
preserving the old architecture. Git tag `bc-v7-before-v2-20261024` identifies the
last pre-pivot implementation. Historical checkpoints/data remain evidence and
controls, not the default model to resume. No RL or simulator-generated supervision
is part of this phase. The first implementation commit is a foundation, not a
validated v2 learner.

### Objective and data

Deliver competent replay-only early-lane imitation, Ahri-mid first and Garen-top
as a contrasting self/unit-targeted kit. Work on corpus A before scaling. Exact
current inventory: 7309 patch26.19 matches /58470 non-jungle perspectives. Ahri
678 total (599train/79val), including592mid (521/71); Garen280 total (251/29),
including236top (210/26). Train/validation are match-disjoint. These counts justify
a focused dev loop; they do not establish an adequate number of every rare event.
Do not classify the regime as insufficient from a failed F1 threshold alone.

Scope stays00:30–14:00,10Hz, alive/in-lane decisions, approximate circular vision.
No buying, leveling, summoners, wards or late-game quality claim. Corpus B is raw
next-patch acquisition, not converted training data: the read-only archive ledger
currently lists23879 verified files /326486357375bytes. Freeze its inventory and
verify decoded coverage/patch separately before use. Do not redownload or start
collector work. Locate target-champion B matches using roster metadata only;
reserve their **entire matches**, excluding every perspective in them from future
pretraining, calibration and model selection. No automatic mixing of the remaining
players of a held-out match into train. The final test is opened after design
freeze, not once per experiment. Patch identity is not a mechanism for repairing
semantic changes; metadata and parser contracts are patch-pinned.

### Event/action contract

Use9 observed semantic categories: continue, move, attack,Q,W,E,R,recall,stop.
No unused reserved head. Remove held-destination copying from the new labels.
Retain raw sub-frame action lists and provenance even when primary10Hz labels
use cast>attack>stop>move and latest tie resolution. Basic attack records currently
come from execution proxies, not proven attack-input receipts. Distinguish attack
starts/target changes from repeated executions; do not relabel every execution as
a new click. Reissue collapse (proposed150units/500ms) is an ablation after review,
not a fact about player intent. Keep event-source and ambiguity flags.

Button-valid and aim-valid masks are independent. A known cast button survives a
missing/hidden target; only its aim term is absent. Self/none has no target;
unit aims select an actual visible entity identity; point aims use a joint spatial
distribution; verified directional aims can use circular angle bins. Keep a safe
point-target representation for Ahri R until variable-distance/recast semantics
are verified. Do not infer a fixed dash or fixed3-charge state from the proposal.
No pseudo-target at the opponent when received aim is missing. No clamped label
silently counted as an exact in-screen destination: preserve world point, clipping
flag and target-valid reason, and audit out-of-view actions before choosing grid
extent or a direction/distance alternative.

Temporal smoothing is experimental, default width0 until an unsmoothed baseline
exists. The supplied add-and-renormalize kernel inflates cast mass. The implemented
`smooth_cast_targets` instead redistributes one unit of mass per received cast
only into valid continue frames and its original event frame, renormalizing at
boundaries/missing rows; nearest-cast ownership resolves overlapping neighborhoods.
It never overwrites another actual event. Aim remains supervised only at its real
cast frame with the actual button; smoothing targets never enter observations.
Compare widths0/1/2/4, report target mass and operational event rates. This is
label uncertainty, not evidence of earlier castability.

### Inputs, metadata, causality

Use named/versioned schemas rather than copied approximate dimensions24/60/12.
Remove never-observed columns only after freezing a field map. Keep current
missingness handling and100ms observation lag. Every extra champion row gets a
patch-stable identity embedding; a projected16-vector cannot simply be added to
128-wide tokens without a projection. New per-entity cast memory must use only
receipts whose caster was visible then, retain last-seen identity correctly, and
not import hidden enemy cast logs. Own received-cast ages retain history beyond
30s (R cooldowns can exceed this); unknown before first receipt remains unknown.

Keep issued-event history separate from confirmed-cast history. An issued keypress
can fail. At deployment, absent confirmation must not masquerade as a successful
cast receipt. Previous target history uses stable unit identity or coordinates,
not a pointer-slot index that can change when entities reorder. Dropout on this
input is an experiment, not a substitute for testing history corruption and
teacher-forcing exposure. No current/future human action is an observation.

Ability metadata comes from pinned client assets/CommunityDragon with hashes and
per-field evidence/availability. No simulator kit defaults. Static metadata is
explicitly authorized by this redesign, separately named from replay measurements.
Use mode, min/max effect range, width/radius, cast time, projectile speed, rank
cooldown reference, recast/charge/transform tags, damage types and target eligibility
only where semantics are qualified. Metadata width is generated from the schema.
Missing values stay masked; never map unknown mode to ground or missing cooldown
to10s. Current Ahri Q/R catalog cast ranges25000 and E10000 are sentinel values,
not usable effect ranges. Raw spell root evidence is not a verified policy mode.
Ahri's recasts require explicit handling; a universal1s refractory mask is not a
replacement for readiness. A base-cooldown-minus-age feature is an explicitly
uncertain reference, not a true cooldown, legality label or hard action mask.
Haste, resets, charges and recasts must be represented or uncertainty retained.

### Model and losses

Target architecture: structured entity transformer plus self token, retained
entity outputs, shared recurrent trunk, button-conditioned target heads. Initial
candidate:128-wide tokens,3pre-LN blocks/4heads/FF256; context256;3x1024MLP→512;
GRU512 with normalized input/residual output. Measure real parameter/memory/step
cost before fixing6–8M or claiming sub-hour runs. Existing2-block/4x1024 trunk
is the capacity control. Do not simultaneously change every depth, dropout,
optimizer and target parameterization when measuring the effect of conditioning.

Distribution: p(button|history) p(target|button,history,known metadata). Train
conditional heads with the received button, infer with the chosen button, and
report both oracle-button and actual-button target metrics. Unit pointer attends
to entity outputs (64-d query/keys) with qualified type/team/padding masks; unknown
eligibility is not silently illegal. Empty candidate set skips aim safely.
Ground/move uses a joint96x54 grid. Circular72-bin angle head is a later isolated
ablation for verified directional abilities. No value head is required for BC.

Baseline uses unweighted button CE and per-mode aim CE, valid-mask normalization;
no5x class weights or held labels. Report actual loss and gradient contribution by
mode. Natural event frequencies are the calibration reference. Any event sampling
or weighting changes must state the effective training distribution and evaluate
rate calibration separately. Auxiliary own-position+2s and opponent-position+1s
are optional ablations, as is own-HP+3s Huber regression (not CE). Future auxiliary
labels never become inputs. Enemy future-position targets require observable
future evidence; deaths, discontinuities, missing units and horizon validity are
masked. Future own displacement is relative to the current anchor, not a future
player-centered camera where own position is identicallyzero.

Start with Adam3e-4/floor3e-5/clip0.5 and existing numerical settings as a control;
AdamW, weight decay, dropout and bf16 are subsequent measured choices. Dev batch64,
window256/burn32, rematerialization; no detachment through burn-in. Benchmark64/128
and fixed update budgets before setting epochs. State carry needs contiguous
windows, resets on perspective/death boundaries, preserved input lag, and an
explicit train/inference-state protocol; never carry hidden state across shuffled
games. Ablate stateful versus reset evaluation to quantify mismatch.

### Evaluation and test isolation

Separate train, checkpoint-selection, threshold/bias-calibration and final-test
match IDs. Hold out some A-training matches for calibration instead of retuning
thresholds on the same scored validation sample. Calibrate biases/peak thresholds
there, then freeze them for validation and test. Human-rate threshold selection
on the scored set is an **oracle diagnostic**, not a deployable metric or useful
rate veto. Frame-dilated AUPRC is a frame-ranking diagnostic, not event AUPRC.

Operational evaluation runs the actual10Hz event emitter. Score one-to-one
predicted/received event matching within0.5s, using maximum-cardinality ordered
matching (nearest-pair greedy can lose valid matches). Report precision/recall/F1,
events/minute, inter-event intervals, missed/burst events, exact and tolerant
rankings, angle error, unit-pointer accuracy, movement point error and joint
button+target quality. Apply all timing masks to exclude burn-in and invalid
intervals. Angle metrics exclude near-zero direction vectors; pointer/angle targets
use consistent delayed entity identity and current frame conventions.

Select a Pareto-qualified checkpoint: event discrimination and correctly emitted
events must improve without material movement/target/calibration regressions.
Freeze metric tolerances from baseline variance and bootstrap by match. Avoid an
arbitrary scalar mixing degrees, F1 and cells with unbounded negative angle terms.
The proposed0.5F1/10deg/0.85pointer/2cell numbers are aspirational, not evidence-based
pass/fail bars or proof that a failure implies too little data. R with few labels
must show counts/confidence. Existing v7 reference2000 does not govern v2.

### Staged experiments and gates

1. Inventory and24-perspective Ahri/Garen raw-source audit; then all target games.
   Verify masks, unit IDs, targeting coverage, event/bin collisions, duplicate
   receipts, inter-event intervals, raw aiming/origin points, source-qualified
   metadata and role/match split counts. Latency and “aim at opponent” distributions
   are diagnostics; a human need not react to every enemy cast or aim every Q at
   the paired opponent. Do not force peaks at200–500ms or0degrees as correctness.
2. Versioned event/aim/history sidecars reuse observations where parity holds.
   Validate known-target synthetic fixtures plus deterministic fixed-window
   memorization and a label-shuffle leakage control. Ten-game validation improvement
   is not a leakage test. Re-run source/lag parity before any new learner.
3. Ahri-mid unsmoothed event baseline; bounded compute benchmark and actual-button
   targeting evaluation. Compare conditional versus unconditional **same target
   head/inputs** first. Then isolate pointer and angle parameterizations.
4. Own-history on/off plus a small conditioning/history2x2 if interaction matters;
   smoothing widths after unsmoothed baseline. Auxiliaries and metadata are separate
   toggles. Evaluate raw and calibrated metrics, shuffled/corrupted history, and
   at least two seeds for close decisions; bootstrap match confidence intervals.
5. All-champion pretrain→80%Ahri/20%mixed fine-tune versus target-only, holding
   target updates, total compute and calibration protocol explicit. Source-known
   simple modes first; complex/unknown modes retain button supervision and use
   point targets only with qualified received points. No need to finish172
   hand-written tables before the Ahri dev loop.
6. Expand to Garen and then frozen corpus-B test after design freeze. Keep a
   learning curve by unique match/event count to distinguish data from architecture.
   Do not run capacity/parameter sweeps on the final test. No RL is automatically
   added if BC stalls.

Live client evaluation is a separate measured dependency, not an existing harness.
Define observable fields and successful-action feedback first. Canonical-to-client
projection is not assumed to be a fixed affine map; calibrate actual camera/world
projection and latency. Replay adapter parity is necessary but does not establish
live observation parity. Blind50ms click retries and1s spell suppression can alter
commands/recasts; neither is a default. Online CS targets require an explicit
buying/leveling/loadout protocol because those behaviors are out of policy scope.
Do not promise competent real-client play without that adapter and scope bridge.

### Research grounding and current build state

[JueWu-SL](https://arxiv.org/abs/2011.12582) reports competitive supervised-only
play with hierarchical actions, intent tasks and scene-based sampling; it is not
proof that RL is necessary for competent BC, nor a guarantee for our smaller and
partially observed corpus. [Tencent's control paper](https://arxiv.org/abs/1912.09729)
provides target attention/action masks/control-dependency motivation.
[DI-star's action head](https://github.com/opendilab/DI-star/blob/main/distar/agent/default/model/head/action_type_head.py)
provides a concrete teacher-forced versus sampled conditioning implementation.
Robotics chunking is a hypothesis to test here, not a demonstrated MOBA improvement.

Implemented v2 modules include event labels, compact causal inputs, received-event
history, button-conditioned pointer/joint-grid policy, losses, batches, bounded
learner/checkpoint resume and stateful replay evaluation. Source audit13041 passed
24 perspectives; target conversion13045/finalizer13047 passed828 Ahri/Garen
perspectives with isolated calibration/validation. Gradient13044 passed basic
wiring; learner13049 completed training/evaluation but failed its bitwise-like
resume comparison. Probe13055 verified exact checkpoint fidelity and bounded
numerical continuation agreement. Strict memorization13063 is outstanding; an
exclusive batch-throughput benchmark is gated after it. No broad Ahri training
or accepted held-out quality result exists. Evaluation uses replay-provided history
even when scoring actual-button targets, so it is offline conditional assessment,
not closed-loop rollout evidence. Calibration, shuffle control and staged ablations
remain to build. See STATUS and the fidelity ledger for subsequent receipts.

V2 job restart protocol: short validation jobs use explicit `--no-requeue` and
`--open-mode=append`. Job13050 demonstrated that Slurm's default requeue can
restart a script against its already-created output path and truncate the earlier
log. Keep unique job/attempt output paths and preserve failed evidence; never make
`exist_ok=True` an implicit checkpoint-resume policy. Until a job wrapper implements
and validates attempt-aware staging, checkpoint restoration and the original total
budget, production learners also use explicit no-requeue and resume via a newly
registered job after terminal feedback. A fresh output directory alone does not
resume a training budget.
