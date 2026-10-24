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
