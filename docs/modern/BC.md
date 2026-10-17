# Behaviour cloning from decoded replays (modern world)

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

v1: the button head predicts noop 99.6% of the time (move recall 0.8%); the one-per-click labels do not
carry enough signal, see section 3. In the sim it samples noop 77% and move 21%, farms half the human CS
against an afk opponent and dies twice in nine minutes, with deaths uncorrelated with its lane distance
(turret and minion deaths from aimless walking rather than wandering off). The gate itself runs 64 full
episodes in 4 minutes on the 5080 (`gate-afk-*.json` next to the checkpoint).

