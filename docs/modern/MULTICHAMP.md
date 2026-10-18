# One policy for every top laner: champion identity, kit descriptors, data plan

Status 2026-10-18: design agreed with Dani (thread 200b87b6, after the RL thread's research pass of
2026-10-15, `/mnt/nfs/shared/research_multichamp_20261015/REPORT.md`). Implementation in progress on branch
`modern/bc`. Nothing here changes the default `modern-world-v1` observation until the regression in §7 passes.

## 1. Decisions carried over from the research

- One shared network for all champions. Per-champion models need ~120k games each (JueWu-SL); we have a few
  hundred per champion.
- Champion identity enters as a learned embedding **and** as kit descriptors (range, cooldown, cast type,
  dash / crowd-control flags per ability), the OpenAI Five recipe. A bare tag over 160 champions with thin
  data per champion would barely train.
- The action space stays generic: Q/W/E/R slots, one screen click, summoners, recall, shop, ward, stop. No
  per-champion heads. Click validity per slot comes from the ability's cast type.
- Mixed-skill data with the player's rank as an input (AlphaStar Unplugged: all tiers plus a high-skill
  fine-tune beat high-skill-only). The 16.19 archive is apex-tier only; the 26.20 low-elo pool adds the
  spread later.
- Class/role heads: untested in the literature; not built. Revisit only if descriptors prove insufficient.

## 2. What the policy sees today (`modern-world-v1`) and what is missing

Own and enemy champion are two one-hots over the 14 kits the sim implements (`obs.MODERN_WORLD_SELF_DIM =
28 + 2*14`). Own cooldown fractions are normalised by the sim kit's cooldown table (`cooldown_row`), so a
champion without a sim kit cannot be the actor. The enemy may be unknown (one-hot all zero; the BC trainer's
`--enemy-id-dropout` trains for it) but then nothing says what its abilities do. The model has to memorise
"Garen's Q is a self-cast on an 8 s cooldown" from the tag.

## 3. Sources that cover every champion on the 16.19 build

| what | source | coverage | provenance |
|---|---|---|---|
| ability definitions: cast type, range by rank, cooldown by rank, cast time, missile speed, spell tags | decoder catalog generator (`add_champion.generate`, frozen v7 snapshot `d3c81556`) run over the saved client | 173 champions (`/mnt/nfs/datasets/rofl_champion_catalogs_2619_20261018/`, job 8177) | per-file sha256 in `generation-manifest.json` |
| base stats, Riot key, resource type | Data Dragon 16.19.1 (`/mnt/nfs/shared/build-research/raw/ddragon/champion.json`) | 173 | file sha256 |
| live stats, cooldown countdowns, ranks, items | the replay export (already read by `bc/convert.py`) | every exported game | decoder commit per batch |

Root spell of a slot: the catalog's `slots[s]` names the ability object; the slot's root spell is the spell row
whose `slot.provenance` has that entry (checked: exactly one per slot for all 14 existing catalogs).

Spell tags that become flags (from `raw_spell_definition.mSpellTags`): dash = `PositiveEffect_MoveBlock` or
`Trait_PlayerSelectedDashDirection`; hard CC = `Trait_ImmobilizingCCSpell` / `Trait_ImmobilizingCCAbility`;
knockback = `Trait_KnockBack`; shield = `Trait_Shield`; attack reset = `Trait_AttackReset`; AoE = `Trait_AoE`;
recast = `Trait_RecastOrReplaceSpell`. Tags are definitions, not runtime behaviour; a flag on a slot means the
root spell declares it, no more. Flags are OR-ed over the slot's root and its children (Camille E's dash is on
the child `CamilleEDash2`).

## 4. Observation profile `modern-world-v2`

Entities (32 x 20) and the global vector (6) are unchanged. The self vector drops the two 14-wide one-hots
and gains, per side (own, enemy):

| block | columns | source |
|---|---|---|
| base stats | attack range/600, move speed/500, HP/1000, AD/100, armor/100, MR/100, resource one-hot (mana, energy, other) | Data Dragon |
| per ability Q/W/E/R (x4) | cast type one-hot (self, unit, point, direction), range/1000, cooldown at rank 1 /60, cooldown at max rank /60, cast time /1, dash, hard CC, knockback, shield, attack reset, AoE, recast | catalog root spell |
| own only | rank/5 per slot (4) | state |

Sizes: base 9 + 4 x 16 = 73 per side, 146 for both, plus 4 ranks. Self vector = 28 (v1 columns without the
one-hots) + 150 = 178 floats.

Three identity inputs ride in the first three self columns as floats (exact small integers, so no signature
changes anywhere): own champion index, enemy champion index, skill tier. Index 0 is "unknown"; champions are
1..173 in Riot-key order from the champion table; tiers are 1..9 Iron..Challenger, 0 unknown. The 16.19
archive is apex-sourced and the footer carries no rank, so its games are recorded as tier 8 (Master) by
convention (`champion_table.ARCHIVE_TIER`), the value the sim reports too. The policy (`PolicyConfig.profile ==
"modern-world-v2"`) embeds them (`nn.Embed`, one 16-wide table shared by own and enemy, 4 for the tier) and
concatenates them with the remaining self columns and the global vector before the context layer. Everything
else in `LanePolicy` is unchanged. Sizes as built: self vector 181 = 3 ids + 28 + 4 ranks + 2 x 73.

For the 14 sim kits the table's cooldowns are the sim's own tables (two catalogs lack a root Cooldown field),
so sim and table agree; 12 slots across other champions have no cooldown in the client data and take the
10 s default. Cast types: the sim casts skillshots without a click ("self" in its kit rules) where the table
says "point"; the labeller keeps the kit rule for the 14 kits and uses the table for everyone else.

Cooldown normalisation (own fractions, enemy cast memory) reads the champion table's cooldown-by-rank, not the
sim kit, so any champion can be the actor. The sim resolves `cfg.champion_ids` (Riot keys) to table rows
through a static (1024,) lookup.

## 5. Action profile

`modern-world-v2` actions stay as they are. The labeller's `CastRules` come from the table's cast types
(self: no click; unit: needs an enemy unit under the cursor; point/direction: a ground click), cross-checked
against the 14 sim kits' `UNIT_TARGET_RANGE` in a test. In the sim only the 14 kits can be cast, as now.

## 6. Pipeline changes (all on branch `modern/bc`)

1. `modern/data/build_champion_table.py` -> `modern/data/26.19/champion_table.json`: one row per champion with
   index, Riot key, Data Dragon id, WAD name, base stats, resource, per-slot descriptors, and the source sha256s.
2. `modern/obs.py`: `build_modern_observation(..., profile=)`; v2 adds the descriptor block and `self_ids`.
   `ModernObservation` gains `self_ids`. `PROFILE` stays v1 until §7 passes.
3. `modern/rl/policy.py`: `PolicyConfig.profile`, embeddings; width check by profile.
4. `bc/convert.py`: self columns by profile; descriptors by name from the table; skill tier from the footer
   (the survey rows carry it); `cast_rules` from the table. `bc/dataset.py` records the profile.
5. `bc/train.py`, `bc/gate.py`, `bc/record.py`, `modern/train.py`: carry `self_ids`; checkpoints record the
   profile so `--init-from` refuses a mismatch.

## 7. Regression before switching the default

Retrain Garen (v3 recipe: class weights x5 on casts, 24 epochs) and Jax on v2 and compare with the v1 runs:
weighted validation NLL, per-button recall, and the 64-episode gate (CS, deaths, lane distance) against the
afk Jax. Pass = no gate metric worse by more than its seed noise (about 3 CS, 0.3 deaths) and validation
NLL within 0.02. Then one run with the enemy embedding dropped out, to see that the descriptors alone carry
the matchup.

## 8. Data plan

Footer survey of the 16.19 archive (`/mnt/nfs/datasets/rofl_top_survey_2619_20261018/summary.json`, job 8155,
14 min, footers only): 11,122 replays, 7,663 on the supported build 16.19.823.0722, 7,658 with one footer-assigned
top laner per team, 15,316 top-lane champion-games over 157 champions, 2,492 distinct matchups.

| champions with at least | champions | top-lane champion-games | share |
|---|---|---|---|
| 1 game | 157 | 15,316 | 100% |
| 20 games | 94 | 14,850 | 97% |
| 50 games | 57 | 13,683 | 89% |
| 100 games | 48 | 13,060 | 85% |
| 200 games | 31 | 10,632 | 69% |

Most played: Jayce 615, Yone 570, Ambessa 552, Fiora 549, Camille 529, Malphite 464, Renekton 435, Aatrox 399,
Jax 379, K'Sante 375, Ornn 374, Garen 366, Darius 326, Urgot 316, Gnar 313. Most common matchups: Jayce-Malphite
41, Ambessa-Renekton 39, Malphite-Yone 38, Ambessa-Camille 35, Camille-Garen 29.

Export cost at decoder v7: about 117 s per actor-export on one desktop CPU; 6 workers gave about 40 games/h
in practice (the node is shared). The low-elo 26.20 pool (1,000 games, Iron..Diamond) joins when the decoder
thread declares 26.20 parity.
