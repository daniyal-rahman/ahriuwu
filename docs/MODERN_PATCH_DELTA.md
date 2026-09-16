# MODERN_PATCH_DELTA.md — what changes between patch 4.20 and modern League

**Scope:** 1v1 Garen top lane, two champions, one lane, roughly ten minutes.
**Audience:** engineers implementing the migration. Not players.
**Modern column pinned to patch V26.18** (10 September 2026) unless a row says otherwise.
**Baseline column** is the vendored LeagueSandbox server's `Content/` data, which targets client
patch 4.20 (Season 4, November 2014) — with important caveats in §3.

This document exists to make Stage J5 of `JAX_REWRITE_PLAN.md` plannable. That plan's D1 states
the migration contract precisely: *"Parity is asserted against the server; the modern patch is a
table swap plus flagged rule changes."* Everything below is sorted into exactly those two bins.

---

## 1. Executive summary — the five changes that would most break a 4.20-trained agent

Ranked by how wrong a policy trained on the vendored server would be when deployed on modern
League. These are not the largest numeric deltas; they are the ones that invalidate something
the agent *learned*.

**1. Last-hitting no longer pulls the enemy minion wave onto you.** (§5.9)
In 4.20 the minion aggro priority list had seven entries, and entry 5 was *"enemy champion
attacking an allied minion."* **That entry was deleted in V26.10, four months ago.** The list had
been stable for eleven years. Under 4.20 rules, every auto-attack on a minion carries a risk of
pulling six minions onto you; under modern rules it carries none. This changes the cost of the
single most frequent action the agent takes, and it is a one-line change in the data that
invalidates a large amount of learned caution.

**2. The lane's clock and its last-hit breakpoints both moved.** (§5.1, §5.5)
The first wave arrives at **0:30** instead of 1:30, and minions now gain HP and AD **every 90
seconds** (`U(t) = 1 + floor((t−30)/90)`) where 4.20 minions were static. Melee minion HP climbs
465 → 675 across the first ten minutes. Every last-hit timing the agent learned is calibrated
against a constant that is no longer constant, and the episode contains 20 waves where the
vendored server produced 15.

**3. Garen's E is a different ability, and it changes his build.** (§4.6)
Judgment now ticks **7 times plus one per 25% bonus attack speed** (it was a fixed 6), **deals
full damage to minions** (the 0.75× modifier is gone), can **critically strike**, and applies
**25% armor shred** after 6 hits. Attack speed went from irrelevant-while-spinning to a primary
damage stat — which is why modern Garen buys Berserker's Greaves and Phantom Dancer. Wave clear,
last-hitting under E, and the entire itemisation logic all move together.

**4. Turret plating adds a reward object that does not exist in the 4.20 world.** (§6.4)
Five plates, **120 gold each, 600 per turret**, and since the 26.1 rework they no longer expire
at 14:00. A 4.20-trained agent has never had a reason to hit a turret in the laning phase and has
no representation for plate value. This changes the *reward landscape* of the lane, not just its
physics — arguably the largest strategic addition in the document.

**5. Spacing priors are wrong in two directions at once.** (§4.1, §4.4)
Garen's attack range went **125 → 175** (+40%), and Q became a **dash with an attack-timer
reset** where it used to be a stationary self-buff. Every "how close is safe" and "can I reach
him" judgment the agent made is calibrated on the wrong geometry. Add the **top-lane-specific
sidelane speed buff** (§5.7, +111 decaying bonus MS on side-lane waves before 14:00, new in
V13.10) and even *where the wave meets* is different every wave.

**Runner-up, worth knowing:** R's missing-health execute shape was already present in 4.20 (the
server's own script confirms it), but its damage type changed from **magic to unconditional
true damage** (§4.7), which removes the target's MR from the kill-threshold calculation
entirely; and **death timers are shorter at levels 3–7** in modern League (§9.3), which is
exactly the band a ten-minute lane lives in — the opposite of the usual assumption.

**The single most useful negative result:** the per-level stat growth curve
`perLevel × (0.65 + 0.035 × Level)` was **introduced by patch 4.20 itself** and is still the live
formula in 2026 (§4.2), and the **XP-to-level table is bit-for-bit identical** (§9.4). Two of the
scariest-looking pieces of engine math need no migration work at all.

---

## 2. How to read this document

### 2.1 The two bins

Every row is classified. This distinction is the document's main product.

| Class | Meaning | Migration cost |
|---|---|---|
| **NUMBER** | A value moved. The rule computing it is unchanged. | **A table swap.** Goes in the patch table, costs nothing but data entry. |
| **RULE** | The logic changed, or a mechanic was added or removed. | **Code, behind a feature flag.** |
| **NO CHANGE** | Verified identical. | **Nothing** — and worth recording so nobody re-derives it. |

Sub-labels used where they help: **RULE (new)** for mechanics that did not exist in 4.20,
**RULE (removed)** for ones that existed and are gone (which can also mean *deleting* code), and
**NUMBER, coupled** for numbers that cannot be swapped independently of another change (§7.1 is
the important instance).

### 2.2 Priority

The reader's scope is two champions, one lane, ten minutes. Priorities reflect *that*, not
League generally.

| Priority | Meaning |
|---|---|
| **HIGH** | Binds inside a ten-minute 1v1 lane. Affects trades, CS, levels, gold, or positioning directly. |
| **MEDIUM** | Real but second-order, or only binds late in the window, or has small magnitude. |
| **LOW** | Exists in the lane but rarely decides anything within scope. |
| **N/A** | Cannot occur in a 1v1 lane episode. Listed only so it can be explicitly skipped. |

Several genuinely large League changes are marked LOW or N/A here — Rift Herald, Objective
Bounties, super minions, tier-2 and inhibitor turrets, post-14:00 wave rules. **That is the
document doing its job.** A mechanic that only appears at dragon or in a teamfight is not this
project's problem.

### 2.3 Volatility

Champion and item numbers change every two weeks. Structures do not.

- **Structurally stable** — the *existence* of turret plating, of Runes Reforged, of ability
  haste, of the 90-second minion upgrade loop, of the `base + k·U` scaling shape. Safe to build
  against. Some of these have been stable for nine to twelve years.
- **Patch-volatile** — every specific damage number, ratio, cooldown, gold value and rune
  coefficient. These belong in a dated constants table keyed to a patch number, never in engine
  logic. Garen received balance changes in at least 8 of the last 20 patches; the minion numbers
  were rewritten wholesale in V26.01 and were still being corrected as recently as V26.10.

**Recommendation:** pin the patch table to **V26.18** explicitly and date it. The 2026 preseason
(V26.01) touched nearly every minion and turret number, and the follow-ups (V26.09, V26.10,
V26.14) were still settling it through mid-2026.

---

## 3. Provenance, and what the 4.20 baseline actually is

### 3.1 Sources

Modern values come from the League of Legends Wiki (`wiki.leagueoflegends.com`), fetched
server-rendered via `api.php?action=parse&page=<X>&prop=wikitext|text` — the `prop=text` form is
necessary because the scaling templates only resolve in rendered HTML. Where possible, values
were confirmed a second way against Riot's shipped data: Data Dragon (`16.18.1`) and
Community Dragon raw game files
(`raw.communitydragon.org/latest/game/data/characters/...`). Rows confirmed twice are marked ✔
in their sections.

Historical 4.20 values were checked against genuine `leagueoflegends.fandom.com` revisions dated
**7 and 15 November 2014** — snapshots from days either side of 4.20 shipping — and against
official patch-note transcriptions ([V4.20](https://wiki.leagueoflegends.com/en-us/V4.20),
[Surrender at 20](https://www.surrenderat20.net/2014/11/patch-420-notes.html)).

Every claim is cited to a page and section. Where a claim could not be confirmed it is marked
**UNVERIFIED** in place, and all such flags are collected in §13.

### 3.2 The uncomfortable finding: the vendored server is not a faithful 4.20 replica

Several baseline numbers do not match what patch 4.20 actually shipped. This is not a research
gap — it is a property of the oracle, and it changes how the migration should be framed.

| Area | Vendored `Content/` | Real patch 4.20 | §  |
|---|---|---|---|
| Outer turret HP / armor | 1550 (1300+250) / 67 | **2000 / 100 flat** | §6.1 |
| Minion stat growth | **none — static all game** | HP growth **kept and buffed**; only resistance growth removed | §5.1 |
| Wave period | **36.4s** (counter quirk) | **30s** | §5.5 |
| Melee minion gold | 20 | 19 (20 arrived in **V4.21**, after 4.20) | §5.3 |
| Caster minion gold | 10 | 14 — **no patch note supports 10** | §5.3 |
| Siege minion gold | 35 blue / 30 red | 40 — **no support for 35 or 30** | §5.3 |
| Minion XP | 77 / 51 / 94 / 500 | base 64 / 32 / 100 / 100 (58.88/29.44/92 solo) | §5.4 |
| Garen R damage type/shape | server: **magic**, `175×rank`, **already has** a missing-health term (§4.7) | wiki narrative (pre-`V9.20` patch notes) reads as flat magic damage with true damage reserved for the Villain | §4.7 |
| Garen Passive regen | server: **level-bracketed** {0.4/0.8/2.0%}/s, lockout {9/6/4}s (§4.3) | a Data-Dragon-`4.20.2` read reported a flat 0.4%/s, 10s lockout | §4.3 |
| Turret `SpellBlock` (MR) | server: **100** flat | patch notes: "100 in each" (armor **and** MR) | §6.1 |

What *does* corroborate exactly: minion HP, AD, armor, attack speeds and attack ranges; Garen's
entire stat block (confirmed against Data Dragon `4.20.2`); Garen's R cooldowns of 160/120/80s
(often assumed to be a LeagueSandbox variant — it is not, it is correct for the era); the
per-level growth formula; and, per §3.3, the server's turret MR of 100 — which, unlike its
armor of 67, actually **matches** the real-4.20 patch note.

**Three consequences for the migration:**

1. **"4.20" is not a reliable semantic anchor for what the agent learned.** The policy learned
   *the server*. On turrets and minion economy the server sits somewhere near, but not on,
   patch 4.20.
2. **Do not frame the migration as a clean 4.20 → modern delta in the economy tables.** Frame it
   honestly: *replacing numbers we could not corroborate with numbers we can corroborate twice.*
3. **Some "modernisations" are really bug fixes.** The 36.4s wave period is the clearest case —
   moving it to 30s is a *4.20 correctness fix*, available without leaving the parity target,
   and the current parity suite would flag it as a regression. Worth separating in the plan from
   genuine modern-patch changes.

None of this argues for changing the 4.20 column. `JAX_REWRITE_PLAN.md` R2 already commits to
*"match the server's behaviour, including its deviations from real League."* It argues for
labelling the column accurately and expecting the deviations to show up as transfer gaps.

### 3.3 The strongest available source for Garen's kit: the server's own spell scripts

Content JSON gives base stats and per-rank effect arrays, but it does not give *formulas* —
those live in the vendored server's executable spell logic, not in a data file. For §4.3–§4.7
this document's 4.20 column is read directly from
`Content/LeagueSandbox-Scripts/Characters/Garen/{Q,W,E,R,CharScriptGaren}.cs` and
`Content/LeagueSandbox-Scripts/Buffs/Garen/{GarenE,GarenW,GarenWPassive,GarenPassiveHeal}.cs`.
This is a stronger source than either the wiki or Data Dragon for the specific question "what
does *our* server do": it is the literal code that runs, not an inference about what code
*should* run given a patch-note trail.

Two corrections fall directly out of reading it, and both matter enough to flag here rather
than only in their sections:

- **R already has a missing-health term in 4.20.** `R.cs` computes
  `damage = 175 × rank + [0.2857, 0.3333, 0.4][rank−1] × (maxHP − currentHP)`, dealt as
  **magic** damage. A wiki-only reading of the pre-`V9.20` patch history — which discusses R's
  damage type only in the context of the since-removed Villain mechanic — could reasonably
  produce "flat magic damage, no missing-health scaling," and an earlier pass in this research
  did exactly that. The server's own code says otherwise: the execute *shape* was already
  present. What changed later (§4.7) is the damage *type* (magic → unconditional true, `V9.20`)
  and the *coefficients* — not the presence of a missing-health term. **Treat the script read as
  authoritative for the 4.20 column; treat the flat-damage framing as a plausible but incorrect
  inference, superseded here.**
- **Perseverance's regen is level-bracketed, not flat.** `GarenPassiveHeal.cs` defines
  `HEALTH_PERCENTAGES = {0.004, 0.008, 0.02}` (0.4%/0.8%/2.0% of max HP per second, by level
  bracket 1–10/11–15/16+) and `OUT_OF_COMBAT_COOLDOWNS = {9, 6, 4}` seconds, ticking every
  1000ms. A separate research pass, reading only a Data-Dragon `4.20.2` snapshot, reported a
  flat 0.4%/s with a 10s lockout — plausible-looking but not what the server executes. §4.3
  uses the script values.

Q, W and E's formulas (§4.4–§4.6) were also cross-checked this way and are noted per-section;
those turned out to agree with the wiki-derived figures already in this document (with one
notational nuance on Q's AD ratio, flagged in §4.4) and needed no correction.

---

## 4. Garen — base stats and kit

### 4.1 Base stats

Source for modern column: [Garen](https://wiki.leagueoflegends.com/en-us/Garen), infobox
("Last changed: V26.14"), cross-checked against Data Dragon `16.18.1`. Source for the 4.20
column: the vendored `Content/.../Stats/Garen/Garen.json`, cross-checked against Data Dragon
`4.20.2` (exact version match — the two agree).

| Stat | 4.20 | Modern (V26.14) | Class | 1v1 priority |
|---|---|---|---|---|
| Base HP | 616.28 | 690 | NUMBER | HIGH |
| HP/lvl | 96 | 98 | NUMBER | HIGH |
| Base HP5 | 7.84 | 8 | NUMBER | MED |
| HP5/lvl | 0.5 | 0.5 | — | LOW |
| Base AD | 57.88 | 69 | NUMBER | HIGH |
| AD/lvl | 3.5 | 4.5 | NUMBER | HIGH |
| Base armor | 27.536 | 38 | NUMBER | HIGH |
| Armor/lvl | 2.7 | 4.2 | NUMBER | HIGH |
| Base MR | 32.1 | 32 | NUMBER (trivial) | LOW |
| MR/lvl | 1.25 | 1.55 | NUMBER | MED |
| Move speed | 345 | 340 | NUMBER | HIGH |
| **Attack range** | **125** | **175** | **NUMBER (but see note)** | **HIGH** |
| Base attack speed | 0.625 | 0.625 | — | HIGH |
| AS growth/lvl | 2.9% | 3.65% | NUMBER | HIGH — feeds E tick count now (§4.6) |
| Gameplay collision radius | absent → engine default 40 | 65 | NUMBER | MED |
| Pathfinding collision radius | 35 | 35 | **unchanged** | MED |
| Selection radius | n/a | 120 (raised from 75 in V14.9) | new field | LOW |

**Attack range 125 → 175 is the sleeper entry in this table.** It is nominally "a number", but
a 40% increase in melee reach changes every spacing decision in the lane: which trades are
available, when a last hit is safe, and how close Garen must walk to start a fight. Any policy
trained on 125 has learned a spacing prior that is simply wrong on 175. Changed in **V5.16**
(2015), alongside the Q rework. Treat it as a table swap to *implement* and a behavioural
break to *expect*.

**Base-stat change timeline** (each confirmed as an itemised bullet in the
[Garen](https://wiki.leagueoflegends.com/en-us/Garen) "Patch history" section) — useful if you
ever want to target an intermediate patch rather than jumping straight to current:

- **V5.16** (2015): HP growth 96→84.25, AD growth 3.5→4.5, armor growth 2.7→3, attack range 125→**175**
- **V5.17**: move speed 345→**340**
- **V7.22**: base AD 57.88→66, base armor 27.536→36 *(this is the Runes Reforged patch — see §7.1; base stats went up because rune stats went away)*
- **V8.3**: base HP5 7→8
- **V9.20** (preseason 2020 rework): AS growth 2.9%→3.65%, base HP 616.28→620, HP growth 84.25→84
- **V10.6**: MR growth 1.25→0.75
- **V11.1**: base MR 32.1→32
- **V12.10**: base HP 620→**690**, HP growth 84→**98**, armor growth 3→**4.2**, MR growth 0.75→**1.55**
- **V13.8**: base armor 36→**38**, base AD 66→**69**

> **Data Dragon trap.** Data Dragon's `attackdamageperlevel` field currently reports `0` for
> Garen. That is a stale/broken field, not a real value. The live wiki infobox and the patch
> history both independently give **4.5** (set V5.16, never changed since). Do not source
> modern Garen AD growth from raw Data Dragon.

### 4.2 The per-level growth formula — **no migration work needed**

This is the most load-bearing negative result in the document.

The server's growth curve, `perLevel * (0.65 + 0.035*Level)`, is not a LeagueSandbox quirk and
is not stale. Per
[Champion statistic](https://wiki.leagueoflegends.com/en-us/Champion_statistic) (stat growth
section), which cites the official 4.20 patch notes directly:

> "Statistics having a 'base' value was established in patch V4.20... The same patch changed
> stat growth from linear to quadratic (growth increases with increasing level)."

The wiki gives the cumulative form as
`Statistic(level n) = base + bonus + g × (n−1) × (0.7025 + 0.0175×(n−1))`, whose per-level
increment is `g × (0.65 + 0.035×n)` — **exactly** the server's formula.

**Patch 4.20 is the patch that introduced the quadratic growth curve, and it is still the live
formula in 2026.** The engine-level growth code ports across unchanged. Attack speed uses the
same curve multiplied by a per-champion AS ratio.

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Per-level growth curve | `g × (0.65 + 0.035×L)` | identical | **NO CHANGE** | HIGH (because it's free) |

### 4.3 Passive — Perseverance

| | 4.20 (server script, §3.3) | Modern | Class |
|---|---|---|---|
| Heal rate | **level-bracketed**: 0.4%/s (levels 1–10), 0.8%/s (11–15), 2.0%/s (16+) of max HP, ticking every 1000ms | **1.5% – 10.1% max HP every 5s**, smooth linear-ish curve in level | **RULE** |
| Out-of-combat delay | **level-bracketed**: 9s (1–10), 6s (11–15), 4s (16+) | 8s flat | NUMBER + RULE (bracketed → flat) |
| Breaks on | any damage, **except** a specific exempt minion unit-tag list (and monsters, from level 11 on) | champion, epic monster, turret, enemy ability, **or enemy summoner spell**; minions never break it, at any level | RULE (trigger set redefined) |

4.20 values are read directly from `Buffs/Garen/GarenPassiveHeal.cs`
(`HEALTH_PERCENTAGES = {0.004, 0.008, 0.02}`, `OUT_OF_COMBAT_COOLDOWNS = {9, 6, 4}`,
`GetCorrectLevelIndex` cutting at levels 11 and 16) — see §3.3 for why this supersedes an
earlier Data-Dragon-based read that reported a flat 0.4%/10s. Modern from
[Garen](https://wiki.leagueoflegends.com/en-us/Garen) → Abilities → Perseverance.

History (modern side): the bracketed shape was smoothed into a continuous curve and the lockout
flattened at **V9.20**, briefly adding a "doubles below 25/50% HP" clause since removed, then
the flat lockout moved 7s → **8s at V10.7** (current).

**Why it's a RULE change, not a number:** bracketed and continuous regen are different
functions of state, and the level-1 comparison is not simply "who regens more" — it flips.
Server-4.20 Garen regenerates at a *higher* rate at level 1 (0.4%/s = 2.0%/5s) than modern
Garen does at level 1 (1.5%/5s); modern overtakes 4.20 by roughly level 5–6 and ends up
regenerating far more by level 18. A sustain model calibrated on one endpoint is wrong at the
other.

**1v1 priority: HIGH.** Out-of-combat sustain is the entire reason Garen can stay in lane
without potions, and it sets the clock on every poke-and-back-off trade pattern.

### 4.4 Q — Decisive Strike

| | 4.20 | Modern | Class |
|---|---|---|---|
| Cooldown | 8s flat, all ranks | **8s flat, all ranks** | **NO CHANGE** — the most stable number in the kit |
| Bonus damage | 30/55/80/105/130 | 30/60/90/120/150 (set V8.17) | NUMBER |
| AD ratio | 40% (server `Q.cs`: `30 + 25·(rank−1) + 1.40×totalAD`, dealt as one physical `DAMAGE_SOURCE_ATTACK` packet — **the "40%" is a bonus on top of the attack's own 100% AD**, matching the wiki's "40% bonus AD" framing; not a separate 140%-AD hit) | **50%** (raised V10.4) | NUMBER |
| Silence | **1.5/1.75/2/2.25/2.5s** (rank-scaling) | **1.5s flat** (flattened V5.16) | NUMBER |
| Move speed | bonus MS + cleanses slows | **35% MS** for 1.4/1.95/2.5/3.05/3.6s (duration retuned V26.05), cleanses slows | NUMBER |
| **Lunge / dash** | **does not exist** | next attack **lunges** at target (~50 units past attack range), uncancellable windup | **RULE (new)** |
| **Attack-timer reset** | **does not exist** | enhanced attack **resets the basic-attack timer** (finalised V9.20) | **RULE (new)** |
| Crit / on-hit | — | applies on-hit effects, **can crit** (bonus damage itself cannot crit) | RULE |

Modern from [Garen](https://wiki.leagueoflegends.com/en-us/Garen) → Abilities → Decisive
Strike. 4.20 values confirmed against the V1.0.0.145 → V8.9 patch-history window, which patch
4.20 falls inside.

**The dash is the change that matters.** In 4.20 Q is a stationary self-buff: you gain speed,
your next hit silences. In modern League Q is a gap-closer with an attack reset — it is an
*engage tool*. Every all-in distance calculation an agent learned on 4.20 is short by a dash.

**1v1 priority: HIGH.** Dash + attack reset changes engage range, all-in sequencing, and the
minimum distance at which an opponent is safe.

### 4.5 W — Courage

| | 4.20 (server `W.cs` / `Buffs/Garen/GarenW*.cs`, §3.3) | Modern | Class |
|---|---|---|---|
| Cooldown | 24/23/22/21/20s | 22/19.5/17/14.5/12s (retuned V25.11) | NUMBER |
| **Passive** | **percentage resist shift**: `Armor.PercentBonus += 0.2` and `Armor.PercentBaseBonus −= 0.2` (same for MR) — a +20%/−20% reshuffle of Garen's *own* armor/MR terms, **not** a kill counter | **permanent kill-stacking**: +0.2 armor +0.2 MR per stack, max **150 stacks** (raised from 120 at V11.9), capped +30/+30 | **RULE** |
| **Active** | flat **30% damage reduction** (`PostMitigationDamage *= 0.7f`) for **`2+(rank−1)`s → 2/3/4/5/6s** (rank-scaling duration), plus flat **30% tenacity** for the whole duration | **25/29/33/37/41% DR for 4s flat**, and for the **first 0.75s only**: a **shield of 65/85/105/125/145 (+18% bonus HP)** and **60% tenacity** | **RULE** |
| Shield | **does not exist** | exists (added V9.20) | **RULE (new)** |

4.20 values are read directly from the server's `W.cs` and `Buffs/Garen/{GarenW,GarenWPassive}.cs`
— see §3.3. Modern from [Garen](https://wiki.leagueoflegends.com/en-us/Garen) → Abilities →
Courage.

**One more rule difference worth pricing precisely.** 4.20's passive is a self-contained
multiplier on stats Garen already owns — it does nothing for a Garen with 0 bonus armor/MR.
Modern's passive is fed by kills, which in an isolated lane means CS: every minion killed is a
stack, so W's defensive value becomes a function of farm rather than of itemisation. That is a
different input variable, not just a different curve. Also note 4.20's active is *flatter and
longer at rank 5* (30% for 6s) than modern's (41% for 4s) — the change trades sustained
mitigation for burst absorption, not a straight buff in either direction.

The **active** also went from one flat window to a **two-phase** effect: a 0.75s burst phase
(shield + tenacity + DR) followed by a 3.25s DR-only tail. That needs a shield object, a
separate tail timer, and a tenacity grant — three pieces of state where 4.20 had one boolean
(and one flat-DR, flat-tenacity window whose *duration* itself scaled with rank).

Timeline: passive reworked **V5.16** (2015); shield added **V9.20** (2019); tenacity-in-burst
added **V7.14** (2017); duration flattened to 4s at **V14.2** (2024); DR made rank-scaling
again at **V25.11** (2025).

**1v1 priority: HIGH** for the shield/DR active. **MEDIUM** for the kill-stacking passive — it
accrues off minion kills so it does tick up during a solo lane, but slowly; it is not the
teamfight-only mechanic it might look like, just a slow one.

### 4.6 E — Judgment

This is the ability with the most new rules and the one most likely to break a transferred
policy.

| | 4.20 | Modern | Class |
|---|---|---|---|
| Duration | 3s | 3s | NO CHANGE |
| **Tick count** | **6 fixed** (every 500ms) | **7 base, +1 per 25% bonus attack speed** (`NumTicks: 7`, `ASPerTick: 0.25`) | **RULE (new)** |
| Damage/tick | `10 + 12.5(r−1) + AD×(0.35 + 0.05(r−1))` → base 10/22.5/35/47.5/60, ratio 35/40/45/50/55% | **4/7/10/13/16 (+40/43/46/49/52% AD)** | NUMBER |
| **Nearest-target bonus** | does not exist | **nearest enemy hit takes +25% damage** per tick | **RULE (new)** |
| **Armor shred** | does not exist | champions hit by **6 spins** get **25% armor reduction for 6s**, refreshing every 6th hit (added V6.24, hit-count set V9.20) | **RULE (new)** |
| **Crit** | code path exists but is dormant (§3.3: `GarenE.cs` multiplies tick damage by `CriticalDamage.Total` on a crit roll) — with base crit chance 0 it never fires | **can crit** (multiplier formula retuned V10.6, V11.6, V12.19, V25.06) | **RULE (new)**, though the hook itself predates 4.20 |
| **Minion damage modifier** | **0.75×** (confirmed in `Buffs/Garen/GarenE.cs`) | **removed — full 100% damage to minions and monsters** (removed V5.16) | **RULE (removed)** |
| Cooldown | 13/12/11/10/9s | **9/8.25/7.5/6.75/6s** | NUMBER |
| Cooldown starts | when the spin **ends** | still when the spin **ends** | NO CHANGE (see note) |
| **Early-cancel CD refund** | does not exist | added V5.16, **removed again V25.11** — current Garen gets **no refund** | RULE (added then removed) |
| Radius | 330 | 325 | NUMBER (UNVERIFIED which patch) |
| Suppresses autos / grants Ghosted | yes — `GarenE.cs` sets `StatusFlags.Ghosted` on activate, clears it on deactivate | yes | NO CHANGE |

Modern values confirmed twice: the rendered
[Garen](https://wiki.leagueoflegends.com/en-us/Garen) tooltip and the raw
`raw.communitydragon.org/latest/game/data/characters/garen/garen.bin.json`
(`BaseDamagePerTick` / `ADRatioPerTick` / `NumTicks` / `ASPerTick` / `Cooldown` arrays) agree
exactly. The 4.20 values are confirmed by the wiki's own "from" values in the V5.16 diff
(*"Base damage changed to 14/18/22/26/30 from 10/22.5/35/47.5/60"*, *"AD ratio reduced to
34-38% from 35/40/45/50/55%"*), which is an unusually strong confirmation of the server data.

Three consequences worth stating plainly:

1. **Attack speed is now a damage stat for Garen.** In 4.20, attack speed did nothing for E —
   the spin suppressed autos and ticked a fixed 6 times. In modern League, every 25% bonus AS
   is another tick. This is why the modern build path runs through Berserker's Greaves and
   Phantom Dancer (§8.3) rather than pure bruiser items. An agent that learned "attack speed is
   irrelevant while spinning" has learned something that is now false.
2. **The 0.75× minion modifier is gone.** Wave-clear speed and last-hit timing under E are both
   different. This directly changes every CS decision the agent makes while spinning.
3. **The cooldown shape reverted.** E was flattened to 9s at all ranks at V9.20 and stayed flat
   for six years, then went *back* to rank-scaling at **V25.11** (2025). Modern E is
   structurally closer to 4.20 (rank-scaling) than the 2019–2025 version was — just faster.

On "cooldown starts when the spin ends": the modern wiki notes discuss a bug about the
cooldown "improperly extending when recasting to end the ability early" (fixed V25.11), which
is only coherent if the cooldown is still keyed to channel end. That is indirect but solid
evidence the 4.20 convention survives. Flagged as inference, not a direct citation.

**What "Ghosted" actually means, settled:** [Ghosting](https://wiki.leagueoflegends.com/en-us/Ghosting)
defines it as a status that disables **unit collision only** — no move-speed bonus, no vision
change, no targetability change. `GarenE.cs` sets exactly that flag (§3.3). It is easy to
confuse with the unrelated *Ghost* summoner spell. If anything anywhere models 4.20 Ghosted as
a movement buff, that is a bug independent of this migration.

**1v1 priority: HIGH on every row.** This is Garen's damage, his wave-clear, and now his
itemisation driver.

**UNVERIFIED:** the exact patch for the 330→325 radius change. The crit-damage multiplier
sub-formula is **highly patch-volatile** (five revisions in six years) — do not hardcode it.

### 4.7 R — Demacian Justice

**Correction to an earlier pass of this research: the missing-health execute is not new.** A
wiki-only reading of the pre-`V9.20` patch history, taken in isolation, can suggest 4.20's R was
flat magic damage with no missing-health term (true damage reserved for a since-removed
"Villain" mechanic). That reading does not survive contact with two independent, stronger
sources: the vendored server's own `R.cs` script (§3.3), and — corroborating it — this
document's *own* R timeline below, whose pre-`V9.20` "changed from" values are the same numbers.

| | 4.20 (server `R.cs`, confirmed by the wiki's own pre-9.20 timeline) | Modern | Class |
|---|---|---|---|
| Cooldown | **160/120/80s** | **120/100/80s** (set V5.7, unchanged since) | NUMBER |
| Base damage | **175 × rank → 175/350/525** | **125/200/275** | NUMBER (retuned repeatedly) |
| Missing-health ratio | **28.57/33.33/40%** (`[0.2857, 0.3333, 0.4]`) | **25/30/35%** | NUMBER |
| Damage type | **magic**, unconditionally (Villain did not exist yet) | **true**, unconditionally (since V9.20) | **RULE** |
| Reveal on cast | does not exist | reveals target for 1s (added V9.20) | RULE (minor) |
| Minion aggro draw | does not exist | draws nearby minion aggro when targeting a champion (added V8.2) | RULE (minor) |
| **Villain** | **does not exist** | **does not exist** (existed V5.16 → V9.19 only) | **no work needed** |

**The 160/120/80s cooldown in the vendored server is correct, not a LeagueSandbox variant.**
That value was set at V1.0.0.145 and held through all of Season 3 and Season 4, dropping to
120/100/80 only at **V5.7** (2015) — after 4.20. Good news for the parity oracle.

Modern values confirmed twice (rendered tooltip + raw `garen.bin.json` `DataValues`:
`BaseDamage` 125/200/275, `ExecuteDamage` 0.25/0.30/0.35).

**On the Villain mechanic:** it was introduced at **V5.16** (2015) — *after* 4.20 — and removed
entirely at **V9.20** (2019). It is not in the current kit. **It requires no implementation.**
Mention it in the migration doc only so nobody wastes a sprint building it from a stale guide.

**Full R timeline** (the kit's most eventful history), read as a single consistent story once
the server-script data is in hand:

1. **Pre-4.20 (baseline):** `175 × rank` + 28.57/33.33/40% of missing health, **magic** damage,
   unconditionally. Confirmed by `R.cs` and matched exactly by the wiki's own "changed from"
   values at the V9.20 entry below.
2. **V5.16** (2015): Villain added; R's damage **against the Villain specifically** became true
   damage — implying it stayed *magic* against everyone else in this 2015–2019 window. This
   detail postdates 4.20 and does not apply to the baseline.
3. **V5.18**: Villain-determination rules tightened.
4. **V9.20** (preseason 2020): **Villain removed**; damage type finalised as **always-true** for
   every target; base 175/350/525 → 150/300/450; missing-HP ratio 28.6/34.3/40% → 20/25/30%;
   reveal added.
5. **V11.14** (2021): missing-HP ratio 20/25/30% → **25/30/35%** (current).
6. **V25.08** (2025): base 150/300/450 → 150/250/350.
7. **V26.14** (2026): base 150/250/350 → **125/200/275** (current).

**Why magic-→-true is a RULE change and the missing-health term is not.** The execute *shape*
— base plus a fraction of missing health — is present start to finish and needs no new formula.
What changed is the damage *type*: 4.20's R is reduced by the target's MR (32.1 + growth, plus
items); modern R ignores resistances entirely. That is a rule change with a concrete side
effect worth stating: **the modern kill threshold is independent of the target's build**, which
is arguably *easier* for a policy to learn than 4.20's MR-dependent one. The target dies to a
clean modern R below `20/23.08/25.93% max HP + 100/153.85/203.7 flat HP` (solving
`125/200/275 + 0.25/0.30/0.35·missing = currentHP` for currentHP); the 4.20 threshold is the
same algebra but the incoming 175/350/525 base is first reduced by `100/(100+MR)`.

**1v1 priority: HIGH** for the damage-type change and the coefficients (it is the kill-confirm).
**LOW** for reveal and minion-aggro-draw.

### 4.8 Garen summary — what needs code vs what needs a number

| Change | Class | 1v1 priority |
|---|---|---|
| E ticks scale with bonus attack speed | **RULE (new)** | **HIGH** |
| E minion damage modifier removed (0.75× → 1.0×) | **RULE (removed)** | **HIGH** |
| R damage type: magic → unconditional true (execute shape unchanged) | **RULE** | **HIGH** |
| Q gains a dash + attack-timer reset | **RULE (new)** | **HIGH** |
| W active becomes two-phase (0.75s shield+tenacity, then DR tail) | **RULE** | **HIGH** |
| W passive: percentage resist shift → permanent kill-stacking | **RULE** | MEDIUM |
| Passive regen: level-bracketed → smooth level-scaling | **RULE** | **HIGH** |
| E armor shred at 6 hits | **RULE (new)** | MEDIUM |
| E can crit; E nearest-target +25% | **RULE (new)** | MEDIUM |
| Attack range 125 → 175 | NUMBER (large behavioural effect) | **HIGH** |
| All base stats / growth values | NUMBER | HIGH |
| All ability damage, ratios, cooldowns | NUMBER (patch-volatile) | HIGH |
| Per-level growth formula | **NO CHANGE** | — |
| Q cooldown 8s flat | **NO CHANGE** | — |
| Villain mechanic | **does not exist in modern League** | **no work** |

---

## 5. Minions

Minions are the lane. For a 1v1 top-lane episode they are the economy, the damage source, the
positioning constraint and the reward signal all at once. This section has the most new rules
in the document, and two of them (**sidelane speed**, **first-wave behaviour**) are
*specifically* top-lane mechanics.

All modern values verified two ways: the server-rendered wiki
([Minion](https://wiki.leagueoflegends.com/en-us/Minion),
[Melee minion](https://wiki.leagueoflegends.com/en-us/Melee_minion),
[Caster minion](https://wiki.leagueoflegends.com/en-us/Caster_minion),
[Siege minion](https://wiki.leagueoflegends.com/en-us/Siege_minion),
[Super minion](https://wiki.leagueoflegends.com/en-us/Super_minion)) and the shipped game files
(`raw.communitydragon.org/latest/game/data/characters/sru_{order,chaos}minion*/*.bin.json`,
`CharacterRecord` blocks). Historical 4.20 values cross-checked against genuine
`leagueoflegends.fandom.com` revisions dated **7 and 15 November 2014** — snapshots from days
before 4.20 shipped.

**Pin:** all modern values are **V26.18** (10 September 2026). Patches V26.14–V26.18 contain no
core minion changes; the live data is V26.13-era, last substantively touched by **V26.09**
(stat tuning) and **V26.10** (aggro).

### 5.1 The 90-second upgrade system — the headline new rule

> "Starting at **0:30** and every **90 seconds** thereafter (regardless of which minion spawn),
> minions gain an '**upgrade**', gaining health, attack damage, and possibly certain buffs
> depending on the game state." — [Minion](https://wiki.leagueoflegends.com/en-us/Minion),
> *Stats mechanics*

The upgrade index is

```
U(t) = 1 + floor((t_seconds - 30) / 90)        for t >= 30s
```

so **the 0:30 wave is already at U = 1**, and U increments at 0:30, 2:00, 3:30, 5:00, 6:30,
8:00, 9:30, 11:00, …

This indexing is easy to get wrong by one, so it was verified four independent ways: the game
files give siege `goldGivenOnDeath = 49.0` while the wiki's gold table shows 50 at 0:30 (⇒ U=1
at 0:30); the wiki's gold table checks out at 14:00 (59 ⇒ U=10), 25:00 (66 ⇒ U=17) and 30:00
(69 ⇒ U=20); every infobox's lower displayed bound equals the formula at U=1, not U=0; and
Riot's V26.09 melee-HP change (`440+25U` for U≤5 → `430+35U`) is *exactly* neutral at U=1,
which only makes sense if wave 1 is U=1.

**The vendored server has no upgrade system at all.** This is a new subsystem.

> **Discrepancy worth recording: real patch 4.20 *did* have minion stat growth.** The Nov-2014
> wiki infoboxes give melee `455 (+20 / 3 min)` HP, caster `290 (+7.5 / 90 sec)` HP, siege
> `700 (+27 / 3 min)` HP, and the V4.20 patch notes read *"Minions no longer gain armor or
> magic resistance over time. All minion health gain over time has been increased to be roughly
> as durable as when they had resistances."* So 4.20 removed *resistance* growth and **kept and
> buffed HP growth**. The vendored server's fully-static minions are a **simplification of**
> 4.20, not 4.20. (The exact 4.20 cadence is **UNVERIFIED** — the 2014 wiki contradicts itself:
> the `Minion` page says every 2 minutes, melee/siege pages say per 3 min, the caster page says
> per 90 sec.)

| | 4.20 (server) | 4.20 (real) | Modern | Class | 1v1 priority |
|---|---|---|---|---|---|
| Stat growth over time | **none** | HP growth yes, resistance growth removed by 4.20 | **+HP/+AD every 90s**, `U(t)` above | **RULE (new)** | **HIGH** |

### 5.2 Base stats and scaling formulas (V26.18)

`U` = upgrade count.

| Stat | Melee | Caster | Siege | Super |
|---|---|---|---|---|
| **Health** | `430 + 35U` (cap 1550) | `275 + 9U` (cap 600) | `750 + 85U` (cap 5850) | `1500 + 100U` (cap 7500) |
| **Attack damage** | `11` for U≤5, then `11 + 3(U−5)` (cap 80) | `19.5 + 1.5U` for U≤5, then `+4`/upgrade (cap 125) | `36 + 1.5U` (cap 126) | `180 + 5U` (cap 480) |
| **Armor** | `0` for U≤5, then quadratic `0.085·(U−6)/2·(U−5)` (cap 20) | **0, never scales** | **0, never scales** | 100, never scales |
| **Magic resist** | **0, never scales** | **0, never scales** | **0, never scales** | −30 |
| Attack speed | 1.25 | 0.667 | 1.00 | 0.85 |
| Attack range | 110 | 550 | **300 (both sides)** | 170 |
| Gameplay radius | 48 | 48 | 65 | 65 |
| Pathing radius | 35.7437 | 35.7437 | 55.7437 | 55.5208 |

**MR never scales for any minion type**, and armor scales for melee only, starting at U=6,
quadratically — and it is **0.000 through U=6 and 0.085 at U=7 (9:30)**. For a ten-minute lane
you can hard-code melee armor to 0 and be wrong by 0.085.

**4.20 vs modern, side by side:**

| | 4.20 HP | modern (U=1) | 4.20 AD | modern (U=1) | 4.20 armor | modern | 4.20 range | modern |
|---|---|---|---|---|---|---|---|---|
| Melee | 455 | **465** | 12 | **11** | 0 | 0 | 110 | 110 |
| Caster | 290 | **284** | 23 | **21** | 0 | 0 | 550 | 550 |
| Siege | 700 | **835** | 40 | **37.5** | **15** | **0** (removed V5.8) | 300/280 | **300** |
| Super | 1500 | 1600 | 180 | 185 | 30 | 100 | 170 | 170 |

The *starting* lane minions are strikingly close (melee 455→465, caster 290→284). Attack speeds
and ranges are **unchanged since 2014**. The structural differences are that siege is much
tankier and lost its base armor, and that everything now grows every 90s — melee HP goes
465 → 675 across the first ten minutes, which moves **every last-hit breakpoint** in the
episode.

**Volatility:** the `X + k·U` *shape* has held since V8.23 (Nov 2018) and is safe to build
against. The *coefficients* are volatile — melee HP was retuned in V25.S1.1 and again in V26.09,
sixteen months apart.

### 5.3 Gold

| Type | Modern gold | Scales? | Game file |
|---|---|---|---|
| Melee | **20** | no (flat since V8.7) | `20.0` ✔ |
| Caster | **14** | no (flat since V8.7) | `14.0` ✔ |
| Siege | **49 + 1 per upgrade** (50 @ 0:30, 55 @ 9:00, 59 @ 14:00) | **yes** | `49.0` ✔ |
| Super | `49 + 1 per upgrade` | yes | `49.0` ✔ |

V26.01 changed melee 21 → 20 and restructured siege from `57 + 3U` (capped 90) to `50 + 1U`
uncapped — a large late-game siege nerf. **Treat siege gold as patch-volatile.**

> **The vendored 4.20 gold numbers do not corroborate.** Content gives 20/10/35 blue/30 red/150;
> the Nov-2014 wiki gives melee `19 (+0.5/3min)`, caster `14 (+0.2/90s)`, siege `40 (+1/3min)`.
> Melee 19→20 happened in **V4.21** — *after* 4.20 — so the Content's 20 looks like a 4.21+
> value. Caster 10 and siege 35/30 have **no patch-note support at all**. Real 4.20 gold also
> grew over time (growth removed for melee/caster only at V8.7).
> **UNVERIFIED / likely reimplementation artifacts.**

**Recommendation:** do not frame the gold migration as a clean 4.20→modern delta. Frame it as
*"replacing numbers we could not corroborate with numbers we can corroborate two ways."*

### 5.4 Experience

| Type | Base XP | Game file | Solo (1 champion) | Scales with time? |
|---|---|---|---|---|
| Melee | **62** | `62.0` ✔ | **62** | no |
| Caster | **31** | `31.0` ✔ | **31** | no |
| Siege | **75** | `75.0` ✔ | **75** | no |
| Super | 75 | `75.0` ✔ | 75 | no |

Share by number of nearby champions (V26.01): **100 / 65 / 43.3 / 32.5 / 26 / 21.7 %**.

> **Solo XP is now 100% of base.** It was **92%** in the 4.20 era (V1.0.0.104), then 93%
> (V9.23), 95% (V12.22), and **100% at V26.01**. For a 1v1 lane this is a direct multiplier on
> every level timing in the episode. **RULE/NUMBER change, HIGH priority.**

XP has not scaled with game time since **V3.14**, so the server's static XP is correct in kind.

**XP radius:** wiki says **1500** (raised from 1400 at V25.S1.1); the game files say `1400.0`.
**UNVERIFIED which is live** — irrelevant in a 1v1 where the last-hitter is always in range.

> **The vendored 4.20 XP numbers do not corroborate either.** Content gives 77/51/94/500. The
> Nov-2014 wiki gives 58.88 / 29.44 / 92 *solo*, which are exactly `base × 0.92` for base
> **64 / 32 / 100 / 100** — corroborated by patch history (V1.0.0.138). Content's melee 77 ≈
> 64×1.2 (the six-champion shared value), but caster 51 ≠ 38.4 and siege 94 ≠ 120, so that
> theory fails too. **Treat the vendored XP numbers as suspect; do not use the 4.20:modern XP
> ratio to justify anything.**

### 5.5 Waves: composition and cadence

| | 4.20 (server) | 4.20 (real) | Modern | Class | 1v1 priority |
|---|---|---|---|---|---|
| **First wave** | 90s | 90s ✔ | **0:30** | **RULE/NUMBER** | **HIGH** |
| Wave period | **36.4s** (counter quirk) | **30s** | **30s** (≤14:00) | NUMBER — see note | **HIGH** |
| Composition | 3 melee + 3 caster | same | same (spawn order: super → 3 melee → siege → 3 caster) | **NO CHANGE** | HIGH |
| First siege | wave 3 | wave 3 (2:30) | **wave 3 (1:30)** | NO CHANGE in cadence | HIGH |
| Siege cadence | every 3rd wave | every 3rd wave | every 3rd wave **until 14:00**, then every 2, then every wave after 25:00 | NUMBER | **LOW** (out of scope) |
| Intra-wave stagger | 800ms | — | **0.792s** | effectively unchanged | LOW |

**First-wave time is the single most impactful cadence change**: 1:30 → 1:15 (V5.22) → 1:05
(V7.22) → **0:30 (V26.01)**. A full extra minute of lane before the modern game reaches the
point the 4.20 game started at, shifting every level and item timing in the episode.

> **The server's 36.4s wave period is a reimplementation artifact, not a 4.20 fact.** The 2014
> wiki says "every 30 seconds" plainly and every 2014 minion infobox carries
> `respawntime = 0:30`. Changing it to 30s is arguably a **4.20 bug fix**, not a modernisation
> — worth separating in the migration plan, since it is a change you could make *without*
> leaving the parity target, and one the parity suite would currently flag as a regression.

**Net effect on a ten-minute episode:** the vendored server (90s + 36.4s) produces **15 waves**;
real 4.20 (90s + 30s) produces **18**; modern (30s + 30s) produces **20 waves and 6 cannons**.

Post-14:00 rules (siege waves drop one melee; post-30:00 drop one caster; siege cadence
tightening) are **entirely outside a ten-minute scope** — a `TODO(post-14min)` comment is
sufficient.

### 5.6 First-wave special behaviour — new, and top-lane specific

> "The **first wave**'s minions have modified behavior. **Top and bottom lanes' minions are
> ghosted for 28 seconds**, while middle lane minions are ghosted for 18 seconds. They will
> **ignore enemy champions** until a champion walks much closer to them or until meeting with
> the enemy wave. **When the waves meet, minions will spread out their attacks on the three
> enemy melee minions regardless of distance.**" — [Minion](https://wiki.leagueoflegends.com/en-us/Minion), *Behavior*

Three sub-rules, none present in 4.20 (the Nov-2014 page describes no first-wave special case
at all):

1. **Ghosting for 28s** (collision disabled) for top/bot. Raised 18 → 28 at **V9.4**;
   introduction patch **UNVERIFIED**.
2. **First wave ignores enemy champions** until very close or until wave contact.
3. **First wave spreads attacks across the three enemy melee minions** regardless of distance —
   deliberately engineered so both first waves' melee minions die simultaneously.

**Class: RULE (new). 1v1 priority: HIGH.** Rule 3 determines the entire opening wave state,
which cascades through the whole episode. **If you implement only one of the three, implement
rule 3.**

### 5.7 Movement speed, and the sidelane buff (top-lane specific)

**Base MS:** 4.20 was **325 flat**. Modern is **350**, rising +25 at 11:00, 16:00, 21:00, 26:00.
Within ten minutes: **flat 350**. Time-based MS did not exist until V7.23.
**Class: NUMBER. Priority: MEDIUM** — 7.7% faster minions change where waves collide.

**Sidelane Speed** — [Minion](https://wiki.leagueoflegends.com/en-us/Minion), *Buffs*:

> "After the first wave and until **14:00** game time, **all top and bottom lane minions** gain
> a movement speed buff upon their spawn. Each spawned minion in the **second** wave gains
> **111 bonus movement speed**. **Each subsequent wave reduces its spawned minions' initial
> speed by 4.5.** The bonus is removed in four stages: the first three last **7 seconds**, the
> fourth lasts **4 seconds** (total **25 seconds**); each stage completion reduces movement
> speed by **15**."

So wave *n* (n ≥ 2, before 14:00) spawns with `111 − 4.5(n−2)` bonus MS, stepping down by 15 at
t+7s, t+14s, t+21s, removed at t+25s. Decays to zero around wave 26 (~13:00). Introduced
**V13.10**, retuned **V14.22**. Riot's stated intent: make side-lane waves meet at the lane
midpoint at the same time mid-lane waves do.

**Class: RULE (new). 1v1 priority: HIGH.** This is a **top-lane-only mechanic in a top-lane
sim**. It is the biggest determinant of *where the wave meets* in the early game, it changes
every wave, and a fixed meeting point will be wrong for every wave of the episode.

**Minion Pushing** — same page, *Buffs*. The team with the higher average champion level gets
bonus minion damage `(5% + 5% × turret_advantage) × level_advantage` (level advantage capped at
3) and a damage-reduction term, **starting at 3:30**, updating live. Introduced **V5.23**
(Nov 2015), not in 4.20.

In a 1v1 with no turrets destroyed this reduces to **up to +15% minion-vs-minion damage for the
champion who is ahead in levels**, from 3:30 onward. **Class: RULE (new). Priority: MEDIUM** —
it is a genuine positive-feedback loop (level lead → wave pushes → more XP) that an RL agent
will find and exploit, but the magnitude is modest and it needs only level tracking you already
have.

### 5.8 Damage modifiers — the two most likely to be missing

**Minions deal 60% damage to champions and structures**
([Minion](https://wiki.leagueoflegends.com/en-us/Minion), *Stats mechanics*).

**This was also 60% in 4.20** (V1.0.0.130 cut it ~15% from 70%; V8.23 later cut it to 50%,
V25.S1.1 raised it to 55%, V26.01 back to **60%**). So the value round-trips — but the action
item is real:

> **Verify the 4.20 sim actually applies the 0.6 modifier.** The Content files carry raw AD
> (melee 12, caster 23). If the multiplier lives in server code rather than Content, it is easy
> to omit. A caster hitting Garen for 23 instead of 13.8 overestimates minion harass by 67%.

**Class: NO CHANGE. 1v1 priority: HIGH** (as a correctness check, not a migration item).

Minor ambiguity: V26.01's notes mention only "*Damage to structures increased to 60% from
55%*", while the minion pages state 60% for both champions and structures. Low-confidence on
that single percentage point.

**Minion-vs-minion bonus current-health damage — new at V25.S1.1:**

| Attacker | Bonus on-hit vs lane minions | History |
|---|---|---|
| Melee | **2% of target's current health** (bonus physical) | added V25.S1.1 |
| Caster | **3.5%** of current health | added at 4%, → 3.5% at V26.09 |
| Siege | **5%** of current health | added at 6%, → 5% at V26.09 |

Super minions have no such on-hit. **Class: RULE (new) — needs a current-HP-proportional on-hit
in the minion-vs-minion damage path. 1v1 priority: HIGH**: it accelerates wave-versus-wave
resolution superlinearly and directly shifts *when* minions enter last-hittable HP ranges,
which is the core timing the agent is learning.

**Turret-bullet modifiers against minions** (for cross-reference with §6): turret attacks deal a
fixed percentage of the minion's **maximum** health — melee **45%**, caster **70%**, siege
**14% / 11% / 8%** by turret tier, super **7%**. Siege minions deal **84%** damage to turrets.
**Priority: LOW** in general, but the melee-45% / caster-70% figures determine how fast a
crashed wave dies under tower, which matters if the agent learns to freeze or dive.

### 5.9 Minion aggro — two rule changes, one of them four months old

**Priority list, modern (6 entries)** — [Minion](https://wiki.leagueoflegends.com/en-us/Minion),
*Behavior → Priority*:

1. Enemy champions attacking an allied champion
2. Enemy minions attacking an allied champion
3. Enemy minions attacking an allied minion
4. Enemy turrets attacking an allied minion
5. The closest enemy minion
6. The closest enemy champion

**4.20 (7 entries)**, from the Nov-2014 snapshot — identical, plus one extra entry at position
5: **"Enemy champion attacking an allied minion."**

That entry was **removed in V26.10 (13 May 2026)**:

> "**Removed:** Previously, when an enemy champion attacked an allied minion, the enemy champion
> would enter the minions' aggro priority list. We're removing this action entirely from
> aggroing minions to prevent it feeling like minions sometimes randomly change their target."

Confirmed by diffing wiki revision 4003116 (28 March 2026, pre-26.10) against current: the
pre-26.10 list is the *identical* 7-entry list to November 2014. **This list was stable for
eleven years and then changed four months ago.**

**Class: RULE CHANGED. 1v1 priority: HIGH — this may be the most behaviourally significant
single line in the document.** Under 4.20 rules, auto-attacking a minion pulls the enemy wave
onto you. Under modern rules it does not. That deletion changes the risk attached to **every
last-hit the agent takes**, which is most of what it does.

Unchanged since 2014: *"Once a minion has chosen a target, it only switches to a new target if
that new target has a **higher** priority. The minion cannot acquire a new target that has the
**same** priority as their current target."* Verbatim in both eras.

**Call for Help triggers — changed at V8.2.** Modern triggers include: a champion standing in
the minion's path with no other targets in range and outside turret range; and a champion
**dealing damage to an enemy champion with a basic attack, most unit-targeted abilities, and a
minority of similar abilities**. Since **V13.10**, Call for Help is ignored while minions are
already attacking a turret.

In 4.20, **only basic attacks drew aggro** — V8.2 added *"Targeted champion abilities, item
actives and summoner spells now draw nearby minion aggro."*

**This matters specifically for Garen.** Q is an empowered auto (so it draws aggro under modern
rules), and E is untargeted AoE — whether E draws aggro depends on the "minority of abilities
marked as being similar" carve-out. Per-ability Call-for-Help flags live in each ability's
*Details* table on the champion page. **Garen's specific flags were not verified — UNVERIFIED,
and worth checking directly since it affects the cost of every E in the wave.**

**Acquisition range — wiki and game files disagree.** The wiki says targets must generally be
within **500 units** (1000 for the ally-attacked Call for Help). The shipped `CharacterRecord`
says:

| | `acquisitionRange` | `firstAcquisitionRange` | `wakeUpRange` |
|---|---|---|---|
| Melee | *(unset, inherits default)* | 1000 | 450 |
| Caster | **700** | 900 | 635 |
| Siege | *(unset)* | *(unset)* | *(unset)* |
| Super | **600** | *(unset)* | *(unset)* |

**UNVERIFIED which governs**; the game files are more likely correct. **Priority: MEDIUM-HIGH** —
acquisition range is the aggro leash the agent must respect when walking up to CS.

**Unchanged since 2014:** minions reevaluate targets between attack windups (this is what makes
aggro "sticky"), and lose/reacquire targets on vision loss.

**Death grace — new:** a minion below **0.35% max HP** taking lethal damage *from another
minion* has its health set to 1 and dies 0.066s later unless damaged from another source; does
not apply if the damage would exceed 190. **Class: RULE (new). Priority: MEDIUM** — it is a
last-hit-steal grace window. **Low confidence on the threshold**: 0.35% of a 465 HP minion is
1.6 HP, which would almost never fire, and a December 2025 wiki edit comment notes *"the display
in game has been wrong and off by a factor of 100 for years"*. Cheap to implement, but the
threshold needs a better source.

### 5.10 Blue/red asymmetry — patched out

The server's D0 asymmetry (siege range 300 blue / 280 red; siege gold 35 blue / 30 red) does
**not** exist in modern League.

- **Range:** [Siege minion](https://wiki.leagueoflegends.com/en-us/Siege_minion) patch history,
  **V10.16**: *"Bug Fix: Red side siege minion attack range increased to 300 from 280."* Riot
  treated it as a bug. Modern infobox lists a single `range = 300`.
- **Gold:** no patch note documents the asymmetry or its removal, but the shipped game files
  are decisive: `sru_orderminionsiege` and `sru_chaosminionsiege` have **byte-identical** value
  sets — both `goldGivenOnDeath: 49.0, expGivenOnDeath: 75.0`. Same for melee (20/62) and
  caster (14/31). (Other records in those files showing 30/35/60/80 gold are *other game modes*
  — ARAM, Arena, Nexus Blitz, Swarm — and appear identically in both sides' files.)

**Class: RULE REMOVED (a simplification). 1v1 priority: HIGH**, and it is good news for the RL
setup specifically: `JAX_REWRITE_PLAN.md` D0 records that this asymmetry contradicts the mirror
assumption `lanerl_rl/obs.py` canonicalises on. **On a modern patch table that contradiction
disappears** — blue and red become genuinely symmetric on minion gold, XP and range, so a
single model can be shared across sides without a side flag for this reason.

### 5.11 Reference: the first ten minutes, modern

Computed from §5.2 with `U(t) = 1 + floor((t−30)/90)`; wave *n* spawns at `30 + 30(n−1)`
seconds; siege on every 3rd wave.

| Wave | Time | U | Melee HP | Melee AD | Caster HP | Caster AD | Siege HP | Siege gold | Siege? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 00:30 | 1 | 465 | 11 | 284 | 21.0 | — | — | |
| 2 | 01:00 | 1 | 465 | 11 | 284 | 21.0 | — | — | |
| 3 | 01:30 | 1 | 465 | 11 | 284 | 21.0 | 835 | 50 | ✔ |
| 4 | 02:00 | 2 | 500 | 11 | 293 | 22.5 | — | — | |
| 6 | 03:00 | 2 | 500 | 11 | 293 | 22.5 | 920 | 51 | ✔ |
| 9 | 04:30 | 3 | 535 | 11 | 302 | 24.0 | 1005 | 52 | ✔ |
| 12 | 06:00 | 4 | 570 | 11 | 311 | 25.5 | 1090 | 53 | ✔ |
| 15 | 07:30 | 5 | 605 | 11 | 320 | 27.0 | 1175 | 54 | ✔ |
| 16 | 08:00 | 6 | 640 | **14** | 329 | **31.0** | — | — | |
| 18 | 09:00 | 6 | 640 | 14 | 329 | 31.0 | 1260 | 55 | ✔ |
| 19 | 09:30 | 7 | 675 | 17 | 338 | 35.0 | — | — | |
| 20 | 10:00 | 7 | 675 | 17 | 338 | 35.0 | — | — | |

Melee armor is 0.000 for U≤6 and 0.085 at U=7. All MR is 0 throughout.

**Totals available in the first 10:00** (20 waves, 6 cannons): **126 CS**, **2355 gold**
(`20 × (3×20 + 3×14)` = 2040, plus siege 50+51+52+53+54+55 = 315), **6030 solo XP**
(`20 × (3×62 + 3×31)` = 5580, plus `6 × 75` = 450).

Note the structure: each upgrade tier lasts exactly three waves before 14:00, and **the siege
wave is always the last wave of its tier** (siege on waves ≡ 0 mod 3; upgrades on waves 1, 4,
7, 10, 13, 16, 19).

**One open implementation question:** siege gold is an upgrading stat, so a siege minion that
spawns at wave 18 (U=6, gold 55) and dies 40s later may be worth 55 or 56 depending on whether
the bounty is latched at spawn or read at death. The wiki's gold table is computed per
game-time, implying **read at death**. Impact ≤1 gold per siege. **UNVERIFIED.**

### 5.12 Minion summary

| Change | Class | 1v1 priority |
|---|---|---|
| **90-second stat upgrade system** | **RULE (new)** | **HIGH** |
| **First wave 1:30 → 0:30** | RULE/NUMBER | **HIGH** |
| **Aggro priority: champion-attacks-minion entry removed (V26.10)** | **RULE** | **HIGH** |
| **Sidelane speed buff (top-lane specific)** | **RULE (new)** | **HIGH** |
| **First-wave spread-attack / ghosting / ignore-champions** | **RULE (new)** | **HIGH** |
| **Minion-vs-minion % current-HP on-hit** | **RULE (new)** | **HIGH** |
| Call for Help now includes abilities (V8.2) | **RULE** | **HIGH** |
| Solo XP 92% → 100% | RULE/NUMBER | **HIGH** |
| Blue/red asymmetry removed | RULE REMOVED | **HIGH** (simplification) |
| Base HP/AD/armor values | NUMBER | **HIGH** |
| Gold and XP values | NUMBER (4.20 baseline uncorroborated) | **HIGH** |
| Minion Pushing buff | **RULE (new)** | MEDIUM |
| Base MS 325 → 350 | NUMBER | MEDIUM |
| Death grace window | RULE (new) | MEDIUM |
| Acquisition-range values | NUMBER (disputed) | MEDIUM |
| 60% damage to champions | **NO CHANGE** — but verify it is implemented | **HIGH** (correctness) |
| Wave period 36.4s → 30s | server bug fix, not a migration item | **HIGH** |
| Siege cadence thresholds (14:00 / 25:00) | NUMBER | LOW |
| Post-14:00 wave thinning | RULE | LOW |
| Super minions | — | **N/A — drop entirely** |
| Intra-wave stagger 800ms → 0.792s | unchanged in practice | LOW |

---

## 6. Turrets

Turrets are the area where the 4.20 baseline and modern League have diverged the most in raw
magnitude, *and* where the largest genuinely-new rule (plating) lives. For a top-lane 1v1 this
section ranks second only to Garen's own kit.

### 6.1 A baseline conflict worth resolving before you build the table

The vendored Content gives the outer turret **1550 HP** (1300 + a 250 bonus), **67 armor**, and
— confirmed by directly reading `SRUAP_Turret_Order3.json`'s `SpellBlock` field — **100 MR**.
The official patch 4.20 notes do not fully agree.

Patch 4.20 was itself a preseason turret overhaul. Per
[V4.20](https://wiki.leagueoflegends.com/en-us/V4.20) and the
[Surrender at 20 patch 4.20 notes](https://www.surrenderat20.net/2014/11/patch-420-notes.html)
(two independent transcriptions of the same official notes, in agreement):

- "Outer turret health reduced to **2000** from 2550" (inner likewise 2000 from 2550)
- "No longer gain armor and magic resistance over time — now have **100** in each at all times"
- Backdoor protection ("Reinforced Armor") resistances increased to **200** from 150
- Nexus turret HP 2500 (from 1925); inhibitor/nexus turrets converted to hitscan lasers with an
  82.5% armor-pen, a target debuff, and a stacking "Heat" mechanic — **outer/inner turrets were
  not given the laser or Heat mechanic**

So the official 4.20 outer turret is **2000 HP / 100 armor / 100 MR**, versus the server's
**1550 HP / 67 armor / 100 MR**. The server's MR happens to **match** the official value even
though its HP and armor do not — the divergence is not uniform across stats on the same unit,
which is itself worth remembering when deciding how much to trust any single Content field.

**Turret gold and XP, also read directly from Content** (not previously pinned down): the
server's `SRUAP_Turret_Order3.json` carries `GlobalGoldGivenOnDeath: 175`,
`LocalGoldGivenOnDeath: 0`, `GlobalExpGivenOnDeath: 100` — i.e. destroying an outer turret in
the 4.20 baseline pays **175 gold, globally, with no local bonus**, plus **100 global XP**. This
was previously an open question in this line of research; it turns out to be a Content-file
fact, not a wiki fact. See §6.7 for the modern comparison.

**One more oddity worth a five-minute check, not a fix:** the same file's `UnitTags` field reads
`"Structure | Structure_Turret | Structure_Turret_Inhib"` — the model this project's turret
roster uses for the **outer** lane turret is tagged, in Riot's own data, as an **inhibitor**
turret. That may be harmless upstream naming (Content sometimes reuses a template file across
tiers), or it may mean `TURRET_MODELS` is pointed at the wrong tier entirely, which would matter
a great deal for which row of §6.2 the modern table should replace. **UNVERIFIED — worth
confirming against `LevelScriptObjects.LoadBuildings` before the modern turret table is frozen**
(register item added to §13.2).

**This is not a research gap — it is a finding about the oracle.** Per `JAX_REWRITE_PLAN.md`
R2 ("match the server's behaviour, including its deviations from real League") and D1 (parity
is asserted against the server), the correct action is *not* to change the 4.20 column. It is
to record that **the vendored LeagueSandbox Content is not a faithful patch-4.20 replica** on
turrets, and therefore that the modern migration is measured from the *server's* baseline, not
from historical League. Two practical consequences:

1. Do not treat "4.20" as a meaningful semantic anchor when reasoning about what a trained
   policy has learned. It learned the *server*, which on turrets is closer to a pre-4.20 build.
2. If anyone later wants a genuine "historical 4.20" target, it is a different data set from
   the one in `Content/`.

Possible benign explanations (none confirmed): the Content reflects a pre-4.20 point release,
or the field being read is not raw `MaxHealth`. Worth a five-minute check of the field name and
units in `Content/` before the table is frozen. **Flagged UNVERIFIED as to cause.**

### 6.2 Turret stats

| | 4.20 (server) | 4.20 (official notes) | Modern (V26.1+) | Class | 1v1 priority |
|---|---|---|---|---|---|
| Outer HP | 1550 | 2000 | **9,000** | NUMBER (≈6×) | **HIGH** |
| Outer AD | 190 | — | **182–350, scaling by game minute** | NUMBER + **RULE** (time-scaling is new) | **HIGH** |
| Outer armor / MR | 67 / **100** | 100 / 100 flat | **60 / 60 base**, decaying from 11:00, capped around −40 armor / −60 MR by 15:00 | NUMBER + **RULE** (decay window is new) | MED (decay starts after the 10-min scope) |
| Range | 750 | — | 750 (edge-to-edge) | **NO CHANGE** | HIGH |
| Attack speed | — | — | 0.833 | — | HIGH |
| Inner (T2) HP | — | 2000 | 5,000 | NUMBER | LOW |
| Inhibitor HP | — | — | 4,750 | NUMBER | LOW |
| Nexus turret HP | — | 2500 | 3,500 | NUMBER | LOW |

Modern figures from [Turret](https://wiki.leagueoflegends.com/en-us/Turret) (stats infobox),
cross-checked against the
[Patch 26.1 notes](https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/),
which is the patch that set them (outer 5,000 ⇒ 9,000; inner 4,000 ⇒ 5,000; inhibitor
3,500 ⇒ 4,750; base resistances 15 ⇒ 60).

> **Do not use the patch 26.16 turret numbers** (outer 3000→3500 etc.). Those appear in the
> 26.16 notes under a **"League Classic"** game-mode section — a separate retro ruleset — not
> standard Summoner's Rift.

**Volatility:** the current numbers date to January 2026 and are only months old at time of
writing. The *2026 rework having happened* is structurally stable; these specific values are
new enough that they may still be in an active tuning window.

### 6.3 Damage ramp ("Warming Up")

| | 4.20 | Modern | Class |
|---|---|---|---|
| Bonus per consecutive hit | **37.5%** | **50%** | NUMBER |
| Hits to reach cap | **2** (cap +75%) | **3** (cap **+150%**) | NUMBER + RULE (shape) |
| Reset | window not confirmed for the era | **5s after the last shot that hits a champion**; does **not** reset on target switch | RULE (specifics new) |

The 4.20-era values come from [V4.11](https://wiki.leagueoflegends.com/en-us/V4.11) ("Turret
damage gained per hit increased to 37.5% from 25%" / "Turrets now finish warming up after 2
hits from 3"). No patch between 4.11 and 4.20 was found that touched this — **flagged
UNVERIFIED (medium confidence)** for exact persistence to 4.20, since patches 4.12–4.19 were
not exhaustively checked.

Modern from [Turret](https://wiki.leagueoflegends.com/en-us/Turret), "Warming Up" section.

An intermediate **40% / +120% cap / 3s reset** version is referenced by bug coverage around
patch 12.14 (2022), suggesting the sequence was 37.5% → 40% → 50%, with the last step in the
26.1 rework. **The exact patch for 40%→50% is UNVERIFIED.**

**A ramp mechanic has existed continuously across the whole span** — this is a rule that is
*stable in kind* but has moved in every parameter. Porting is a formula-shape change (2-hit cap
vs 3-hit cap) plus new reset semantics, not a pure table swap.

**1v1 priority: HIGH.** This determines how many turret shots a diving Garen can absorb, which
is the core quantity in every dive decision.

### 6.4 Turret plating — the biggest new rule

**Did not exist in 4.20.** Introduced in **patch 8.23** (19–20 November 2018, preseason 2019).
Per [V8.23](https://wiki.leagueoflegends.com/en-us/V8.23) and
[Blog of Legends' 8.23 turret breakdown](https://blogoflegends.com/2018/11/19/patch-8-23-breakdown-turret-changes/):

- Outer turret HP raised to 5000 from 3800 in the same patch to make room for plates
- **5 plates**, each broken by **1000 damage** to the turret
- **160 gold** per plate, split among nearby champions
- Plates granted scaling bonus armor/MR by plates remaining, plus a "Bulwark" stack per break
- **Plating fell off entirely at 14:00**
- First Turret gold reduced 300 → 150 to offset the new plate income

**Reworked again in patch 26.1 (January 2026).** Current state per
[Turret](https://wiki.leagueoflegends.com/en-us/Turret) and the
[26.1 notes](https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/):

| | 8.23–25.x | Modern (26.1+) | Class |
|---|---|---|---|
| Plate threshold | fixed **1000 damage** each | **% of missing HP: 10/25/45/70/100%** | RULE |
| Gold per plate | 160 → 125 → 120 | **120** | NUMBER |
| Max plate gold | 800 → 600 | **600** | NUMBER |
| Which turrets | outer only | **outer, inner and inhibitor** | RULE |
| **14:00 expiry** | yes | **no — plating no longer expires on a clock** | **RULE (removed)** |

The removal of the 14:00 expiry is corroborated by
[GameRiv's 2026 turret changes summary](https://gameriv.com/all-new-turret-changes-coming-to-league-of-legends-in-2026/)
("turret plating will not be removed at 14:00") and, negatively, by the current wiki plating
section containing no mention of any expiry. The time-pressure role that plating expiry used to
play has been handed to the new **resistance decay window** (11:00–15:00, §6.2) instead.

**The 160→125→120 gold history is only partially pinned to patches — flagged UNVERIFIED on the
intermediate steps.**

**1v1 priority: HIGH — this is the single highest-value new lane mechanic in the document.**
Plates are a gold objective that does not exist anywhere in the 4.20 environment. A policy
trained on 4.20 has never seen a reason to hit a turret early, and has no representation for
"a plate is worth 120g and there are five of them". Adding plating changes the reward landscape
of the lane, not just its physics.

**Also new in 26.1: Crystalline Overgrowth**, a turret true-damage mechanic on a ~90s cycle
(30s in Swiftplay) scaling from 2–3.3% of turret max HP at level 1 to 8.8–18.9% at level 18 by
team level. Two sources describe the trigger differently (turret-hits-champion vs
minions-attacking-under-turret) — **trigger condition UNVERIFIED**, existence and rough scaling
corroborated across two sources. **1v1 priority: MEDIUM** — it is turret damage in the lane, so
it matters for dives, but the numbers are unsettled and it is brand new.

Also in the same rework: **minion damage to turrets increased to 60%**. Melee champions
continue to deal **20% increased damage to turrets** (predates 2026, unchanged).

### 6.5 "Fortification" — a premise that does not survive checking

The research brief assumed an early-game "turret fortification" buff that plating later
replaced. **No source supports that narrative.** What is actually true:

- The only thing literally named "Fortification" traces to the **Fortify** summoner spell,
  removed **15 November 2011** in V1.0.0.129 ([Fortify](https://wiki.leagueoflegends.com/en-us/Fortify)).
  The wiki's `Fortification` page is now a redirect with no distinct content.
- The 8.23 plating coverage explicitly says turrets were "relatively untouched" between patch
  6.22 and 8.23 and does **not** describe plating as replacing any named buff.
- **The real mechanic in this space is the one 4.20 itself deleted**: before 4.20, turrets
  *gained armor and MR over the course of the game*, and patch 4.20 removed that in favour of
  flat 100/100. That is a genuine "turrets used to get tankier over time" story — it just
  resolves a season earlier than assumed, and in the opposite direction from plating.
- The living descendant of 4.20's backdoor protection is **Reinforced Armor**: **80% damage
  reduction, including true damage, when no enemy minions (or Rift Herald) are nearby**, with a
  3s reactivation delay after minions leave. 4.20's version was 200 flat resistance (75% DR).
  Implementation-wise that is a **RULE change** — flat resistance became flat percentage DR.

**Recommendation: drop "fortification as plating's ancestor" from the migration narrative.**
Cite the pre-4.20 resistance-over-time removal instead if a historical beat is wanted.

**1v1 priority for Reinforced Armor: MEDIUM.** It only binds when a champion attacks a turret
with no friendly minions present — which is exactly the "solo-diving a tower with no wave"
situation a 1v1 lane sim can produce, so it is not ignorable.

### 6.6 Aggro and targeting

Per [Turret](https://wiki.leagueoflegends.com/en-us/Turret), target-selection section:

- **Priority order** (highest first): most pets/summoned units → siege and super minions →
  Mist Walkers → **melee minions** → **caster minions** → certain summoned champions' units →
  **champions last**. Within a priority tier, closest target.
- **Champion-aggro trigger:** if an enemy champion **deals damage to an allied champion within
  1400 range of the turret**, the turret switches to the enemy champion. Documented as a
  single-hit trigger.
- Otherwise standard sticky targeting: attack the first unit in range until it dies, leaves
  range, or becomes untargetable.

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Minions before champions | assumed same | confirmed — champions are last in a 7-tier list | **RULE STABLE** | **HIGH** |
| Ally-damaged aggro switch | assumed same | damage an ally within **1400** range → switch | RULE STABLE, number UNVERIFIED for 2014 | **HIGH** |
| "24 attacks" tagging rule | — | **NOT FOUND on the wiki at all** | **treat as folklore** | — |
| "45% reduced champion damage to plating unless minions present" | — | **NOT FOUND** | **likely a conflation with Reinforced Armor's 80%** | — |

Two sub-details the wiki does not settle, both **UNVERIFIED**: whether an ability that hits a
champion and a minion simultaneously counts as damaging the ally for aggro purposes, and the
precise aggro-drop timing beyond "dies / leaves range / untargetable". Both matter for dive
modelling; both will need to be measured empirically against the live game rather than read.

**Note for the parity suite:** `JAX_REWRITE_PLAN.md` R2 already records that the server's
`LaneMinionAI` deliberately deviates from real League (first-wave exception not implemented,
`CountUnitsAttackingUnit` disabled). Those deviations are in the *minion* AI, but the same
caution applies here — the modern aggro rules above are what real League does, which is not
necessarily what the oracle does.

### 6.7 Turret and plate gold

| | 4.20 (server Content, §6.1) | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Turret destroy gold | **175 global, 0 local** (`GlobalGoldGivenOnDeath`/`LocalGoldGivenOnDeath`) | outer **50g** global (see row below); most of the value moved to plates | NUMBER — **shape inverts** (one lump at death → mostly-local income while it's still standing) | **HIGH** |
| Turret destroy XP | **100 global** (`GlobalExpGivenOnDeath`) | not separately itemised on the wiki's turret page | NUMBER | LOW |
| Plate gold | **n/a — no plates** | **120** local per plate, 600 max per turret | **RULE (new)** | **HIGH** |
| Outer turret local bounty | — | **250g**, shared among champions alive within 1200 range | NUMBER | MED |
| Global structure bounty | — | outer **50g** to the whole team; inner 25g; inhibitor 25g; nexus turret 50g | NUMBER | LOW |
| First Turret bonus | UNVERIFIED for 4.20 (not in Content — turret gold is server-code, not Content, per §6.1) | **300 per the general wiki page**, but 8.23 reduced it to 150 — **CONFLICT, unresolved** | NUMBER | MED |

The First Turret figure could not be reconciled: the 2018 patch notes reduce it 300→150, while
a current wiki summary states 300, implying an unlocated restoration somewhere between 2018 and
2026. **Re-check against the live wiki before freezing this number.**

A pre-2026 source computed the maximum gold from a single outer turret before 14:00 as ~1075g
(600 plating + 300 bounty + 50 global + 150 first-turret). That total is **stale** — plating no
longer expires and thresholds are percentage-based now, so it needs recomputation.

### 6.8 Turret summary

| Change | Class | 1v1 priority |
|---|---|---|
| **Turret plating exists at all** | **RULE (new, 8.23)** | **HIGH** |
| Plating thresholds → % missing HP, no 14:00 expiry | RULE (26.1) | **HIGH** |
| Plate gold as a lane objective | RULE (new) | **HIGH** |
| Outer turret HP 1550 → 9,000 | NUMBER | **HIGH** |
| Turret AD scales with game clock | **RULE (new)** | **HIGH** |
| Damage ramp 37.5%/2-hit → 50%/3-hit, 5s reset | NUMBER + RULE | **HIGH** |
| Reinforced Armor: flat resistance → 80% DR incl. true damage | RULE | MED |
| Resistance decay window 11:00–15:00 | RULE (new) | MED (outside a 10-min scope) |
| Crystalline Overgrowth | RULE (new, unsettled) | MED |
| Minion damage to turrets 60% | NUMBER | MED |
| Aggro rules (minions first, 1400-range ally trigger) | RULE STABLE | HIGH (verify, don't rebuild) |
| Tier 2 / inhibitor / nexus turret stats | NUMBER | LOW |

---

## 7. Runes → Runes Reforged

### 7.1 The structural change, and a correction to the usual date

Patch 4.20 used **old Runes** (marks/seals/glyphs/quintessences bought with IP and socketed
into pages) **plus Masteries** (a 30-point tree). At 4.20 the mastery tree was still the
original **Offense / Defense / Utility** layout — the Ferocity/Cunning/Resolve relabel did not
happen until **V5.22** (November 2015), a year later
([Summoner Mastery](https://wiki.leagueoflegends.com/en-us/Summoner_Mastery)).

Both systems were removed and replaced by **Runes Reforged** in a single patch:
**V7.22, released 8 November 2017** (preseason 2018). The patch page states it directly:
*"Masteries and Old Runes Removed"* ([V7.22](https://wiki.leagueoflegends.com/en-us/V7.22)).

> **Correction to the common claim.** Runes Reforged is often dated to patch 8.1 / Season 8.
> 8.1 was merely the first patch of *ranked* Season 8 (January 2018). The actual removal and
> replacement happened at **V7.22**, two months earlier. Use V7.22.

**The migration trap in that same patch.** V7.22 also states that *"all champions received stat
adjustments to compensate for the removal of rune-based stat bonuses."* Riot moved power out of
runes and into **champion base stats** simultaneously. This is visible directly in Garen's own
history (§4.1): base AD 57.88 → 66 and base armor 27.536 → 36 *at V7.22*.

The consequence for this project is specific and important:

> **You cannot model the modern rune system by taking 4.20 base stats and swapping the rune
> bonuses.** The base stats and the rune bonuses were rebalanced against each other in the same
> patch. Both columns must come from modern data together, or the champion will be wrong by
> roughly the amount of the old rune page.

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Customisation system | Runes (IP-bought, socketed) + 30-point Masteries | Runes Reforged: 2 paths + 3 stat shards | **RULE** | **HIGH** |
| Launch patch | — | **V7.22, 2017-11-08** | — | — |
| Base stats | pre-compensation | post-compensation (raised at V7.22) | NUMBER, but **coupled** | **HIGH** |

### 7.2 Modern structure and stat shards

Per [Rune](https://wiki.leagueoflegends.com/en-us/Rune):

- **Primary path:** 1 keystone + 3 runes (one per remaining row)
- **Secondary path:** 2 runes from a different tree
- **3 stat shards**, one per row:

| Row | Current options |
|---|---|
| Offense | Adaptive Force **+5.4 AD or +9 AP** · **+10% attack speed** · **+8 ability haste** |
| Flex | Adaptive Force (+5.4 AD / +9 AP) · **+2.5% move speed** · Health scaling **+10 to +200** (by level) |
| Defense | **+65 health** · **+15% tenacity and slow resist** · Health scaling (+10–200) |

Trees: Precision, Domination, Sorcery, Resolve, Inspiration. Adaptive force converts at 0.6 AD
or 1.0 AP per point ([Adaptive force](https://wiki.leagueoflegends.com/en-us/Adaptive_force)).

> **Structural fact with direct consequences for this migration: there is no flat armor or flat
> MR stat shard on Summoner's Rift.** The old system's flat **+9.0 armor at level 1 has no
> modern rune-based equivalent at all.** That power now lives in base stats and items. An
> earlier search result claiming "Armor: 6 / MR: 8" shards could not be corroborated against
> the wiki page and appears to conflate Arena-mode shards or a reverted variant —
> **flagged UNVERIFIED, treat the no-armor/MR table above as authoritative.**

**Architecture note:** the *set of available shard options* has itself changed over the system's
life, not just the numbers inside it. Model the shard table as versioned/patch-keyed data, not
as a fixed option list with swappable values.

### 7.3 What a modern Garen top actually runs

Cross-checked across two aggregators (Mobalytics, U.GG) for patch 26.18. Direct fetches were
blocked (HTTP 403); this rests on search summaries, so **treat the specific page as
meta-volatile and lightly sourced**, while the *shape* of the answer is solid.

- **Keystone: Conqueror** (Precision). The wiki's own
  [Conqueror](https://wiki.leagueoflegends.com/en-us/Conqueror) page names Garen as an example
  champion for the rune — his E interacts naturally with the stacking mechanic. That is a
  first-party confirmation the pairing is intended, not incidental.
- **Primary:** Conqueror → Triumph → Legend: Haste → Last Stand
- **Secondary (Sorcery):** Axiom Arcanist + Celerity
- **Shards:** Adaptive Force / Adaptive Force / Health scaling

**Why not Grasp or Aftershock** (the "tanky top" alternatives the brief anticipated):
**Aftershock requires immobilising an enemy champion**, and Garen has no immobilise in his kit
— Q is a silence and a slow-cleanse, not a root or stun. Aftershock is a poor fit for Garen
specifically. Grasp of the Undying remains a plausible off-meta alternative.

### 7.4 Replacing the "+137.97 HP / +20.25 AD / +9.0 armor" line

There is no clean single number to swap in. The honest answer has two parts.

**Part 1 — unconditional level-1 stats from the modern page:**

| Stat | 4.20 (runes + masteries) | Modern (runes reforged) |
|---|---|---|
| HP | **+137.97** | **≈ +10** (health-scaling shard is near its floor at level 1) |
| AD | **+20.25** | **+10.8** (two adaptive-force shards × 5.4) |
| Armor | **+9.0** | **+0** — no shard source exists |
| Move speed | — | +1% (Celerity's flat component) |

**A modern level-1 Garen gets roughly half the AD, a fourteenth of the HP, and none of the
armor that the 4.20 rune page gave him.** Some of that gap is genuinely gone; most of it moved
into base stats at V7.22 (§7.1) — which is exactly why the two columns cannot be mixed.

**Part 2 — the rest of the power is now stateful, and is code, not a number:**

| Rune | What it actually is | Implementation shape |
|---|---|---|
| **Conqueror** | 0 at start; stacks on champion damage (max 12, 5s, refreshing) for **1.08–2.56 AD per stack** (level-scaled, ~13–31 AD at max), plus **8% of post-mitigation damage healed** at full stacks for melee | combat state machine + heal hook |
| **Triumph** | on **takedown** only: +2.5% max HP, 5% missing-HP heal, +20 gold | discrete event handler |
| **Legend: Haste** | 0 at start; **+1.5 ability haste per stack up to 15**, stacks from a point economy (100/champion takedown, 25/large monster, 4/minion ≈ 25 CS per stack) | point accumulator across the lane phase |
| **Last Stand** | **5%–11% bonus damage**, only below 60% / 30% max HP | own-health-conditional damage multiplier |
| **Axiom Arcanist** | +14% ult damage (9% AoE); 7% ult CD refund on takedown | irrelevant before level 6 |
| **Celerity** | +7% amplification of *other* move-speed sources | multiplier on other effects |
| *Grasp (alt.)* | 3.5% max-HP magic damage + 1.3% max-HP heal on a 4-stack proc, **+5 permanent max HP per proc** | proc timer + permanent accumulator |

**Recommendation for the patch table.** Put **+10.8 AD / +10 HP / +0 armor** in the modern
column as the unconditional level-1 floor, and book **Conqueror stacking** as the single
highest-value rune mechanic to implement for Garen (the wiki names him as a target champion,
and it scales with exactly the sustained-damage pattern E produces). Everything else in the
table above can be deferred and flagged.

| Change | Class | 1v1 priority |
|---|---|---|
| Rune/mastery system replaced wholesale | **RULE** | **HIGH** |
| Base stats rebalanced in the same patch (coupling) | NUMBER, **coupled** | **HIGH** |
| Flat level-1 rune stats drop sharply; armor shard gone | NUMBER | **HIGH** |
| Conqueror stacking | **RULE (new)** | **HIGH** |
| Legend: Haste point economy | **RULE (new)** | MED |
| Last Stand missing-HP damage ramp | **RULE (new)** | MED |
| Triumph takedown heal/gold | **RULE (new)** | MED (1v1: only on the kill itself) |
| Celerity / Axiom Arcanist | **RULE (new)** | LOW |

### 7.5 Stability

Runes Reforged has been structurally stable for **nine years** (V7.22, 2017 → present). The
architecture — two paths, keystone, three shard rows — is safe to build against. Every number
inside it is balance-patched frequently and belongs in data, not in engine logic.

---

## 8. Items

### 8.1 Season 4 starting items

Item stats below are wiki-verified for the 4.20 era. **The Garen-specific meta claim is
UNVERIFIED** — the wiki does not retroactively document per-champion Season 4 build meta.

| Item | Stats at ~4.20 | Source |
|---|---|---|
| Doran's Shield | 450g; +80 HP, +6 HP/5s | [Doran's Shield](https://wiki.leagueoflegends.com/en-us/Doran's_Shield), Season 4 (V4.3) |
| Doran's Blade | 450g; +7 AD, +70 HP, +3% lifesteal (V4.10) | [Doran's Blade](https://wiki.leagueoflegends.com/en-us/Doran's_Blade) |
| Cloth Armor | 300g; +15 armor (confirmed present *at* V4.20) | [Cloth Armor](https://wiki.leagueoflegends.com/en-us/Cloth_Armor) |

The Cloth Armor page describes it as a starter "to improve lane sustainability against physical
damage dealers", explicitly contrasted with Doran's Shield — i.e. the two were a
matchup-dependent pair. Treat **Doran's Shield + potions** and **Cloth Armor + 5 potions** as
equally defensible defaults, chosen by matchup config rather than asserted as fact.

### 8.2 The Mythic era — introduced and removed, skip it entirely

| | Patch | Date |
|---|---|---|
| Mythic items introduced | **V10.23** | 10 November 2020 (preseason 2021) |
| Mythic items **removed** | **V14.1** | January 2024 (Season 14) |

Removal confirmed three separate times across item pages:
[Mythic item](https://wiki.leagueoflegends.com/en-us/Mythic_item) ("was removed in V14.1"),
[Stridebreaker](https://wiki.leagueoflegends.com/en-us/Stridebreaker), and
[Trinity Force](https://wiki.leagueoflegends.com/en-us/Trinity_Force) ("reclassified from a
Mythic item in Patch V14.1 ... removing the previous Mythic Passive").

The system: Mythics were finished items with higher stats than Legendaries, each carrying a
**Mythic Passive** that granted bonuses per *other* Legendary owned, with a **one Mythic per
build** limit.

**Implementation instruction: do not build it.** For a 2026 target the Mythic tier is a
transient historical feature that existed for exactly three seasons. It appears in a great deal
of secondary literature written between 2021 and 2023, so it is worth an explicit "do not
implement" line to stop someone porting it from a stale guide.

**1v1 priority: N/A.** Only listed to be ruled out.

### 8.3 Current representative Garen build (patch 26.18)

Highly meta-volatile — this list will be stale within a patch or two. It is here to show which
*stat primitives* a modern build exercises, not as a build to hardcode. Item stats below are
from direct wiki item-page fetches; the build ordering is from aggregator summaries.

| Slot | Item | Stats and passive |
|---|---|---|
| Start | Doran's Shield | as above, still the recommended Garen starter |
| Core | **Stridebreaker** (Legendary since V14.1) | 3,300g; +40 AD, +25% AS, +450 HP; **Cleave** on-hit AoE (40% AD melee) + **Breaking Shockwave** AoE knockback-slow dash |
| Boots | Berserker's Greaves (U.GG) or Boots of Swiftness (Mobalytics) | 1,100g; +30% AS, +45 MS — most matchup-volatile slot |
| Mid | **Phantom Dancer** | 2,650g; +65% AS, +25% crit, +10% MS, permanent unit-ghosting |
| Mid | **Mortal Reminder** | 3,000g; +35 AD, +30% armor pen, +25% crit, applies **Grievous Wounds**; limited to one "Fatality"-class item |
| Late | Dead Man's Plate | 2,900g; +55 armor, +350 HP, +4% MS; momentum passive (its old on-hit slow was removed at V14.1) |
| Late | Sterak's Gage | 3,200g; +400 HP, +20% tenacity, +50% base AD as bonus AD; **Lifeline** shield below 30% HP |

**Read this build against §4.6.** Two of the six items are attack-speed items and two are crit
items. That is not a coincidence: modern E gains a tick per 25% bonus AS and can crit. The
build path is downstream of the E rework, which is why "Garen builds attack speed now" is a
statement about *rules*, not fashion.

### 8.4 Item stats an implementation must support

Confirmed present in the current system ([Item](https://wiki.leagueoflegends.com/en-us/Item)
plus individual pages):

**AD · Armor · MR · Health · Health regen · Attack speed · Ability haste · Critical strike
chance · Critical strike damage · Life steal · Omnivamp · On-hit physical/magic damage ·
Move speed · Tenacity · Armor penetration**

**Terminology, resolved.** Life steal was **not** renamed to "Physical Vamp". Per
[Vamp](https://wiki.leagueoflegends.com/en-us/Vamp), three distinct stats exist under a "vamp"
umbrella:

| Stat | Heals from | Status |
|---|---|---|
| **Life steal** | basic attacks only | **live** (e.g. Bloodthirster 15%) |
| **Omnivamp** | all damage | **live** (e.g. Doran's Blade +2.5%) |
| Physical vamp | post-mitigation physical damage generally | **legacy** — last source removed V11.13, removed from the HUD V14.1; **no current item grants it** |

**Implementation takeaway: support Life Steal and Omnivamp as two separate healing hooks gated
by damage source.** One unified lifesteal stat (as 4.20 had) is not sufficient. Physical vamp
can be modelled as present-but-sourceless or omitted.

### 8.5 Item rule changes vs number changes

| Mechanic | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| **Cooldown reduction** | **CDR%**, `CD × (1 − CDR/100)`, **capped 40%** | **Ability haste**, `CD / (1 + AH/100)`, **uncapped**, introduced **V10.23** | **RULE** — different formula, different curve, no cap | **HIGH** |
| Mythic tier + Mythic passives | n/a | existed 2021–2023, **removed V14.1** | RULE, transient | **N/A — do not build** |
| Grievous Wounds | existed as item-specific "reduced healing", ~50%, inconsistent stacking | standardised debuff, generally **40%**, non-stacking, persists through shields | NUMBER (mechanic predates 4.20) | MED |
| Life steal vs omnivamp | one unified lifesteal stat | **two separate stats**, gated by damage source | **RULE** | MED |
| Item class-exclusion tags | some unique passives/auras existed, different taxonomy | items tagged into mutually-exclusive families (Fatality, Momentum, Hydra, Phantom Dancer…) | **RULE** | MED — HIGH if the sim ever picks items automatically |
| Spellblade | Sheen-style "next auto after ability" existed | 200% base AD on-hit, 1.5s internal CD, 10s window | NUMBER (rule survived) | MED |
| All item stat lines | — | — | NUMBER | HIGH |

**The ability-haste formula is the one non-negotiable item-side rule change.** Garen's Q and E
cooldowns are the spine of his damage rotation; getting the cooldown function wrong makes every
damage-per-window calculation wrong. `CD/(1+AH/100)` is a different curve from
`CD × (1 − CDR/100)` and, crucially, has no cap — so late-build cooldowns can go places the
4.20 engine could not represent.

---

## 9. Lane and economy mechanics

### 9.1 Ambient gold

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Rate | 0.95g / 500ms = **1.9 g/s** | **20.4 gold per 10s = 2.04 g/s**, ticking every 0.5s | NUMBER (+7.4%) | **HIGH** |
| Start time | **90s** | **1:05 (65s)** | NUMBER | **HIGH** |
| Scales with game time? | no | **no — flat all game** | NO CHANGE | — |

Source: [Gold (League of Legends)](https://wiki.leagueoflegends.com/en-us/Gold_(League_of_Legends)),
*Passive gold gain → Base global gold generation*.

> **Correcting a common assumption:** the base passive tick does **not** scale with game time in
> modern League. The impression that late-game income accelerates comes from siege-minion gold
> scaling (§5.3) and from gold-income items (support-role tools, not top-lane relevant), not
> from the ambient tick.

**UNVERIFIED:** the exact patch that moved 1.9 → 2.04 g/s and 90s → 65s.

### 9.2 Ambient XP

| | 4.20 | Modern | Class |
|---|---|---|---|
| Passive XP | **0** | **0** | **NO CHANGE** |

Confirmed against [Experience (champion)](https://wiki.leagueoflegends.com/en-us/Experience_(champion)).
The only passive-XP mechanic in the game is the Howling Abyss aura (ARAM), which does not apply
to Summoner's Rift. **Priority: N/A** — nothing to model.

### 9.3 Champion respawn timers

This is a **rule** change, not a number change: the clean linear formula became a hand-tuned
lookup table, and a second dimension was added that did not exist at all in 4.20.

Modern, per [Death](https://wiki.leagueoflegends.com/en-us/Death), *Death timer → Summoner's Rift*:

**Base Respawn Wait by level (1–18):**
`10 / 10 / 12 / 12 / 14 / 16 / 20 / 25 / 28 / 32.5 / 35 / 37.5 / 40 / 42.5 / 45 / 47.5 / 50 / 52.5` seconds

**Plus** a game-time multiplier that only activates after **15:00** (+0.425%/30s from 15–30 min,
+0.3%/30s from 30–45 min, +1.45%/30s from 45–55 min, hard cap +50%). `Total = BRW × (1 + TIFx)`.

**The time multiplier is zero before 15:00, so for a ten-minute episode only the table matters.**

| Level | 4.20 (`7.5 + 2.5(L−1)`) | Modern BRW | Δ |
|---|---|---|---|
| 1 | 7.5 | **10** | **+2.5 longer** |
| 2 | 10 | 10 | 0 |
| 3 | 12.5 | 12 | −0.5 |
| 4 | 15 | **12** | **−3 shorter** |
| 5 | 17.5 | **14** | **−3.5 shorter** |
| 6 | 20 | **16** | **−4 shorter** |
| 7 | 22.5 | **20** | **−2.5 shorter** |
| 8 | 25 | 25 | 0 |
| 9+ | 27.5 → 50 | 28 → 52.5 | ≈ +2.5 longer |

> **This is counter-intuitive and worth stating explicitly: modern deaths are *shorter* at
> levels 3–7**, which is exactly the level band a ten-minute top lane lives in. Do not assume
> "comeback-era League = shorter timers" uniformly — at level 1 and from level 9 up, modern
> timers are *longer*.

**Class: RULE CHANGED. 1v1 priority: HIGH** — death cost is the single largest term in the
risk side of every trade the agent evaluates, and it moved in *both* directions depending on
level.

**UNVERIFIED:** the patch at which the linear formula was replaced by the table. Both endpoint
states are solidly sourced.

### 9.4 The XP curve — identical, no work needed

The level-up threshold table is **bit-for-bit identical** between 4.20 and today.

Per-level deltas: `280/380/480/580/680/780/880/980/1080/1180/1280/1380/1480/1580/1680/1780/1880`
Cumulative (levels 2–18): `280 / 660 / 1140 / 1720 / 2400 / 3180 / 4060 / 5040 / 6120 / 7300 /
8580 / 9960 / 11440 / 13020 / 14700 / 16480 / 18360`

This matches the given 4.20 baseline exactly, including 18360 at level 18. The table was set at
**V3.14** (preseason 4, ~Nov 2013) and **no subsequent patch note changes it** through V26.01.
Source: [Experience (champion)](https://wiki.leagueoflegends.com/en-us/Experience_(champion)),
main table + patch history.

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| XP-to-level table | as given | **identical** | **NO CHANGE** | HIGH (to confirm, not to build) |
| Level cap | 18 | 18 | NO CHANGE | — |

**What did change is the XP you receive**, not the thresholds — see §5.4 (solo minion XP 92% →
100%) and §9.5.

### 9.5 Shared XP and kill/assist XP

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Minion XP radius | **1400** (raised from 1250 at V4.11, so 4.20 already had 1400) | **1500** (V25.S1.1) | NUMBER | LOW in a 1v1 |
| Champion kill/assist XP radius | UNVERIFIED | **1600** | NUMBER | LOW |
| Solo minion XP | 92% of base | **100%** (V26.01) | RULE/NUMBER | **HIGH** |
| Shared pool | — | 2 champs 65% each; down to 21.7% at 6 | NUMBER, volatile | LOW in a 1v1 |
| Comeback kill-XP bonus | cruder integer-level version | decimal-level scaling, up to −60% when ahead (tuned V10.8, V13.4) | RULE | LOW in a 1v1 |

For an isolated 1v1 the radius and sharing terms almost never bind — the last-hitter is always
in range and there is nobody to share with. **The one that matters is solo XP 92% → 100%**,
which is a straight 8.7% multiplier on every level timing in the episode.

### 9.6 Kill gold, first blood, and bounties — the task's premise was wrong here

> **Correction: the tiered kill/death-streak bounty system already existed in patch 4.20.** It
> was set at **V3.9** (January 2014) and then remained **completely unchanged for a decade**,
> until **V14.10** (2024). Shutdown gold is not a post-4.20 invention. Source:
> [Champion gold bounties/History](https://wiki.leagueoflegends.com/en-us/Champion_gold_bounties/History).

**The table live during patch 4.20:**

| Streak | Kill gold | Death (consecutive) |
|---|---|---|
| First Blood | **400 total** (300 base + **100 bonus**) | — |
| 1st | 300 | 300 |
| 2nd | 300 | 274 |
| 3rd (Killing Spree) | 450 | 220 |
| 4th (Rampage) | 600 | 176 |
| 5th (Unstoppable) | 700 | 140 |
| 6th (Dominating) | 800 | 112 |
| 7th (Godlike) | 900 | 100 (floor) |
| 8th+ (Legendary) | 1000, +100/kill up to 3500 at the 33rd | — |

Note also that **patch 4.20 itself removed** a pre-existing early-kill gold discount (kills
before 4:00 paid 75–100% of normal). As of 4.20 shipping, there is no early-game kill discount.

**Modern (post-V14.21, October 2024 — a genuine rework):**

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| First Blood bonus | **+100g** | **+100g** | **NO CHANGE** (despite removal at V25.S1.1 and re-add at V26.01) | **HIGH** |
| Base kill gold | **flat 300** at every level | **scales with victim level**: 300 ×6, then 310/320/…/420 (V25.09, 2025) | **RULE (new)** | **HIGH** |
| Bounty accrual | discrete streak tiers | **continuous**: +1 bounty per 3g from kills/assists; +1 per 20g farmed while ahead, +1 per 7g while behind | **RULE** | **HIGH** |
| Shutdown | tier-based reset | accrued bounty beyond [base + 100]; single-kill payout capped at [base + 700], **excess rolls over** to the next kill (rollover added V8.23) | **RULE** | MED |
| Assist bounty | half kill, cap 150 | half kill, cap 50% of base (up to 210 now) | NUMBER | LOW in a 1v1 |
| Objective Bounties | **did not exist** | 10% of team gold deficit per objective, cap 1000g (introduced **V11.23**, 2021; current formula V25.09) | **RULE (new)** | **LOW** — keyed on *team* gold deficit, not lane state |

**For a 1v1 lane the load-bearing items are:** first blood is still exactly +100g (so that
constant ports unchanged), and **base kill gold now scales with the victim's level**, which
changes the payoff of a kill at level 7+ relative to 4.20. The continuous-accrual bounty
machinery matters much less in a two-champion episode where streaks stay short, but it is a
real rule change if the sim ever runs long enough for a lead to compound.

### 9.7 Minion gold scaling over time

Covered in detail in §5.3. Summary here because it is an economy-curve property, not a minion
stat: melee (20g) and caster (14g) are **flat** on Summoner's Rift; only **siege minions scale**
(`49 + 1 per 90s upgrade`). Average wave value runs **118.66g at 0:30 → 115g at 14:00 → 121.5g
at 15:00 → 148g at 25:00**.

> **Refinement of the brief's assumption:** minion gold does scale with time, but the mechanism
> is narrower than "all minions get more gold" — it is siege-minion upgrade gold only. Within a
> ten-minute window the effect is modest (~118 → ~121 g/wave), so flattening it is a defensible
> approximation for a short episode, and a bad one for a long one.

---

## 10. Map and geometry

**Bottom line: reuse the existing map data. No citable evidence of top-lane geometric change
exists between 4.20 and 2026.** This is also the least-verified section in the document, and it
is worth saying why.

### 10.1 Why this section is weaker than the others

The wiki **does not maintain a patch-history section on the
[Summoner's Rift](https://wiki.leagueoflegends.com/en-us/Summoner's_Rift) page** — its table of
contents is Environment / Gameplay / Shopkeepers / Versions / Trivia, with no history log. That
is unlike essentially every ability, item and mechanic page, all of which track patch-by-patch
changes. Geometry changes are evidently either rare enough or under-the-hood enough that the
wiki does not track them the way it tracks numbers.

**Consequence: a negative result here is weaker evidence than a negative result elsewhere in
this document.** "No patch note says the brush moved" is not the same as "the brush did not
move." If lane geometry turns out to matter to the agent's behaviour, it should be verified by
measuring the live game, not by reading.

### 10.2 The Season 5 visual rework

The Summoner's Rift Visual and Gameplay Update shipped in **patch 5.1** (January 2015). Its
*documented* gameplay-relevant changes were: texture and LOD sharpening, a turret-aggro/Flash
interaction bugfix, **base gates added between middle and outer lanes**, jungle AI and leash
tuning, jungle XP-cliff tuning, a Baron Nashor stat rebalance, and Howling Abyss/Smite changes.

**Nothing in the patch notes describes altered lane length, brush repositioning, or turret
coordinates — for top lane or any lane.**

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Top lane corridor geometry | — | no documented change | **NO CHANGE** (UNVERIFIED at strength) | HIGH if wrong, LOW if right |
| Art / textures / props | old assets | rebuilt in 5.1 | ASSET ONLY | N/A |
| Base gates between lanes | absent | added 5.1 | RULE (new) | N/A — not in the top lane corridor |

Riot's public communication at the time characterised the update as deliberately
geometry-preserving — same wall shapes, brush hitboxes and lane topology, new art on top —
specifically to avoid invalidating competitive pathing and vision knowledge. **This is
background knowledge, not a fresh citation — flagged UNVERIFIED.**

### 10.3 Later map updates

- **Preseason 2021 / 2020-era jungle changes** (camp composition, jungle plants such as Blast
  Cone, Scryer's Bloom, Honeyfruit) are **jungle-interior** changes, not lane-corridor changes.
  Some plants spawn in jungle brush adjacent to the top-lane jungle entrance, which could matter
  for vision or dive modelling right at the lane's jungle-facing edge. **UNVERIFIED** as to
  exact spawn-location stability near top lane.
- **The 2022 "Durability Update"** was a global damage/time-to-kill rebalance, **not** a
  geometry change. It is, however, a very real and high-impact change for a laning sim — see
  §11.6.
- No wiki-documented change was found to the **top lane tribrush position, top-side jungle wall
  shapes, or turret coordinates** in any patch. Turret *stats* changed enormously (§6); turret
  *positions* appear stable.

### 10.4 Verdict

**Reuse the 4.20 navgrid and lane geometry.** The verified deltas near top lane are economy and
timing mechanics layered on top of the same geometry (turret stats and decay, minion spawn
timing, sidelane speed), not the geometry itself.

Practical note for this project: `JAX_REWRITE_PLAN.md` §1.4 records that the AIMesh navgrid was
already parsed out of `Content/.../AIPath.aimesh_ngrid`, and §1.11 reports movement/pathfinding
parity measured at 0.062 against the server. **That work is not invalidated by the migration**,
which is a meaningful saving: the single most expensive piece of the sim to re-derive is the
piece that does not need re-deriving.

**Confidence: high, but not exhaustively closed.** Flag it as such rather than as settled.

---

## 11. Mechanics that did not exist in 4.20

Scoped to what could plausibly matter in a 1v1 top lane inside ten minutes. Two of the brief's
four candidates turned out to already exist in 4.20 — those corrections are below, and they are
useful because they are work you do *not* have to do.

### 11.1 Ability Haste — genuinely new, and a formula change

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Stat | **Cooldown Reduction (CDR%)** | **Ability Haste** | **RULE** | **HIGH** |
| Formula | `CD × (1 − CDR/100)` | `CD × 100/(100 + AH)`, i.e. `CDR% = AH/(AH+100)` | **RULE** | **HIGH** |
| Cap | hard cap **40%** | **no meaningful cap** (soft cap ~500) | **RULE** | **HIGH** |
| Introduced | — | **V10.23**, preseason 2021 | — | — |

Per [Ability haste](https://wiki.leagueoflegends.com/en-us/Ability_haste): *"In Pre-Season 2021,
haste replaced cooldown reduction, a statistic that multiplicatively increased casting speed and
mandated an artificial cap and limited access from items and runes."*

**Why this needs code, not a column.** The old stat was multiplicative with a hard ceiling; the
new one is additive in the stat and hyperbolic in the output, with no ceiling. Garen has no
innate haste scaling, but every haste source on his items and runes (Legend: Haste, §7.4) will
behave differently from an old CDR item, and late-build cooldowns can reach values the 4.20
engine could not represent at all. Q and E cooldowns are the spine of Garen's damage rotation —
get this wrong and every damage-per-window number is wrong.

### 11.2 Tenacity — already existed in 4.20, including on Garen's own W

> **Correction: tenacity is not new.** The stat name dates to **V1.0.0.118**, pre-Season 1,
> when Mercury's Treads' "35% crowd control reduction" was renamed "+35% tenacity".

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Mercury's Treads | **35%** | **30%** (35 → 20 at V5.22 → 30 at V6.4, stable since) | NUMBER | MED |
| Tenacious mastery | 10% CC-duration reduction | masteries gone; now a **"Tenacity and Slow Resist" rune shard** (15%) | RULE (system moved) | MED |
| Cap | — | **100%**, via a 3-group stacking formula (additive within group, multiplicative across) | RULE (formula) — **UNVERIFIED** date | LOW |
| **Garen W tenacity** | **granted tenacity** (removed at V5.16, so it existed through 4.20) | **60% for the first 0.75s** of the active (re-added V7.14, stable 9 years) | RULE (removed then re-added) | **HIGH** |

**The cap is 100%, not 75%** as the brief assumed. And note the Garen-specific detail: his W
granted tenacity in 4.20, lost it in 2015, and got it back in 2017 in a different shape (a short
burst window rather than the full duration). That interacts directly with §4.5.

### 11.3 Grievous Wounds — already existed in 4.20, and barely matters against Garen

> **Correction: Grievous Wounds is not new either.** The named debuff and its items (Bramble
> Vest, Morellonomicon, Ignite) are Season 1–3 vintage. The wiki's patch history for the page
> starts at V5.22, which reads *"Now only affects self-healing... Healing reduction reduced to
> 40% from 50%"* — pinning the 4.20-era state at **50%, affecting healing from all sources**,
> i.e. *stronger and broader* than it became.

Value history: **50% (4.20)** → 40% and self-healing only (V5.22) → all sources again (V6.9) →
two-tier standard/enhanced 40%/60% (V10.23) → cut to 25%/40% (V12.10–12.11) → **unified at 40%
for all sources (V13.3, 2023)**, which is the current state.

| | 4.20 | Modern | Class | 1v1 priority |
|---|---|---|---|---|
| Healing reduction | 50%, all sources | **40%, unified across all sources** (V13.3) | NUMBER + RULE (unification) | **LOW vs Garen** |
| Sources | Ignite, Bramble Vest, Morellonomicon, Executioner's Calling | above plus Chempunk Chainsword, Mortal Reminder, Oblivion Orb, Thornmail | NUMBER | MED |

**Garen-specific, straight from the wiki's own strategy notes:** *"Some champions only have
sustain when out of combat. This includes **Garen**... **Grievous Wounds is not effective
against those champions** (as the debuff usually doesn't last long enough to have effect on
them)."*

**So: LOW priority when Garen is the target** — his sustain is Perseverance, which only runs out
of combat, and the debuff expires before it applies. It becomes **MEDIUM-to-HIGH only if the
modelled opponent has in-combat sustain.** In a Garen-mirror 1v1 (which is this project's
Stage J4 setup), Grievous Wounds is close to irrelevant.

### 11.4 Teleport — one of the most-changed mechanics found

If Teleport is in scope for the sim at all, this deserves attention: it changed repeatedly, and
the most recent change is barely a year old.

| | 4.20 | Modern | Class |
|---|---|---|---|
| Base cooldown | **300s** | **300s** — but only after a decade-long detour (360s at V8.14; a level-scaling 420→210s from V10.6 to V12.1; back to 360s; **300s again at V25.S1.1**) | NUMBER (net), RULE (the level-scaling era came and went) |
| Channel time | **3.5s** (set V4.4) | **3s** (reduced from 4s at V25.S1.1) | NUMBER |
| **Turret cooldown refund** | **yes** — teleporting to an allied turret cut the cooldown to 240s | **removed at V5.22** | **RULE (removed)** |
| Arrival | **instant blink** after the channel | **channel (3s) then a separate dash phase of 0.5–5s by distance**, untargetable and unable to act during it (**V25.S1.1**, Jan 2025) | **RULE (new)** |
| **Unleashed Teleport** | **does not exist** | at **10:00** game time (was 14:00; introduced V12.1, moved to 10:00 at V13.10) TP upgrades: arrives with **+50% move speed for 3s** and a faster dash | **RULE (new)** |
| Targets | turrets, wards (minion targeting timeline UNVERIFIED) | turrets, minions, wards (minion/ward targeting removed V12.1, **re-added V25.S1.1**) | RULE |
| Level requirement | 6 | 7 (V7.22) | NUMBER |

**1v1 priority: HIGH if Teleport is modelled, N/A if it is not.** Top lane is the Teleport lane,
so this is likely in scope eventually. The blink→dash change (January 2025) is the easiest one
to miss when working from older documentation, and it changes total time-to-arrive.

### 11.5 Rift Herald and Objective Bounties — exist, but out of scope

| Mechanic | Introduced | 1v1 priority |
|---|---|---|
| **Rift Herald** | **V5.22–5.24 (Nov 2015)** — much older than the brief assumed, not preseason 2021 | **N/A** — a jungle objective requiring jungler participation; no interaction with an isolated 1v1 lane |
| **Objective Bounties** | **V11.23 (Nov 2021)**; current formula V25.09 | **LOW** — keyed on whole-team gold deficit, not lane state; it does suppress shutdown gold when active, so note it exists if kill payouts are ever computed holistically |

Listed only to be explicitly ruled out, so nobody spends time on them.

### 11.6 The 2022 Durability Update — a real change that fits neither category

The 2022 Durability Update lengthened time-to-kill across nearly all champions and items. It is
**not a mechanic** — it is thousands of individual number changes — so it does not fit the
NUMBER/RULE frame cleanly. But it is genuinely new since 4.20 and it directly changes laning
combat, which is this project's entire subject.

**Implication:** the modern numbers in §4 already embody it, so no separate implementation is
needed. But when comparing 4.20 and modern *trade outcomes*, expect modern trades to resolve
more slowly and fights to last longer than the 4.20 baseline, beyond what any single row in
these tables explains. **1v1 priority: HIGH as an expectation-setter, N/A as an implementation
item.**

### 11.7 A trap: "League Classic" is not a 4.20 reference

Riot shipped **League Classic** (live 29 July 2026), a retro Featured Game Mode recreating
roughly Season 3-era League, sitting in the mode picker alongside Arena and URF.

> **Patch 26.16's turret and death-timer changes ("Classic Turrets... same effective HP they had
> in 2013", "Death Timers 12/14/16/19/22/25/28/30 → 10/12/14/17/20/23/26/29") apply ONLY to
> League Classic, not to standard Summoner's Rift.** Source:
> [League Classic](https://wiki.leagueoflegends.com/en-us/League_Classic) plus the raw 26.16
> patch notes, which explicitly reference 2013 values and "porting jungle monsters to modern
> systems".

Two independent research passes hit this trap, which is why it gets its own subsection. League
Classic is a tempting reference point for "old League" data and it is the wrong one twice over:
it is not standard SR, and it is not a faithful 4.20 recreation either — it is a curated
Season-3-ish greatest-hits mode. **Never source a "4.20" or a "modern SR" number from a League
Classic patch-note section.**

---

## 12. Consolidated migration plan

### 12.1 Every rule change in the document, ranked

These need code behind a feature flag. Ordered by value for a ten-minute 1v1 top lane.

| # | Rule change | § | Priority | Rough shape of the work |
|---|---|---|---|---|
| 1 | Minion aggro: drop the "champion attacks allied minion" priority entry | 5.9 | **HIGH** | Delete one entry from the priority list. Trivial code, large behavioural effect. |
| 2 | Minion 90-second stat upgrade system | 5.1 | **HIGH** | A global `U(t)` counter feeding HP/AD lookups. Self-contained. |
| 3 | First wave at 0:30; 20 waves per 10 min | 5.5 | **HIGH** | Constants, but re-times the whole episode. |
| 4 | Garen E: tick count scales with bonus attack speed | 4.6 | **HIGH** | Tick loop reads a stat instead of a constant. |
| 5 | Garen E: minion damage modifier removed | 4.6 | **HIGH** | Delete the 0.75× branch. |
| 6 | Turret plating (5 plates, %-missing-HP thresholds, 120g each, no expiry) | 6.4 | **HIGH** | New structure state + a gold event. The largest single new subsystem. |
| 7 | Garen R: damage type magic → unconditional true (execute shape unchanged) | 4.7 | **HIGH** | Drop the MR mitigation step for R specifically; also changes the kill-threshold surface the agent sees. |
| 8 | Garen Q: dash + attack-timer reset | 4.4 | **HIGH** | New movement + AA-timer interaction. |
| 9 | Minion-vs-minion % current-HP on-hit (2 / 3.5 / 5%) | 5.8 | **HIGH** | New on-hit in the minion-vs-minion damage path only. |
| 10 | Sidelane speed buff (top-lane specific, 4-stage decay) | 5.7 | **HIGH** | Per-wave spawn buff with a staged timer. Changes where waves meet. |
| 11 | First-wave behaviour: spread attacks, ghosting 28s, ignore champions | 5.6 | **HIGH** | Special-case the opening wave. Implement the spread-attack rule first. |
| 12 | Garen W: two-phase active (0.75s shield + tenacity, then DR tail) | 4.5 | **HIGH** | Shield object + tail timer + tenacity grant. |
| 13 | Garen passive: flat → level-scaling regen | 4.3 | **HIGH** | Formula change. |
| 14 | Ability haste replaces CDR% (`CD × 100/(100+AH)`, uncapped) | 11.1 | **HIGH** | New stat type + new cooldown function. |
| 15 | Call for Help now includes targeted abilities | 5.9 | **HIGH** | Aggro trigger set widens. Check Garen Q/E flags. |
| 16 | Solo minion XP 92% → 100% | 5.4 | **HIGH** | One constant; 8.7% on every level timing. |
| 17 | Turret damage ramp: 3-hit / +150% cap / 5s reset | 6.3 | **HIGH** | Formula-shape change plus new reset semantics. |
| 18 | Turret AD scales with game clock | 6.2 | **HIGH** | Turret damage reads the clock. |
| 19 | Base kill gold scales with victim level | 9.6 | **HIGH** | Lookup instead of a constant. |
| 20 | Respawn timer: linear formula → lookup table | 9.3 | **HIGH** | Table; note it is *shorter* at levels 3–7. |
| 21 | Conqueror stacking (and the rune procs generally) | 7.4 | **HIGH** | Combat state machine. The highest-value single rune for Garen. |
| 22 | Garen E: armor shred at 6 hits; crit interaction; nearest-target +25% | 4.6 | MED | Three independent additions. |
| 23 | Garen W: permanent kill-stacking passive | 4.5 | MED | Accumulator; slow to matter in a solo lane. |
| 24 | Minion Pushing buff (level-lead damage bonus from 3:30) | 5.7 | MED | Positive-feedback loop the agent will exploit. |
| 25 | Reinforced Armor: flat resistance → 80% DR incl. true damage | 6.5 | MED | Binds when diving with no wave present. |
| 26 | Life steal vs omnivamp as separate stats | 8.4 | MED | Two healing hooks gated by damage source. |
| 27 | Death grace window (1 HP, 0.066s) | 5.9 | MED | Cheap; threshold is low-confidence. |
| 28 | Crystalline Overgrowth (turret true damage, 2026) | 6.4 | MED | New and unsettled — defer. |
| 29 | Item class-exclusion tags (Fatality, Momentum, Hydra…) | 8.5 | MED | Only needed if the sim picks items automatically. |
| 30 | Turret resistance decay 11:00–15:00 | 6.2 | MED | Outside a 10-minute window. |
| 31 | Bounty accrual: streak tiers → continuous | 9.6 | MED | Rarely binds in a 2-champion episode. |
| 32 | Teleport: blink → channel+dash; Unleashed TP at 10:00 | 11.4 | HIGH *if modelled* | Only if summoner spells are in scope. |

**Rules that can be deleted rather than built:** the blue/red minion asymmetry (§5.10) is gone in
modern League — which also retires the D0 contradiction with the mirror assumption in
`lanerl_rl/obs.py`. Super minions, the Villain mechanic, and the Mythic item tier should never be
built at all (§5.2, §4.7, §8.2).

### 12.2 Verified no-ops — do not spend time here

| Thing | § | Status |
|---|---|---|
| Per-level stat growth `g × (0.65 + 0.035×L)` | 4.2 | **Introduced by 4.20, still live in 2026.** Identical. |
| XP-to-level table (280…18360) | 9.4 | **Bit-for-bit identical.** Unchanged since V3.14. |
| Level cap 18 | 9.4 | Unchanged. |
| Garen Q cooldown 8s flat | 4.4 | Unchanged since V1.0.0.145. |
| Garen R cooldowns 160/120/80 at 4.20 | 4.7 | The server is **correct** for the era. |
| Ambient XP = 0 | 9.2 | Still zero on Summoner's Rift. |
| Wave composition 3 melee + 3 caster + conditional siege | 5.5 | Unchanged since 2009. |
| Siege every 3rd wave (before 14:00) | 5.5 | Unchanged. |
| Minion attack speeds and attack ranges | 5.2 | Unchanged since 2014. |
| Minions deal 60% damage to champions | 5.8 | 60% then, 60% now — **but verify it is implemented at all.** |
| Minion pathfinding/gameplay radii | 5.2 | Unchanged. |
| Turret range 750 | 6.2 | Unchanged. |
| Turret priority: minions before champions | 6.6 | Unchanged. |
| E cooldown starts when the spin *ends* | 4.6 | Convention survives (inferred, §4.6). |
| First Blood bonus +100g | 9.6 | Same value, despite a 2025 detour to zero. |
| Top lane geometry, brush, turret positions | 10 | No documented change. Reuse the navgrid. |
| Intra-wave stagger ~800ms | 5.5 | 0.792s — the same constant, tick-rounded. |

### 12.3 Suggested sequencing

The dependency structure is shallower than the list length suggests.

**Phase 1 — the data table.** Every NUMBER row. Garen's stats and ability values, minion base
stats, turret stats, gold and XP tables, respawn table. Mechanically simple; no engine changes.
Do this first because it is the part D1 already promises, and because several rule changes are
untestable until the numbers are right.

**Phase 2 — the four cheap, high-value rules.** Items 1, 5, 16 and the `U(t)` counter (item 2).
Each is a small, local, self-contained change with a large behavioural payoff: one deleted aggro
entry, one deleted damage branch, one constant, one global counter. This is the best
value-per-line in the migration.

**Phase 3 — Garen's kit.** Items 4, 7, 8, 12, 13, 22, 23. These are independent of each other
and each is a contained change to one ability.

**Phase 4 — the lane systems.** Items 3, 9, 10, 11, 15, 24. Wave timing, the new minion damage
path, sidelane speed, first-wave behaviour, aggro triggers. These interact, so land them
together and re-baseline after.

**Phase 5 — turrets.** Items 6, 17, 18, 25. Plating is the big one and is worth its own
milestone, because it adds a reward object and therefore changes the RL problem, not just the
simulator.

**Phase 6 — the long tail.** Runes (21), ability haste (14), items, and anything marked MEDIUM.

**Validation note.** Stage J5 in `JAX_REWRITE_PLAN.md` already records the hard part: *"a fresh
validation story because the oracle is gone."* There is no modern LeagueSandbox to diff against,
so the Tier-1 differential that carries the 4.20 work does not exist here. Two partial
substitutes: keep the 4.20 parity suite running against the 4.20 table as a **regression gate on
the engine** (the logic is shared; only the table and the flags differ), and validate the modern
table itself against the two independent sources this document used (rendered wiki and shipped
game files), which is how most rows here were confirmed in the first place.

---

## 13. UNVERIFIED register

Every gap, in one place. An honest gap is more useful than a confident wrong number, and several
of these are cheap to close with direct measurement rather than more reading.

### 13.1 Conflicts between the vendored server and the historical record

| Item | Conflict | § | Suggested resolution |
|---|---|---|---|
| Outer turret HP / armor | Server 1550 / 67 vs patch notes 2000 / 100 (MR agrees at 100 both ways) | 6.1 | Check the `Content/` field name and units. Then label the column honestly rather than "fixing" it. |
| Minion gold (caster 10, siege 35/30) | No patch note supports these values at any point | 5.3 | Accept as reimplementation artifacts; migrate to corroborated modern values. |
| Minion XP (77/51/94/500) | Era base was 64/32/100/100; no multiplier explains all four | 5.4 | Same. Do not use the 4.20:modern XP ratio for anything. |
| Wave period 36.4s | Real 4.20 was 30s | 5.5 | Treat fixing it as a 4.20 bug fix, tracked separately from the migration. |
| Minion stat growth | Server static; real 4.20 had HP growth | 5.1 | Note that the migration *adds* scaling the baseline never had. |
| Garen R damage type/shape | A wiki-only reading suggested flat magic damage with no missing-health term; the server's `R.cs` (§3.3) and this document's own pre-9.20 patch-note values agree the missing-health term was already present | 4.7 | **Resolved** — use the `R.cs`-derived formula; no further action needed. |
| Garen Passive regen | A Data-Dragon-`4.20.2`-based read reported a flat 0.4%/s, 10s lockout; the server's `GarenPassiveHeal.cs` (§3.3) shows a level-bracketed {0.4/0.8/2.0%}/s with a {9/6/4}s lockout | 4.3 | **Resolved** — use the script-derived brackets. |

### 13.2 Modern values that could not be confirmed

| Item | What is unknown | § | How to close |
|---|---|---|---|
| Garen E radius 330 → 325 | Which patch changed it; whether 325 is exact | 4.6 | Data Dragon spell data. |
| Garen E crit multiplier | Exact current sub-formula (5 revisions in 6 years) | 4.6 | Re-read at implementation time; do not hardcode. |
| Turret model tagged as inhibitor tier | `SRUAP_Turret_Order3`'s own `UnitTags` reads `Structure_Turret_Inhib`, on the model this project's roster uses as the **outer** turret | 6.1 | Check `LevelScriptObjects.LoadBuildings` before freezing the modern turret table — it determines which modern row replaces this one. |
| Turret damage ramp 40% → 50% | Which patch | 6.3 | Patch-note search; low stakes. |
| Plate gold 160 → 125 → 120 | Intermediate patch numbers | 6.4 | Low stakes; current value is confirmed. |
| First Turret gold | Wiki says 300; V8.23 reduced it to 150; unreconciled | 6.7 | **Re-check the live wiki before freezing.** |
| Crystalline Overgrowth trigger | Two sources describe it differently | 6.4 | Defer the mechanic until it settles. |
| Turret aggro: 1400 range at 4.20 | Era-specific value unconfirmed | 6.6 | Low stakes. |
| Turret aggro: AoE hitting champion + minion | Whether it counts as damaging the ally | 6.6 | **Measure in the live game.** |
| Turret aggro drop timing | Beyond "dies / leaves range / untargetable" | 6.6 | **Measure in the live game.** |
| Minion acquisition range | Wiki says 500; game files say 700 caster / unset melee | 5.9 | Trust the game files; verify empirically. |
| Minion XP radius | Wiki 1500 vs game files 1400 | 5.4 | Irrelevant in a 1v1. |
| Death grace threshold | 0.35% may be off by 100× per a wiki edit note | 5.9 | Needs a better source before relying on it. |
| Siege gold latch timing | Read at spawn or at death? | 5.11 | Assume death-time; ≤1 gold impact. |
| First-wave ghosting | Introduction patch (V9.4 raised 18 → 28) | 5.6 | Low stakes. |
| **Garen Q/E Call-for-Help flags** | Whether E draws minion aggro under the modern rule | 5.9 | **Worth closing — it affects the cost of every E in the wave.** |
| Ambient gold change | Patch that moved 1.9 → 2.04 g/s and 90s → 65s | 9.1 | Low stakes; endpoints confirmed. |
| Respawn formula → table | Which patch | 9.3 | Low stakes; endpoints confirmed. |
| Champion kill XP radius at 4.20 | Era value | 9.5 | Low stakes in a 1v1. |
| Tenacity stacking formula | Patch that formalised the 3-group rule | 11.2 | Low stakes. |
| ~~Garen W tenacity % at 4.20~~ | **Resolved (§3.3):** flat 30%, whole active duration (`StatsModifier.Tenacity.PercentBonus += 30` in `GarenW.cs`). | 11.2 | — |
| Season 4 Garen starting items | No wiki source documents per-champion S4 meta | 8.1 | Accept as config, not fact. |
| Armor/MR stat shards | A search result claimed they exist; uncorroborated | 7.2 | Treat the no-armor/MR table as authoritative. |
| Modern Garen rune page / build | Aggregators only (direct fetch 403) | 7.3, 8.3 | Meta-volatile by nature; treat as config. |
| Top lane geometry | **No patch-history section exists on the SR wiki page** | 10 | A negative result here is weaker than elsewhere. **Measure if it matters.** |
| Jungle plant spawns near top lane | Stability near the lane's jungle edge | 10.3 | Only matters for vision/dive modelling. |

### 13.3 Claims from the brief that research did not support

Recorded so they are not carried forward from older notes or secondary sources.

| Claim | Finding | § |
|---|---|---|
| "Runes Reforged launched in patch 8.1" | **V7.22**, 8 November 2017. 8.1 was merely the first ranked patch of Season 8. | 7.1 |
| "Turret fortification was an early-game buff that plating replaced" | **No source supports this.** The real mechanic is pre-4.20 resistance-over-time, removed *by* 4.20. | 6.5 |
| "Turrets need ~24 attacks to tag a champion" | **Not found on the wiki at all.** Treat as folklore. | 6.6 |
| "Champions deal 45% reduced damage to plating unless minions are present" | **Not found.** Likely a conflation with Reinforced Armor's 80%. | 6.5 |
| "Shutdown/bounty gold did not exist in Season 4" | **It did.** Set at V3.9 (Jan 2014), unchanged until V14.10 (2024). | 9.6 |
| "Tenacity is new since 4.20" | **It is not.** The stat name dates to V1.0.0.118; Garen's own W granted it in 4.20. | 11.2 |
| "Grievous Wounds was standardised after 4.20" | Existed in 4.20 at **50%, all sources** — broader than it later became. Unified at 40% in V13.3. | 11.3 |
| "Tenacity caps at 75%" | **100%.** | 11.2 |
| "Rift Herald was introduced around preseason 2021" | **V5.22–5.24, November 2015.** | 11.5 |
| "Modern ambient gold scales with game time" | The base tick is **flat all game**. Scaling comes from siege gold. | 9.1 |
| "Siege minion replaced the name cannon minion" | Both names coexisted in 2014 and still do. Riot files say *Siege*; players say *cannon*. | 5.2 |
| "Lifesteal was renamed Physical Vamp" | Three distinct stats. Life steal and omnivamp are live; physical vamp is legacy and sourceless. | 8.4 |
| "Modern deaths are longer (comeback era)" | **Shorter at levels 3–7**, longer at 1–2 and 9+. | 9.3 |

### 13.4 A trap worth repeating

**League Classic is not a 4.20 reference.** Patch 26.16's turret and death-timer numbers apply to
that retro mode only, not to standard Summoner's Rift. Two independent research passes hit this.
See §11.7.
