# Is the enemy `gold_total` stream real, or a stale held value?

**Date:** 2026-08-27. **Scope:** the *content* of the lane opponent's
`visible_heroes[opp].gold_total` in `labels.json` — not whether the field is
populated (that was already checked) but whether the numbers in it are a live
per-event read. **Trigger:** `docs/REWARD_SIGNAL_VIABILITY.md` (7f0e06a) and the
2026-08-27 CORRECTION in `docs/DESIGN_DECISIONS.md` §1 recommend flipping the
reward to zero-sum gold-diff (`use_solo_gold=False`) on the strength of
"present on 99.81% of frames". *Present* and *correct* are different claims.
Only the first had been tested.

**Corpus:** all 146 labeled games in `/srv/nfs/datasets/lol_replays_16_9_772`,
**4,176,465 frames = 58.0 game-hours** at 20 fps, 99.880% of them labeled.
3,738,079 consecutive-frame deltas analysed. Every number below is measured on
that corpus. Analysis scripts are in the session scratchpad
(`eg/extract.py`, `eg/analyze.py`, `eg/summarize.py`, `eg/addendum.py`,
`eg/fieldscan.py`); they only read `labels.json`. No code was changed and no
training was run.

**Status key:** CONFIRMED (measured here) · LIKELY (measured but confounded) ·
SPECULATIVE (reasoned).

**Control throughout:** Garen's own `champion_stats.gold_total`, which is
definitely a live read. Every metric is reported for the opponent *and* for
Garen over the same frames. Nothing below rests on an absolute threshold; it
rests on own-vs-opponent parity.

---

## Verdict

**Yes — the enemy `gold_total` stream is trustworthy enough to build a reward
on.** CONFIRMED. It is a genuine live per-event read of the opponent's
cumulative earned gold, indistinguishable from Garen's own stream on every
statistic tested, and it does not degrade when the opponent is off screen.

The suspected failure mode — a value refreshed only when the enemy is visible,
held constant in between — is **falsified, not merely unsupported**:

> Across 58 game-hours, the lane opponent's `gold_total` never holds a constant
> value for more than **2.10 seconds**. The corpus-wide maximum hold for
> **Garen's own gold is also 2.10 seconds.** The opponent is off screen ~51% of
> the time; a vision-gated value would show holds of tens of seconds every time
> he recalls.

What *is* broken is a different field (`gold`, not `gold_total`) and the
*identity* plumbing around the opponent, neither of which blocks the reward
change. Details in §7.

The strategic conclusion in the task brief still stands and is unchanged by
this: **for the Bronze milestone (wave management, CSing) solo gold is the
right reward and none of this is needed.** This document removes the *data*
objection to zero-sum, not the *sequencing* argument for leaving it until
trading and denying matter.

---

## 1. Provenance: the corpus is two sub-corpora, and this was not previously recorded

CONFIRMED, and it changes how several prior numbers should be read.

`labels.json` files split cleanly by mtime into two populations:

| | n | `lane_opponent` persisted | `visible_heroes[*].screen` |
|---|---|---|---|
| **live-pipeline** (written 2026-05) | 87 | yes, all 87 | valid (projected) |
| **backfilled** (rewritten 2026-06-29) | 59 | **null, all 59** | **always `None`** |

The 59 were rebuilt post-hoc by `scripts/aggregation/backfill_visible_heroes.py`
from the already-captured `raw_mem.json`, because the pre-2026-05-08 pipeline
emitted only `hp`/`hp_max`/`level` for `visible_heroes` and dropped `gold_total`
and off-screen heroes entirely. Each carries `_backfilled_at` and
`_backfill_stats` at the root of `labels.json`.

Three consequences:

1. **"87 of 146 games have a resolvable lane opponent" is wrong.** All 146 do.
   The 59 nulls are a *persistence* gap, not a resolution failure — and
   `resolve_lane_opponent()` already re-derives from frames when the cached
   value is falsy (`src/ahriuwu/data/lane_opponent.py:104-112`). Re-deriving
   succeeds on **146/146**. The n=87 in `REWARD_SIGNAL_VIABILITY.md` §5 is an
   undercount by 68%.
2. **The backfill script does not recompute `lane_opponent`** — that is the
   whole cause of the nulls.
3. **The "opponent on screen 49.1%" figure is only meaningful on the 87
   live-pipeline games**, where I measure mean 0.4923 (min 0.277, max 0.817) —
   consistent. On the 59 backfilled games `screen` is structurally `None` for
   every hero including Garen, because the backfill has no camera to reproject
   with. Every visibility test in §4 below is therefore restricted to the 87.

**Backfill join quality — CONFIRMED sound.** The backfill joins frames to mem
samples by nearest `gt` with *no* gap tolerance (`_nearest_mem`, unlike the live
pipeline's `MAX_MEM_GAP`), so it could silently pull a distant sample. It
doesn't. Built-in control: in backfilled games `champion_stats` still comes from
the original tolerance-checked join while `visible_heroes[Garen]` comes from the
new untoleranced one, so they can be differenced.

| | max abs disagreement | frac frames disagreeing | by >20 g |
|---|---|---|---|
| backfilled (n=59) | median 15.1 g | 0.00058 | 0.000014 |
| live-pipeline (n=87) | 0.00 g | 0 | 0 |

Median disagreement affects 0.058% of frames, and 99.99% of frames agree to
within 20 g. The join is fine.

---

## 2. Presence and finiteness — the original claim, re-measured

CONFIRMED, and slightly stronger than reported.

| metric | measured |
|---|---|
| labeled frames with a finite opponent `gold_total` | **100.000%** (min over 146 games: 1.00000) |
| frames labeled at all | 99.880% pooled (min 99.518%) |
| ⇒ opponent `gold_total` present, all frames | **99.880%** |
| games with any absurd (\|value\|>1e6) `gold_total`, any of 10 heroes | **0 / 146** |
| negative deltas (gold going backwards), corpus-wide | **0** own, **0** opponent |

The prior 99.81% figure is right. `gold_total` is monotone non-decreasing for
every hero in every game — no resets, no rollbacks.

---

## 3. Update cadence — the crux

CONFIRMED live. This is the test that would have caught a held value, and it
is the strongest result in this document.

League grants passive gold on a ~0.5 s engine tick. A live read of any player
shows a change roughly every 0.5 s forever. A vision-gated read shows a flat
line whenever that player is unobserved.

Per-game statistics, restricted to `gt > 150 s` (after passive income starts)
and to consecutive frames 0.02–0.12 s apart:

| statistic | Garen (control) | lane opponent |
|---|---|---|
| update rate | 2.110 /s | 2.118 /s |
| median gap between updates | 0.500 s | 0.500 s |
| p99 gap | 0.550 s | 0.550 s |
| **max gap, worst game of 146** | **2.10 s** | **2.10 s** |
| fraction of time inside any >5 s hold | **0.00000** | **0.00000** |
| passive drip | 2.0127 g/s | 2.0190 g/s |
| negative deltas | 0 | 0 |

Per-game paired ratios opponent/own: update rate **1.0040** (p05 0.978, p95
1.037), drip rate **1.0032** (p05 0.991, p95 1.030). The two streams tick in
near-lockstep — `P(opponent's gold changes | Garen's gold changes) = 0.933`
median, Jaccard 0.871 — which is what the shared passive tick predicts and what
an independent stale-refresh schedule cannot produce.

**All ten heroes, as a fog-of-war test.** Allies are perpetually in team vision;
enemies are not. If the client only updated observed units, the two groups would
separate. They do not:

| group | n (hero, game) | max update gap | passive drip | update rate |
|---|---|---|---|---|
| focus (Garen) | 146 | med 1.00 s | 2.013 g/s | 2.110 /s |
| 4 allies | 584 | med 1.00 s | 2.027 g/s | 2.103 /s |
| **5 enemies** | **730** | **med 1.00 s** | **2.020 g/s** | **2.106 /s** |

Enemies are statistically identical to allies and to the focus champion.
`visible_heroes` is a full memory read of all ten hero structs, as claimed, and
the *values* in it are live, not merely present.

---

## 4. Visibility gating — the specific bug that was feared

CONFIRMED absent. Live-pipeline games only (n=87), where `screen` is valid.

**Conditioning on whether the opponent is projected on screen:**

| | opp on screen | opp off screen | off/on ratio |
|---|---|---|---|
| **opponent's** gold change rate | 0.10719 /frame | 0.10480 /frame | **0.978** |
| **Garen's** change rate, same frames (control) | 0.10577 | 0.10489 | **0.992** |
| opponent's passive drip | 2.018 g/s | 2.022 g/s | 1.002 |

The opponent's update rate falls by 2.2% off screen. Garen's own falls by 0.8%
over the identical frames — so nearly all of the modulation is a shared
gameplay confound (both players earn more while they are in lane together, which
is exactly when the opponent is on camera), not a gate on the enemy's read.

**Long unobserved stretches.** Over 87 games there are 1,105 maximal off-screen
runs of ≥10 s, totalling 13.4 hours of "opponent not on camera". Inside them:

| inside ≥10 s off-screen runs | measured |
|---|---|
| opponent's gold updates | **2.090 /s** (median 2.089, min 1.990) |
| opponent's passive drip | **2.021 g/s** |
| opponent's *lump* (last-hit/kill) gold | **3.59 g/s** |

Not one game shows the flat line a held value would produce.

**Refresh-on-visibility test.** A held-then-refreshed value dumps its accumulated
backlog the instant the unit becomes observable. Mean |delta| at an off→on-screen
transition is **0.336 g** against a baseline of **0.328 g** — no spike. Gold
gained in the 0.5 s after onset is 3.04 g, i.e. one passive tick plus ordinary
laning income.

**Worked example** (NA1_5549981347, live-pipeline, opponent Kayle): the longest
single off-screen stretch runs `gt` 323.1 → 383.0 s, 59.9 s and 1,199 frames
during which Kayle is never projected on screen. Over it his `gold_total` goes
1925.3 → 2094.7, through **123 distinct values in 122 change frames**, with
deltas `{1.0 ×95, 1.1 ×24, 14.0 ×2, 20.0 ×1}` — the passive tick, two caster
minions and one melee minion, all captured while he was invisible to the camera.

---

## 5. Jump quantities — are the last-hit lumps really there?

CONFIRMED. Pooled over all 146 games, every delta ≥3 g ("lump"): 23,767 for
Garen, 24,229 for the opponent. The quantisation is identical.

| lump size | Garen | opponent | Garen % | opp % |
|---|---|---|---|---|
| 14 g (caster minion) | 6,668 | 7,130 | 28.06% | 29.43% |
| 15 g | 1,699 | 1,856 | 7.15% | 7.66% |
| 20 g (melee minion) | 6,314 | 7,113 | 26.57% | 29.36% |
| 21 g | 1,614 | 1,824 | 6.79% | 7.53% |
| 28–29 g | 1,463 | 903 | 6.16% | 3.73% |
| 34–35 g | 718 | 245 | 3.02% | 1.01% |
| 60–70 g (cannon / plate band) | 1,121 | 1,182 | 4.72% | 4.88% |

**Total variation distance between the two pooled lump distributions: 0.079.**
Both peak on the same two spikes at 14 g and 20 g. Per-game: fraction of lumps in
the 12–16 g caster band 0.346 (own) vs 0.360 (opp); in the 17–23 g melee band
0.334 vs 0.369. Lump *count* per game: 174 own, 175 opponent.

The residual asymmetry is at 28–35 g, where Garen has ~2× the opponent's share.
That is a Garen fact, not a data fact: those are double-lump frames — Garen's Q
and E hit multiple minions in one 50 ms frame, so two minions land in the same
delta (14+14, 14+20, 20+14). LIKELY, on the shape of the excess (28 = 14+14,
34 = 14+20). It is not a defect in the enemy stream; if anything it is evidence
the enemy stream is *not* a copy of Garen's.

**Off-screen lumps stay minion-shaped.** For the opponent, 71% of off-screen
lumps fall in 12–23 g vs 80% of on-screen ones. A batched refresh would show the
opposite signature — off-screen "lumps" would be large aggregates. (The mild
difference is expected: off-screen time is disproportionately recall, roams and
jungle camps, which have a wider gold spectrum.)

---

## 6. Divergence over time, on a balanced outcome sample

CONFIRMED: no persistent one-sided drift.

**Outcome proxy, and its limitations.** There is no `garen_win` manifest on disk
and no outcome field anywhere in the dataset — CONFIRMED, I searched. To avoid
circularity I built the proxy from **Garen's own stream only**, using nothing
from the enemy read that is under test:

> `score = z(own gold per minute) − z(own deaths per minute)`, split at the
> median into 73 "lane-winning" and 73 "lane-losing" games.

Separation achieved: own gpm 445.1 vs 323.5, own deaths/min 0.075 vs 0.185.

*Limitations, stated plainly:* this is an **absolute-farm-and-survival proxy,
not a head-to-head lane outcome**. A game where both laners farmed well scores
as a win; a short game scores differently from a long one even at equal skill
(gpm normalises duration but not phase mix); and team-level state (a fed jungler
farming for you, or a lost game that ends early) leaks in. It is directionally a
"did Garen play well" split, not "did Garen win lane".

**It nonetheless predicts things it was not given.** The split was built without
touching the enemy stream, yet it recovers, *from* the enemy stream: final gold
diff +2,396 (win half) vs −1,581 (lose half), and opponent deaths 5.78 vs 3.08.
A stale or fabricated enemy series could not track an outcome variable
constructed in ignorance of it. That is independent corroboration.

**Gold-diff trajectory, balanced sample:**

| t (s) | win half | lose half | all 146 |
|---|---|---|---|
| 300 | −43 | −251 | **−147** |
| 420 | −20 | −444 | **−228** |
| 600 | +57 | −609 | **−224** |
| 780 | +293 | −765 | **−154** |
| 900 | +556 | −894 | **−56** |
| final | +2,396 | −1,581 | **+407** |

The sign flips between halves, as it must if the diff is measuring gameplay.

**Systematic-drift check.** A defective enemy stream that under-counted would
produce a monotone positive drift in `own − opp`. Measured over all 146:

- mean final gold diff **+407 g**, 95% CI **[−73, +888]** — includes zero.
- fraction of games with a positive final diff: 0.555.
- team-mean final gold, 4 allies minus 5 enemies: **+248 g**, 95% CI
  **[−17, +513]** — includes zero.
- own gpm 384.3 vs opponent 374.0 (2.7%); own final level 15.12 vs 14.73.

And the pooled trajectory does not drift one way at all — it **crosses zero**:

| t (s) | mean own | mean opp | mean diff | sd diff | opp/own |
|---|---|---|---|---|---|
| 180 | 1,086.8 | 1,130.0 | −43.2 | 225.9 | 1.040 |
| 300 | 1,675.3 | 1,822.3 | −147.1 | 368.1 | 1.088 |
| 420 | 2,324.1 | 2,551.7 | −227.5 | 535.8 | 1.098 |
| 600 | 3,354.5 | 3,578.5 | −224.0 | 749.1 | 1.067 |
| 900 | 5,441.1 | 5,497.5 | −56.3 | 1,478.1 | 1.010 |
| 1200 | 7,903.7 | 7,843.8 | +60.0 | 2,033.8 | 0.992 |
| 1500 | 10,416.7 | 10,191.5 | +225.2 | 2,636.8 | 0.978 |

The opponent is *ahead* through the first ~17 minutes and behind after — the
opposite direction from an under-counting defect, and a well-known lane shape
(melee Garen into a pool that is heavily ranged: Teemo, Vayne, Kennen, Kayle,
Quinn, Gangplank, Urgot, Jayce, Cassiopeia, Varus, Lucian all appear as resolved
opponents). `corr(own gold(t), opp gold(t))` has median **0.990** per game: the
two top laners track each other, exactly as the brief predicted sound data
would. LIKELY that the early deficit is genuine gameplay; CONFIRMED that it is
not the drift signature of a broken read.

---

## 7. Opponent identity — the weakest link, but it holds

CONFIRMED correct on this corpus; the plumbing around it is fragile.

| check | result |
|---|---|
| games with a resolvable opponent | **146 / 146** (87 persisted + 59 re-derived) |
| persisted value == freshly re-derived value | **87 / 87** |
| spawn-side team assignment yields 5 enemies + 4 allies | **146 / 146** |
| spawn-side team agrees with `labels["team"]` | **146 / 146** |
| Garen assigned to top lane by position | **146 / 146** |
| **resolved opponent assigned to top lane** | **146 / 146** |
| resolved opponent's time-in-jungle fraction | median 0.054, **max 0.253** |
| separation margin (2nd-nearest enemy − nearest, laning phase) | median **5,524** units, min **3,006** |
| games with an ambiguous margin (<500 units) | **0** |

**Does the identified opponent change mid-game?** Re-running the nearest-enemy
argmin in independent 60 s windows: per-game agreement with the resolved
opponent is median **1.00**, mean **0.966**, min 0.667. 109/146 games have a
single winner across every window; 34 have two; 3 have three. All six games
below 0.85 agreement still have the resolved opponent as the plurality winner
(e.g. Shen 6 windows / Zoe 2 / Ekko 1) — these are post-laning roams, not
misidentification.

**Is it ever a jungler who wandered past?** No. The four `XinZhao`, two
`Anivia`, one `Ahri`, one `FiddleSticks`, one `RekSai`, one `Warwick`, one
`Sejuani` and one `Kayn` resolutions look alarming by champion name, but each is
a genuine top-lane resident by position: e.g. NA1_5550028932 resolves XinZhao
with laning jungle-fraction **0.006** and a mean distance to Garen of **784**
units against **7,813** for the next-nearest enemy. Off-meta or autofilled top
laners, correctly identified. The 1,400-unit lane-corridor test never puts a
resolved opponent in the jungle.

**The fragility is in persistence, not resolution.** `lane_opponent` is null on
40% of the corpus (§1); anything that reads the field directly instead of
calling `resolve_lane_opponent()` will silently see 87 games instead of 146.
`src/ahriuwu/rewards/reward.py:92` calls the resolver, so the reward path is
correct as written.

---

## 8. What IS broken (found while validating; none of it blocks the reward)

1. **`gold` (current/unspent) is a dead memory offset — CONFIRMED, corpus-wide.**
   Scanned all numeric fields on 24 games (12 backfilled / 12 live-pipeline).
   `gold` is **constant for the entire game in 240/240 (hero, game) pairs** and
   in 24/24 focus-champion cases; 67/240 hold absurd magnitudes
   (range −2.69e35 … +2.58e35), 2/240 are non-finite. On NA1_5549995114
   `champion_stats.gold` is the single value `-3.7744e22` across all 11,301
   frames while `gold_total` moves normally through 1,063 values (500.0 →
   3,218.5). It is `OFFSETS["gold_current"]` pointing at the wrong word in the
   hero struct on patch 16.9.772 — the same wrong word for the focus champion as
   for enemies, which is itself reassuring: the enemy read path is not special.

   *Blast radius: nil for training.* Nothing under `src/ahriuwu/` reads it —
   `grep` confirms only `gold_total` reaches the reward. `overlay.py:216` reads
   it but already guards with `0 <= gc < 1e6` and falls back to `gold_total`.
   `backfill_visible_heroes.py:106` copies it, but copies the same garbage the
   live pipeline writes at `pipeline.py:906/1544/1582`, so it propagates nothing
   new. `src/ahriuwu/live/client_api.py:54` reads `currentGold` from the Riot
   Live Client API — a different, valid source, unaffected.

   **Use `gold_total`. Never `gold`.** Worth deleting the field or renaming it
   `gold_current_BROKEN` so nobody reaches for it later.

2. **`lane_opponent` is null on 59/146 games** (§1). Recoverable, but a trap for
   any consumer that reads the field instead of calling the resolver.

3. **`visible_heroes[*].screen` is `None` on all 59 backfilled games**, including
   for Garen. Any screen-space statistic over the whole corpus is diluted by 40%
   of games where the answer is structurally "never on screen".

4. **Every numeric field other than `gold` is clean.** `gold_total`, `hp`,
   `hp_max`, `level`, `world_x`, `world_z` — 0/240 constant, 0/240 absurd, 0/240
   non-finite, all ranges physically sensible (`gold_total` 500 → 18,500; `hp`
   0 → 4,860; `level` 1 → 20; world 52 → 14,700).

5. **`movement.speed` has a ~0.3–0.7% tail of impossible values** (median
   163–185, p99 ~434–486, max ~14,000 — the map diagonal). It is computed from
   a 10-frame position lookahead, so recalls, deaths/respawns and blinks read as
   teleports. Explainable, but it should be clipped or masked before use.

6. **Single-frame gold jumps >1,000 g exist and are real** — 6 for Garen and 11
   for opponents out of 3,738,079 frame-pairs (max 1,090 g own, 1,583 g
   opponent). Multi-kills with shutdown bounties landing in one 50 ms frame.
   Under `gold_scale=1e-3` a 1,090 g jump is a reward spike of 1.09 against a
   ±3 twohot range (`DESIGN_DECISIONS.md` §2) — a real but rare tail worth
   knowing about when the bucket range is re-tuned. Under
   `gold_diff_scale=5e-5` it is 0.079 and harmless.

7. **Enemy `hp` is live too** (bonus check, n=30 games): median max hold 46.3 s
   for the opponent vs **50.5 s for Garen himself** — the opponent's stream
   holds *less* than the control. Whenever a trading/poke reward term is built,
   the enemy HP read will not be the obstacle either.

---

## 9. What this changes

- **`REWARD_SIGNAL_VIABILITY.md` §5 and `DESIGN_DECISIONS.md` §1 CORRECTION are
  upheld on the data question, and their n is too small.** The lane opponent is
  resolvable on **146/146** games, not 87. Both documents should be updated.
- **The data objection to `use_solo_gold=False` is fully retired.** It is not
  merely "present"; it is live, event-accurate, monotone, correctly attributed,
  and unaffected by visibility.
- **The sequencing argument is untouched.** Bronze targets wave management and
  CSing, which are solo-gold concepts — you do not need the opponent's gold to
  learn to last-hit, and the reward that teaches it is the one already
  configured. Zero-sum buys trading and denying, which is Emerald-tier. This
  document says the switch is *safe whenever you want it*, not that it is *due*.
- **Before flipping it,** note `gold_diff_scale=5e-5` is 20× smaller than
  `gold_scale=1e-3`, so the flip changes the return scale as well as its
  content, and the ±3 twohot range is already mis-tuned
  (`DESIGN_DECISIONS.md` §2: 39 of 255 buckets ever used). Those two constants
  should be set together from the realised return distribution, not
  independently.

## 10. Limitations of this validation

- **No external ground truth.** There is no Riot match-API dump to check
  `gold_total` against; every test here is internal-consistency plus own-vs-
  opponent parity. A defect that corrupted all ten heroes identically and
  preserved minion quantisation, tick cadence and monotonicity would survive
  every test above. I consider that SPECULATIVE to the point of negligible.
- **The outcome proxy is not a win/loss label** (§6) — see the limitations
  stated there. The balanced split is a "did Garen play well" split.
- **The field scan (§8) is a 24-game stratified sample**, not all 146. The
  cadence, visibility, lump and identity results are all n=146.
- **Screen-space conclusions rest on 87 games**, not 146, for the reason in §1.
- **One patch, one champion, one lane.** Everything here is Garen top on
  16.9.772. The `gold_current` offset being wrong on this patch is a warning
  that offsets are patch-fragile; `gold_total` should be re-validated after any
  client update, and the tick-cadence test in §3 is the cheap way to do it.
