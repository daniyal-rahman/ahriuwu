# Where does enemy gold come from, and can it be trusted?

Provenance audit of the `visible_heroes` block, run 2026-08-27. Companion to the
statistical gold-diff analysis; this document answers only the **provenance**
question: what wrote these numbers, and is the enemy's `gold_total` a live read
or a held value.

**Scope of evidence.** n = 146 matches with a `labels.json` under
`/srv/nfs/datasets/lol_replays_16_9_772` (147 dirs; `NA1_5554140394` has no
labels), 4,176,465 frames, 4,171,462 labeled (99.880%). Every number below was
measured, not read off a comment. Claims are tagged CONFIRMED (I ran it),
LIKELY (I read the code), or SPECULATIVE.

---

## Verdict

**The enemy's `gold_total` is a live, full-heap read of all ten champion structs,
updated every frame regardless of camera or fog. It is not a held value.**
CONFIRMED.

`docs/archive/DESIGN_DECISIONS.md` §1's original claim — that a gold-diff reward "needs
the opponent resolved **and visible**, which fails exactly when the opponent
leaves screen" — is **false for the schema that produced this corpus**. It was
already false on the day it was written: the screen-space gate was deleted from
the recorder on 2026-05-08 (`ea8a4af`), and DESIGN_DECISIONS was written
2026-08-13 (`1644fd0`), three months later. The claim describes a schema that
had not existed for a quarter.

The later CORRECTION is right on the substance and slightly off on the number:
coverage is **99.880%** of all frames, not 99.81% — and, more usefully, the
opponent's `gold_total` is present on **exactly the same frames as Garen's own**,
to six decimal places. The 0.12% shortfall is unlabeled frames (mem sample >100 ms
away), which drop *everyone*, control included.

**But** the block containing that good field also contains a demonstrably
garbage one, `gold` (current/unspent), which is a single constant per hero for
an entire game in 130/130 hero-games audited. Field-by-field trust, not
block-level trust, is the correct posture. See §5.

For the BRONZE milestone (wave management, CS) none of this is needed — those
are solo-gold concepts. Nothing here forces the reward change; it only removes
provenance as a reason *against* it.

---

## 1. The data path, recorder to label

```
League client process memory
  └─ init_heroes()                pipeline.py:681-732   enumerate hero_array → all 10 hero structs
      └─ _mem_loop()              pipeline.py:891-942   50 Hz; per hero, read hp/hp_max/gold/gold_total/level/pos
          └─ raw_mem.json                               {wall, gt, heroes:{name:{...}}} per tick
              └─ post_process()   pipeline.py:1528-1546 → labels.json frames[].label.visible_heroes
              └─ backfill_visible_heroes.py             (alternate path, 59 matches — see §6)
```

**There is no vision, fog, or screen gate anywhere on that path.** LIKELY (read
the code, and it is short). `_mem_loop` iterates `hero_ptrs.items()` — the
pointer table built once by `init_heroes` from the game's own `hero_array` — and
dereferences each struct directly:

```python
# scripts/aggregation/pipeline.py:898-908
for name, hinfo in hero_ptrs.items():
    hp = hinfo["ptr"]
    pos = m.vec3(hp + o["position"])
    if not pos: continue
    entry = {
        ...
        "gold_total": round(m.f32(hp + o["gold_earned"]) or 0, 1),
```

The only way a hero can be dropped is the `if not pos: continue` guard. Measured:
it never fires. `raw_mem.json` for `NA1_5549995114` holds **10/10 heroes on
37,757 of 37,757 ticks**, median tick spacing 0.0500 s (the intended 50 Hz).
CONFIRMED.

`post_process` then copies every hero through, and says so:

```python
# scripts/aggregation/pipeline.py:1529-1534
# Include ALL heroes (not just camera-visible). Off-screen heroes get
# screen=None but their stats (gold_total, level, hp) are still
# captured every frame ...
```

**Why this is legitimate at all.** These are *in-client replays*, not live games.
The `.rofl` carries the full server-side state — that is what lets a replay
viewer switch to any player's camera and read anyone's scoreboard. The client
therefore genuinely holds all ten heroes' gold in memory. In a *live* game the
original DESIGN_DECISIONS instinct would have been closer to right; on replays it
does not apply. LIKELY (domain reasoning, consistent with every measurement
below).

### Provenance gap I could not close

**The recorder that produced this corpus is not the `pipeline.py` in this repo.**
CONFIRMED. Every native `labels.json` and every `raw_mem.json` on disk carries a
per-hero `inventory` field; the repo's `_mem_loop` never reads one, and
`git log -S'"inventory"' -- scripts/aggregation/pipeline.py` returns nothing in
any commit. A newer copy — presumably the Windows-side working tree — wrote this
data.

The divergence is **purely additive**, which is why the code above is still a
valid description of the gold read. Across 377,570 hero-ticks in one `raw_mem.json`
there are exactly two key sets, both a strict superset of what the repo version
emits:

```
321010  (gold, gold_total, hp, hp_max, inventory, level, pos)
 56560  (cast_target, gold, gold_total, hp, hp_max, inventory, level, pos, spell)
```

I reason from the repo source plus the data. I could not read the actual
recorder that ran.

---

## 2. Does the field name lie? Yes — it is historical

`visible_heroes` **was** a visibility filter. Commit `ea8a4af` (2026-05-08,
"Surface all-hero gold_total + world in labels.json") deleted the gate:

```diff
-            if sp:
-                visible.append({... "screen": sp, "hp":..., "level":...})
+            visible.append({
+                "name": name,
+                "screen": sp,                          # None when off-screen
+                "world": p,                            # always present
+                "gold_total": hd.get("gold_total", 0),
```

The name survived the semantics. **The count does not vary with what is on
screen**: pooled over all 146 matches,

| entries per frame | frames | share |
|---|---|---|
| **10** | 4,171,462 | **100.000000%** |

No frame anywhere in the corpus has 9 or fewer. CONFIRMED. The "full memory
read" claim is exactly right and the name is exactly wrong.

Per-frame *presence* of `gold_total`, on labeled frames:

| | fraction |
|---|---|
| own (`champion_stats`) | 1.000000 |
| lane opponent | 1.000000 |
| all ten heroes | 1.000000 |

= **99.880%** of all frames including unlabeled ones, identically for the
opponent and for the control.

---

## 3. Fog of war — the decisive tests

The failure mode to rule out is a value that moves only while the enemy is
observed and is held otherwise. Four independent tests, all negative.

### 3.1 Passive-tick synchrony (the strongest one)

League grants every champion a small passive gold drip **simultaneously**, on the
server's clock. So: take the frames where **Garen's own** `gold_total` rose by
≤2 g — a drip tick, located by the known-good control — and ask whether each
other hero ticked *on that same frame*. A held value cannot do this; it would tick
only when the hero came back on camera.

Pooled over 13 matches (25,150 own-drip frames):

| | on screen | **off screen** | **in fog (proxy)** |
|---|---|---|---|
| lane opponent | 8523/8728 = **0.9765** | 16124/16422 = **0.9819** | 7152/7295 = **0.9804** |
| all enemies | 1918/1963 = 0.9771 | 96963/98637 = **0.9830** | 41643/42437 = **0.9813** |
| allies | 1876/1900 = 0.9874 | 97530/98700 = 0.9881 | 45885/46400 = 0.9889 |
| **Garen (control)** | 15285/15285 = 1.0000 | 9805/9865 = 0.9939 | 20222/20266 = 0.9978 |

CONFIRMED. Off-screen synchrony is *equal to or higher than* on-screen. In one
match Yasuo and Kai'Sa were **never once on camera** (0/0 on-screen drip frames)
and still ticked on 980/990 = 99.0% of Garen's drip frames.

Two honest caveats. Garen's on-screen 1.0000 is tautological — the drip frames
are *defined* by Garen ticking; the real control is that the other nine hit
0.977–0.989 with no dependence on visibility. And the residual ~2% is 50 Hz mem
sampling landing between two 20 fps frames, which shifts a tick one frame late;
it affects the focus champion too (0.9939 off-screen).

The same test on a **backfilled** match (`NA1_5550028932`, a different code path)
gives 0.9892–0.9961 for all ten heroes. CONFIRMED.

### 3.2 Accrual rate by visibility

Full sweep, 146 matches, lane opponent only:

| state | frames | gold gained | rate |
|---|---|---|---|
| on screen | 997,704 | 360,244 | 7.221 g/s |
| **off screen** | 3,171,591 | 983,655 | **6.203 g/s** |
| in allied vision (proxy) | 2,618,887 | 943,817 | 7.208 g/s |
| **in fog (proxy)** | 1,550,408 | 400,082 | **5.161 g/s** |

CONFIRMED. A held value gives ~0 g/s off screen. The passive floor alone is
~2 g/s; off-screen accrual is 3× that, so the opponent is visibly **farming
through fog** in our labels. The on/off gap (7.2 vs 6.2) is the real thing it
should be: when the enemy top laner is on our camera he is usually in lane
CSing, and when he is not he is often walking, recalling, or dead.

### 3.3 Gold lumps and level-ups in fog

Two more, on the same 6 native matches, both independent of the drip:

- **486 enemy gold lumps ≥3 g while in fog** (median 17 g, p90 60 g, max 300 g) —
  last-hits and kill bounties landing entirely out of our vision. Lane opponent
  alone: 55.
- **49 enemy level-ups while in fog.** `level` is a *different memory offset*
  (`19800` vs `10384`), so this is a second field independently demonstrating
  live updates through fog.

CONFIRMED.

### 3.4 Position continuity — liveness without touching gold

If the recorder saw through fog for gold, it should see through fog for
position. Per-frame speed between consecutive 20 fps frames, 6 native matches:

| role | on screen | n | median | p99 | frac == 0 | frac > 3000 u/s |
|---|---|---|---|---|---|---|
| enemy | **False** | 259,829 | 231.9 | 1131.3 | 0.278 | 0.00135 |
| enemy | True | 2,615 | 200.8 | 884.1 | 0.301 | 0.00038 |
| lane opp | **False** | 26,876 | 217.7 | 2501.3 | 0.383 | 0.00700 |
| lane opp | True | 38,735 | 190.5 | 761.8 | 0.271 | 0.00026 |
| **Garen (control)** | True | 65,605 | 239.0 | 1014.0 | 0.278 | 0.00123 |

CONFIRMED. An off-screen enemy's position is as smooth as Garen's own. A held
position would show `frac == 0` near 1.0 punctuated by teleport-sized jumps;
instead `frac == 0` is 0.278 off-screen versus 0.301 on-screen. The residual
>3000 u/s frames are recalls, deaths and Flash.

### 3.5 The frozen runs are a game-clock artifact, not a fog artifact

The obvious counter-evidence — long stretches where enemy gold does not move —
is real but is not about visibility. The longest frozen run per hero is
**identical for the always-visible control and for the opponent, to the frame**:

```
NA1_5549995114  Garen      1257f (62.9s)  gt=1.132 → 64.982
NA1_5549995114  Fiora      1257f (62.9s)  gt=1.132 → 64.982
NA1_5550013959  Garen      1250f (62.5s)  gt=1.274 → 64.974
NA1_5550013959  Gangplank  1250f (62.5s)  gt=1.274 → 64.974
```

CONFIRMED. It is the window at game start before the first income of any kind;
pooled across 13 matches the median max-run is 1262 f for the focus champion and
1262 f for the lane opponent. Same number. Nothing visibility-shaped.

### 3.6 Integrity

`gold_total` is cumulative earned and must be monotone non-decreasing.

- **Zero** decreases in own gold across 146 matches. **Zero** in opponent gold.
- Across all ten heroes and ~41.7 M hero-frames: **2** decreases total, one on
  Ivern and one on Zac (both champions with unusual state mechanics). 5 × 10⁻⁸.
- Zero jumps > 2000 g in a single frame, own or opponent.

CONFIRMED. `reward.py`'s "loud-fail on numerical pathology" guard has nothing to
fire on here.

---

## 4. The reward code path actually runs

`use_solo_gold=False` had reportedly never been executed. It has now, on **146
matches × 3 configurations**, via `compute_episode_reward`:

| config | ok | exceptions | warnings | all-zero episodes | nonzero frames/ep (median) | negative frames/ep (median) |
|---|---|---|---|---|---|---|
| `solo` (default) | 146/146 | 0 | 0 | 0 | 3261 | 3 |
| `use_solo_gold=False` | **146/146** | **0** | **0** | **0** | 1131 | 578 |
| + `use_lane_anchor` + `use_outcome` | 146/146 | 0 | 0 | 0 | 1132 | 579 |

CONFIRMED. It is not dead code that errors or silently zeroes: it produces a
dense, signed, per-frame signal on every match in the corpus. `resolve_lane_opponent`
returned `None` zero times, so the "no lane opponent" warning branch never fired.

One structural property worth recording, because it falls out of §3.1: the
gold-diff has **35.5% as many nonzero frames as solo-gold** (median ratio 0.355).
The synchronised passive drip cancels exactly in the difference. Zero-summing the
reward therefore removes the passive-income component for free — which is a point
in its favour that the viability doc does not make.

*(Not in scope here: whether `gold_diff_scale = 5e-5` is the right constant, or
what the diff does to return magnitudes. That is the sibling analysis.)*

---

## 5. What in this block is NOT trustworthy

Establish field trust by measurement. The block is not uniformly good.

### 5.1 `gold` (current/unspent) — GARBAGE, confirmed, everywhere

Audited 13 matches = 130 hero-games:

| field | hero-games | single constant all game | every value \|x\| > 1e10 | uniq (min/med/max) |
|---|---|---|---|---|
| `hp` | 130 | 0 | 0 | 505 / 1508 / 4816 |
| `hp_max` | 130 | 0 | 0 | 5 / 18 / 119 |
| **`gold`** | 130 | **130 (100%)** | **25** | **1 / 1 / 1** |
| `gold_total` | 130 | 0 | 0 | 823 / 2287 / 4065 |
| `level` | 130 | 0 | 0 | 5 / 10 / 19 |

CONFIRMED. `gold` is a *different* constant per hero per game and never changes:
Garen −3.7744e+22, Vladimir 1.4594e+31, Yasuo −5.2098e+16, Nami 2.2036e+12,
Alistar 8.4254e+17, Fiora 0.0. Our own champion is affected identically — this is
not an enemy-only problem.

**Root cause, and it is instructive.** `scan_offsets.py` validates the two gold
offsets with very different rigour:

```python
# scripts/aggregation/scan_offsets.py:439-446  — gold_earned
if (... 100 < v1 < 200_000 and 100 < v2 < 200_000
        and v2 > v1 and (v2 - v1) < 100_000):      # MUST HAVE MOVED

# scripts/aggregation/scan_offsets.py:450-456  — gold_current
if (... 0 <= v1 < 30_000 and 0 <= v2 < 30_000
        and (v1 > 10 or v2 > 10)):                 # static range only
```

`gold_earned` had to *change between two snapshots* to be accepted, so it is a
behaviourally validated offset. `gold_current` only had to sit in a plausible
range — **a field that never moves passes that test**. It was then scanned on one
champion in one game and shipped. This is the same class of bug the movement
work hit twice: an acceptance test that a held value can satisfy.

Known consumers: `overlay.py:214-224` already range-gates it
(`0 <= gc < 1e6`) and falls back to `gold_total`, so display is safe;
`backfill_visible_heroes.py:106` (`"gold": hd.get("gold", 0)`) copies the garbage
straight into stored labels, so 59 matches now carry it on disk. Nothing in
`src/` reads it. Harmless today, a trap tomorrow.

### 5.2 `inventory[].uc` — garbage, no consumers

Per match, `uc` takes only 2–8 distinct values, one of which is a large
implausible constant present on 10.8%–43.1% of item slots (59393, 131073, 57857,
16897 alongside a sane `1`). Same signature. CONFIRMED.

`inventory[].lf` and `inventory[].id` are clean: `lf` spans 0–210 with 48–49
distinct values, `id` yields real League item IDs (1001, 1036, 2003, 3340, …).
`lf` is the one that matters — `replay_dataset.py:973` uses it for Stridebreaker
detection — and it survives the audit.

### 5.3 Nothing else has the signature

Scanned across 12 matches: `champion_world.{x,z}`, `champion_screen.{x,y}`,
`cursor.world.*`, `cursor.screen.*`, `movement.speed`,
`movement.heading_{world,screen}[*]`, `action.screen.*`. **Zero** constant-all-game,
**zero** values with |x| > 1e10, all ranges physically sensible. CONFIRMED.

`champion_stats.level` (8–19 distinct/game) and `hp_max` (18–50 distinct/game)
are *rarely changing*, not constant — genuine, as expected.

`label.waypoint` is `None` on 100% of frames in every match checked
(`_mem_loop` never writes a `waypoint` key, so `post_process`'s
`if waypoint else None` always takes the `None` branch). Dead field, not garbage.

### 5.4 The backfilled 59 have no per-hero `screen`

**59 of 146 matches** were rebuilt by `backfill_visible_heroes.py`, which hard-codes

```python
# scripts/aggregation/backfill_visible_heroes.py:101
"screen": None,    # backfill can't reproject without cam
```

CONFIRMED: in `NA1_5550028932`, **0 of 322,660** hero entries have a `screen`.
Their `gold_total`, `world`, `hp`, `level` are all fine — the backfill reads the
same `raw_mem.json` the pipeline did, and §3.1's tick test passes on it — but
anything keyed on per-hero `screen` is silently empty for 40% of the corpus.
(`label.champion_screen`, one level up, is untouched and valid.)

This is why the "opponent on screen 49.1%" figure must be measured on native
matches only: native median **0.4768**, mean 0.4923, n=87 — which reproduces the
earlier number, and explains why that investigation reported n=87 rather than 146.
Pooled over all 146 it would read 0.36 and be meaningless.

One thing I expected to be a bug and is not: `backfill`'s `_nearest_mem` has no
gap limit, unlike the pipeline's `MAX_MEM_GAP = 0.1` (`pipeline.py:177`), so in
principle it could clamp to a far-away sample. It cannot in practice — it only
rewrites frames that *already carry a label*, and the pipeline only labels frames
with a mem sample within 100 ms. `n_no_mem_match` was 0 on the match inspected.

---

## 6. Lane-opponent resolution is not the weak link

`src/ahriuwu/data/lane_opponent.py`. Two heuristics: team from earliest
(lowest-coord-sum) position in the first 30 s; opponent = enemy with smallest
mean distance to Garen over gt 60–300 s.

Full sweep, n = 146:

- `resolve_lane_opponent` returned `None` **0 times**.
- `identify_teams` found exactly 10 heroes and exactly 5 enemies in **146/146**.
- The pipeline-persisted `lane_opponent` (present in the 87 native matches)
  agrees with the live re-derivation in **87/87**. Zero disagreements.

CONFIRMED. And it is not a close call. On 40 matches:

- Gap between nearest and second-nearest enemy (mean distance over laning):
  median **5368 units**, p10 4206, **min 3547**. The map is ~14,800 units across.
- The resolved opponent is the *single nearest enemy* on median **88.5%** of
  laning frames; min 74.2%; **below 50% in 0/40**.
- Garen's mean laning position is in the top-lane corridor in 40/40.

Four picks look off-meta (XinZhao ×2, Ahri, Anivia). All four are unambiguous by
the same margin — e.g. `NA1_5550028932`: XinZhao at 784 units and nearest on
**100%** of laning frames, runner-up Xerath at 7813, and that enemy comp
(XinZhao/Xerath/Talon/Morgana/KogMaw) contains no other top laner. These are real
off-meta top laners, not resolution failures. LIKELY.

**The one genuine caveat.** The identity is resolved **once**, from gt 60–300 s,
and applied to the whole episode. The nearest enemy changes between later
4-minute windows in **76/146 = 54.7%** of matches — which is exactly what should
happen once laning ends and the map opens up. Nothing re-resolves. So in the late
game the diff is taken against someone who stopped being a lane counterpart
twenty minutes earlier. For a laning-phase reward that is arguably correct; for a
full-game reward it is a modelling decision nobody has made explicitly. It is not
a data-integrity problem.

---

## 7. What I could not verify

- **No external oracle.** Neither the dataset nor the repo contains Riot match-API
  participant data (`MANIFEST.json` has no per-participant gold; no outcomes
  manifest exists on disk). I could not check a single hero's final `gold_total`
  against ground truth. Every liveness argument above is internal consistency +
  physics + the control champion. A systematic offset shared by all ten heroes
  would be invisible to all of it — though it would also cancel exactly in a
  gold *diff*.
- **The fog split is a proxy**: opponent > 2000 units from every ally, with no
  ward data. Some "fog" frames were really warded. This biases *against* the
  conclusion — true fog frames are a subset of what I measured, and accrual there
  is still 5.16 g/s.
- **The actual recorder source is not in this repo** (§1). I reasoned from the
  checked-in `pipeline.py` plus the on-disk data, and confirmed the divergence is
  additive, but I did not read the code that ran.
- **Offsets were not re-validated against a live client.** That needs the Windows
  side and a running game.
- I did not check whether `gold_earned` matches Riot's `goldEarned` *definition*
  (it starts at 500 = League starting gold, which is the right shape).

---

## 8. What this changes

1. **`docs/archive/DESIGN_DECISIONS.md` §1's visibility objection is dead.** It was
   describing a schema deleted three months before the doc was written. The
   correction already in that file stands; the coverage figure should read
   **99.880%**, and the sharper statement is that opponent coverage *equals own
   coverage exactly*.
2. **Gold-diff is not blocked by data provenance.** The branch runs on 146/146
   matches with no errors and produces a dense signed signal, and the input it
   depends on is a live read. Whether to switch is a reward-design question, not
   a data question.
3. **For BRONZE, don't.** Wave management and CS are solo-gold concepts; the
   enemy term buys nothing for that milestone, and deferring it costs nothing now
   that we know the data will still be there.
4. **`gold` (current) should be deleted or fixed, not carried.** It is 100%
   garbage across every hero-game audited, `backfill_visible_heroes.py:106` is
   actively writing it to disk, and it sits one key away from the good field in
   the same dict. The fix in `scan_offsets.py` is to require `gold_current` to
   *change* between snapshots, the way `gold_earned` already must.
5. **`inventory[].uc` is garbage too** (no consumers). `lf` and `id` are fine.
6. **Any future use of per-hero `screen` must exclude the 59 backfilled matches**,
   where it is `None` by construction.

---

## Reproducing

Everything here is derived from `labels.json` / `raw_mem.json` under
`/srv/nfs/datasets/lol_replays_16_9_772` with `PYTHONPATH=src` and
`/home/dani/miniconda3/envs/ml/bin/python`. Analysis scripts were scratch and are
not committed; the tests that matter are, in order of decisiveness:

1. On frames where own `gold_total` rises by ≤2 g, count how many of the other
   nine heroes rise on the same frame, split by `screen is None` and by a fog
   proxy. Expect ≈0.98 in every cell.
2. Compare enemy per-frame position deltas on- versus off-screen. Expect the same
   distribution.
3. Locate the longest constant-`gold_total` run for the focus champion and the
   opponent. Expect identical frame indices.
4. `compute_episode_reward(labels, True, RewardConfig(use_solo_gold=False))` over
   the corpus. Expect 146/146 clean.
