# What the non-farming baseline policies actually do (2026-09-24)

Written from HEAD `a4a1bd0`. The working tree also had an uncommitted metrics-only edit to `trainer.py`, `reward_*` logging, which is not on the rollout path. Uncommitted.

Scratch is `lanerl_jax/runs/nofarm/` (gitignored). `/tmp/nofarm` is a symlink to it.
- Scripts: `rollout.py`, `analyse.py`, `probe.py`, `stuck.py`, `chase.py`, `rollout_dbg.py`, `table.py`.
- Outputs: `roll_<tag>.npz`, `an_<tag>.json`, `probe_<tag>.json`, `stuck.json`, `table.md`.
- Slurm job: 1335 (log `job-1335.out`).

## 0. Headline

1. **No mechanics bug explains the non-farming.**
   - The attack path works in every checkpoint that uses it:
     - An ATTACK order on a hostile unit sets the target 88-99% of the time. Almost every miss is an order issued while the champion is dead.
     - A ready, in-range ATTACK on an enemy minion starts a swing on that decision 90-100% of the time (n = 37-3,189 per checkpoint).
   - Spells, deaths and credits all behave.
   - No target slot pointed at nothing: 0 invalid-slot picks in 5.76 M decisions.
2. **The failures spread across several distinct behaviours, which points to learning rather than one bug.** The non-farming checkpoints split into four behaviours (§2):
   - reach lane and never attack minions;
   - trade with the champion;
   - one side never leaves the fountain;
   - farm with E while every auto-attack is cancelled.
3. **The failing and farming policies form one continuum on the same ingredients.**
   - Share of the target pointer on enemy minions: 2-5% → 52%.
   - Auto-attacks landed on minions: 0 → 63 per episode.
   - What separates them is whether the pointer learned to pick enemy minions and stay on them.
4. **One thing every checkpoint shares is structural.** A large share of the target-pointer mass sits on the champion's **own turret** (15-85%). An attack on an ally is a legal no-op in this sim, and the server's only disengage (`ENT-01`).
   - The non-farmers use `attack_move` on their own turret as a do-nothing / hold action: 56-85% of the pointer, and 13-45% of all decisions are `attack_move`.
5. **Two structural issues that are bugs or near-bugs.** Details in §4.
   - **(R1) Screen-head credit assignment.**
     - `attack_move` decodes to ATTACK in **100.0%** of samples in every checkpoint (5.76 M decisions), because the own turrets are always visible, so a valid slot always exists.
     - Yet `ppo._head_usage` counts the screen heads for `attack_move`, so the movement heads take policy gradient from 22-75% of decisions on which they had no effect.
     - The movement heads sit at 88-96% of max entropy in every checkpoint, including diag1b after 2,000 updates.
   - **(R2) Side asymmetry of the reflected lane frame at the blue fountain.**
     - The same canonical click is off-grid (x < 0) from the blue fountain.
     - The server's raw two-point walk (`PATH-008`) then parks blue against the map edge.
     - In `c13s1`, blue spends **62%** of the game at the fountain against red's 6%. Four of the 16 blue champions never leave.
6. **The CS@10 readout at update 280 understates the final policy.**
   - A fresh full-episode rollout of the same parameters scores higher in 7 of 9 checkpoints. `c13s2` and `a36s2` score lower (0.0 vs 0.20, 0.84 vs 1.27). Six of them:

     | Checkpoint | Training readout | This rollout |
     |---|---|---|
     | `c13s1` | 0.02 | 3.28 |
     | `c13s0a` | 1.86 | 9.88 |
     | `a36s1` | 3.61 | 7.62 |
     | `2d0fdb2` | 11.85 | 19.5 |
     | `c13s0b` | 18.69 | 22.0 |
     | base1 | 0.34 | 0.56 |

   - `c13s0a` is **not** a non-farmer: 28 of 32 champions reach ≥ 5 CS.
   - Likely cause: the readout averages episodes played over the previous ~141 updates, one episode's length. This is unverified.
   - Consequence: RL-007's discovery-rate tally should be re-read on rollouts of the final parameters.

## 1. Method

**Rollout** (`rollout.py`):
- The trainer's own `train.rollout` with `rollout_steps=1`: the same `_obs`, sampling, `orders_from` decoder and `env_apply`/`env_advance`.
- `SimConfig.training` with the `map1_garen_r35_o50_v2` route table, mirror play, stochastic sampling.
- 16 envs x 18,000 decisions (one 600 s episode from t = 0). That is 32 champion-episodes per checkpoint, on the desktop 5080.
- Seed 123 for every checkpoint (common random numbers).
- A second pass rebuilds the observation of the pre-step state. This recovers the target unit, order kind, per-head logits/entropy and cast legality, which the trainer does not log.

**Clock detail:** the float clock reaches 599,963 ms at decision 18,000, one decision short of `done`. CS@10 is therefore read from the last state, not from the trainer's `cs_done`.

**Replication:** diag1b scores **29.5 CS** (audit §2: 31.4 on 8 envs, seed 123). The action mix and deaths also match the audit (attack_move 75.5% vs 75.4%, deaths 3.0 vs 3.1).

**Definitions:**
- *in lane*: `lane_corridor_distance == 0`.
- *fountain*: within 1,100 u of the spawn point.
- *lane fraction*: s/L in the champion's own frame. 0 is its own outer turret and 1 the enemy's. Negative values are behind its own turret.
- *landed swing*: the windup completes and the target loses ≥ 0.5 x AA damage, or dies.
- *lethal-able*: the target's HP before landing is ≤ the champion's AA damage D(L).
- *effective press*: the spell's `can_cast` was true in the state the press was made in. For R the enemy champion must also be the target and within 400 u.
- *dmg to enemy champion*: the enemy champion's HP drop on ticks where its `hit_flag_by` is the champion (exact last-hitter attribution).
- *E damage to minions*: enemy-minion HP drops within 400 u while E is spinning. **This is an approximation**: it also counts minion-on-minion damage in the radius.

## 2. Per-checkpoint behaviour

Per champion-episode, averaged over 32. The first five columns are the non-farmers by training readout. `c13s0a` and `a36s1` are partly farming. The last three are farming controls. Full table with every measure: `table.md`.

| measure | base1 | c13s1 | c13s2 | a36s0 | a36s2 | c13s0a | a36s1 | 2d0fdb2 | c13s0b | diag1b |
|---|---|---|---|---|---|---|---|---|---|---|
| training readout CS@10 | 0.34 | 0.02 | 0.20 | 0.18 | 1.27 | 1.86 | 3.61 | 11.85 | 18.69 | ~29 |
| **CS@10, this rollout (mean / median)** | **0.56 / 0** | **3.28 / 3** | **0.0 / 0** | **0.59 / 0** | **0.84 / 0** | 9.88 / 9 | 7.62 / 8 | 19.5 / 21 | 22.0 / 23.5 | 29.5 / 29 |
| champions ≥ 5 CS (of 32) | 1 | 10 | 0 | 1 | 1 | 28 | 24 | 32 | 32 | 32 |
| move % / attack_move % | 50 / 45 | 86 / 7.5 | 53 / 41 | 82 / 14 | 68 / 26 | 29 / 67 | 41 / 56 | 32 / 61 | 32 / 64 | 22 / 76 |
| Q / W / E / R / recall % | 1.1/.2/1.6/1.1/.5 | .1/0/4.1/1.1/.1 | 1.4/1.0/1.8/1.1/.3 | .1/0/2.1/.2/0 | .7/1.2/1.9/1.2/.4 | .6/.1/2.2/.7/.1 | .1/.2/2.2/.6/.1 | 1.7/.4/2.2/1.4/.2 | 1.4/.2/1.8/.9/.2 | .5/0/1.0/.5/0 |
| entropy / max: button, target | .44, .33 | .26, .52 | .47, .66 | .29, .40 | .42, .44 | .39, .52 | .41, .45 | .44, .46 | .41, .46 | .31, .36 |
| entropy / max: screen x, y | .92, .92 | .92, .88 | .93, .89 | .91, .88 | .95, .95 | .93, .89 | .93, .94 | .95, .95 | .96, .94 | .95, .96 |
| pointer → own turret | **.85** | .36 | .56 | **.75** | .68 | .16 | .41 | .34 | .23 | .15 |
| pointer → enemy minion | .03 | .32 | .05 | .05 | .02 | .21 | .25 | .39 | .46 | .53 |
| pointer → ally minion / enemy champ / enemy turret | .06/.02/.04 | .04/.20/.08 | .36/.04/.00 | .04/.00/**.17** | .30/0/0 | .63/0/0 | .34/0/0 | .08/.11/.09 | .17/.13/.02 | .32/0/0 |
| occupancy: in lane / walking / fountain / dead | .56/.38/.03/.03 | .39/.20/**.34**/.07 | .42/.49/.04/.05 | .61/.30/.07/.01 | .63/.31/.03/.04 | .65/.23/.09/.04 | .49/.40/.05/.06 | .49/.34/.07/.11 | .47/.30/.14/.10 | .56/.27/.07/.09 |
| lane fraction p10 / p50 / p90 | -.91/.44/.59 | -1.54/**-.67**/.73 | -1.02/.43/.59 | -1.27/.33/.55 | -.95/.43/.58 | -1.33/.35/.66 | -1.15/.18/.76 | -1.29/.02/.73 | -1.48/-.03/.74 | -1.25/.15/.73 |
| nearest enemy minion (alive) p10/p50 | 810/1208 | 332/4760 | 923/1299 | 702/1383 | 773/1180 | 142/957 | 147/1789 | 108/1870 | 98/2297 | 96/1199 |
| share of alive time ≤ 300 u from an enemy minion | .017 | .092 | **.000** | .016 | .011 | .189 | .189 | .263 | .291 | .330 |
| AA swings started on enemy minions | **0** | 40 | **0** | **0** | 2.7 | 88 | 34 | 142 | 190 | 214 |
| … landed / lethal-able / credited last hits | 0/0/0 | 4.2/1.0/0.7 | 0/0/0 | 0/0/0 | .25/.16/.09 | 2.3/1.5/0.9 | 2.5/1.3/0.9 | 23/13/9.8 | 36/15/12 | 63/20/18 |
| HP/D of the target at swing start (p50) | - | 2.2 | - | - | 0.8 | 1.6 | 1.5 | 1.3 | 2.0 | 2.2 |
| AA swings started on the enemy champion | 0.3 | 29 | **51** | 0 | 0.2 | 0.4 | 0 | 70 | 61 | 0.2 |
| damage to minions: AA / E (approx) | 0 / 524 | 329 / 1840 | 0 / 2 | 0 / 350 | 13 / 405 | 120 / **5288** | 146 / **4656** | 1410 / 5731 | 2711 / 5674 | 5099 / 6938 |
| damage to enemy champion (hit-flag) | 1720 | 1115 | 2652 | 810 | 1874 | 859 | 153 | 1437 | 1448 | 965 |
| deaths (by minion / turret / champion) | 1.0 (.5/.06/.47) | 2.4 (.56/**.91**/.91) | 1.5 (0/0/**1.5**) | **0.3** (.12/.03/.16) | 1.0 (.25/0/.75) | 1.2 (.97/0/.19) | 2.1 (1.4/.56/.12) | 3.5 (1.3/.9/1.2) | 3.3 (1.4/.75/1.1) | 3.0 (1.9/.28/.81) |
| kills | .47 | .88 | 1.5 | .16 | .75 | .19 | .12 | 1.3 | 1.2 | .84 |
| recall presses / starts / longest channel | 82/72/0.9 s | 9/8/0.7 s | 53/47/0.9 s | 7/7/0.6 s | 70/63/0.7 s | 14/13/1.1 s | 12/11/1.3 s | 41/32/1.1 s | 28/23/1.3 s | 4/3.5/1.7 s |
| E presses / effective / spins started | 283/57/36 | 739/74/39 | 316/57/36 | 383/66/38 | 345/61/37 | 390/64/38 | 392/62/37 | 390/59/36 | 333/56/34 | 182/40/28 |
| Q / W / R presses (effective) | 196(27)/37(9)/201(0) | 18(7)/7(2)/195(.4) | 247(28)/174(14)/198(.7) | 8(5)/7(3)/42(0) | 118(24)/208(15)/219(0) | 107(22)/9(4)/117(0) | 21(7)/35(7)/104(0) | 297(27)/70(10)/243(.6) | 248(26)/29(7)/169(.8) | 98(23)/3(2)/82(0) |
| level / gold at 10 min | 7.2/1134 | 5.8/1283 | 8.4/1410 | 6.7/1019 | 7.8/1221 | 7.4/1143 | 6.1/1108 | 8.3/1740 | 8.2/1729 | 8.0/1722 |

**How behaviour changes over the episode:**
- The action mix barely moves after minute 1 in any checkpoint. For example, base1's attack_move is 58% in minute 0 and 42-44% in minutes 2-9.
- E use roughly doubles from minute 0 to minute 9 everywhere (1.0-2.8% → 1.6-4.4%).
- Minute-0 in-lane is 0 for all checkpoints, because the walk from the fountain takes about 39 s.
- By minute 1:
  - base1, `c13s2`, `a36s2` and `c13s1` are already 31-83% in lane.
  - `a36s0`, `c13s0a`, `a36s1`, `2d0fdb2` and `c13s0b` are at 0-2%: they walk in later, via attack-chases.
- CS by minute is flat near 0 for all five non-farmers, e.g. base1 reads 0.28 at minute 8.

### Classification

- **base1 (0.56): reaches lane, never attacks a minion. The attack button is a no-op on its own turret, and it pokes the champion with E.**
  - In lane from minute 1 (51% → 75%), holding at lane fraction 0.3-0.5, which is its own half.
  - It is a median 1,208 u from the nearest enemy minion and within 300 u only 1.7% of the time.
  - 85% of the pointer mass is on its own turret, so the 45% `attack_move` decisions do nothing.
  - **Zero swings started on minions** in 32 episodes.
  - Its damage is E. It spins E off cooldown (36 spins; 14.6% of alive time spinning), dealing 1,720 to the champion and ~520 to minions.
  - Deaths 1.0: half to minions, half to the champion. First death at a median 423 s.
- **c13s2 (0.0): trades with the champion instead of farming.**
  - 51 AA swings started on the enemy champion per episode and **0 on minions**.
  - Never within 300 u of an enemy minion (0.0%).
  - Highest damage to the champion (2,652). All 1.5 deaths are to the champion and it gets 1.5 kills, a mirror slugfest.
  - Ally minions take 36% of the pointer (a second no-op target). W is pressed 174 times, with 14 effective.
- **a36s0 (0.59): passive; the slowest to reach lane; never attacks minions; almost never dies.**
  - 0% in lane in minute 1, then 77-90% from minute 3, holding back at lane fraction ~0.35.
  - 75% of the pointer is on its own turret and 17% on the **enemy turret**.
  - The enemy-turret orders produce 3,187 samples of the champion standing still while chasing an enemy turret a median 4 km away. That is the straight-line chase (`PATH-009`), 0.6% of alive time.
  - Lowest deaths (0.31) and lowest damage dealt.
- **a36s2 (0.84): same mode as base1.**
  - 68% of the pointer on its own turret and 30% on ally minions.
  - 2.7 minion swings per episode, of which 86 of 91 across the run were cancelled by its own next order.
  - Pokes the champion (1,874 damage).
- **c13s1 (3.28): side-split.**
  - **Blue:** 62% of the game at the fountain; 4 of 16 blue champions at 100%, and 12 of 16 at ≥ 44%.
  - Red: 6% at the fountain.
  - It navigates by `move` (86% of decisions, the lowest button entropy of the set). Its click distribution points "wall-ward", which is off-grid from the blue fountain (see R2).
  - The red champions walk in, swing at the champion (29 per episode) and at minions (40 per episode, 4 landed), and die to the **turret** (0.91 per episode, the most of the non-farmers). They are diving.
  - CS: red 4.7, blue 1.9.
- **c13s0a (9.88; readout 1.86): a farmer that farms with E, not auto-attacks.**
  - 28/32 champions ≥ 5 CS. 88 minion swings started per episode, but only **2.3 landed**.
  - 97% were cancelled by its own next order; 74% of those cancels were an `attack_move` onto an ally minion.
  - Only 0.9 of its 9.9 CS are credited AA last hits. The rest come from E (≈5,300 damage to minions).
- **a36s1 (7.62): the same as c13s0a.** 34 swings, 2.5 landed, 0.9 AA last hits, E ≈ 4,700 damage to minions. It dies to minions (1.4 per episode).

**The farming controls** (`2d0fdb2`, `c13s0b`, diag1b) differ from the E-farmers on exactly one axis:
- They land 23-63 swings, 10-18 of them credited last hits.
- Their cancel rate is still high (72-84%). This matches the audit's 70% for diag1b, so the cancel itself is not specific to failing policies.
- The pointer is 39-53% on enemy minions.
- `2d0fdb2` and `c13s0b` also trade with the champion (61-70 swings) and die ~3.4 times.

## 3. Bug or learning

Evidence that it is learning (a local optimum per seed), not a broken mechanic:

- **Every mechanic the farmers rely on works identically for the non-farmers whenever they exercise it.**
  - Ready, in-range ATTACK → swing: 0.91 (`c13s1`) and 1.00 (`a36s2`), vs 0.96-0.98 for the farmers.
  - A hostile ATTACK sets the target: 0.91-0.99. The misses are orders issued while dead: 315/325, 1,763/1,806, 952/1,069 and 1,196/1,232.
  - The few remaining misses (≤ 30 per checkpoint) are targets that became invalid on the same tick.
- **All checkpoints run the same code, sim and seed.** The non-farmers do not share a single failure signature:
  - never-attack plus own-turret hold (base1, `a36s2`);
  - champion trading (`c13s2`);
  - passive and far (`a36s0`);
  - blue fountain trap (`c13s1`);
  - E-farming with AA cancellation (`c13s0a`, `a36s1`).
- **CS rises smoothly with the learned quantities:** pointer share on enemy minions, time spent within 300 u of minions, and landed swings. There is no cliff that would mark a switch.
- **The diag1b policy scores 27-33 CS frozen on every one of these commits (`RL-007`).** The sim can be farmed by a policy that knows how.

What *is* shared, and is structural rather than seed-specific (§4):
- the own-turret no-op dominating the pointer;
- untrained movement heads;
- recall and R as dead buttons.

These make "never discover last-hitting" an easy basin to settle in. They do not stop a policy that finds it.

## 4. Red flags

**R1. The screen heads are credited for `attack_move` decisions they never influence. (Credit-assignment bug.)**
- `train/ppo.py:135-149` `_head_usage` sets `uses_screen[attack_move] = True`. The reasoning in the comment: the screen point reaches the wire "when the chosen slot holds [no] visible unit".
- Measured: `attack_move` decoded to a unit ATTACK in **100.0%** of cases in all 10 checkpoints (0 MOVE, 0 NOOP; `attack_move_decode` in `an_*.json`).
- The reason: the two nearest turrets are always slotted, and the own turret is always visible (`obs/builder.py:161`), so a valid slot always exists and the masked pointer always hits one.
- So `factored_log_prob` adds the screen-x/y log-probs, and `factored_entropy` their entropy weighted by p(attack_move), on 22-75% of decisions where they had no effect.
- The screen heads therefore take PPO gradient that is pure noise on those decisions.
- This is consistent with screen entropy at 88-96% of max in every checkpoint, diag1b included (audit §2: "the movement heads are essentially untrained").
- Fix direction: gate the screen term on `~has_target` per sample. That needs `slot_unit` or a has-target flag stored in the Transition, not a per-button constant.

**R2. The blue fountain is a trap for a click-navigating policy. (Side asymmetry.)**
- The shared policy acts in a reflected lane frame (`obs/frame.py`, `actions.orders_from`): `(s,n)→(L-s,n)`, with the same world normal for both sides. The lane maps exactly, but the two bases do not: Map1 is rotationally, not reflectively, symmetric.
- From the blue fountain, a canonical click toward n > 0 (the policy's bias in `c13s1`) is often off-grid.
  - Debug rollout `dbg_c13s1.npz`: mean click (-375, +1,097); 65% of clicks at x < 0.
  - 2,463 of 3,090 Moves routed `SERVER_NULL` (status 8), i.e. the raw two-point walk to the unprojected click (`PATH-008`, `local_pathing.py:80-87`).
  - The champion slides against the edge at x ≈ -100…-170 for the whole game.
- **Across all ten checkpoints, blue spends more time at the fountain than red in 9/10** (e.g. diag1b 0.098 vs 0.040; `c13s0b` 0.183 vs 0.089; `probe_*.json`). In `c13s1` it is 0.62 vs 0.06.
- `PATH-008` states the server walks the same raw line, so this may be faithful. If so, it is an environment asymmetry the mirrored policy has to learn around, and it goes unseen in side-averaged metrics.
- Worth a direct server check: does a blue champion at the fountain, clicked off-grid, really end up at x < 0?

**R3. "Attack own turret" is a free, always-available no-op, and it dominates the non-farmers.**
- The own turret takes 56-85% of the pointer in four of the five non-farmers, against 15-34% in the farmers.
- In the sim an ATTACK on an ally is held with no swing, no chase and no hold (`sim/step.py:823-838`, `ENT-01`, server-faithful). It also releases a sticky hostile target: a Move does *not* release one (`orders.py` target commit; `step.py` 3b re-engages).
- So `attack_move` is the policy's only disengage and its cheapest "do nothing".
- Not a bug, but it means `attack_move %` is not "attacking". The training dashboard should report attack orders by target class.

**R4. Recall has never completed, in any checkpoint.**
- 3.5-72 recall starts per episode. The longest windup + channel observed in 5.76 M decisions is **1.73 s**, against 8.5 s needed. Recall completions: **0.00** everywhere.
- It is always ended by the next `move` (2,272 of 2,308 endings in base1).
- A recall press also stops the champion (it clears the path; `orders.py` `stop_for_recall`), so recall is in effect a *stop* button.
- A stochastic policy with p(move) ≥ 0.2 cannot hold still for 255 decisions, so recall is unreachable by exploration.
- Not a bug, but a dead action.

**R5. R and Q/W are pressed far more than they can act.**
- R: 42-243 presses per episode, ≤ 0.8 effective. It is rank 0 until level 6 (end rank ≈ 0.7-1.2), and only an enemy-champion target in range is legal (`orders.py:233-248`).
- Q: 5-37% effective. W: 4-45% effective.
- A high-use button that almost never has an effect is the pattern the brief asked about. Here the gating is correct (the observation shows the cooldown and rank as 1.0 = unavailable). The policy has just not learned it, because the entropy bonus pays for the spread.

**R6. The straight-line attack chase (`PATH-009`) sometimes parks champions.**
- Stuck samples (alive, outside the fountain, < 10 u moved in 1 s, ATTACK_TO a hostile > 1,000 u away): ≤ 0.8% of alive time everywhere.
- The largest case is `a36s0`: 3,187 samples chasing an enemy turret a median 4 km away.
- Controlled check (`chase.py`, blue at spawn, red champion pinned):
  - A chase to a visible target up the lane arrives (154 u after 30 s), like a routed Move (56 u).
  - An invisible target is dropped at once and the champion does not move.
- Minor, already booked.

**R7. The training CS readout lags the parameters it is attributed to.**
- See §0 item 6. The fresh rollout of the final parameters scored higher than the training readout in 7 of 9 checkpoints; `c13s2` (0.0 vs 0.20) and `a36s2` (0.84 vs 1.27) scored lower.
- The readout averages 62-90 champion-episodes that finished in the last ~20 updates. Each of those episodes was played across the previous ~141 updates.
- Not verified beyond the comparison above: one seed, 32 champion-episodes per checkpoint.
- **What it changes:** `c13s0a` ("1.86", listed non-farming) farms 9.9 CS with 28/32 champions ≥ 5, and `c13s1` is side-split, not dead.

**Checked and clean:**
- No pointer onto an empty slot (0 in 5.76 M).
- No attack order that failed to start a swing while ready and in range, beyond the 0-10% explained by same-tick target changes.
- No champion frozen in lane while issuing Moves beyond ~1-2% of alive time. That fraction is attack-ally holds and far chases, not frozen orders.
- Deaths are always attributed. The killer's `hit_flag_by` was set in every death (0 unknown).

## 5. Caveats

- One seed (123) and 32 champion-episodes per checkpoint. Class assignments rest on large, qualitative differences (0 vs 40 vs 214 swings), not on small ones.
- Damage-to-minions by E is an upper-bound approximation (§1). AA damage and champion damage are exact to the tick.
- Rolled out at HEAD, not at each checkpoint's training commit. `RL-007` shows the mechanics are unchanged across these commits for a frozen policy (diag1b 27-33 on all of them).
- The GPU rollout is not bit-reproducible, because the default XLA ops are nondeterministic.
