#!/usr/bin/env python
"""Play a trained JAX policy against the scripted C# bot, INSIDE the C# server.

WHY THIS IS THE ONLY EVAL THAT SETTLES ANYTHING
-----------------------------------------------
Every RL number this project has comes from mirror self-play inside the JAX
port. `cs@10min` is an absolute metric and is why RL-002 is readable at all, but
"beats the C# bot's 33 CS" has never been tested by *playing the bot*: it is a
comparison of two numbers measured in two different engines, on two different
days, by two different harnesses. Gate 1 measures one tick. Gate 2 measures
divergence. ACCEPT-001 measures a scripted heuristic in both engines. None of
them puts the trained network in the real server.

This does, and so it tests the port, the observation builder and the policy
together, which is the only combination that matters at deployment.

HOW THE OBSERVATION IS BUILT, AND THE ONE PLACE IT IS NOT EXACT
---------------------------------------------------------------
The rule is the one the spike's `eval_vs_bot.py` states and this project has
violated twice: an eval that builds its observation differently from training
measures the wrong thing. So the observation here is built by the TRAINING code
-- `obs.builder.build_observation` on a `LaneState` -- and the only new code is
the reconstruction of that `LaneState` from the server's control-channel frame.

That reconstruction is tractable because `build_observation` reads exactly
seventeen state fields (`grep -o 'state\\.\\w*'`): x, y, team, kind, alive, hp,
max_hp, model, level, gold, cs, t_ms, spell_level, spell_cooldown,
recall_channel_ms, buffs, observed_enemy_cast_ms. The control wire
(`LanerlControl.BuildObservation`) already publishes all but the last two:
position, hp/mhp, team, kind, minion subtype (`mt`), and for champions
gold/xp/lvl/cs, spell ranks (`sl`), per-slot cooldowns (`cd0..cd3`) and the
recall channel flag (`rc`).

**The deviation, stated up front.** `buffs` and `observed_enemy_cast_ms` are
not on the wire, so they stay at their empty-state values. Consequences, both
bounded and both reported by this script:

* `buffs` feeds `has_w_passive` and the Q/E `cast_locked` flags. **E IS
  recovered from the wire** (`OBS-01`, 2026-09-23): `E.cs` swaps slot 2 for
  `GarenECancel` with a 1 s cooldown for the length of the spin, so `cd2`
  reads ~1000 ms falling to 0 during the spin and then jumps to the full
  rank cooldown when the spin ends or is cancelled. A rising edge of `cd2`
  to <= 1.1 s is therefore a spin START and a rising edge above it a spin
  END, and `StateRebuilder` sets `buffs.e.active`/`elapsed_s` from that, so
  the policy sees E exactly as in training: locked for the first
  `E_CANCEL_MIN_S` of the spin, then AVAILABLE, because a press from 1.0 s on
  is a CANCEL on both sides (`STRUCT-001` made the training observation say
  so; it used to report E locked for the whole spin while the sim accepted
  the cancel). Before `OBS-01` the reconstructed obs showed E READY from the
  START of every spin. Q is NOT recoverable: `Q.cs` sets the
  cooldown to 0 for the window, so `cd0` reads 0 before, during and after
  the cast and only the window's END is visible (`SPELL-008`, `OBS-02`);
  the obs shows Q ready during its own window here and locked in
  training. `--report-casts` counts every cast the policy issues.
* `observed_enemy_cast_ms` is witnessed-event memory that training accumulates
  over an episode. Left at the -1 sentinel it renders as the saturated
  "long ago" value, which is what a never-seen cast is supposed to look like.

Fixing the first properly means emitting buffs on the control wire the way
`LanerlStateDump.BuffDescriptor` already does for the parity dump. That is the
right fix and is deliberately NOT done here: this script is read-only against
the vendored server, and a rebuild is a separate change with its own
verification.

ONE SERVER PROCESS PER EPISODE
------------------------------
Inherited from the spike script, and it is a measurement decision. The server's
in-process episode reset strips the champion's rune and mastery page: max hp
672 -> 616, attack damage 78.14 -> 57.88. A run of N episodes in one process is
one game by a runed champion and N-1 by a weaker one, which is how "BC = 37.3
CS" got quoted from three games that were not the same game. `VecLaneEnv` is
therefore constructed fresh per batch with `auto_restart=False`, and the
starting stats of every episode are compared before any mean is printed.

    python tools/rl_eval_vs_server.py \
        --checkpoint lanerl_jax/runs/train/<run>/ckpt_latest.msgpack \
        --episodes 4 --parallel 4

`--checkpoint random` runs the untrained network, which is the reference every
absolute number needs.

The DRIVER (wire frame -> `LaneState`, checkpoint loading, the jitted act,
the wire encoding) now lives in `lanerl_jax/parity/policy_driver.py` and is
imported from there (`PARITY-001`): the policy-divergence gate
(`python -m lanerl_jax.parity.policy_divergence`) drives the server with the
same code, so there is one copy. This script keeps the episode loop, the
counters, `--replay` and `--selftest`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from lanerl_jax.obs.builder import build_observation                  # noqa: E402
from lanerl_jax.parity.lanerl_lane import LanerlLane                  # noqa: E402
from lanerl_jax.obs.frame import make_lane_frame                      # noqa: E402
from lanerl_jax.sim.init import TOP_OUTER_TURRET, lane_params         # noqa: E402
from lanerl_jax.sim.orders import OrderKind                           # noqa: E402
from lanerl_jax.sim.state import MI_SLICE, TU_SLICE, Kind, Team       # noqa: E402
from lanerl_jax.train.trainer import BLUE_NEXUS                       # noqa: E402
# The driver lives in ONE place (`PARITY-001`): the policy-divergence gate
# drives the server with exactly this code. Re-exported here so the eval's
# names (and anything that imported them from this script) keep working.
from lanerl_jax.parity.policy_driver import (                         # noqa: E402,F401
    E_CANCEL_ARMED_MAX_MS, WIRE_MT_TO_SIM, WIRE_TEAM, _MINION_KINDS,
    CreationRankMap, StateRebuilder, _turret_slot_map, load_params,
    make_driver, order_to_wire, pending_rank_up)


class HeterogeneousEpisodes(RuntimeError):
    """Episodes that must not be averaged, because they were not the same game."""


def start_stats(frame: dict | None) -> dict:
    """The blue champion's stats on an episode's FIRST frame.

    The fingerprint of the rune and mastery page, and therefore of whether this
    episode ran in a freshly launched server or after an in-process reset that
    stripped it. Absent keys are omitted rather than defaulted: a guessed stat
    is how this project ended up with three different wrong attack-damage
    constants.
    """
    for u in (frame or {}).get("u", []):
        if u.get("k") == "Champion" and u.get("tm") == 100:
            return {k: float(u[k]) for k in ("mhp", "ad", "ar", "mr", "lvl")
                    if u.get(k) is not None}
    return {}


def check_homogeneous(rows: list[dict]) -> None:
    """Refuse to average episodes whose champion did not start the same."""
    seen: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        seen.setdefault(json.dumps(r.get("start_stats") or {}, sort_keys=True),
                        []).append(i)
    if len(seen) > 1:
        detail = "; ".join(f"episodes {v}: {k}" for k, v in sorted(seen.items()))
        raise HeterogeneousEpisodes(
            "these episodes did not start from the same champion stats, so "
            "their mean measures nothing: " + detail + ". The known cause is "
            "the server's in-process episode reset stripping the rune page "
            "(mhp 672 -> 616, ad 78.14 -> 57.88); every episode is supposed to "
            "get its own server process.")


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------

def replay_driver(path: Path, counts: dict):
    """An OPEN-LOOP driver: send a recorded sim order stream, ignore the server.

    The isolation this exists for. The same checkpoint scores 53 CS in the JAX
    sim and 0 in the C# server, and the two runs also show different action
    mixes (attack 3.4% in the sim against 8.9% here), so the closed-loop
    comparison cannot separate "the engines behave differently" from "my
    reconstructed observation makes the policy behave differently". Feeding the
    sim's own orders to the server removes the observation from the loop: if the
    server champion STILL never reaches lane, it is the engine.

    MOVE carries world coordinates and CAST carries a slot, so both replay
    exactly. ATTACK carries a sim UNIT INDEX, which has no meaning on the
    server. It is mapped through CREATION RANK when the line carries the
    target's ``target_spawn_seq`` (the sim's `LaneState.spawn_seq` of the
    targeted unit at record time): `spawn_seq` -> champion by team, turret by
    position, or the k-th LaneMinion NetId this server has created, where
    ``k = spawn_seq - first_minion_seq`` (`policy_driver.CreationRankMap`;
    both engines create minions in the same order, red before blue within a
    wave, `RESET-004`). A line WITHOUT ``target_spawn_seq`` (every recording
    made before 2026-09-23), or whose rank names a minion this server has not
    created, is dropped and counted -- never guessed at. Those were 3.4% of
    the sim's orders and the CS-producing ones (`SPELL-001`).

    JSONL schema, one decision per line::

        {"kind": <OrderKind>, "x": float, "y": float,
         "target": <sim slot>, "target_spawn_seq": <int, optional>}
    """
    orders = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ranks = CreationRankMap()
    i = 0

    def drive(frame: dict) -> dict:
        nonlocal i
        ranks.observe(frame)
        if i >= len(orders):
            counts["noop"] = counts.get("noop", 0) + 1
            return {"t": "noop"}
        o = orders[i]
        i += 1
        kind = int(o["kind"])
        netid = np.zeros(0, np.int64)
        tgt = -1
        if kind == OrderKind.ATTACK:
            seq = o.get("target_spawn_seq")
            nid = ranks.netid_of_spawn_seq(int(seq)) if seq is not None else None
            if nid:
                netid, tgt = np.asarray([nid], np.int64), 0
                counts["attack_mapped"] = counts.get("attack_mapped", 0) + 1
            else:
                counts["attack_dropped"] = counts.get("attack_dropped", 0) + 1
        wire = order_to_wire(kind, o["x"], o["y"], tgt, netid)
        counts[wire["t"]] = counts.get(wire["t"], 0) + 1
        return wire

    drive.counts = counts
    drive.rebuilder = None
    drive.ranks = ranks
    return drive


def play_batch(policy, params, *, n: int, max_game_ms: int, seed: int,
               deterministic: bool, red: str, bot_config=None,
               log_dir: Path, server_dir=None, trace_every_s: float = 0.0,
               replay: Path | None = None) -> list[dict]:
    """`n` episodes in `n` FRESH server processes, all stepped in lockstep.

    Parallel because the servers are independent processes and the policy step
    is one small jitted call per instance; `auto_restart=False` because a
    restart mid-episode would silently hand us an episode played at BASE stats
    (the in-process reset strips the rune page), which is the exact blending
    this script exists to refuse.
    """
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
    # `ENT-05`: `LanerlHooks._autoBuyUndriven = LANERL_AUTOBUY != "0"`, and
    # `vec.py` never set it, so both server champions started with a 475
    # wallet and bought Doran's Shield and potions -- against a sim with no
    # items. `ServerLaunchSpec.environment` copies `os.environ`.
    os.environ.setdefault("LANERL_AUTOBUY", "0")

    spec = ServerLaunchSpec(
        # None lets `ServerLaunchSpec.resolved_server_dir()` pick, which is
        # bin/Release. `--server-dir` exists because bin/Trace is a SEPARATE
        # build and any env-gated diagnostic added to the vendored source is
        # absent from Release until Release is rebuilt -- silently: the variable
        # is set, the server ignores it, and the run looks normal.
        server_dir=server_dir,
        toponly=True, freerun=True, step_ticks=2,
        # 'purple' hands RED to the scripted bot; blue is driven by us. 'none'
        # leaves red undriven in the fountain -- an UNCONTESTED lane, which is
        # the only way to separate "cannot last-hit" from "is being contested".
        # Uncontested CS is NOT an upper bound (the spike measured 37.5
        # uncontested against 45.4 versus the bot: with nobody killing your
        # minions your own wave takes the kills), so it is a second reference,
        # not a ceiling.
        bot_teams="purple" if red == "bot" else "none",
        bot_config=Path(bot_config) if bot_config else None,
    )
    env = VecLaneEnv(n, spec=spec, log_dir=log_dir, step_timeout_s=180.0,
                     auto_restart=False)
    drivers = [
        replay_driver(replay, {"cast": 0, "attack": 0, "move": 0, "noop": 0,
                               "recall": 0, "level": 0})
        if replay is not None else
        make_driver(policy, params, deterministic=deterministic, seed=seed + i)
        for i in range(n)]
    rows = [{"episode": i} for i in range(n)]
    res = env.start()
    try:
        if not any(res.alive):
            raise RuntimeError("no server booted")
        for i, o in enumerate(res.obs):
            rows[i]["start_stats"] = start_stats(o)
            rows[i]["booted"] = bool(res.alive[i])
        track = [{"path": 0.0, "prev": None, "hp_lost": 0.0, "prev_hp": None,
                  "deaths": 0, "was_alive": True, "seen": set(), "died": set(),
                  "alive_prev": set(), "trace": [], "next_trace": 0.0,
                  "spins": 0, "prev_cd2": 0.0}
                 for _ in range(n)]
        # WHERE the champion was, sampled on a clock. A policy that never
        # leaves the fountain and one that farms both score 0 deaths and a
        # plausible-looking gold total; only the movement stats tell them apart,
        # which is the spike eval's own argument for recording them. Lane
        # fraction is the sim's `lane_dist` seen from the other side: 0.0 is
        # blue's outer turret, 1.0 is red's, and the waves meet at ~0.5.
        lane = LanerlLane()          # TOP_LANE_DEFAULT
        last = list(res.obs)
        while True:
            live = [i for i in range(n) if env.alive[i] and last[i] is not None
                    and int(last[i].get("t", 0)) < max_game_ms]
            if not live:
                break
            actions = [None] * n
            for i in live:
                actions[i] = {"blue": drivers[i](last[i])}
            res = env.step(actions)
            for i in range(n):
                o = res.obs[i]
                if o is None:
                    continue
                last[i] = o
                t = track[i]
                units = o.get("u", [])
                # Enemy minions alive now; anything alive last frame and gone
                # now died. Keyed by netid, and this eval sees the whole map, so
                # a disappearance IS a death rather than a fog event. This is
                # THE DENOMINATOR: CS alone cannot say whether 45 is good,
                # because the number of enemy minions that die in a game is not
                # fixed and moves with how the wave is played.
                now = {u["id"] for u in units
                       if u.get("k") in _MINION_KINDS and u.get("tm") != 100}
                t["seen"] |= now
                t["died"] |= (t["alive_prev"] - now)
                t["alive_prev"] = now
                b = next((u for u in units if u.get("k") == "Champion"
                          and u.get("tm") == 100), None)
                if b is None:
                    continue
                if t["prev"] is not None:
                    t["path"] += math.dist(t["prev"], (b["x"], b["y"]))
                t["prev"] = (b["x"], b["y"])
                hp = float(b.get("hp", 0))
                if t["prev_hp"] is not None and hp < t["prev_hp"]:
                    t["hp_lost"] += t["prev_hp"] - hp
                t["prev_hp"] = hp
                alive = hp > 0
                if t["was_alive"] and not alive:
                    t["deaths"] += 1
                t["was_alive"] = alive
                cd2 = float(b.get("cd2") or 0.0)
                if cd2 > t["prev_cd2"] + 1.0:
                    t["spins"] += 1
                t["prev_cd2"] = cd2
                if trace_every_s:
                    tsec = int(o.get("t", 0)) / 1000.0
                    if tsec >= t["next_trace"]:
                        t["next_trace"] = tsec + trace_every_s
                        t["trace"].append({
                            "t_s": round(tsec, 1),
                            # `along_of` is distance ALONG the lane in game
                            # units; divided by the polyline length it is the
                            # 0..1 fraction the gate-3 work quotes.
                            "lane": round(lane.along_of((b["x"], b["y"]))
                                          / lane.length, 3),
                            "off": round(lane.distance_to((b["x"], b["y"]))),
                            "cs": b.get("cs"), "gold": b.get("gold"),
                            "lvl": b.get("lvl"),
                            "hp": round(100.0 * hp / max(1.0, float(b.get("mhp", 1)))),
                            # Every rising edge of cd2 is one Garen E spin that
                            # ACTUALLY started -- a refused cast leaves it
                            # counting down. This is how "the policy pressed E
                            # 8,325 times" becomes "N spins landed", which is
                            # the only version of the number that can be
                            # compared across engines.
                            "spins": t["spins"],
                            "sl": b.get("sl"),
                            "cds": [b.get(f"cd{j}") for j in range(4)],
                        })
        for i in range(n):
            o, t = last[i], track[i]
            champs = {u["tm"]: u for u in (o or {}).get("u", [])
                      if u.get("k") == "Champion"}
            b, r = champs.get(100, {}), champs.get(200, {})
            died = len(t["died"])
            rows[i].update(
                t_s=int((o or {}).get("t", 0)) / 1000.0,
                blue_cs=b.get("cs"), red_cs=r.get("cs"),
                blue_gold=b.get("gold"), red_gold=r.get("gold"),
                blue_lvl=b.get("lvl"), red_lvl=r.get("lvl"),
                enemy_minions_died=died,
                conversion=(100.0 * (b.get("cs") or 0) / died) if died else None,
                deaths=t["deaths"], hp_lost=round(t["hp_lost"]),
                distance=round(t["path"]),
                actions=dict(drivers[i].counts),
                # Non-zero means the buff deviation in the module docstring was
                # LIVE for this episode; zero means it was inert.
                casts=drivers[i].counts.get("cast", 0),
                spins=t["spins"],
                minions_dropped=(drivers[i].rebuilder.dropped_minions
                                 if drivers[i].rebuilder is not None else 0),
                trace=t["trace"],
            )
    finally:
        env.close()
    return rows


def selftest() -> int:
    """Assert the wire -> `LaneState` mapping on a synthetic frame.

    The eval runs either way; this is what says it measures the right thing.
    Every failure mode here is silent in a live run -- a swapped minion subtype
    gives the policy a caster's stat row for a cannon, a swapped team makes an
    enemy minion read as an ally, and both produce a plausible CS number.
    Three of those swaps are live traps in this codebase: wire
    `MinionSpawnType` is not the sim's `MinionType` (caster and cannon are
    exchanged), wire team ids are 100/200 against compact 0/1, and turret tier
    is absent from the wire entirely.
    """
    from lanerl_jax.sim.profiles import PROFILES

    rb = StateRebuilder()
    base = rb.base
    # A cannon on the wire is mt=2; a caster is mt=3. If the mapping were the
    # identity these two would be exchanged, which is the whole point.
    frame = {"t": 12345, "u": [
        {"id": 11, "k": "Champion", "tm": 100, "x": 100, "y": 200, "hp": 500,
         "mhp": 672, "gold": 550, "xp": 120, "lvl": 3, "cs": 7, "rc": 0,
         "sl": [1, 0, 2, 0], "cd0": 1500, "cd1": 0, "cd2": 3000, "cd3": 0},
        {"id": 12, "k": "Champion", "tm": 200, "x": 900, "y": 800, "hp": 616,
         "mhp": 616, "gold": 300, "xp": 60, "lvl": 2, "cs": 3, "rc": 1,
         "sl": [0, 0, 1, 0], "cd0": 0, "cd1": 0, "cd2": 0, "cd3": 0},
        {"id": 21, "k": "LaneMinion", "tm": 100, "x": 300, "y": 300,
         "hp": 455, "mhp": 455, "mt": 0},                      # melee, blue
        {"id": 22, "k": "LaneMinion", "tm": 200, "x": 400, "y": 300,
         "hp": 100, "mhp": 290, "mt": 3},                      # CASTER, red
        {"id": 23, "k": "LaneMinion", "tm": 200, "x": 420, "y": 300,
         "hp": 700, "mhp": 700, "mt": 2},                      # CANNON, red
    ]}
    # A turret straight off the placed set, so the position match has an answer.
    ti = TU_SLICE.start + 3
    frame["u"].append({"id": 31, "k": "LaneTurret",
                       "tm": 100 if int(base.team[ti]) == Team.BLUE else 200,
                       "x": int(base.x[ti]), "y": int(base.y[ti]),
                       "hp": 1000, "mhp": 1300})

    st, netid = rb.rebuild(frame)
    fail = []

    def chk(cond, msg):
        if not cond:
            fail.append(msg)

    chk(float(st.t_ms) == 12345.0, f"t_ms {float(st.t_ms)}")
    # champions in slots 0/1 by TEAM, not by wire order
    chk(int(st.kind[0]) == Kind.CHAMPION and int(st.team[0]) == Team.BLUE,
        "blue champion is not slot 0")
    chk(int(st.kind[1]) == Kind.CHAMPION and int(st.team[1]) == Team.RED,
        "red champion is not slot 1")
    chk(int(st.level[0]) == 3 and int(st.cs[0]) == 7
        and abs(float(st.gold[0]) - 75.0) < 1e-6,     # OBS-04: 550 wallet - 475
        "blue champion scalars")
    chk(list(np.asarray(st.spell_level[0])) == [1, 0, 2, 0], "spell ranks")
    # ms on the wire, SECONDS in the state -- build_observation divides by the
    # seconds-valued cooldown tables.
    chk(abs(float(st.spell_cooldown[0, 0]) - 1.5) < 1e-6
        and abs(float(st.spell_cooldown[0, 2]) - 3.0) < 1e-6, "spell cooldowns")
    chk(float(st.recall_channel_ms[1]) > 0 and float(st.recall_channel_ms[0]) == 0,
        "recall flag")
    # E spin inference (`OBS-01`): a cd2 edge to ~1000 ms opens the spin,
    # the buff stays through cd2's fall to 0, and the edge to the full
    # cooldown closes it.
    rb2 = StateRebuilder()
    def champ(t, cd2):
        return {"t": t, "u": [{"id": 11, "k": "Champion", "tm": 100,
                "x": 0, "y": 0, "hp": 1, "mhp": 1, "sl": [0, 0, 1, 0],
                "cd0": 0, "cd1": 0, "cd2": cd2, "cd3": 0}]}
    seq = [(1000, 0, False, None), (1033, 983, True, 0.0),
           (2500, 0, True, 1.467), (2533, 0, True, 1.5),
           (4100, 8500, False, None), (4133, 8467, False, None)]
    for t, cd2, want_on, want_el in seq:
        s2, _ = rb2.rebuild(champ(t, cd2))
        on = bool(s2.buffs.e.active[0])
        chk(on == want_on, f"E spin at t={t} cd2={cd2}: on={on} want {want_on}")
        if want_on:
            chk(abs(float(s2.buffs.e.elapsed_s[0]) - want_el) < 0.01,
                f"E elapsed at t={t}: {float(s2.buffs.e.elapsed_s[0])}")

    mi = {int(netid[i]): i for i in range(rb.n_units) if netid[i]}
    for nid, want in ((21, (0, Team.BLUE)), (22, (1, Team.RED)), (23, (2, Team.RED))):
        i = mi.get(nid)
        if i is None:
            fail.append(f"minion {nid} not placed")
            continue
        chk(MI_SLICE.start <= i < MI_SLICE.stop, f"minion {nid} outside MI_SLICE")
        kind_, sub, team_ = PROFILES[int(st.model[i])]
        chk(kind_ == Kind.LANE_MINION, f"minion {nid} profile kind {kind_}")
        chk((sub, team_) == want,
            f"minion {nid} profile (subtype,team)=({sub},{team_}) want {want} "
            "-- wire MinionSpawnType is NOT the sim's MinionType")
        chk(int(st.team[i]) == want[1], f"minion {nid} team")
        chk(bool(st.alive[i]), f"minion {nid} alive")

    t = mi.get(31)
    chk(t is not None and TU_SLICE.start <= t < TU_SLICE.stop,
        "turret not matched into TU_SLICE")
    if t is not None:
        # tier and team come from init_lane, NOT from the wire: the wire has no
        # tier and PROFILES has one row per tier per team.
        chk(int(st.model[t]) == int(base.model[t]),
            "turret profile changed -- the tier from init_lane was overwritten")
        chk(abs(float(st.hp[t]) - 1000.0) < 1e-6, "turret hp not taken from wire")

    # slot stability: the same frame twice must give the same unit indices, and
    # a frame missing a minion must FREE its slot for reuse.
    _, netid2 = rb.rebuild(frame)
    chk(list(np.asarray(netid)) == list(np.asarray(netid2)),
        "slot assignment is not stable across identical frames -- Orders.target "
        "is a unit index, so an unstable map retargets the policy's choice")
    gone = {"t": 12400, "u": [u for u in frame["u"] if u["id"] != 22]}
    rb.rebuild(gone)
    chk(22 not in {int(x) for x in np.asarray(rb.rebuild(gone)[1]) if x},
        "a departed minion kept its slot")

    # and the whole point: the TRAINING observation builder accepts it
    fr = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE], TOP_OUTER_TURRET[Team.RED],
                         BLUE_NEXUS)
    obs = build_observation(st, 0, fr, params=lane_params())
    chk(np.isfinite(np.asarray(obs.entities)).all(), "non-finite entity feature")
    chk(np.isfinite(np.asarray(obs.self_vec)).all(), "non-finite self_vec")
    chk(np.isfinite(np.asarray(obs.global_vec)).all(), "non-finite global_vec")
    placed = {int(netid[i]) for i in np.asarray(obs.slot_unit) if i >= 0}
    chk({22, 23} <= placed,
        f"enemy minions missing from the observation slots: {sorted(placed)}")

    for m in fail:
        print(f"  FAIL {m}")
    print(f"selftest: {'FAILED' if fail else 'ok'} "
          f"({len(fail)} failures)")
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="",
                    help="path to a RunDir ckpt_*.msgpack, or 'random'")
    ap.add_argument("--selftest", action="store_true",
                    help="assert the wire -> LaneState mapping and exit. Needs "
                         "no server. Run it after any change to the wire or to "
                         "PROFILES.")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=0,
                    help="servers in flight at once (default: all episodes)")
    ap.add_argument("--max-game-ms", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax instead of sampling. OFF by default: a "
                         "deterministic policy froze the champion in this "
                         "project's anchor eval and the CS number still looked "
                         "plausible. Training sampled, so sampling is the "
                         "like-for-like comparison.")
    ap.add_argument("--red", choices=("bot", "idle"), default="bot")
    ap.add_argument("--bot-config", default=None)
    ap.add_argument("--server-dir", type=Path, default=None)
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--trace-every-s", type=float, default=0.0,
                    help="sample lane position / CS / gold / spell ranks every N "
                         "game-seconds and print the timeline. 'never left the "
                         "fountain' and 'farmed badly' look identical in the "
                         "summary row and completely different here.")
    ap.add_argument("--replay", type=Path, default=None,
                    help="JSONL of recorded sim orders to send OPEN-LOOP, "
                         "ignoring the policy and the server's observations. "
                         "The isolation that separates an engine difference "
                         "from an observation difference.")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.replay is not None and not a.checkpoint:
        # The replay supplies every order, so no network is consulted; a fresh
        # one is built only because `load_params` returns the module too.
        a.checkpoint = "random"
    if not a.checkpoint:
        ap.error("--checkpoint is required (or pass --selftest or --replay)")

    policy, params, label = load_params(a.checkpoint)
    if a.replay is not None:
        label = f"replay:{a.replay.name}"
    log_dir = a.log_dir or Path(tempfile.mkdtemp(prefix="rl_eval_vs_server_"))
    par = a.parallel or a.episodes
    print(f"{label}: {a.episodes} episodes vs red={a.red}, {par} servers at a "
          f"time, logs in {log_dir}")

    rows: list[dict] = []
    done = 0
    while done < a.episodes:
        k = min(par, a.episodes - done)
        batch = play_batch(policy, params, n=k, max_game_ms=a.max_game_ms,
                           seed=a.seed + done, deterministic=a.deterministic,
                           red=a.red, bot_config=a.bot_config,
                           log_dir=log_dir, server_dir=a.server_dir,
                           trace_every_s=a.trace_every_s, replay=a.replay)
        for r in batch:
            r["episode"] = done
            done += 1
            print(f"  ep{r['episode']}: t={r['t_s']:.0f}s cs={r['blue_cs']} "
                  f"(red {r['red_cs']}) died={r['enemy_minions_died']} "
                  + (f"conv={r['conversion']:.0f}% " if r['conversion'] else "")
                  + f"gold={r['blue_gold']} lvl={r['blue_lvl']} "
                  f"deaths={r['deaths']} dist={r['distance']} "
                  f"spins={r['spins']} acts={r['actions']}", flush=True)
            for tr in (r.get("trace") or []):
                print(f"      t={tr['t_s']:>6.0f}s lane={tr['lane']:+.3f} "
                      f"off={tr['off']:>6} cs={tr['cs']:>3} gold={tr['gold']:>5} "
                      f"lvl={tr['lvl']} hp={tr['hp']:>3}% sl={tr['sl']}")
        rows += batch

    # Before the mean, not after. The whole output is one number, and a number
    # averaged over two populations is worse than none.
    check_homogeneous(rows)

    def agg(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return statistics.mean(v) if v else float("nan")

    cs = [r["blue_cs"] for r in rows if isinstance(r.get("blue_cs"), (int, float))]
    print()
    if len(cs) >= 2:
        sd = statistics.stdev(cs)
        print(f"  CS@10 sd={sd:.1f} over n={len(cs)}, se={sd / math.sqrt(len(cs)):.1f}"
              f"  -- the self-play sd is 7.3, so n=4 carries a 95% CI near +-11 CS. "
              f"Do not read a difference smaller than that.")
    drops = sum(r.get("minions_dropped") or 0 for r in rows)
    casts = sum(r.get("casts") or 0 for r in rows)
    print(f"  {label}: n={len(rows)}  CS={agg('blue_cs'):.1f} vs red "
          f"{agg('red_cs'):.1f}  gold={agg('blue_gold'):.0f}  "
          f"lvl={agg('blue_lvl'):.1f}  deaths={agg('deaths'):.2f}  "
          f"dist={agg('distance'):.0f}")
    print(f"  references: C# LanerlBot 33 CS (bot_cs_probe), torch BC prior "
          f"37.3 (spike eval_vs_bot, n=3), JAX mirror self-play 46 "
          f"(RL-002, 4 episodes)")
    print(f"  casts issued: {casts} -- the docstring's buff deviation is "
          + ("LIVE, so Q/E availability was misreported on some frames"
             if casts else "INERT for this run")
          + f". minion slots dropped: {drops}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"label": label, "checkpoint": a.checkpoint, "red": a.red,
             "deterministic": a.deterministic, "rows": rows}, indent=2))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
