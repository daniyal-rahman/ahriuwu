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
recall_channel_ms, buff_id, observed_enemy_cast_ms. The control wire
(`LanerlControl.BuildObservation`) already publishes all but the last two:
position, hp/mhp, team, kind, minion subtype (`mt`), and for champions
gold/xp/lvl/cs, spell ranks (`sl`), per-slot cooldowns (`cd0..cd3`) and the
recall channel flag (`rc`).

**The deviation, stated up front.** `buff_id` and `observed_enemy_cast_ms` are
not on the wire, so they stay at their empty-state values. Consequences, both
bounded and both reported by this script:

* `buff_id` feeds `has_w_passive` and the Q/E `cast_locked` flags. Garen's E
  cooldown starts AFTER the 3 s spin, so during a spin the wire's `cd2` reads 0
  -- indistinguishable from ready -- and no derivation from the wire can
  recover it. The reconstructed observation therefore tells the policy Q/E are
  available during their own active window. This can only cost a wasted cast,
  and `--report-casts` counts every cast the policy issues so the exposure is a
  measured number rather than an assumption. Zero casts means the deviation is
  inert for that episode.
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

Lives in `tools/` rather than `lanerl_jax/parity/` for one release only: a new
file under `lanerl_jax/` changes `run_manifest.source_fingerprint()`, and this
was written while a seven-arm sweep was in flight whose arms must all report the
same fingerprint. Move it once that campaign is closed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from lanerl_jax.obs.builder import build_observation                  # noqa: E402
from lanerl_jax.obs.frame import make_lane_frame                      # noqa: E402
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane, lane_params  # noqa: E402
from lanerl_jax.sim.orders import OrderKind                           # noqa: E402
from lanerl_jax.sim.profiles import profile_id                        # noqa: E402
from lanerl_jax.sim.state import (CH_SLICE, MI_SLICE, TU_SLICE, Kind,  # noqa: E402
                                  Team)
from lanerl_jax.train.actions import orders_from                      # noqa: E402
from lanerl_jax.train.policy import LanePolicy, PolicyConfig          # noqa: E402
from lanerl_jax.train.trainer import BLUE_NEXUS, RED_NEXUS, _sample   # noqa: E402

#: wire ``MinionSpawnType`` -> `sim.targeting.MinionType`. The server numbers
#: them MELEE=0 SUPER=1 CANNON=2 CASTER=3; the sim uses `MinionWaveTypes`'
#: spawn order, MELEE=0 CASTER=1 CANNON=2 SUPER=3. Caster and cannon are
#: swapped, which is why this is not the identity map -- see
#: `parity.last_hit_drive.WIRE_MINION_TYPE`, whose docstring records the same
#: trap for the patch-table keys.
WIRE_MT_TO_SIM = {0: 0, 1: 3, 2: 2, 3: 1}

#: Server team ids on the wire -> compact `Team` indices.
WIRE_TEAM = {100: Team.BLUE, 200: Team.RED}

_MINION_KINDS = ("LaneMinion", "Minion")


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
# wire frame -> LaneState
# ---------------------------------------------------------------------------

def _turret_slot_map(base) -> dict:
    """Wire netid -> turret unit index, resolved by POSITION.

    Turret tier is not on the control wire, and tier selects the stat profile
    (`PROFILES` has one row per tier per team, which is the bug `TurretTier`
    exists to fix). `init_lane` already places all 24 turrets with the right
    tiers, so matching a wire turret to the nearest placed turret recovers the
    tier without inventing it. Turrets never move, so the match is exact rather
    than approximate.

    Returned as a builder because netids are only known once a frame arrives.
    """
    tx = np.asarray(base.x[TU_SLICE])
    ty = np.asarray(base.y[TU_SLICE])
    off = TU_SLICE.start

    def match(units):
        out = {}
        for u in units:
            d = (tx - float(u["x"])) ** 2 + (ty - float(u["y"])) ** 2
            j = int(np.argmin(d))
            if d[j] > 100.0 ** 2:      # 100 u: turrets are >1,000 u apart
                continue               # not one of the placed 24 (e.g. inhibitor)
            out[int(u["id"])] = off + j
        return out
    return match


class StateRebuilder:
    """Rebuilds a training-shaped `LaneState` from each control-channel frame.

    Stateful for one reason: minion slot assignment must be STABLE across
    frames within an episode. Entity slots in the observation are chosen by
    `top_k` on distance with ties to the lowest index, so a minion that hops
    between unit slots frame to frame would reorder the observation under a tie
    and, more importantly, would make `Orders.target` (a unit index) point at a
    different minion than the one the policy selected. netid -> slot is
    therefore assigned once, on first sight, and freed when the slot's minion
    is no longer in the frame.
    """

    def __init__(self):
        self.base = init_lane()
        self.params = lane_params()
        self._turret_match = _turret_slot_map(self.base)
        self._turrets: dict[int, int] = {}
        self._minion: dict[int, int] = {}
        self._free = list(range(MI_SLICE.start, MI_SLICE.stop))
        self._dropped = 0
        self.n_units = int(self.base.x.shape[0])

    def rebuild(self, frame: dict):
        """Returns `(state, netid_of_unit)`; `netid_of_unit[i]` is 0 if empty."""
        n = self.n_units
        x = np.zeros(n, np.float32)
        y = np.zeros(n, np.float32)
        hp = np.zeros(n, np.float32)
        mhp = np.zeros(n, np.float32)
        alive = np.zeros(n, bool)
        kind = np.zeros(n, np.int32)
        team = np.asarray(self.base.team, np.int32).copy()
        model = np.asarray(self.base.model, np.int32).copy()
        level = np.asarray(self.base.level, np.int32).copy()
        gold = np.zeros(n, np.float32)
        cs = np.zeros(n, np.int32)
        netid = np.zeros(n, np.int64)
        spell_level = np.zeros((2, 4), np.int32)
        spell_cd = np.zeros((2, 4), np.float32)
        recall = np.zeros(n, np.float32)

        units = frame.get("u", [])
        turrets = [u for u in units if "Turret" in str(u.get("k", ""))]
        if turrets and not self._turrets:
            self._turrets = self._turret_match(turrets)

        # Minion slots first, so a slot freed this frame can be reused by a
        # minion that spawned in the same frame.
        seen = {int(u["id"]) for u in units
                if u.get("k") in _MINION_KINDS}
        for nid in [k for k in self._minion if k not in seen]:
            self._free.append(self._minion.pop(nid))

        for u in units:
            k = str(u.get("k", ""))
            tm = WIRE_TEAM.get(int(u.get("tm", -1)))
            if tm is None:
                continue                       # neutral/unowned: not modelled
            if k == "Champion":
                i = CH_SLICE.start + int(tm)
                kind[i] = Kind.CHAMPION
                model[i] = profile_id(Kind.CHAMPION, -1, int(tm))
                level[i] = max(1, int(u.get("lvl", 1)))
                gold[i] = float(u.get("gold", 0.0))
                cs[i] = int(u.get("cs", 0))
                sl = u.get("sl") or [0, 0, 0, 0]
                spell_level[int(tm)] = [int(v) for v in sl[:4]]
                # wire cd<slot> is ms; `state.spell_cooldown` is seconds, as
                # `build_observation` divides it by the seconds-valued
                # Q_COOLDOWN / *_COOLDOWNS tables.
                spell_cd[int(tm)] = [max(0.0, float(u.get(f"cd{s}", 0)) / 1000.0)
                                     for s in range(4)]
                recall[i] = 1.0 if int(u.get("rc", 0)) else 0.0
            elif k in _MINION_KINDS:
                nid = int(u["id"])
                i = self._minion.get(nid)
                if i is None:
                    if not self._free:
                        self._dropped += 1     # slot table full -- see below
                        continue
                    i = self._free.pop(0)
                    self._minion[nid] = i
                kind[i] = Kind.LANE_MINION
                mt = WIRE_MT_TO_SIM.get(int(u.get("mt", 0)), 0)
                model[i] = profile_id(Kind.LANE_MINION, mt, int(tm))
            elif "Turret" in k:
                i = self._turrets.get(int(u["id"]))
                if i is None:
                    continue
                kind[i] = Kind.TURRET       # model/team come from init_lane
            else:
                continue
            team[i] = int(tm)
            x[i] = float(u["x"])
            y[i] = float(u["y"])
            hp[i] = float(u.get("hp", 0.0))
            mhp[i] = float(u.get("mhp", 0.0))
            alive[i] = hp[i] > 0.0
            netid[i] = int(u["id"])

        state = self.base.replace(
            x=jnp.asarray(x), y=jnp.asarray(y), hp=jnp.asarray(hp),
            max_hp=jnp.asarray(mhp), alive=jnp.asarray(alive),
            kind=jnp.asarray(kind), team=jnp.asarray(team),
            model=jnp.asarray(model), level=jnp.asarray(level),
            gold=jnp.asarray(gold), cs=jnp.asarray(cs),
            spell_level=jnp.asarray(spell_level),
            spell_cooldown=jnp.asarray(spell_cd),
            recall_channel_ms=jnp.asarray(recall),
            t_ms=jnp.asarray(float(frame.get("t", 0)), jnp.float32))
        return state, netid

    @property
    def dropped_minions(self) -> int:
        """How many live minions did not fit in the slot table.

        `N_MINIONS` is 40, sized for a top-only lane, and both teams' minions
        share it. A frame carrying more live minions than that means the
        observation was built from a SUBSET, which is a different measurement --
        so it is counted and reported rather than tolerated silently.

        Counted at the point of the drop. Deriving it from `len(self._minion)`
        would always have returned 0, because a dropped minion never enters
        that dict.
        """
        return self._dropped


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

def load_params(path: str):
    """Params out of a `RunDir` checkpoint, or a fresh untrained set.

    The checkpoint payload is `{"params": ..., "opt_state": ...}` serialised by
    `flax.serialization.to_bytes`, which needs a TARGET of the right structure
    to deserialise into -- so the network is constructed the way the trainer
    constructs it (`LanePolicy(PolicyConfig())`, initialised on a real
    observation) and the bytes are read into that. Building it any other way is
    how an eval ends up silently measuring a different network from the one that
    was trained.
    """
    from flax.serialization import from_bytes

    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    policy = LanePolicy(PolicyConfig())
    obs0 = build_observation(init_lane(), 0, frame, params=lane_params())
    fresh = policy.init(jax.random.key(0), obs0.entities, obs0.entity_pad_mask,
                        obs0.self_vec, obs0.global_vec)
    if path == "random":
        return policy, fresh, "random"
    payload = from_bytes({"params": fresh, "opt_state": None},
                         Path(path).read_bytes())
    return policy, payload["params"], Path(path).parent.name


def order_to_wire(kind: int, ox: float, oy: float, target_unit: int,
                  netid: np.ndarray) -> dict:
    """One semantic `Orders` row -> one control-channel action.

    An ATTACK whose target slot resolved to an empty unit becomes a noop rather
    than an attack on netid 0. The server's complaint counter would catch it,
    but a silently retargeted attack would not be caught by anything.
    """
    if kind == OrderKind.MOVE:
        return {"t": "move", "x": float(ox), "y": float(oy)}
    if kind == OrderKind.ATTACK:
        nid = int(netid[target_unit]) if 0 <= target_unit < len(netid) else 0
        return {"t": "attack", "id": nid} if nid else {"t": "noop"}
    for slot, k in enumerate((OrderKind.CAST_Q, OrderKind.CAST_W,
                              OrderKind.CAST_E, OrderKind.CAST_R)):
        if kind == k:
            return {"t": "cast", "slot": slot}
    if kind == OrderKind.RECALL:
        return {"t": "recall"}
    return {"t": "noop"}


def make_driver(policy, params, *, deterministic: bool, seed: int):
    """A `frame -> (wire action, stats)` callable.

    `deterministic` defaults OFF. A deterministic argmax policy has already made
    an evaluation in this project measure nothing at all: it froze the champion
    and the CS number that came out looked plausible for weeks. Sampling is also
    what training did, so it is the like-for-like comparison.
    """
    frame_blue = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                                 TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)

    @jax.jit
    def act(state, key):
        obs = build_observation(state, 0, frame_blue, params=lane_params())
        logits = policy.apply(params, obs.entities, obs.entity_pad_mask,
                              obs.self_vec, obs.global_vec)
        if deterministic:
            action = (jnp.argmax(logits.button), jnp.argmax(logits.screen_x),
                      jnp.argmax(logits.screen_y), jnp.argmax(logits.target))
        else:
            action, _ = _sample(logits, key)
        # `orders_from` is the TRAINING decoder and expects the (2, ...) batch
        # of both champions. Only blue is driven here, so the row is doubled and
        # row 0 is used -- doubling rather than reshaping keeps the exact
        # per-side lane-frame handling in `orders_from` (`side = +-1` by team)
        # instead of reimplementing it.
        act2 = tuple(jnp.stack([a, a]) for a in action)
        slots = jnp.stack([obs.slot_unit, obs.slot_unit])
        orders = orders_from(act2, state, slots, frame_blue)
        return (orders.kind[0], orders.x[0], orders.y[0], orders.target[0],
                action[0])

    rb = StateRebuilder()
    key = jax.random.key(seed)
    counts = {"cast": 0, "attack": 0, "move": 0, "noop": 0, "recall": 0}

    def drive(frame: dict) -> dict:
        nonlocal key
        key, k = jax.random.split(key)
        state, netid = rb.rebuild(frame)
        kind, ox, oy, tgt, _btn = act(state, k)
        wire = order_to_wire(int(kind), float(ox), float(oy), int(tgt), netid)
        counts[wire["t"]] = counts.get(wire["t"], 0) + 1
        return wire

    drive.counts = counts
    drive.rebuilder = rb
    return drive


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------

def play_batch(policy, params, *, n: int, max_game_ms: int, seed: int,
               deterministic: bool, red: str, bot_config=None,
               log_dir: Path, server_dir=None) -> list[dict]:
    """`n` episodes in `n` FRESH server processes, all stepped in lockstep.

    Parallel because the servers are independent processes and the policy step
    is one small jitted call per instance; `auto_restart=False` because a
    restart mid-episode would silently hand us an episode played at BASE stats
    (the in-process reset strips the rune page), which is the exact blending
    this script exists to refuse.
    """
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

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
    drivers = [make_driver(policy, params, deterministic=deterministic,
                           seed=seed + i) for i in range(n)]
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
                  "alive_prev": set()} for _ in range(n)]
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
                minions_dropped=drivers[i].rebuilder.dropped_minions,
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
        and abs(float(st.gold[0]) - 550.0) < 1e-6, "blue champion scalars")
    chk(list(np.asarray(st.spell_level[0])) == [1, 0, 2, 0], "spell ranks")
    # ms on the wire, SECONDS in the state -- build_observation divides by the
    # seconds-valued cooldown tables.
    chk(abs(float(st.spell_cooldown[0, 0]) - 1.5) < 1e-6
        and abs(float(st.spell_cooldown[0, 2]) - 3.0) < 1e-6, "spell cooldowns")
    chk(float(st.recall_channel_ms[1]) > 0 and float(st.recall_channel_ms[0]) == 0,
        "recall flag")

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
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.checkpoint:
        ap.error("--checkpoint is required (or pass --selftest)")

    policy, params, label = load_params(a.checkpoint)
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
                           log_dir=log_dir, server_dir=a.server_dir)
        for r in batch:
            r["episode"] = done
            done += 1
            print(f"  ep{r['episode']}: t={r['t_s']:.0f}s cs={r['blue_cs']} "
                  f"(red {r['red_cs']}) died={r['enemy_minions_died']} "
                  + (f"conv={r['conversion']:.0f}% " if r['conversion'] else "")
                  + f"gold={r['blue_gold']} lvl={r['blue_lvl']} "
                  f"deaths={r['deaths']} dist={r['distance']} "
                  f"acts={r['actions']}", flush=True)
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
