"""The checkpoint-backed policy driver for the C# server, in ONE copy.

Factored out of `tools/rl_eval_vs_server.py` (`PARITY-001`), which imports it
from here, so the external eval and the policy-divergence gate
(:mod:`lanerl_jax.parity.policy_divergence`) drive the server with the same
code: the wire frame -> `LaneState` rebuild (:class:`StateRebuilder`), the
TRAINING observation builder and decoder (`build_observation`,
`train.actions.orders_from`), and the wire encoding (:func:`order_to_wire`).
See the tool's module docstring for why the observation is built this way and
the one place it is not exact (buffs are not on the control wire; E is
recovered from `cd2` edges, Q is not). `StateRebuilder`'s docstring lists the
three wire quantities it translates into the sim's meaning rather than copying
(gold, witnessed enemy casts, `GarenWPassive`: `OBS-04/05/06`).

What this module adds on top of the move
----------------------------------------
* :class:`PolicyDriver` -- ``step(frame) -> DriverStep(wire, orders, state,
  netid)``: the wire order sent, the semantic sim `Orders` row it came from
  (target as a REBUILDER slot), the rebuilt state the policy saw, and the
  rebuilder's slot -> NetId table. `make_driver` is kept, unchanged in
  behaviour, as the eval's ``frame -> wire`` closure.
* :class:`CreationRankMap` -- NetId <-> sim slot by CREATION RANK. A sim unit
  index means nothing on the server and a NetId means nothing in the sim; what
  both engines share is the ORDER in which units were created. Server NetIds
  are handed out by a counter in creation order; the sim stamps every unit with
  `spawn_seq` (turrets, then the two champions, then each minion as it spawns,
  red before blue within a wave -- `RESET-004`). So the k-th LaneMinion NetId
  the server ever created is the minion whose `spawn_seq` is
  ``first_minion_seq + k``. Champions map by team, turrets by their fixed
  position (the same match `StateRebuilder` uses).
* :class:`PolicyActionLog` -- per decision, the wire order (attack targets as
  NetIds) AND the sim order AND the target's NetId/`spawn_seq`, plus the full
  sorted list of minion NetIds seen, so a replay can re-derive every mapping.
  `to_action_log()` gives the plain `record.ActionLog` the rest of the parity
  apparatus aligns on.
* :class:`PolicyPairDriver` -- a ``driver(obs, i)`` for `record.record_trace`
  that drives BOTH champions with the policy (mirror self-play, the training
  distribution) or blue only with red idle, and fills a `PolicyActionLog`.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..obs.builder import build_observation
from ..obs.frame import make_lane_frame
from ..sim.init import TOP_OUTER_TURRET, init_lane, lane_params
from ..sim.orders import OrderKind
from ..sim.profiles import profile_id
from ..sim.spells import E_DURATION_S
from ..sim.state import CH_SLICE, MI_SLICE, TU_SLICE, Kind, Team
from ..train.actions import orders_from
from ..train.policy import LanePolicy, PolicyConfig
from ..train.trainer import BLUE_NEXUS, RED_NEXUS, _sample

__all__ = [
    "WIRE_MT_TO_SIM", "WIRE_TEAM", "E_CANCEL_ARMED_MAX_MS", "STARTING_GOLD",
    "StateRebuilder",
    "load_params", "order_to_wire", "pending_rank_up", "make_driver",
    "DriverStep", "PolicyDriver", "CreationRankMap", "PolicyActionLog",
    "PolicyPairDriver", "WIRE_ORDER_CAST_SPELL", "CAST_FREEZE_MS",
    "CastFreezeDetector", "scan_cast_freeze",
]

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


#: `GarenECancel`'s cooldown is 1000 ms; one 30 Hz frame of slack.
E_CANCEL_ARMED_MAX_MS = 1100.0

#: `LanerlConfig.StartingGold`: the wire's ``gold`` is the WALLET and starts
#: here. The sim's `state.gold` is EARNINGS from 0 (`OBS-04`).
STARTING_GOLD = 475.0

#: How long before its wire cooldown RISE each spell was pressed, in ms, for
#: the witnessed-cast memory (`OBS-05`). W and E (start) write their cooldown
#: at the press, so the rise is seen on the next frame and the press was the
#: previous frame. R writes it at `FinishCasting`, `R_CAST_TIME_S` after the
#: press. Q writes 0 at the press (`Q.cs`) and the real cooldown when its
#: window closes, which may be up to 4.5 s later and is not recoverable from
#: the wire -- so Q is recorded at the window's end (late; `OBS-05` row).
_CAST_RISE_LAG_MS = (0.0, 0.0, 0.0, 435.0)

#: A wire cooldown must move up by more than this (ms) to count as a rise.
_RISE_EPS_MS = 1.0


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

    Three more per-champion memories make the state mean what the SIM's state
    means, since the observation is built from it by the training builder:

    * **gold** (`OBS-04`): the wire carries the WALLET (starts at
      :data:`STARTING_GOLD`, drops on a purchase); the sim's `state.gold` is
      EARNINGS from 0 -- minion last-hits, ambient gold from 90 s, champion and
      turret kills (`step.py`: ``state.gold + rw.gold + amb + ckr.gold +
      tk_gold``), and nothing is ever subtracted. So earnings are
      ``wallet - 475`` on first sight, then the sum of positive wallet deltas,
      which ignores a purchase instead of reading it as negative income. The
      wire rounds the wallet DOWN to an integer, so this is at most 1 gold
      below the sim's float (3e-4 of the feature's 3000 scale).
    * **witnessed enemy casts** (`OBS-05`): `orders._record_observed_enemy_casts`
      mirrored from wire cooldown RISES, since casts are not on the wire. A
      rise counts when the caster is alive and visible to the observer's team
      (the sim's `visible_to`) and within `OBSERVED_CAST_SCREEN_RADIUS` of the
      alive observer, judged on the frame the order was issued (the previous
      rebuilt frame), as the sim judges it on the pre-tick state. E rises
      TWICE per spin (to `GarenECancel`'s ~1000 ms at the start, to the rank
      cooldown at the end); only the first is a cast, as `cast_e` reports
      only a start. Timing per :data:`_CAST_RISE_LAG_MS`.
    * **`GarenWPassive`** (`OBS-06`): the sim grants it the tick W first has a
      rank and never removes it; the builder reads it for own armour/MR. It
      was never set here, so from W's first rank the server-path observation
      showed the pre-passive resists.
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
        # Per-champion E-spin inference from `cd2` edges (module docstring).
        self._e = [{"prev_cd2": 0.0, "spin_start_ms": None} for _ in range(2)]
        # OBS-04: per champion, last wallet seen and earnings so far.
        self._wallet_prev: list = [None, None]
        self._earned = [0.0, 0.0]
        # OBS-05: per champion, last wire cooldowns (ms), None before first
        # sight; per OBSERVER row, the frame time of each witnessed enemy cast
        # (NaN = never); the previous frame's witness matrix and time.
        self._cd_prev: list = [None, None]
        self._cast_t = np.full((2, 4), np.nan)
        self._prev_witness: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        # OBS-06: GarenWPassive, sticky once granted.
        self._w_passive = [False, False]

    def _earnings(self, tm: int, wallet: float) -> float:
        prev = self._wallet_prev[tm]
        if prev is None:
            self._earned[tm] = max(0.0, wallet - STARTING_GOLD)
        elif wallet > prev:
            self._earned[tm] += wallet - prev
        self._wallet_prev[tm] = wallet
        return self._earned[tm]

    @staticmethod
    def _witness_matrix(x, y, kind, team, alive) -> np.ndarray:
        """``(2, 2)`` bool, [observer, caster]: `_record_observed_enemy_casts`'s
        ``can_witness`` on this frame's rebuilt arrays."""
        from ..obs.fog import visible_to
        from ..sim.orders import OBSERVED_CAST_SCREEN_RADIUS
        ch = slice(CH_SLICE.start, CH_SLICE.start + 2)
        seen = {t: np.asarray(visible_to(t, jnp.asarray(x), jnp.asarray(y),
                                         jnp.asarray(kind), jnp.asarray(team),
                                         jnp.asarray(alive)))[ch]
                for t in (Team.BLUE, Team.RED)}
        cx, cy = x[ch].astype(np.float64), y[ch].astype(np.float64)
        live = (kind[ch] == Kind.CHAMPION) & alive[ch]
        out = np.zeros((2, 2), bool)
        for o in range(2):
            for c in range(2):
                if o == c or team[CH_SLICE.start + o] == team[CH_SLICE.start + c]:
                    continue
                d2 = (cx[c] - cx[o]) ** 2 + (cy[c] - cy[o]) ** 2
                out[o, c] = bool(live[o] and live[c]
                                 and seen[int(team[CH_SLICE.start + o])][c]
                                 and d2 <= OBSERVED_CAST_SCREEN_RADIUS ** 2)
        return out

    def _observed_casts(self, t_now: float, cds: dict, witness: np.ndarray
                        ) -> np.ndarray:
        """Update the witnessed-cast memory from this frame's cooldowns and
        return `observed_enemy_cast_ms` ``(2, 4)``: ms since each observer
        last witnessed the enemy cast each slot, -1 if never."""
        judge = self._prev_witness if self._prev_witness is not None else witness
        t_press = self._prev_t if self._prev_t is not None else t_now
        for c, cd in cds.items():
            prev = self._cd_prev[c]
            self._cd_prev[c] = cd
            if prev is None:
                continue                       # first sight: no edge yet
            for s in range(4):
                if cd[s] <= prev[s] + _RISE_EPS_MS:
                    continue
                if s == 2 and cd[s] > E_CANCEL_ARMED_MAX_MS:
                    continue                   # spin END, not a cast
                for o in range(2):
                    if judge[o, c]:
                        self._cast_t[o, s] = t_press - _CAST_RISE_LAG_MS[s]
        self._prev_witness = witness
        self._prev_t = t_now
        return np.where(np.isnan(self._cast_t), -1.0,
                        np.maximum(0.0, t_now - np.nan_to_num(self._cast_t)))

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
        # (N, 4), the state's own shape: only the champion rows (slots 0/1,
        # = team) are filled, but `spells.status` reads every row.
        spell_level = np.zeros((n, 4), np.int32)
        spell_cd = np.zeros((n, 4), np.float32)
        recall = np.zeros(n, np.float32)
        # Only E's spin is recoverable from the wire (below); every other
        # buff record stays at the base state's (empty) value.
        e_active = np.asarray(self.base.buffs.e.active).copy()
        e_elapsed = np.asarray(self.base.buffs.e.elapsed_s).copy()
        w_passive = np.asarray(self.base.buffs.w_passive).copy()
        cds: dict = {}                 # team -> wire cooldowns (ms), this frame
        t_now = float(frame.get("t", 0))

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
                # OBS-04: earnings, not the wallet (class docstring).
                gold[i] = self._earnings(int(tm), float(u.get("gold", 0.0) or 0.0))
                cs[i] = int(u.get("cs", 0))
                sl = u.get("sl") or [0, 0, 0, 0]
                spell_level[int(tm)] = [int(v) for v in sl[:4]]
                # OBS-06: `grant_w_passive` -- alive with W ranked; sticky.
                if float(u.get("hp", 0.0)) > 0.0 and int(sl[1]) >= 1:
                    self._w_passive[int(tm)] = True
                w_passive[i] = self._w_passive[int(tm)]
                # wire cd<slot> is ms; `state.spell_cooldown` is seconds, as
                # `build_observation` divides it by the seconds-valued
                # Q_COOLDOWN / *_COOLDOWNS tables.
                cds[int(tm)] = [max(0.0, float(u.get(f"cd{s}", 0) or 0.0))
                                for s in range(4)]
                spell_cd[int(tm)] = [v / 1000.0 for v in cds[int(tm)]]
                recall[i] = 1.0 if int(u.get("rc", 0)) else 0.0
                # E spin from the cd2 edge: <= 1.1 s is `GarenECancel`
                # being armed (a spin began); anything larger is the
                # rank cooldown (the spin ended, by time or by cancel).
                e = self._e[int(tm)]
                cd2_ms = float(u.get("cd2", 0) or 0.0)
                if cd2_ms > e["prev_cd2"] + 1.0:
                    e["spin_start_ms"] = (
                        t_now if cd2_ms <= E_CANCEL_ARMED_MAX_MS else None)
                e["prev_cd2"] = cd2_ms
                if e["spin_start_ms"] is not None:
                    el = (t_now - e["spin_start_ms"]) / 1000.0
                    if el >= E_DURATION_S:
                        e["spin_start_ms"] = None
                    else:
                        e_active[i] = True
                        e_elapsed[i] = el
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

        observed = self._observed_casts(
            t_now, cds, self._witness_matrix(x, y, kind, team, alive))

        state = self.base.replace(
            x=jnp.asarray(x), y=jnp.asarray(y), hp=jnp.asarray(hp),
            max_hp=jnp.asarray(mhp), alive=jnp.asarray(alive),
            kind=jnp.asarray(kind), team=jnp.asarray(team),
            model=jnp.asarray(model), level=jnp.asarray(level),
            gold=jnp.asarray(gold), cs=jnp.asarray(cs),
            spell_level=jnp.asarray(spell_level),
            spell_cooldown=jnp.asarray(spell_cd),
            recall_channel_ms=jnp.asarray(recall),
            buffs=self.base.buffs.replace(
                e=self.base.buffs.e.replace(
                    active=jnp.asarray(e_active),
                    elapsed_s=jnp.asarray(e_elapsed,
                                          self.base.buffs.e.elapsed_s.dtype)),
                w_passive=jnp.asarray(w_passive)),
            observed_enemy_cast_ms=jnp.asarray(
                observed, self.base.observed_enemy_cast_ms.dtype),
            t_ms=jnp.asarray(t_now, jnp.float32))
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



def pending_rank_up(champ: Mapping) -> Optional[int]:
    """The slot the SIM would have ranked up by now, or None.

    THE BUG THIS FIXES, and it is worth stating because the first run of
    the eval scored 1.5 CS against 44.6 in the sim and this was the whole
    difference. The JAX sim ranks spells automatically: `step.py` indexes
    `_RANK_TABLE = RANKS_BY_LEVEL` by champion level on every tick, so a
    level-3 champion HAS E rank 2 with no action spent. The C# server ranks
    spells in `AutoLevel`, which belongs to the SCRIPTED BOT -- and the eval
    drives blue itself, so blue's `sl` stayed `[0,0,0,0]` for the whole game
    while red (bot-driven) had E at rank 1 from the first frame.

    A policy that presses E on 72% of its decisions (measured, same
    checkpoint, both engines) was therefore casting an UNRANKED spell.
    `Spell.Cast` does not check spell level -- the cast goes through and the
    cooldown starts -- so nothing complained and nothing looked wrong; the
    spin simply did rank-0 damage. `docs` records "spell rank 0" as one of
    the five original BC blockers, which is the same bug in a different
    harness.

    So the driver sends the rank-ups itself, following `spells.SKILL_ORDER`
    (E first: 2,0,1,2,2,3,...) via `RANKS_BY_LEVEL`, which is the exact table
    training uses. It costs the decision it is sent on -- the wire takes one
    order per side per step -- which is at most 18 of 18,000 decisions, and
    the count is reported. The sim has no rank-up order, so a replay of the
    log turns these into NOOPs (`PolicyActionLog` marks them ``rank_up``).
    """
    from ..sim.spells import RANKS_BY_LEVEL

    lvl = int(champ.get("lvl") or 1)
    want = RANKS_BY_LEVEL[min(lvl, len(RANKS_BY_LEVEL) - 1)]
    have = [int(v) for v in (champ.get("sl") or [0, 0, 0, 0])[:4]]
    for slot in range(4):
        if have[slot] < want[slot]:
            return slot
    return None


def _lane_frames():
    blue = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                           TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    red = make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                          TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS)
    return blue, red


def _make_act(policy, params, *, deterministic: bool, team: int):
    """The jitted ``(state, key) -> one Orders row`` for champion ``team``.

    For ``team == BLUE`` this is, line for line, the closure that lived in
    `rl_eval_vs_server.make_driver`. For red it is the trainer's own red
    path: the observation is built from red's seat with red's lane frame
    (`trainer.py`: ``build_observation(state, 1, red_frame, ...)``) and the
    decoder is called with the BLUE frame for both rows exactly as the trainer
    calls it -- `orders_from` flips the lane axis per row from ``state.team``.
    """
    frame_blue, frame_red = _lane_frames()
    own = frame_blue if int(team) == Team.BLUE else frame_red
    row = int(team)

    @jax.jit
    def act(state, key):
        obs = build_observation(state, row, own, params=lane_params())
        logits = policy.apply(params, obs.entities, obs.entity_pad_mask,
                              obs.self_vec, obs.global_vec)
        if deterministic:
            action = (jnp.argmax(logits.button), jnp.argmax(logits.screen_x),
                      jnp.argmax(logits.screen_y), jnp.argmax(logits.target))
        else:
            action, _, _ = _sample(logits, key, ~obs.entity_pad_mask)
        # `orders_from` is the TRAINING decoder and expects the (2, ...) batch
        # of both champions. One champion is driven here, so the row is
        # doubled and row `team` is used -- doubling rather than reshaping
        # keeps the exact per-side lane-frame handling in `orders_from`
        # (`side = +-1` by team) instead of reimplementing it.
        act2 = tuple(jnp.stack([a, a]) for a in action)
        slots = jnp.stack([obs.slot_unit, obs.slot_unit])
        orders = orders_from(act2, state, slots, frame_blue)
        return (orders.kind[row], orders.x[row], orders.y[row],
                orders.target[row], action[0])

    return act


class DriverStep(NamedTuple):
    """One decision of one champion.

    ``orders`` is the semantic sim order the wire order was encoded from,
    ``{"kind","x","y","target"}`` with ``target`` a REBUILDER slot (valid only
    against ``netid``), or None on a rank-up decision, which has no sim
    counterpart. ``state``/``netid`` are what the policy saw, None on a
    rank-up (the frame is not rebuilt then, as in the original eval).
    """

    wire: dict
    orders: Optional[dict]
    state: object
    netid: Optional[np.ndarray]


class PolicyDriver:
    """One champion driven by a checkpoint: ``step(frame) -> DriverStep``.

    ``deterministic`` defaults OFF. A deterministic argmax policy has already
    made an evaluation in this project measure nothing at all: it froze the
    champion and the CS number that came out looked plausible for weeks.
    Sampling is also what training did, so it is the like-for-like comparison.

    Several drivers may share one ``rebuilder`` (both champions of a mirror
    game see the same rebuilt state, and the minion slot table must be one
    table for `Orders.target` to mean the same unit to both).
    """

    def __init__(self, policy, params, *, team: int = Team.BLUE,
                 deterministic: bool = False, seed: int = 0,
                 rebuilder: Optional["StateRebuilder"] = None):
        self.team = int(team)
        self.wire_team = 100 if self.team == Team.BLUE else 200
        self.rebuilder = rebuilder if rebuilder is not None else StateRebuilder()
        self.key = jax.random.key(seed)
        self.counts = {"cast": 0, "attack": 0, "move": 0, "noop": 0,
                       "recall": 0, "level": 0}
        self._act = _make_act(policy, params, deterministic=deterministic,
                              team=self.team)

    def rank_up(self, frame: Mapping) -> Optional[dict]:
        """The ``level`` wire order to send this decision, or None."""
        me = next((u for u in frame.get("u", [])
                   if u.get("k") == "Champion" and u.get("tm") == self.wire_team),
                  None)
        if me is None:
            return None
        slot = pending_rank_up(me)
        if slot is None:
            return None
        self.counts["level"] += 1
        return {"t": "level", "slot": slot}

    def decide(self, state, netid: np.ndarray) -> DriverStep:
        """Act on an already-rebuilt state."""
        self.key, k = jax.random.split(self.key)
        kind, ox, oy, tgt, _btn = self._act(state, k)
        kind, ox, oy, tgt = int(kind), float(ox), float(oy), int(tgt)
        wire = order_to_wire(kind, ox, oy, tgt, netid)
        # `SPELL-010`: the server casts an UNRANKED spell (`Spell.Cast` never
        # checks the level); the sim and real League refuse it. The rank-up
        # order is sent first, but the policy can press before it lands, so
        # the wire order is gated here on the rank the frame reported --
        # the same rule `orders.py` applies -- and counted separately.
        if wire.get("t") == "cast":
            rank = int(state.spell_level[int(self.team), int(wire["slot"])])
            if rank <= 0:
                self.counts["cast_unranked"] = self.counts.get("cast_unranked", 0) + 1
                wire = {"t": "noop"}
                kind = int(OrderKind.NOOP)
        self.counts[wire["t"]] = self.counts.get(wire["t"], 0) + 1
        return DriverStep(wire, {"kind": kind, "x": ox, "y": oy, "target": tgt},
                          state, netid)

    def step(self, frame: Mapping) -> DriverStep:
        wire = self.rank_up(frame)
        if wire is not None:
            return DriverStep(wire, None, None, None)
        state, netid = self.rebuilder.rebuild(frame)
        return self.decide(state, netid)


#: `OrderType.CastSpell` as the control wire's ``mo`` carries it
#: (`GameServerCore/Enums/OrderType.cs:77`, ``CastSpell = 0xF``).
WIRE_ORDER_CAST_SPELL = 15

#: How long a champion may hold ``mo == CastSpell`` before it is FROZEN, in
#: game ms. The longest legitimate stretch is a cast windup: GarenQAttack
#: 250 ms, R ~435 ms, the recall pill 500 ms. Measured over seven recordings
#: (sweepA-a4 300 s, diag1b and obsharden 120 s, aa005_full, tier15,
#: champ_dynamic_audit, buff001): every healthy stretch was <= 501 ms. The one
#: exception was sweepA-a4's blue from 213,934 ms to the end of the recording,
#: an 86 s freeze (`SERVER-001`). 3 s is 6x the longest legitimate stretch and
#: still flags a freeze within 90 decisions.
CAST_FREEZE_MS = 3000.0


class CastFreezeDetector:
    """Flag a server champion stuck in ``MoveOrder == CastSpell`` (`SERVER-001`).

    The server can leave a champion with ``_castingSpell`` set and
    ``MoveOrder == CastSpell`` for the rest of the process. `ObjAIBase.CanMove`,
    `CanChangeWaypoints`, `CanAttack` and `CanCast` then all refuse, so every
    later order is silently dropped. It survives death and respawn. The
    policy's observation cannot tell: position, hp and gold still update. The
    eval's CS number then measures the freeze, not the policy (diag1b: 1 CS
    against the bot's 40, frozen from 159.5 s).

    Detection reads ONLY the wire's ``mo``, which `LanerlControl.BuildObservation`
    already publishes for champions. It adds nothing to what the policy sees.
    A frame whose champions carry no ``mo`` cannot be checked. That is recorded
    (``unchecked``) and makes the run invalid: an unscored run is not a pass.

    ``observe(frame)`` once per frame; ``report()`` is JSON-able. ``invalid``
    is True once any champion has held CastSpell for ``threshold_ms``.
    """

    def __init__(self, threshold_ms: float = CAST_FREEZE_MS):
        self.threshold_ms = float(threshold_ms)
        self._open: Dict[int, dict] = {}       # wire team -> current stretch
        self.frozen: List[dict] = []            # stretches past the threshold
        self.frames = 0
        self.champion_rows = 0
        self.rows_without_mo = 0

    def observe(self, frame: Optional[Mapping]) -> None:
        if not frame:
            return
        self.frames += 1
        t = float(frame.get("t", 0))
        for u in frame.get("u", []):
            if u.get("k") != "Champion":
                continue
            self.champion_rows += 1
            tm = int(u.get("tm", -1))
            mo = u.get("mo")
            if mo is None:
                self.rows_without_mo += 1
                continue
            if int(mo) != WIRE_ORDER_CAST_SPELL:
                self._open.pop(tm, None)
                continue
            s = self._open.get(tm)
            if s is None:
                s = self._open[tm] = {
                    "team": tm, "start_t_ms": t, "last_t_ms": t,
                    "x": u.get("x"), "y": u.get("y"),
                    "died_while_stuck": False, "flagged": False}
            s["last_t_ms"] = t
            if float(u.get("hp", 1)) <= 0:
                s["died_while_stuck"] = True
            if not s["flagged"] and t - s["start_t_ms"] >= self.threshold_ms:
                s["flagged"] = True
                self.frozen.append(s)       # by reference: keeps extending

    @property
    def unchecked(self) -> bool:
        """No champion row carried ``mo``: the check could not run."""
        return self.champion_rows == 0 or self.rows_without_mo == self.champion_rows

    @property
    def invalid(self) -> bool:
        return bool(self.frozen) or self.unchecked

    def frozen_teams(self) -> List[int]:
        return sorted({int(s["team"]) for s in self.frozen})

    def reason(self) -> Optional[str]:
        if self.frozen:
            parts = []
            for s in self.frozen:
                side = {100: "blue", 200: "red"}.get(int(s["team"]), str(s["team"]))
                parts.append(
                    f"{side} champion held MoveOrder=CastSpell from "
                    f"t={s['start_t_ms'] / 1000:.1f}s to {s['last_t_ms'] / 1000:.1f}s "
                    f"({(s['last_t_ms'] - s['start_t_ms']) / 1000:.1f}s"
                    + (", through a death" if s["died_while_stuck"] else "") + ")")
            return ("server cast freeze (SERVER-001): " + "; ".join(parts)
                    + ". Every order after that was dropped by the server, so "
                      "the counters measure the freeze, not the policy.")
        if self.unchecked:
            return ("freeze check impossible: no champion row carried the wire "
                    "field 'mo' (a server build older than the control "
                    "channel's tgt/atk/mo fields?)")
        return None

    def report(self) -> dict:
        return {"threshold_ms": self.threshold_ms, "frames": self.frames,
                "unchecked": self.unchecked, "invalid": self.invalid,
                "frozen": [{k: v for k, v in s.items() if k != "flagged"}
                           for s in self.frozen],
                "reason": self.reason()}


def scan_cast_freeze(frames, threshold_ms: float = CAST_FREEZE_MS) -> CastFreezeDetector:
    """Run a detector over an iterable of wire frames (e.g. a recorded
    ``*_obs.jsonl``, one JSON frame per line, or already-parsed dicts)."""
    det = CastFreezeDetector(threshold_ms)
    for fr in frames:
        if isinstance(fr, (str, bytes)):
            fr = fr.strip()
            if not fr:
                continue
            fr = json.loads(fr)
        det.observe(fr)
    return det


def make_driver(policy, params, *, deterministic: bool, seed: int):
    """A ``frame -> wire action`` closure for blue, as the eval uses it.

    Carries ``.counts`` and ``.rebuilder`` (the eval reads both) and
    ``.driver``, the underlying :class:`PolicyDriver`.
    """
    d = PolicyDriver(policy, params, team=Team.BLUE,
                     deterministic=deterministic, seed=seed)

    def drive(frame: dict) -> dict:
        return d.step(frame).wire

    drive.counts = d.counts
    drive.rebuilder = d.rebuilder
    drive.driver = d
    return drive


# ---------------------------------------------------------------------------
# NetId <-> sim slot, by creation rank
# ---------------------------------------------------------------------------

class CreationRankMap:
    """NetId <-> `spawn_seq` <-> sim slot, recovered from creation order.

    Neither engine's identifier means anything to the other: a sim unit index
    is a recycled slot, a server NetId a process-global counter. Both engines
    create units in the same ORDER, and both record it -- the server in the
    NetId itself (a monotone counter), the sim in `LaneState.spawn_seq`. So:

    * champions by team (`spawn_seq` from `init_lane`, slot `CH_SLICE.start +
      team`);
    * turrets by their fixed position against `init_lane`'s placement (the
      match `StateRebuilder` uses; turret NetIds come from the map file, not
      the counter, so rank would be wrong for them);
    * lane minions by RANK: the k-th smallest LaneMinion NetId ever seen is
      the minion with ``spawn_seq == first_minion_seq + k``. Red-before-blue
      within a wave (`RESET-004`) holds in both engines, so no tie-break is
      needed. The rank is only as good as the NetId list is complete -- every
      minion must have been OBSERVED -- which is why the list is recorded in
      full (`PolicyActionLog.minion_netids`) and ranks are recomputed at
      replay time rather than trusted from record time.

    A minion rank maps to a sim SLOT only while a live minion with that
    `spawn_seq` exists in the sim; once the engines disagree about a death,
    :meth:`sim_slot` returns None rather than guess (the slot may already hold
    a newer minion).
    """

    def __init__(self, base=None):
        base = base if base is not None else init_lane()
        seq = np.asarray(base.spawn_seq)
        kind = np.asarray(base.kind)
        self.first_minion_seq = int(base.next_spawn_seq)
        self._champ_seq = {t: int(seq[CH_SLICE.start + t])
                           for t in (Team.BLUE, Team.RED)}
        self._turret_match = _turret_slot_map(base)
        self._turret_seq = {int(i): int(seq[i])
                            for i in range(TU_SLICE.start, TU_SLICE.stop)
                            if kind[i] == Kind.TURRET}
        self.champions: Dict[int, int] = {}      # team -> netid
        self.turrets: Dict[int, int] = {}        # netid -> slot
        self.minions: List[int] = []             # sorted netids
        self._minion_set: set = set()

    # -- learning NetIds -----------------------------------------------------
    def add_minions(self, netids) -> None:
        for n in netids:
            n = int(n)
            if n not in self._minion_set:
                self._minion_set.add(n)
                bisect.insort(self.minions, n)

    def observe(self, frame: Mapping) -> None:
        """Learn every NetId on one control-channel frame."""
        units = frame.get("u", [])
        turrets = []
        for u in units:
            k = str(u.get("k", ""))
            tm = WIRE_TEAM.get(int(u.get("tm", -1)))
            if tm is None or "id" not in u:
                continue
            if k == "Champion":
                self.champions[int(tm)] = int(u["id"])
            elif k in _MINION_KINDS:
                self.add_minions((u["id"],))
            elif "Turret" in k:
                turrets.append(u)
        if turrets and not self.turrets:
            self.turrets = self._turret_match(turrets)

    # -- lookups ---------------------------------------------------------------
    def _champ_team(self, netid: int) -> Optional[int]:
        for t, n in self.champions.items():
            if n == netid:
                return t
        return None

    def rank_of(self, netid: int) -> Optional[int]:
        netid = int(netid)
        if netid not in self._minion_set:
            return None
        return bisect.bisect_left(self.minions, netid)

    def spawn_seq_of(self, netid: int) -> Optional[int]:
        netid = int(netid)
        t = self._champ_team(netid)
        if t is not None:
            return self._champ_seq[t]
        if netid in self.turrets:
            return self._turret_seq.get(self.turrets[netid])
        r = self.rank_of(netid)
        return None if r is None else self.first_minion_seq + r

    def netid_of_spawn_seq(self, seq: int) -> Optional[int]:
        """The reverse map, for replaying SIM orders into the server."""
        seq = int(seq)
        for t, s in self._champ_seq.items():
            if s == seq:
                return self.champions.get(t)
        for nid, slot in self.turrets.items():
            if self._turret_seq.get(slot) == seq:
                return nid
        r = seq - self.first_minion_seq
        if 0 <= r < len(self.minions):
            return self.minions[r]
        return None

    def sim_slot(self, netid: int, spawn_seq, alive, kind) -> Optional[int]:
        """The sim unit index this NetId names in a live sim state, or None."""
        netid = int(netid)
        t = self._champ_team(netid)
        if t is not None:
            return CH_SLICE.start + t
        if netid in self.turrets:
            return self.turrets[netid]
        s = self.spawn_seq_of(netid)
        if s is None:
            return None
        hits = np.flatnonzero((np.asarray(spawn_seq) == s) & np.asarray(alive)
                              & (np.asarray(kind) == Kind.LANE_MINION))
        return int(hits[0]) if hits.size else None

    def slot_table(self, spawn_seq, alive, kind) -> Dict[int, int]:
        """NetId -> sim slot for every NetId resolvable against this state.

        The shape `action_replay.decision_to_orders` takes.
        """
        out: Dict[int, int] = {}
        for nid in list(self.champions.values()) + list(self.turrets):
            out[int(nid)] = int(self.sim_slot(nid, spawn_seq, alive, kind))
        seq = np.asarray(spawn_seq)
        live = np.asarray(alive) & (np.asarray(kind) == Kind.LANE_MINION)
        for i in np.flatnonzero(live):
            r = int(seq[i]) - self.first_minion_seq
            if 0 <= r < len(self.minions):
                out[self.minions[r]] = int(i)
        return out

    # -- persistence ------------------------------------------------------------
    def to_json(self) -> dict:
        return {"first_minion_seq": self.first_minion_seq,
                "champions": {str(k): v for k, v in self.champions.items()},
                "turrets": {str(k): v for k, v in self.turrets.items()},
                "minions": list(self.minions)}

    def load_json(self, d: Mapping) -> "CreationRankMap":
        if int(d["first_minion_seq"]) != self.first_minion_seq:
            raise ValueError(
                f"recorded first_minion_seq {d['first_minion_seq']} != "
                f"init_lane's {self.first_minion_seq}: the turret/champion "
                "placement changed since this log was recorded, so its "
                "creation ranks do not line up with this sim")
        self.champions = {int(k): int(v) for k, v in d["champions"].items()}
        self.turrets = {int(k): int(v) for k, v in d["turrets"].items()}
        self._minion_set = set()
        self.minions = []
        self.add_minions(d["minions"])
        return self


# ---------------------------------------------------------------------------
# the action log
# ---------------------------------------------------------------------------

@dataclass
class PolicyActionLog:
    """What was sent to the server, per decision, and what it meant in the sim.

    ``blue``/``red`` are the wire orders exactly as sent (attack targets are
    server NetIds) -- the same lists `record.ActionLog` holds. ``blue_sim`` /
    ``red_sim`` hold, per decision, the sim order the wire order was encoded
    from plus the target's NetId and `spawn_seq` at record time, or
    ``{"rank_up": True, ...}`` / None (undriven side). ``ranks`` is the
    :class:`CreationRankMap` state at the end of the recording, which a
    replay uses to recompute every NetId -> `spawn_seq` mapping.
    """

    t_ms: List[int] = field(default_factory=list)
    blue: List[dict] = field(default_factory=list)
    red: List[dict] = field(default_factory=list)
    blue_sim: List[Optional[dict]] = field(default_factory=list)
    red_sim: List[Optional[dict]] = field(default_factory=list)
    ranks: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def append(self, t_ms: int, blue: dict, red: dict,
               blue_sim: Optional[dict], red_sim: Optional[dict]) -> None:
        self.t_ms.append(int(t_ms))
        self.blue.append(blue)
        self.red.append(red)
        self.blue_sim.append(blue_sim)
        self.red_sim.append(red_sim)

    def __len__(self) -> int:
        return len(self.t_ms)

    def to_action_log(self):
        from .record import ActionLog
        return ActionLog(t_ms=list(self.t_ms), blue=list(self.blue),
                         red=list(self.red))

    def rank_map(self, base=None) -> CreationRankMap:
        m = CreationRankMap(base)
        return m.load_json(self.ranks) if self.ranks else m

    def to_json(self) -> dict:
        return {"t_ms": self.t_ms, "blue": self.blue, "red": self.red,
                "blue_sim": self.blue_sim, "red_sim": self.red_sim,
                "ranks": self.ranks, "meta": self.meta}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), separators=(",", ":")))

    @classmethod
    def from_json(cls, d: Mapping) -> "PolicyActionLog":
        out = cls(t_ms=[int(t) for t in d["t_ms"]], blue=list(d["blue"]),
                  red=list(d["red"]), blue_sim=list(d["blue_sim"]),
                  red_sim=list(d["red_sim"]), ranks=dict(d.get("ranks") or {}),
                  meta=dict(d.get("meta") or {}))
        n = len(out.t_ms)
        if not all(len(v) == n for v in (out.blue, out.red, out.blue_sim,
                                         out.red_sim)):
            raise ValueError(
                "policy action log lengths disagree "
                f"({n}/{len(out.blue)}/{len(out.red)}/{len(out.blue_sim)}/"
                f"{len(out.red_sim)}): its decisions do not line up and a "
                "replay would apply orders at the wrong ticks")
        if any(b <= a for a, b in zip(out.t_ms, out.t_ms[1:])):
            raise ValueError("policy action log times are not strictly increasing")
        return out

    @classmethod
    def load(cls, path) -> "PolicyActionLog":
        return cls.from_json(json.loads(Path(path).read_text()))


class PolicyPairDriver:
    """``driver(obs, i) -> {"blue": wire, "red": wire}`` for `record_trace`.

    ``red="policy"`` drives BOTH champions with the same parameters -- mirror
    self-play, the distribution the checkpoint was trained on (`trainer.py`:
    "both champions act under the same parameters"). ``red="idle"`` sends red
    NOOPs, which the sim replays exactly. There is no ``"bot"``: the scripted
    C# bot has no sim counterpart, so its orders could not be replayed.

    Both champions share ONE `StateRebuilder`, rebuilt once per frame.
    """

    def __init__(self, policy, params, *, red: str = "policy",
                 deterministic: bool = False, seed: int = 0,
                 meta: Optional[dict] = None):
        if red not in ("policy", "idle"):
            raise ValueError(f"red must be 'policy' or 'idle', not {red!r}")
        self.rebuilder = StateRebuilder()
        self.ranks = CreationRankMap(self.rebuilder.base)
        self.drivers: Dict[int, PolicyDriver] = {
            Team.BLUE: PolicyDriver(policy, params, team=Team.BLUE,
                                    deterministic=deterministic, seed=seed,
                                    rebuilder=self.rebuilder)}
        if red == "policy":
            self.drivers[Team.RED] = PolicyDriver(
                policy, params, team=Team.RED, deterministic=deterministic,
                seed=seed + 1_000_003, rebuilder=self.rebuilder)
        self.log = PolicyActionLog(meta={"red": red,
                                         "deterministic": bool(deterministic),
                                         "seed": int(seed), **(meta or {})})

    def _sim_record(self, st: DriverStep) -> dict:
        o = dict(st.orders)
        o["target_netid"] = 0
        o["target_spawn_seq"] = None
        if o["kind"] == OrderKind.ATTACK:
            t = o["target"]
            nid = int(st.netid[t]) if 0 <= t < len(st.netid) else 0
            o["target_netid"] = nid
            o["target_spawn_seq"] = (self.ranks.spawn_seq_of(nid) if nid else None)
        return o

    def __call__(self, obs: Optional[Mapping], i: int) -> Optional[Dict[str, dict]]:
        if obs is None:
            return None
        self.ranks.observe(obs)
        state = netid = None
        out: Dict[str, dict] = {}
        sims: Dict[str, Optional[dict]] = {}
        for team, side in ((Team.BLUE, "blue"), (Team.RED, "red")):
            d = self.drivers.get(team)
            if d is None:
                out[side], sims[side] = {"t": "noop"}, None
                continue
            wire = d.rank_up(obs)
            if wire is not None:
                out[side] = wire
                sims[side] = {"kind": OrderKind.NOOP, "x": 0.0, "y": 0.0,
                              "target": -1, "rank_up": True,
                              "slot": wire["slot"]}
                continue
            if state is None:
                state, netid = self.rebuilder.rebuild(obs)
            st = d.decide(state, netid)
            out[side] = st.wire
            sims[side] = self._sim_record(st)
        self.log.append(int(obs.get("t", -1)), out["blue"], out["red"],
                        sims["blue"], sims["red"])
        self.log.ranks = self.ranks.to_json()
        return out

    @property
    def counts(self) -> Dict[str, dict]:
        return {("blue" if t == Team.BLUE else "red"): dict(d.counts)
                for t, d in self.drivers.items()}
