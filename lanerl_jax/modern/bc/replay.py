"""A decoded League replay, read onto the modern policy's decision grid.

The source is one game of a native-event export of the rofl decoder (`event-training/` under a game's
`export/`), read with the decoder's own loader and position model, imported read-only from its checkout
(`DECODER_SCRIPTS`). Replay state is at native event times; this module puts it on a fixed grid of decision
times (`decision_times`, 10 a second), the modern trainer's rate (`train.ModernVecConfig.step_ticks` 3).

What is read, and how it is put on the grid:

* positions of champions and top-lane minions: the decoder's accepted position model (integrated from received
  waypoint paths; about 8-40 units median error). Turrets: their received placement.
* health, maximum health, level, experience, current gold (`stat.mGold`), mana, shield, ability haste: the
  decoder's zero-order hold (`scalar_asof`) at each decision time.
* alive: a minion from its received creation to its received death; a champion from the start, dead from a
  received death to its first positive health; a turret until its received death.
* last hits: a minion death whose last received damage notice came from the champion (the death notices name
  no killer).
* champion combat: damage notices between the champion and any enemy champion (`in_combat`).
* the controlled champion's actions (`Action`): received movement paths, executed basic attacks with their
  target, and casts with slot, target and cast point (abilities, summoners, trinket, recall, items). These are
  what the server received or executed, not the clicks. Summoner and trinket casts carry no resolved target;
  their cast record's cursor vector stands in for the click point.
* summoner spells in the policy's slot order (`canonical_summoners`): Flash is always D.
* the lane opponent's casts, for "time since I saw the enemy cast".
* both laners' combat stats as the game replicated them (`STAT_NAMES`; items, runes and level growth
  included), zero-order hold.

``cooldown.*`` fields (the server's received countdowns: slot, remaining seconds at receipt, duration) are
read only from exports whose decoder includes the countdown fix (tag ``2619-received-countdowns-v2``, commit
19cbb838; ``countdowns_received``). Older exports have known-wrong ability cooldowns; for them `convert.py`
rebuilds cooldowns from the casts with the champion's kit rules (Garen only).

Not in the replay: fog of war (the replay sees the whole map), the player's camera and clicks, the click
latency, and wards (not decoded as units).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

__all__ = ["DECODER_SCRIPTS", "DECISION_DT", "BLUE", "RED", "MINION_SUB", "STAT_NAMES", "STAT_SCALE",
           "Unit", "Action", "ReplayGame", "decision_times", "decoder_commit", "read_game", "TOP_LANE",
           "lane_distance"]

DECODER_SCRIPTS = Path("/mnt/nfs/projects/ahriuwu-rofl-decoder-2619/scripts/rofl_decode")
#: The repository every worktree (decoder, this one) shares; readable from both nodes under /mnt/nfs.
MAIN_GIT_DIR = Path("/mnt/nfs/projects/_archive/ahriuwu-main-administrative-20261001/.git")
DECISION_DT = 0.1
BLUE, RED = 0, 1                      # the modern world's team ids (``core.types``: team = champion slot)
_TEAM = {"ORDER": BLUE, "CHAOS": RED}
#: replay minion type -> the modern world's minion ``sub`` (melee, caster, siege, super)
MINION_SUB = {"melee": 0, "ranged": 1, "siege": 2, "super": 3}
#: Replicated combat stats kept beside the observation (raw, then divided by ``STAT_SCALE``).
STAT_NAMES = ("attack_damage", "ability_power", "armor", "magic_resist", "attack_speed", "move_speed",
              "ability_haste", "max_hp", "max_mana", "attack_range", "level")
STAT_SCALE = (200.0, 200.0, 200.0, 200.0, 2.5, 500.0, 100.0, 3000.0, 2000.0, 1000.0, 18.0)
# The 26.19 map's top lane: blue inhibitor, inner and outer turrets, the corner, red outer, inner and inhibitor
# turrets (from the received turret placements). Used only to decide which minions walk the top lane.
TOP_LANE = ((1169., 4287.), (1512., 6699.), (981., 10441.), (1700., 13200.), (4318., 13875.),
            (7943., 13411.), (10481., 13650.))
_TOP_LANE_RADIUS = 1500.0
_RIVER = 14800.0                      # x + y below this is blue's half of the map
_LAST_HIT_TOLERANCE_S = 0.034         # damage notices of the death's own tick count


@dataclass(frozen=True)
class Unit:
    id: int
    family: str          # "champion", "minion" or "turret"
    team: int            # BLUE / RED
    subtype: str         # champion name, minion type ("melee", ...), turret tier ("outer", ...)


class Action(NamedTuple):
    """One received action of the controlled champion."""
    t: float
    kind: str                                 # "path", "attack" or "cast"
    target: Optional[int] = None              # unit id: attack target, cast target
    point: Optional[Tuple[float, float]] = None   # path end, or a cast's target point
    slot: int = -1                            # cast slot (0-3 Q/W/E/R, 4-5 summoners, 12 trinket, 13 recall)
    n_waypoints: int = 0


class ReplayGame(NamedTuple):
    game: str
    provenance: dict
    me: int                               # the controlled champion's and the lane opponent's unit ids
    opponent: int
    opponent_name: str
    summoners: Tuple[int, int]            # the champion's summoner spell ids in the policy's order (Flash = D)
    summoner_slot: Dict[int, int]         # replay cast slot (4, 5) -> policy summoner slot (0 = D, 1 = F)
    units: List[Unit]
    t: np.ndarray                         # (T,) game seconds of each decision
    x: np.ndarray                         # (T, U) per unit; NaN where the position model has none
    y: np.ndarray
    hp: np.ndarray
    max_hp: np.ndarray
    alive: np.ndarray
    level: np.ndarray                     # (T,) the controlled champion
    xp: np.ndarray
    gold: np.ndarray                      # current gold
    cs: np.ndarray                        # last hits since the game start
    mana: np.ndarray
    max_mana: np.ndarray
    shield: np.ndarray
    haste: np.ndarray                     # ability haste
    spell_rank: np.ndarray                # (T, 2, 4): Q/W/E/R ranks of the champion and the opponent
    champion_stats: np.ndarray            # (T, 2, N_STATS) raw stats of the champion and the opponent
    last_champion_combat: np.ndarray      # (T,) time of the last damage between the champion and an enemy champion
    actions: List[Action]
    opponent_casts: List[Tuple[float, int]]
    cooldown_events: List[Tuple[float, int, float, float, int]]  # received countdowns (t, slot, remaining, duration, mode); [] if not read
    turret_xy: Dict[Tuple[int, int, str], Tuple[float, float]]   # {(team, lane, tier): (x, y)}
    turret_id: Dict[Tuple[int, int, str], int]

    def index(self, unit_id: int) -> int:
        return next(i for i, u in enumerate(self.units) if u.id == unit_id)


def decision_times(start_s: float, end_s: float) -> np.ndarray:
    """Decision times from `start_s` (inclusive) to `end_s` (exclusive), 10 a second."""
    return start_s + DECISION_DT * np.arange(int(round((end_s - start_s) / DECISION_DT)))


def decoder_commit(dataset: Path) -> str:
    """The decoder commit a dataset was exported with, as the dataset records it."""
    snapshot = dataset / "decoder-snapshot" / "COMMIT"
    if snapshot.exists():
        return snapshot.read_text().split()[0][:8]
    reexport = dataset / "native-reexport-result.json"
    if reexport.exists():
        return json.loads(reexport.read_text())["code_commit"][:8]
    provenance = dataset / "batch-provenance.json"          # batches prepared by the modern-bc thread
    if provenance.exists():
        return json.loads(provenance.read_text())["decoder_commit"][:8]
    raise ValueError(f"{dataset} records no decoder commit")


#: The decoder commit that corrected the received ability countdowns (tag ``2619-received-countdowns-v2``).
COUNTDOWN_FIX_COMMIT = "19cbb838"


@lru_cache(maxsize=None)
def countdowns_received(dataset: Path) -> bool:
    """Whether a dataset's decoder commit descends from ``COUNTDOWN_FIX_COMMIT``, asked of the shared repository
    (``MAIN_GIT_DIR``: the decoder worktree's own ``.git`` names a path only the login node has). A commit git
    cannot resolve raises rather than silently counting as not fixed."""
    import subprocess
    commit = decoder_commit(dataset)
    r = subprocess.run(["git", f"--git-dir={MAIN_GIT_DIR}", "merge-base", "--is-ancestor", COUNTDOWN_FIX_COMMIT,
                        commit], capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"git could not tell whether {commit} has the countdown fix: {r.stderr.strip()}")
    return r.returncode == 0


def lane_distance(x, y, polyline=TOP_LANE) -> np.ndarray:
    """Distance from points to a polyline (the top lane's centre line by default)."""
    p = np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float)), -1)
    best = np.full(p.shape[:-1], np.inf)
    for a, b in zip(polyline[:-1], polyline[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        u = np.clip(((p - a) @ d) / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - a - u[..., None] * d, axis=-1))
    return best


def _decoder():
    """The decoder's loader and position model (read-only import from its checkout)."""
    if str(DECODER_SCRIPTS) not in sys.path:
        sys.path.append(str(DECODER_SCRIPTS))
    import event_training_loader
    import position_model
    return event_training_loader, position_model


class _Fields:
    """Row access to one game's native-event field tables."""

    def __init__(self, native, loader):
        self.native = native
        self.loader = loader
        tracks = native["index"]["tracks"]
        self.track_id = np.asarray([t[0] if t[0] is not None else -1 for t in tracks], np.int64)
        self.track_family = np.asarray([str(t[2]) for t in tracks])
        self._index = {}       # per instance, so a finished game is freed

    def rows(self, name, family=None):
        """Available rows of a field in stream order: (rows, times, entity ids)."""
        f = self.native["fields"].get(name)
        if f is None:
            return np.zeros(0, int), np.zeros(0), np.zeros(0, np.int64)
        keep = f["available"].astype(bool)
        if family is not None:
            keep &= self.track_family[f["entity_track"]] == family
        rows = np.flatnonzero(keep)
        rows = rows[np.argsort(f["update_index"][rows], kind="stable")]
        return rows, f["time_seconds"][rows], self.track_id[f["entity_track"][rows]]

    def value(self, name, row):
        return self.loader.structured_value(self.native, name, int(row))

    def _by_entity(self, name):
        if name not in self._index:
            f = self.native["fields"][name]
            ent = self.track_id[f["entity_track"]]
            order = np.lexsort((f["update_index"], ent))
            ids, starts = np.unique(ent[order], return_index=True)
            self._index[name] = dict(zip(ids.tolist(), np.split(order, starts[1:])))
        return self._index[name]

    def series(self, name, entity_id):
        """One entity's scalar series, in the form `scalar_asof` takes."""
        f = self.native["fields"][name]
        rows = self._by_entity(name).get(entity_id, np.zeros(0, int))
        return {"storage": f["storage"], "time_seconds": f["time_seconds"][rows],
                "values": f["values"][rows], "available": f["available"][rows].astype(bool)}

    def asof(self, name, entity_id, t, default=np.nan):
        if name not in self.native["fields"]:
            return np.full(len(t), default, float)
        held = self.loader.scalar_asof(self.series(name, entity_id), t)
        return np.where(held["available"], held["values"].astype(np.float64), default)


def _first_after(times: np.ndarray, t: float) -> float:
    i = np.searchsorted(times, t, side="right")
    return float(times[i]) if i < len(times) else math.inf


_LAST_FIELDS: list = [None, None]


def _fields_of(path: Path, loader) -> _Fields:
    """The game's field tables; the last game's are kept, so its other laner converts without a second read."""
    if _LAST_FIELDS[0] != path:
        _LAST_FIELDS[:] = [None, None]
        _LAST_FIELDS[:] = [path, _Fields(loader.load_event_game(path), loader)]
    return _LAST_FIELDS[1]


def read_game(game_dir: Path, t: np.ndarray, champion: str = "Garen") -> ReplayGame:
    """Read one exported game onto the decision times `t`. `game_dir` holds `export/`."""
    loader, position_model = _decoder()
    export = game_dir / "export"
    manifest = json.loads((export / "manifest.json").read_text())
    if manifest["position_model"] != position_model.VERSION:
        raise ValueError(f"export used position model {manifest['position_model']}, "
                         f"the decoder checkout has {position_model.VERSION}")
    fields = _fields_of(export / "event-training", loader)
    dataset = game_dir.parent.parent
    fixed = countdowns_received(dataset)
    provenance = dict(dataset=str(dataset), game=game_dir.name, decoder_commit=decoder_commit(dataset),
                      position_model=manifest["position_model"], replay_sha256=manifest["replay_sha256"],
                      cooldown_fields_read=fixed)

    roster = manifest["initial_roster"]
    me_row = next(r for r in roster if r["name"] == champion)
    opp_row = next(r for r in roster if r["team"] != me_row["team"]
                   and r["assigned_role"] == me_row["assigned_role"])
    me, opponent = me_row["object_id"], opp_row["object_id"]
    summoners, summoner_slot = canonical_summoners(me_row.get("selected_summoner_ids", (4, 12)))
    champions = [Unit(r["object_id"], "champion", _TEAM[r["team"]], r["name"]) for r in roster]
    champion_ids = [u.id for u in champions]
    enemy_champions = {u.id for u in champions if u.team != _TEAM[me_row["team"]]}

    # Turrets: received placements (a turret can be received more than once).
    placement, turret_xy, turret_id = {}, {}, {}
    rows, _, ids = fields.rows("identity.turret_receipt")
    for row, tid in zip(rows, ids):
        placement.setdefault(int(tid), fields.value("identity.turret_receipt", row)["placement"])
    turrets = [Unit(tid, "turret", _TEAM[p["team"]], p["tier"]) for tid, p in placement.items()]
    for tid, p in placement.items():
        key = (_TEAM[p["team"]], p["lane"], p["tier"])
        turret_xy[key] = tuple(p["position_xz"])
        turret_id[key] = tid

    # Minions: creation, type, movement; kept if they walk the top lane.
    born = {}
    rows, times, ids = fields.rows("event.lane_creation", "minion")
    for tm, mid in zip(times, ids):
        born.setdefault(int(mid), float(tm))
    mtype = {}
    rows, _, ids = fields.rows("identity.type", "minion")
    for row, mid in zip(rows, ids):
        mtype.setdefault(int(mid), fields.value("identity.type", row))
    paths = []
    first_path = {}
    for family in ("champion", "minion"):
        rows, times, ids = fields.rows("movement.received_path", family)
        for row, tm, uid in zip(rows, times, ids):
            uid = int(uid)
            if family == "champion" and uid not in champion_ids:
                continue
            v = fields.value("movement.received_path", row)
            paths.append((float(tm), uid, v))
            if family == "minion":
                first_path.setdefault(uid, v["waypoints_xz"])
    top = sorted(m for m, w in first_path.items()
                 if m in born and m in mtype and mtype[m] in MINION_SUB
                 and lane_distance(*w[-1]) < _TOP_LANE_RADIUS)
    minions = [Unit(m, "minion", BLUE if sum(first_path[m][0]) < _RIVER else RED, mtype[m]) for m in top]

    units = champions + minions + turrets
    index = {u.id: i for i, u in enumerate(units)}
    n_t, n_u = len(t), len(units)

    death_times: Dict[int, List[float]] = {}
    rows, times, _ = fields.rows("event.unit_death_received_proxy")
    for row, tm in zip(rows, times):
        victim = fields.value("event.unit_death_received_proxy", row)["victim_id"]
        death_times.setdefault(int(victim), []).append(float(tm))

    hp = np.full((n_t, n_u), np.nan)
    max_hp = np.full((n_t, n_u), np.nan)
    alive = np.zeros((n_t, n_u), bool)
    for i, u in enumerate(units):
        hp[:, i] = fields.asof("stat.mHP", u.id, t)
        max_hp[:, i] = fields.asof("stat.mMaxHP", u.id, t)
        deaths = sorted(death_times.get(u.id, []))
        if u.family == "minion":
            end = deaths[0] if deaths else math.inf
            alive[:, i] = (t >= born[u.id]) & (t < end)
        elif u.family == "turret":
            alive[:, i] = t < (deaths[0] if deaths else math.inf)
        else:
            hs = fields.series("stat.mHP", u.id)
            positive = hs["time_seconds"][hs["available"] & (hs["values"] > 0)]
            alive[:, i] = True
            for d in deaths:
                alive[:, i] &= ~((t >= d) & (t < _first_after(positive, d)))
    hp = np.where(alive, hp, 0.0)

    model = position_model.MinionPositions(top, phase_entity_ids=champion_ids, champion_ids=champion_ids)
    for tm, uid, v in sorted(paths, key=lambda p: p[0]):
        if uid in index:
            model.movement(tm, [v])
    for u in champions + minions:
        s = fields.series("stat.mMoveSpeed", u.id)
        for tm, v, ok in zip(s["time_seconds"], s["values"], s["available"]):
            if ok:
                model.speed(float(tm), u.id, float(v))
    x = np.full((n_t, n_u), np.nan)
    y = np.full((n_t, n_u), np.nan)
    for i, u in enumerate(units):
        if u.family == "turret":
            x[:, i], y[:, i] = placement[u.id]["position_xz"]
            continue
        for k in np.flatnonzero(alive[:, i]):
            e = model.estimate(u.id, float(t[k]))
            if e["position_available"]:
                x[k, i], y[k, i] = e["position_xz"]

    level = fields.asof("stat.mLevelRef", me, t, default=1).astype(np.int32)
    spell_rank = np.stack([np.stack([fields.asof(f"ability.{s}.rank", c, t, default=0) for s in range(4)], -1)
                           for c in (me, opponent)], 1).astype(np.int32)
    last_hit_times = _last_hits(fields, me, death_times)
    cs = np.searchsorted(np.asarray(last_hit_times), t, side="right").astype(np.int32)
    champion_stats = np.stack([_combat_stats(fields, c, t) for c in (me, opponent)], 1)
    actions, opponent_casts = _actions(fields, me, opponent)
    cooldown_events = _cooldown_events(fields, me) if fixed else []
    return ReplayGame(
        game=game_dir.name, provenance=provenance, me=me, opponent=opponent, opponent_name=opp_row["name"],
        summoners=summoners, summoner_slot=summoner_slot, units=units, t=t, x=x, y=y, hp=hp, max_hp=max_hp, alive=alive, level=level,
        xp=fields.asof("stat.mExp", me, t, default=0.0), gold=fields.asof("stat.mGold", me, t, default=0.0),
        cs=cs, mana=fields.asof("stat.mPAR", me, t, default=0.0),
        max_mana=fields.asof("stat.mMaxPAR", me, t, default=0.0),
        shield=fields.asof("stat.mAllShield", me, t, default=0.0),
        haste=fields.asof("stat.mAbilityHasteMod", me, t, default=0.0),
        spell_rank=spell_rank, champion_stats=champion_stats,
        last_champion_combat=_champion_combat(fields, me, enemy_champions, t),
        actions=actions, opponent_casts=opponent_casts, cooldown_events=cooldown_events, turret_xy=turret_xy,
        turret_id=turret_id)


def _combat_stats(fields: _Fields, champion: int, t) -> np.ndarray:
    """(T, N_STATS) raw stats in `STAT_NAMES` order from the replicated totals; 0 before first received."""
    s = lambda name: fields.asof("stat." + name, champion, t, default=0.0)  # noqa: E731
    return np.stack([
        s("mBaseAttackDamage") + s("mFlatPhysicalDamageMod"),
        s("mFlatMagicDamageMod") * (1.0 + s("mPercentMagicDamageMod")),
        s("mArmor"), s("mSpellBlock"), s("mAttackSpeedMod"), s("mMoveSpeed"),
        s("mAbilityHasteMod"), s("mMaxHP"), s("mMaxPAR"), s("mAttackRange"), s("mLevelRef"),
    ], -1)


def _damage_notices(fields: _Fields):
    """(time, receiver, source) of every received damage notice that names a source."""
    out = []
    rows, times, _ = fields.rows("event.received_damage_notice")
    for row, tm in zip(rows, times):
        d = fields.value("event.received_damage_notice", row)
        p = d["participant_ids"]
        if len(p) > 1:
            out.append((float(tm), int(d["receiver_id"]), int(p[1])))
    return out


def _last_hits(fields: _Fields, me: int, death_times) -> List[float]:
    """Times of `me`'s last hits on minions (any lane)."""
    last_source: Dict[int, List[Tuple[float, int]]] = {}
    for tm, receiver, source in _damage_notices(fields):
        last_source.setdefault(receiver, []).append((tm, source))
    out = []
    _, _, ids = fields.rows("event.unit_death_received_proxy", "minion")
    for mid in set(ids.tolist()):
        tm = death_times[mid][0]
        hits = [h for h in last_source.get(mid, ()) if h[0] <= tm + _LAST_HIT_TOLERANCE_S]
        if hits and hits[-1][1] == me:
            out.append(tm)
    return sorted(out)


def _champion_combat(fields: _Fields, me: int, enemies: set, t) -> np.ndarray:
    """(T,) time of the last damage notice between `me` and an enemy champion (-1e9 if none yet)."""
    times = sorted(tm for tm, receiver, source in _damage_notices(fields)
                   if (receiver == me and source in enemies) or (source == me and receiver in enemies))
    times = np.asarray(times, float)
    i = np.searchsorted(times, t, side="right") - 1
    return np.where(i >= 0, times[np.maximum(i, 0)] if len(times) else -1e9, -1e9)


FLASH = 4


def _cast_point(d) -> Optional[Tuple[float, float]]:
    """A cast's target point: the decoder's resolved point; for summoners, items and the trinket (slots 4+),
    whose records carry no resolved target, the record's cursor vector (``0xd0``). An unresolved ability cast
    (slots 0-3, e.g. a champion whose catalog the batch lacked) has no point: ``0xd0`` is then the caster's
    own position, not an aim point (decoder thread, 2026-10-18)."""
    p = d.get("target_point_world_xz")
    if p is not None:
        return (float(p[0]), float(p[1]))
    if d.get("slot") is not None and 0 <= int(d["slot"]) < 4:
        return None
    w = d.get("wire_vectors_xyz") or {}
    v = w.get("0xd0")
    return (float(v[0]), float(v[2])) if v is not None and len(v) == 3 else None


def canonical_summoners(ids) -> Tuple[Tuple[int, int], Dict[int, int]]:
    """``(summoner ids in the policy's slot order, replay slot -> policy slot)``: Flash is always the
    policy's D (slot 0), the other spell F, whatever keys the player used; without Flash the order stands."""
    ids = [int(s) for s in ids[:2]] + [0] * (2 - len(ids[:2]))
    if ids[1] == FLASH and ids[0] != FLASH:
        return (ids[1], ids[0]), {4: 1, 5: 0}
    return (ids[0], ids[1]), {4: 0, 5: 1}


def _cooldown_events(fields: _Fields, me: int):
    """The champion's received countdown events ``(t, slot, remaining, duration, mode)`` in time order. Each is
    one server message: ``slot`` 0-3 abilities (Jax's Q sends none), 4/5 summoners, 6-11 items, 12 trinket;
    ``remaining`` seconds left at receipt (the timer the client shows), ``duration`` the full countdown, ``mode``
    the native operand (1: a remaining-time correction)."""
    names = ("cooldown.slot", "cooldown.remaining_at_receipt", "cooldown.received_duration", "cooldown.mode")
    if any(n not in fields.native["fields"] for n in names):
        return []
    s = [fields.series(n, me) for n in names]
    ok = np.logical_and.reduce([x["available"] for x in s])
    t = s[0]["time_seconds"]
    return sorted((float(t[i]), int(s[0]["values"][i]), float(s[1]["values"][i]), float(s[2]["values"][i]),
                   int(s[3]["values"][i])) for i in np.flatnonzero(ok))


def _actions(fields: _Fields, me: int, opponent: int):
    """`me`'s received actions, and the opponent's casts."""
    actions, opponent_casts = [], []
    rows, times, ids = fields.rows("movement.received_path", "champion")
    for row, tm, uid in zip(rows, times, ids):
        if uid == me:
            w = fields.value("movement.received_path", row)["waypoints_xz"]
            actions.append(Action(float(tm), "path", point=tuple(w[-1]), n_waypoints=len(w)))
    rows, times, ids = fields.rows("event.basic_attack_execution_proxy", "champion")
    for row, tm, uid in zip(rows, times, ids):
        if uid == me:
            d = fields.value("event.basic_attack_execution_proxy", row)
            actions.append(Action(float(tm), "attack", target=d["target_id"]))
    rows, times, ids = fields.rows("event.ability_cast_received_proxy", "champion")
    for row, tm, uid in zip(rows, times, ids):
        if uid not in (me, opponent):
            continue
        d = fields.value("event.ability_cast_received_proxy", row)
        if uid == me:
            actions.append(Action(float(tm), "cast", target=d.get("target_id"), point=_cast_point(d),
                                  slot=d["slot"]))
        else:
            opponent_casts.append((float(tm), d["slot"]))
    actions.sort(key=lambda a: a.t)
    return actions, opponent_casts
