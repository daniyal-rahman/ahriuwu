"""Behaviour-cloning data from a decoded replay: the modern policy's observations and the replay champion's
actions, 10 decisions a second over the lane window.

    python -m lanerl_jax.modern.bc.convert OUT_DIR GAME_DIR [GAME_DIR ...] [--lag 1 | --lag 1-2]

For each decision the replay state (`replay.read_game`) is written into a modern ``ModernState`` of the
top-lane world (two champions, the 40 top-lane minions nearest the champion, the top lane's structures),
the world's own fog is computed over it (`views.visibility`: sight radii, walls and brush of the 26.19
map) and the observation is built by the world's own `obs.build_modern_observation`. Every feature has the
policy's layout, normalisation, selection and fog rules by construction; what differs is only what the state
could be filled with (docs/modern/BC.md). Actions are labelled by `labels.label_actions`. The world is used
only to format observations and to read labels back with `modern_orders_from` as a check: no simulation
step runs in producing the data.

Filled from the replay: positions, health, unit kinds, teams and minion types, turret life, both champions'
levels and spell ranks, the champion's experience, current gold, last hits, mana, shield, recall channel,
summoner cooldowns (from summoner casts), champion combat (from damage notices), the enemy's witnessed casts.

Filled the world's way, then patched from the replay: the self block's attack damage, ability power, armour,
magic resist, move speed and attack range are the world's own composition of a Garen with starting items,
which the replay's replicated totals replace (items, runes and growth included). Spell cooldowns are rebuilt
from the champion's received casts with the kit's cooldown table and the replay's ability haste
(`spell_state`); the replay's ``cooldown.*`` fields are not read (known-wrong before decoder tag
``2619-received-countdowns-v2``).

Not available: wards (not decoded), the role quest (0), trinket charges (the world's starting value), jungle
monsters and the other eight champions (decisions with a third champion near are out of scope).

Output per game: ``<game>.npz`` (arrays below, leading axis = decision) and ``<game>.json`` (provenance:
dataset, decoder commit, converter commit; counts; label checks).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .. import champions as K
from .. import obs as OB
from .. import world as MS
from ..actions import MODERN_BUTTON_INDEX, MODERN_BUTTONS, modern_orders_from
from ..champions import garen as G
from ..champions.core import cooldown_row
from ..champions.summoners import COOLDOWN as SUMMONER_COOLDOWN
from ..core import types as W
from ..data.champions import cooldowns as kit_cooldowns
from ..economy import RECALL_CAST, RECALL_CHANNEL
from ..frame import make_lane_frame
from ..train import LANE_HALF_WIDTH, default_loadouts
from ..world.config import FOUNTAINS, NEXUS_POSITIONS, TIERS
from ..world.views import visibility
from .labels import Labels, label_actions, cast_rules
from .replay import (BLUE, MINION_SUB, RED, STAT_SCALE, ReplayGame, decision_times, lane_distance, lane_line,
                     read_game)
from ..data import champion_table as CT
from ..data import loadouts as DL

__all__ = ["LANE_WINDOW_S", "OBSERVATION_FIELDS", "BCGame", "convert_game", "spell_state", "parse_lag",
           "world_config", "cooldown_tables", "stand_in_kit"]

#: Game clock 0:30 (the first wave leaves the base) to 14:00.
LANE_WINDOW_S = (30.0, 840.0)
#: A decision is in scope when the champion is alive, within the trainer's lane corridor (`LANE_HALF_WIDTH`
#: around the top lane, extended to the fountains) and no champion but the two laners is within
#: `THIRD_CHAMPION_RADIUS`.
THIRD_CHAMPION_RADIUS = 2000.0
_TOP = 2
#: The lane each replay role is converted in (0 bot, 1 mid, 2 top), and its index in ``obs.ROLES``.
ROLE_LANE = {"top": 2, "mid": 1, "bot": 0, "support": 0}
_RECALL_S = RECALL_CAST + RECALL_CHANNEL
_RECALL_SLOT = 13
_CHUNK = 500
#: Rank-1 cooldowns used to normalise "time since the enemy cast" for an opponent without a kit module.
_UNKNOWN_KIT_RANK1_CD = (10.0, 10.0, 10.0, 100.0)
_HORIZON_S = 600.0            # the trainer's default episode (``episode_s``): the clock feature's normaliser
#: The arrays that are the policy's view of the game, which `BCGame.lagged` delays.
OBSERVATION_FIELDS = ("entities", "entity_pad_mask", "self_vec", "global_vec", "champion_stats")


def parse_lag(text: str):
    """``"1"`` -> 1; ``"1-2"`` -> (1, 2), a range of decisions drawn per decision."""
    lo, _, hi = text.partition("-")
    return (int(lo), int(hi)) if hi else int(lo)


class BCGame(NamedTuple):
    entities: np.ndarray        # (T, 32, 20) float32
    entity_pad_mask: np.ndarray  # (T, 32) bool, True = empty row
    self_vec: np.ndarray        # (T, 56) float32
    global_vec: np.ndarray      # (T, 6) float32
    champion_stats: np.ndarray  # (T, 2, 11) float32, me then the enemy (zero while it has no row)
    button: np.ndarray          # (T,) int32, `actions.MODERN_BUTTONS`
    screen_x: np.ndarray        # (T,) int32
    screen_y: np.ndarray        # (T,) int32
    uses_screen: np.ndarray     # (T,) float32, the trainer's `actions.screen_usage`
    click_valid: np.ndarray     # (T,) bool: the click loss applies
    click_on_unit: np.ndarray   # (T,) bool: the click is on a unit (plain target), else ground (sloped)
    label_valid: np.ndarray     # (T,) bool: in scope and representable, the decisions to learn from
    held: np.ndarray            # (T,) bool: the label is the order still active, not an event (`labels.py`)
    in_scope: np.ndarray        # (T,) bool
    obs_index: np.ndarray       # (T,) int32: the decision whose observation is shown (itself until `lagged`)
    t: np.ndarray               # (T,) float64 game seconds
    meta: dict

    def lagged(self, lag, seed: int = 0) -> "BCGame":
        """Each action paired with the observation of `lag` decisions earlier: a reaction time. `lag` is a
        count of decisions, or ``(lo, hi)`` to draw one per decision uniformly from lo..hi (seeded).
        Decisions with no observation that old are not learned from."""
        T = len(self.t)
        lo, hi = lag if isinstance(lag, tuple) else (lag, lag)
        k = np.random.default_rng(seed).integers(lo, hi + 1, T) if hi > lo else np.full(T, lo)
        src = np.arange(T) - k
        delayed = {f: getattr(self, f)[np.maximum(src, 0)] for f in OBSERVATION_FIELDS}
        meta = dict(self.meta, observation_lag=dict(decisions=[int(lo), int(hi)], seed=seed,
                                                    seconds_per_decision=0.1))
        return self._replace(**delayed, obs_index=np.maximum(src, 0).astype(np.int32),
                             label_valid=self.label_valid & (src >= 0), meta=meta)

    def save(self, out: Path, name: str | None = None):
        out.mkdir(parents=True, exist_ok=True)
        game = name or self.meta["provenance"]["game"]
        np.savez_compressed(out / f"{game}.npz", **{k: v for k, v in self._asdict().items() if k != "meta"})
        (out / f"{game}.json").write_text(json.dumps(self.meta, indent=1))


def _converter_commit() -> str:
    """This checkout's commit, ``-dirty`` if `lanerl_jax` has uncommitted changes. The worktree's git
    directory is recorded under /srv/nfs; on the cluster nodes the same tree is at /mnt/nfs."""
    repo = Path(__file__).resolve().parents[3]
    git_dir = Path((repo / ".git").read_text().split("gitdir:")[1].strip())
    if not git_dir.exists():
        git_dir = Path(str(git_dir).replace("/srv/nfs/", "/mnt/nfs/", 1))
    git = ["git", f"--git-dir={git_dir}", f"--work-tree={repo}"]
    head = subprocess.run(git + ["rev-parse", "--short=8", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain", "--", str(repo / "lanerl_jax")],
                           capture_output=True, text=True, check=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def _loadout(name: str):
    """`train.default_loadouts`, or for a kit with no top page (Ezreal) its first recommended page of any role."""
    try:
        return default_loadouts((name,))[0]
    except KeyError:
        for role in ("middle", "bottom", "support", "jungle"):
            try:
                return MS.Loadout(name, items=(1055, 2003), rune_page=DL.rune_pages(name, role)[0])
            except KeyError:
                continue
        raise


@lru_cache(maxsize=8)        # ~0.3 GB of host arrays each
def world_config(blue: str, red: str, lane: int = _TOP):
    """The one-lane modern world for a (blue, red) champion pair (host-side, cached)."""
    return MS.build_config((_loadout(blue), _loadout(red)), lanes=(lane,), jungle=False, objectives=False,
                           lane_structures=True)


def _hasted(cd: float, haste: float) -> float:
    return cd * 100.0 / (100.0 + max(haste, 0.0))


def spell_state(t, casts, attack_times, rank_at, haste_at):
    """(T, 4) seconds of cooldown left at times `t`, rebuilt from Garen's received casts with the kit's
    cooldown table (`data.champions.cooldowns`), the replay's ability haste and the kit's rules: Q's window
    runs from the cast to the next attack (the empowered hit) or 4.5 s, then its cooldown; W's from the cast;
    E spins 3 s, or until a press at least 1 s in, then its cooldown; R's from the end of its 0.435 s cast.
    A press while the rebuilt cooldown or window says it would do nothing, within ``t``, is counted in
    ``conflicts`` (a refund, or haste the replay did not replicate) and starts the spell anyway, since it
    was received. Returns ``(cooldown (T, 4), conflicts)``."""
    t = np.asarray(t, float)
    tables = [list(kit_cooldowns(G.NAME, s)) for s in "QWER"]
    cooldown = np.zeros((len(t), 4))
    attack_times = np.sort(np.asarray(attack_times, float))
    ready_at = np.zeros(4)
    busy_until = np.full(4, -np.inf)       # Q's window or E's spin
    conflicts = 0

    def cool(slot, start, duration):
        later = t >= start
        cooldown[later, slot] = np.maximum(start + duration - t[later], 0.0)
        ready_at[slot] = start + duration

    casts = sorted(casts)
    for tc, slot in casts:
        if slot == 2 and tc <= busy_until[2]:
            continue                           # a press during the spin (or the one ending it)
        if t[0] <= tc <= t[-1] and (tc < ready_at[slot] - 0.05 or tc < busy_until[slot]):
            conflicts += 1
        rank = max(rank_at(tc, slot), 1)
        base = _hasted(tables[slot][min(rank, len(tables[slot])) - 1], haste_at(tc))
        if slot == 0:
            i = np.searchsorted(attack_times, tc, side="right")
            end = min(tc + G.Q_WINDOW, attack_times[i] if i < len(attack_times) else np.inf)
            cooldown[(t >= tc) & (t < end), 0] = 0.0
            busy_until[0] = end
            cool(0, end, base)
        elif slot == 1:
            cool(1, tc, base)
        elif slot == 2:
            presses = [c for c, s in casts if s == 2 and tc + G.E_MIN_SPIN <= c < tc + G.E_DURATION]
            end = min([tc + G.E_DURATION] + presses)
            cooldown[(t >= tc) & (t < end), 2] = 0.0
            busy_until[2] = end
            cool(2, end, base)
        else:
            cool(3, tc + G.R_CAST_TIME, base)
    return cooldown, conflicts


_RECEIPT_WINDOW_S = 1.5      # a cast whose slot gets no countdown message within this is timed from the kit table


def cooldown_tables(champion: str) -> list[list[float]]:
    """Base cooldowns by rank per slot: the sim kit's table for a registered kit, else the champion table's."""
    if champion in K.BY_NAME:
        return [list(kit_cooldowns(champion, s)) for s in "QWER"]
    idx = CT.index_of(champion)
    if idx == 0:
        raise ValueError(f"{champion!r} has neither a sim kit nor a champion-table row")
    row = CT.champions()[idx - 1]
    return [list(CT._cooldown_ranks(d, c))[:c] for d, c in zip(row["slots"], CT.rank_cap(row))]   # no kit: catalog


def received_spell_state(t, events, casts, kit_name: str, rank_at, haste_at):
    """(T, 4) seconds of cooldown left from the server's received countdowns (``replay.cooldown_events``,
    exports with the countdown fix): after each message for slot ``s`` the timer is ``t_msg + remaining - t``
    until the slot's next message. Abilities the server sends no countdown for (Jax's Q) fall back to the kit
    table hasted at the cast, from the cast time: every cast of a slot without a message within
    ``_RECEIPT_WINDOW_S`` is counted in ``fallback``; a slot with no table either (champion-table rows without
    cooldown values) stays at 0 there. Returns ``(cooldown (T, 4), fallback)``."""
    t = np.asarray(t, float)
    cooldown = np.zeros((len(t), 4))
    tables = cooldown_tables(kit_name)
    if not any(tables):
        raise ValueError(f"{kit_name}: no ability data (no decoder catalog, so its casts are not resolved either)")
    fallback = 0
    for slot in range(4):
        msgs = sorted((te, rem) for te, sl, rem, _dur, _mode in events if sl == slot)
        times = np.asarray([te for te, _ in msgs], float)
        for i, (te, rem) in enumerate(msgs):
            end = msgs[i + 1][0] if i + 1 < len(msgs) else np.inf
            sel = (t >= te) & (t < end)
            cooldown[sel, slot] = np.maximum(te + rem - t[sel], 0.0)
        for tc, sl in sorted(casts):
            if sl != slot:
                continue
            j = int(np.searchsorted(times, tc - 0.05))
            if j < len(times) and times[j] <= tc + _RECEIPT_WINDOW_S:
                continue
            fallback += 1
            table = tables[slot] if slot < len(tables) else []
            if not table:
                continue
            rank = max(rank_at(tc, slot), 1)
            base = _hasted(table[min(rank, len(table)) - 1], haste_at(tc))
            until = min(tc + base, times[j] if j < len(times) else np.inf)
            sel = (t >= tc) & (t < until)
            cooldown[sel, slot] = np.maximum(cooldown[sel, slot], tc + base - t[sel])
    return cooldown, fallback


def _fill(a):
    """Forward-fill NaN rows of an (T, 2) array, then back-fill the start."""
    a = a.copy()
    ok = np.all(np.isfinite(a), axis=1)
    idx = np.where(ok, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    first = np.argmax(ok)
    a[:first] = a[first]
    return a


def _frames(g: ReplayGame, cfg, lane: int = _TOP):
    """The world's lane frames, checked against the replay's turret placements."""
    blue = np.asarray(g.turret_xy[(BLUE, lane, "outer")])
    red = np.asarray(g.turret_xy[(RED, lane, "outer")])
    world = OB.modern_frames(cfg, lane)
    for f, own in ((world[0], blue), (world[1], red)):
        if np.hypot(*(np.asarray(f.origin) - own)) > 10.0:
            raise ValueError(f"replay lane {lane} outer turret {own} is not where the modern world has it {f.origin}")
    return world


def _unit_fields(g: ReplayGame, cfg, me_xy, extra_champions: bool = False):
    """Per-decision arrays of the unit columns the observation and the click decoder read, and the replay
    unit each world slot holds (-1 none). Slots: 0/1 the blue/red laners, the minion block (nearest 40),
    the structures (template positions; turrets take the replay's life and health); with ``extra_champions``
    the other champions in the ward block (no wards are decoded), for the observation and fog only."""
    T = len(g.t)
    lay = cfg.layout
    N = lay.n_units
    i_me, i_opp = g.index(g.me), g.index(g.opponent)
    s_me = g.units[i_me].team                 # BLUE = slot 0, RED = slot 1
    kind0, sub0, team0 = (np.asarray(a) for a in (cfg.unit_kind, cfg.unit_sub, cfg.unit_team))
    x0, y0 = np.asarray(cfg.unit_x), np.asarray(cfg.unit_y)
    f = dict(kind=np.tile(kind0, (T, 1)).astype(np.int32), sub=np.tile(sub0, (T, 1)).astype(np.int32),
             team=np.tile(team0, (T, 1)).astype(np.int32), x=np.tile(x0, (T, 1)).astype(np.float32),
             y=np.tile(y0, (T, 1)).astype(np.float32), hp=np.zeros((T, N), np.float32),
             max_hp=np.zeros((T, N), np.float32), alive=np.zeros((T, N), bool))
    unit_of_slot = np.full((T, N), -1, np.int64)
    known = g.alive & np.isfinite(g.x) & np.isfinite(g.hp) & np.isfinite(g.max_hp)

    def put(k, slot, i, alive):
        unit_of_slot[k, slot] = i
        f["alive"][k, slot] = alive
        f["hp"][k, slot] = g.hp[k, i]
        f["max_hp"][k, slot] = g.max_hp[k, i]

    # Champions.
    for k in range(T):
        put(k, s_me, i_me, g.alive[k, i_me])
        put(k, 1 - s_me, i_opp, known[k, i_opp])
    f["x"][:, s_me], f["y"][:, s_me] = me_xy[:, 0], me_xy[:, 1]
    opp_xy = np.stack([g.x[:, i_opp], g.y[:, i_opp]], -1)
    f["x"][:, 1 - s_me] = np.where(known[:, i_opp], opp_xy[:, 0], x0[1 - s_me])
    f["y"][:, 1 - s_me] = np.where(known[:, i_opp], opp_xy[:, 1], y0[1 - s_me])
    # Minions: the nearest 40 known, in the top lane's block.
    minions = np.asarray([i for i, u in enumerate(g.units) if u.family == "minion"])
    n_slots = lay.monster0 - lay.minion0
    if len(minions):
        d2 = (g.x[:, minions] - me_xy[:, :1]) ** 2 + (g.y[:, minions] - me_xy[:, 1:]) ** 2
        d2 = np.where(known[:, minions], d2, np.inf)
        nearest = np.argsort(d2, axis=1)[:, :n_slots]
        for k in range(T):
            for j, m in enumerate(nearest[k]):
                if np.isfinite(d2[k, m]):
                    i = minions[m]
                    slot = lay.minion0 + j
                    put(k, slot, i, True)
                    u = g.units[i]
                    f["kind"][k, slot] = W.KIND_MINION
                    f["sub"][k, slot] = MINION_SUB[u.subtype]
                    f["team"][k, slot] = u.team
                    f["x"][k, slot], f["y"][k, slot] = g.x[k, i], g.y[k, i]
    if extra_champions:
        others = [i for i, u in enumerate(g.units) if u.family == "champion" and i not in (i_me, i_opp)]
        assert len(others) <= lay.struct0 - lay.ward0
        for j, i in enumerate(others):
            slot = lay.ward0 + j
            unit_of_slot[:, slot] = i
            f["alive"][:, slot] = known[:, i]
            f["hp"][:, slot], f["max_hp"][:, slot] = g.hp[:, i], g.max_hp[:, i]
            f["kind"][:, slot], f["team"][:, slot] = W.KIND_CHAMPION, g.units[i].team
            f["x"][:, slot] = np.where(known[:, i], g.x[:, i], 0.0)
            f["y"][:, slot] = np.where(known[:, i], g.y[:, i], 0.0)
    # Structures: the template's, with the replay's turret life and health where the replay has the turret.
    lane0 = np.asarray(cfg.unit_lane)
    for slot in range(lay.struct0, N):
        f["alive"][:, slot] = True
        f["hp"][:, slot] = f["max_hp"][:, slot] = 1.0
        if kind0[slot] != W.KIND_TURRET:
            continue
        tier = next(name for name, v in TIERS.items() if v == sub0[slot])
        key = (int(team0[slot]), int(lane0[slot]), tier)
        if key in g.turret_id:
            i = g.index(g.turret_id[key])
            unit_of_slot[:, slot] = i
            f["alive"][:, slot] = g.alive[:, i]
            f["hp"][:, slot] = np.nan_to_num(g.hp[:, i], nan=1.0)
            f["max_hp"][:, slot] = np.nan_to_num(g.max_hp[:, i], nan=1.0)
    f["hp"] = np.nan_to_num(f["hp"]).astype(np.float32)
    f["max_hp"] = np.nan_to_num(f["max_hp"]).astype(np.float32)
    structure = np.asarray(W.is_structure(cfg.unit_kind))
    f["targetable"] = f["alive"] & ((f["kind"] != W.KIND_NONE) & ~structure[None] | structure[None])
    return f, unit_of_slot, s_me


def _own_fields(g: ReplayGame, s_me: int, T: int, champion: str = G.NAME):
    """The champion layer, economy, summoner and combat fields the observation reads, per decision; the
    other slot keeps the template's values (replaced where the opponent's are known: ranks, life).
    Cooldowns come from the received countdowns when the export has them (``received_spell_state``), else
    from Garen's kit rules (``spell_state``); ``conflicts`` counts kit fallbacks or rule conflicts."""
    t = g.t
    casts = [(a.t, a.slot) for a in g.actions if a.kind == "cast" and 0 <= a.slot < 4]
    attacks = [a.t for a in g.actions if a.kind == "attack"]
    at = lambda tc: int(np.clip(np.searchsorted(t, tc), 0, T - 1))  # noqa: E731
    rank_at = lambda tc, s: int(g.spell_rank[at(tc), 0, s])        # noqa: E731
    haste_at = lambda tc: float(g.haste[at(tc)])                   # noqa: E731
    if g.provenance.get("cooldown_fields_read"):
        cooldown, conflicts = received_spell_state(t, g.cooldown_events, casts, champion, rank_at, haste_at)
    elif champion == G.NAME:
        cooldown, conflicts = spell_state(t, casts, attacks, rank_at, haste_at)
    else:
        raise ValueError(f"{champion}: the export has no received countdowns and the converter only has "
                         f"Garen's kit rules")
    # A recall channels for 8.5 s unless the champion does anything else first.
    recalling = np.zeros(T, bool)
    for a in g.actions:
        if a.kind == "cast" and a.slot == _RECALL_SLOT:
            later = [b.t for b in g.actions if b.t > a.t and (b.kind != "path" or b.n_waypoints > 1)]
            end = min([a.t + _RECALL_S] + later[:1])
            recalling[(t >= a.t) & (t < end)] = True
    # Summoner spells: ready when cast time + cooldown (haste not replicated per spell; unhasted).
    ready_at = np.zeros((T, 3))
    for a in g.actions:
        if a.kind == "cast" and a.slot in (4, 5):
            slot = g.summoner_slot[a.slot]
            ready = a.t + SUMMONER_COOLDOWN.get(g.summoners[slot], 300.0)
            later = t >= a.t
            ready_at[later, slot] = np.maximum(ready_at[later, slot], ready)

    def both(me_v, other_v, dtype):
        out = np.zeros((T, 2) + np.shape(me_v)[1:], dtype)
        out[:, s_me] = me_v
        out[:, 1 - s_me] = other_v
        return out
    i_opp = g.index(g.opponent)
    fields = dict(
        ranks=both(g.spell_rank[:, 0], g.spell_rank[:, 1], np.int32),
        cooldowns=both(cooldown, np.zeros((T, 4)), np.float32),
        mana=both(g.mana, 0.0, np.float32), cs=both(g.cs, 0, np.int32),
        level=both(g.level, np.maximum(g.champion_stats[:, 1, 10], 1), np.int32),
        xp=both(g.xp, 0.0, np.float32), gold=both(g.gold, 0.0, np.float32),
        dead=both(~g.alive[:, g.index(g.me)], ~g.alive[:, i_opp], bool),
        recall_channeling=both(recalling, False, bool),
        summoner_ready_at=both(ready_at, np.zeros((T, 3)), np.float32),
        last_champion_combat=both(g.last_champion_combat, -1e9, np.float32),
        t=t.astype(np.float32))
    return fields, conflicts


def _seen_casts(g: ReplayGame, can_see, T: int):
    """(T, 4) time the champion's team last saw the opponent cast each of Q/W/E/R (-1e9 never), as the
    world's ``champ.seen_cast``: a cast counts when the opponent was visible to the team at that decision."""
    t = g.t
    seen = np.full((T, 4), -1e9, np.float32)
    last = np.full(4, -1e9)
    casts = sorted((tc, s) for tc, s in g.opponent_casts if 0 <= s < 4)
    j = 0
    for k in range(T):
        while j < len(casts) and casts[j][0] < t[k]:
            tc, s = casts[j]
            if can_see[int(np.clip(np.searchsorted(t, tc), 0, T - 1))]:
                last[s] = tc
            j += 1
        seen[k] = last
    return seen


def _corridor(cfg, lane: int = _TOP):
    """The lane's centre line from fountain to fountain (world coordinates)."""
    path = ([tuple(p) for p in np.asarray(cfg.lane_path).tolist()] if lane == _TOP
            else [tuple(map(float, p)) for p in lane_line(lane)])
    return (FOUNTAINS[0], NEXUS_POSITIONS[0], *path, NEXUS_POSITIONS[1], FOUNTAINS[1])


def _scope(g: ReplayGame, me_xy, corridor, extra_champions: bool = False):
    """(in scope, in the lane corridor, a third champion near) per decision; a third champion only takes a
    decision out of scope when the observation cannot show it (no ``extra_champions``)."""
    i_me = g.index(g.me)
    others = [i for i, u in enumerate(g.units) if u.family == "champion" and u.id not in (g.me, g.opponent)]
    d = np.hypot(g.x[:, others] - me_xy[:, :1], g.y[:, others] - me_xy[:, 1:])
    crowded = np.any(g.alive[:, others] & (np.nan_to_num(d, nan=np.inf) < THIRD_CHAMPION_RADIUS), axis=1)
    in_lane = lane_distance(me_xy[:, 0], me_xy[:, 1], corridor) < LANE_HALF_WIDTH
    return g.alive[:, i_me] & in_lane & (extra_champions | ~crowded), in_lane, crowded


_JITTED: "OrderedDict[tuple, Any]" = OrderedDict()
_JITTED_MAX = 12      # each holds its world constants on the device; 8 GPU workers share one card


def _vmapped(fn, fields, key=None):
    """`fn` over the leading (decision) axis of `fields`, in chunks. ``key`` names everything `fn` closes over
    (world, side, profile, role, ...): later games with the same key reuse the first one's compiled function."""
    T = len(next(iter(jax.tree.leaves(fields))))
    run = _JITTED.pop(key, None) if key is not None else None
    if run is None:
        run = jax.jit(jax.vmap(fn))
    if key is not None:
        _JITTED[key] = run
        while len(_JITTED) > _JITTED_MAX:
            _JITTED.popitem(last=False)
    parts = [jax.device_get(run(jax.tree.map(lambda a: a[i:i + _CHUNK], fields))) for i in range(0, T, _CHUNK)]
    return jax.tree.map(lambda *a: np.concatenate(a), *parts)


def _self_names(profile: str = OB.PROFILE) -> list[str]:
    """Names of the self-vector columns of a profile (`obs.build_modern_observation`)."""
    head = ["s", "n", "hp", "level", "gold", "cs", "cd_q", "cd_w", "cd_e", "cd_r", "ad", "ap", "armor", "mr",
            "dead", "recalling"]
    tail = ["mana", "shield", "summ_d", "summ_f", "next_level", "quest", "quest_done", "in_combat", "ms",
            "range", "skill_points", "trinket"]
    if profile in OB.TABLE_PROFILES:
        names = ["own_id", "enemy_id", "tier", *head, *tail, "rank_q", "rank_w", "rank_e", "rank_r",
                 *CT.side_columns("own"), *CT.side_columns("enemy")]
        names += [f"role_{r}" for r in OB.ROLES] if profile == "modern-world-v3" else []
    else:
        names = [*head, *(f"own_{x.NAME}" for x in K.KITS), *(f"enemy_{x.NAME}" for x in K.KITS), *tail]
    assert len(names) == OB.self_dim(profile)
    return names


def _self_columns(profile: str = OB.PROFILE):
    """Column index of each named self feature (`obs.build_modern_observation`)."""
    return {n: i for i, n in enumerate(_self_names(profile))}


def stand_in_kit(champion: str) -> str:
    """The sim kit a champion without one borrows for the world template (its identity, descriptors, cooldown
    normalisation and stats are overwritten from the replay and the champion table): Ezreal when ranged by the
    table's attack range, else Garen."""
    if champion in K.BY_NAME:
        return champion
    idx = CT.index_of(champion)
    if idx == 0:
        raise ValueError(f"{champion!r} is not in the champion table")
    return "Ezreal" if CT.champions()[idx - 1]["stats"]["attack_range"] > 300 else "Garen"


def _label_check(g, labels, valid, order, unit_of_slot, slot_unit):
    """How the world reads the labels (`modern_orders_from` on the rebuilt state), over the decisions to
    learn from: attacks and targeted casts on the intended unit, that unit in the observation, moves to the
    received path's end, stops as stops."""
    T = len(g.t)
    acts = [g.actions[i] if i >= 0 else None for i in labels.source]
    b = MODERN_BUTTON_INDEX
    ids = np.asarray([u.id for u in g.units])

    def unit_of(slot):
        u = unit_of_slot[np.arange(T), np.maximum(slot, 0)]
        return np.where((slot >= 0) & (u >= 0), ids[np.maximum(u, 0)], -1)
    target = np.asarray([-2 if a is None or a.target is None else a.target for a in acts])
    is_kind = lambda kind: np.asarray([a is not None and a.kind == kind for a in acts]) & valid  # noqa: E731
    attack = is_kind("attack") & (labels.button == b["move"])
    attack_unit = unit_of(order["attack"])
    r = is_kind("cast") & (labels.button == b["r"])
    cast_unit = unit_of(order["cast_target"])
    seen = np.any(slot_unit == order["attack"][:, None], axis=1) & (order["attack"] >= 0)
    move = is_kind("path") & (labels.button == b["move"]) & ~labels.clamped
    end = np.asarray([a.point if a is not None and a.kind == "path" else (np.nan, np.nan) for a in acts])
    miss = np.hypot(order["move_x"][move] - end[move, 0], order["move_y"][move] - end[move, 1])
    stop = valid & (labels.button == b["stop"])
    return dict(
        attacks=int(attack.sum()), attacks_resolved=int((attack & (attack_unit == target)).sum()),
        attacks_target_in_obs=int((attack & seen & (attack_unit == target)).sum()),
        r_casts=int(r.sum()), r_resolved=int((r & (order["cast_slot"] == 3) & (cast_unit == target)).sum()),
        moves=int(move.sum()), moves_read_as_attack=int((move & (order["attack"] >= 0)).sum()),
        stops=int(stop.sum()), stops_resolved=int((stop & order["stop"]).sum()),
        move_point_error_median=float(np.median(miss)) if len(miss) else None,
        move_point_error_p90=float(np.percentile(miss, 90)) if len(miss) else None)


def convert_game(game_dir: Path, window=LANE_WINDOW_S, champion: str = "Garen", hold: bool = True,
                 profile: str = OB.PROFILE, tier: int = CT.ARCHIVE_TIER, role: str = "top",
                 extra_champions: bool = False) -> BCGame:
    """One game's decisions over `window` (game seconds) for `champion`, no lag. ``profile`` is the observation
    layout (`obs.PROFILES`); under ``modern-world-v2`` any champion of the build can be the actor (a sim kit stands
    in for the world template) and ``tier`` is the skill tier the self vector reports. ``role`` (``obs.ROLES``)
    picks the lane the world is built in (``ROLE_LANE``) and the v3 role columns; ``extra_champions`` puts the
    other eight champions in the observation and keeps the decisions they are near."""
    if profile not in OB.PROFILES:
        raise ValueError(f"unknown observation profile {profile!r}")
    v2 = profile in OB.TABLE_PROFILES
    lane = ROLE_LANE[role]
    if champion not in K.BY_NAME and not v2:
        raise ValueError(f"{champion}: no sim kit; convert with profile modern-world-v2")
    t = decision_times(*window)
    g = read_game(game_dir, t, champion, lane)
    T = len(t)
    i_me, me_team, registered, me_kit, pair, cfg, frames, me_xy = _setup(g, champion, lane)
    frame = frames[me_team]

    f, unit_of_slot, s_me = _unit_fields(g, cfg, me_xy, extra_champions)
    own, conflicts = _own_fields(g, s_me, T, champion)
    f.update(own)
    template = MS.init_state(cfg)
    other = 1 - s_me

    def state_of(x):
        x = {k: v for k, v in x.items() if k != "label"}
        s = template
        champ = s.champ._replace(ranks=x["ranks"], cooldowns=x["cooldowns"], mana=x["mana"], cs=x["cs"],
                                 seen_cast=x.get("seen_cast", s.champ.seen_cast))
        econ = s.econ._replace(gold=x["gold"], xp=x["xp"], level=x["level"], dead=x["dead"],
                               recall=s.econ.recall._replace(channeling=x["recall_channeling"]))
        combat = s.combat._replace(clocks=s.combat.clocks._replace(last_champion_combat=x["last_champion_combat"]))
        s = s._replace(t=x["t"], kind=x["kind"], sub=x["sub"], team=x["team"], x=x["x"], y=x["y"], hp=x["hp"],
                       max_hp=x["max_hp"], alive=x["alive"], targetable=x["targetable"], champ=champ, econ=econ,
                       summoners=s.summoners._replace(ready_at=x["summoner_ready_at"]), combat=combat)
        # Extra champions sit in ward slots, which the ward inputs would give ward sight (none placed).
        vis, sight, _ = visibility(cfg, s.x, s.y, s.kind, s.sub, s.team, s.alive, s.reveal, s.t,
                                   wards=None if extra_champions else s.wards, level=s.econ.level)
        return s._replace(visible=vis, sight=sight)

    def enemy_seen(x):
        s = state_of(x)
        return s.visible[s_me, other] & s.alive[other]
    f["seen_cast"] = np.zeros((T, 2, 4), np.float32) - 1e9
    # The closures read only the world (pair, lane), the side and the observation settings.
    key = (pair, lane, s_me, profile, tier, role, extra_champions)
    f["seen_cast"][:, other] = _seen_casts(g, _vmapped(enemy_seen, f, key + ("enemy_seen",)), T)

    labels = _labels(g, champion, me_team, frame, me_xy, hold)
    f["label"] = np.stack([labels.button, labels.screen_x, labels.screen_y], -1)

    def observe(x):
        s = state_of(x)
        obs = OB.build_modern_observation(s, s_me, frame, cfg, horizon_s=_HORIZON_S, profile=profile, tier=tier,
                                          role=OB.ROLES.index(role), extra_champions=extra_champions)
        action = tuple(jnp.zeros(2, jnp.int32).at[s_me].set(v) for v in x["label"])
        o = modern_orders_from(action, s, frames)
        order = dict(attack=o.attack[s_me], move_x=o.move_x[s_me], move_y=o.move_y[s_me],
                     cast_slot=o.cast_slot[s_me], cast_target=o.cast_target[s_me], stop=o.stop[s_me])
        return (obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, obs.slot_unit), order
    (entities, pad, self_vec, global_vec, slot_unit), order = _vmapped(observe, f, key + ("observe",))

    # Patches from the replay: the self block's combat stats and the enemy identity.
    col = _self_columns(profile)
    me_stats, opp_stats = g.champion_stats[:, 0], g.champion_stats[:, 1]
    self_vec = np.array(self_vec)
    self_vec[:, col["ad"]] = me_stats[:, 0] / OB.NORM_AD
    self_vec[:, col["ap"]] = me_stats[:, 1] / OB.NORM_AD
    self_vec[:, col["armor"]] = me_stats[:, 2] / OB.NORM_AD
    self_vec[:, col["mr"]] = me_stats[:, 3] / OB.NORM_AD
    self_vec[:, col["mana"]] = np.where(g.max_mana > 0, g.mana / np.maximum(g.max_mana, 1.0), 0.0)
    self_vec[:, col["shield"]] = g.shield / np.maximum(me_stats[:, 7], 1.0)
    self_vec[:, col["ms"]] = me_stats[:, 5] / 500.0
    self_vec[:, col["range"]] = me_stats[:, 9] / 600.0
    patched = ["ad", "ap", "armor", "mr", "mana", "shield", "ms", "range"]
    global_vec = np.array(global_vec)
    own_idx, opp_idx = CT.index_of(champion), CT.index_of(g.opponent_name)
    if v2:
        # Identity, descriptors and cooldown normalisation from the champion table, whatever kits the world
        # template borrowed; the skill tier the dataset declares.
        self_vec[:, col["own_id"]], self_vec[:, col["enemy_id"]], self_vec[:, col["tier"]] = own_idx, opp_idx, tier
        desc = CT.descriptors()
        self_vec[:, col["own_attack_range"]:col["own_attack_range"] + CT.SIDE_DIM] = desc[own_idx]
        self_vec[:, col["enemy_attack_range"]:col["enemy_attack_range"] + CT.SIDE_DIM] = desc[opp_idx]
        ranks = np.maximum(f["ranks"][:, s_me], 1)
        base = CT.cooldowns()[own_idx][np.arange(4)[None, :], np.clip(ranks - 1, 0, CT.MAX_RANKS - 1)]
        frac = np.where(f["ranks"][:, s_me] > 0, np.clip(f["cooldowns"][:, s_me] / np.maximum(base, 1e-3), 0, 1), 1.0)
        for i, nm in enumerate(("cd_q", "cd_w", "cd_e", "cd_r")):
            self_vec[:, col[nm]] = frac[:, i]
        patched += ["identity and descriptors (champion table)", "cooldown fractions (champion table)"]
        if not registered:
            since = t[:, None] - f["seen_cast"][:, other]
            rank1 = CT.cooldowns()[opp_idx][:, 0] if opp_idx else np.asarray(_UNKNOWN_KIT_RANK1_CD)
            global_vec[:, 2:] = np.where(f["seen_cast"][:, other] > -1e8, np.clip(since / rank1, 0.0, 1.0), 1.0)
            patched += ["enemy cast memory (no kit: champion-table rank-1 cooldowns)"]
    elif not registered:
        for k in K.KITS:
            self_vec[:, col[f"enemy_{k.NAME}"]] = 0.0
        since = t[:, None] - f["seen_cast"][:, other]
        global_vec[:, 2:] = np.where(f["seen_cast"][:, other] > -1e8,
                                     np.clip(since / np.asarray(_UNKNOWN_KIT_RANK1_CD), 0.0, 1.0), 1.0)
        patched += ["enemy identity (no kit: zeros)", "enemy cast memory (no kit: default rank-1 cooldowns)"]
    stats = (g.champion_stats / np.asarray(STAT_SCALE)).astype(np.float32)
    stats[:, 1] *= global_vec[:, 1:2]

    in_scope, in_lane, crowded = _scope(g, me_xy, _corridor(cfg, lane), extra_champions)
    label_valid = in_scope & labels.representable
    meta = dict(
        provenance=dict(g.provenance, converter_commit=_converter_commit(),
                        world_profile=cfg.profile, obs_profile=profile, action_profile="modern-world-v2"),
        champion=champion, role=role, lane=lane, extra_champions=extra_champions,
        champion_kit=me_kit, champion_index=own_idx, opponent=g.opponent_name,
        opponent_kit=g.opponent_name if registered else None, opponent_index=opp_idx, skill_tier=tier,
        team="blue" if me_team == BLUE else "red", summoners=list(g.summoners), window_s=list(window),
        decisions=T,
        counts=dict(in_scope=int(in_scope.sum()), label_valid=int(label_valid.sum()),
                    dead=int((~g.alive[:, i_me]).sum()), out_of_lane=int((~in_lane).sum()),
                    third_champion=int(crowded.sum()), not_representable=int((~labels.representable).sum()),
                    clicks_clamped=int(labels.clamped.sum()), click_valid=int((labels.click_valid & label_valid).sum()),
                    superseded=labels.n_superseded, repeat_attacks=labels.n_repeat_attacks,
                    held=int((labels.held & label_valid).sum()),
                    held_on_unit=int((labels.held & labels.click_on_unit & label_valid).sum()),
                    cooldown_conflicts=conflicts,
                    buttons={b: int(np.sum((labels.button == i) & label_valid)) for i, b in enumerate(MODERN_BUTTONS)}),
        label_check=_label_check(g, labels, label_valid, order, unit_of_slot, slot_unit),
        patched_from_replay=patched,
        not_from_replay=[("spell cooldowns: received countdowns, kit table where the server sends none "
                          "(received_spell_state)") if g.provenance.get("cooldown_fields_read")
                         else "spell cooldowns: rebuilt from casts (spell_state)", "wards: none",
                         "quest: 0", "trinket charges: the world's starting value",
                         "summoner cooldowns: unhasted"],
        observation_lag=dict(decisions=[0, 0]))
    return BCGame(entities=entities, entity_pad_mask=pad, self_vec=self_vec.astype(np.float32),
                  global_vec=global_vec.astype(np.float32), champion_stats=stats, button=labels.button,
                  screen_x=labels.screen_x, screen_y=labels.screen_y, uses_screen=labels.uses_screen,
                  click_valid=labels.click_valid, click_on_unit=labels.click_on_unit, label_valid=label_valid,
                  held=labels.held, in_scope=in_scope, obs_index=np.arange(T, dtype=np.int32), t=t, meta=meta)


def _setup(g: ReplayGame, champion: str, lane: int):
    """The world a laner's game is converted in: (index of the champion, its team, opponent kit registered,
    its stand-in kit, the world's kit pair, config, lane frames, the champion's position per decision)."""
    i_me = g.index(g.me)
    me_team = g.units[i_me].team
    registered = g.opponent_name in K.BY_NAME
    opp_kit = g.opponent_name if registered else ("Jax" if champion != "Jax" else "Garen")   # stand-in kit
    me_kit = stand_in_kit(champion)
    pair = (me_kit, opp_kit) if me_team == BLUE else (opp_kit, me_kit)
    cfg = world_config(*pair, lane)
    me_xy = _fill(np.stack([g.x[:, i_me], g.y[:, i_me]], -1))
    return i_me, me_team, registered, me_kit, pair, cfg, _frames(g, cfg, lane), me_xy


def _labels(g: ReplayGame, champion: str, me_team: int, frame, me_xy, hold: bool):
    enemies = {u.id for u in g.units if u.team != me_team}
    unit_xy = {u.id: np.stack([g.x[:, i], g.y[:, i]], -1) for i, u in enumerate(g.units)}
    return label_actions(g.t, g.actions, me_xy, unit_xy, np.asarray(frame.axis), np.asarray(frame.normal),
                         enemies, g.summoner_slot, hold=hold, rules=cast_rules(champion), me=g.me)


def relabel_game(game_dir: Path, champion: str, role: str = "top", hold: bool = True,
                 window=LANE_WINDOW_S) -> Labels:
    """The labels `convert_game` gives the game now, without building observations (labels depend only on
    the replay's actions and positions and the lane frame)."""
    lane = ROLE_LANE[role]
    g = read_game(game_dir, decision_times(*window), champion, lane)
    i_me = g.index(g.me)
    me_team = g.units[i_me].team
    me_xy = _fill(np.stack([g.x[:, i_me], g.y[:, i_me]], -1))
    frames = _frames(g, world_config(*_FRAME_PAIR, lane), lane)    # the lane frames do not depend on the kits
    return _labels(g, champion, me_team, frames[me_team], me_xy, hold)


_FRAME_PAIR = ("Garen", "Jax")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("games", type=Path, nargs="+")
    parser.add_argument("--lag", type=parse_lag, default=1, help="observation lag in decisions, N or LO-HI")
    parser.add_argument("--seed", type=int, default=0, help="for a lag range")
    parser.add_argument("--no-hold", action="store_true", help="label click events only, not held orders")
    args = parser.parse_args()
    for game_dir in args.games:
        bc = convert_game(game_dir, hold=not args.no_hold).lagged(args.lag, args.seed)
        bc.save(args.out)
        print(game_dir.name, json.dumps(bc.meta["counts"]), flush=True)


if __name__ == "__main__":
    main()
