"""Gate 3's live lead: characterise the in-reach HP asymmetry and mortality.

WHY THIS EXISTS
----------------
`hp_band.py` already answers "what fraction of in-reach decision-minion
samples are inside the one-shot band" for both engines. A follow-up census
(2026-09-18, ad hoc, quoted in the task that produced this module) went one
level up: of every enemy minion the held champion could physically reach, how
many are ALREADY one-shot (`hp_band.py`'s own `in_band` definition) versus
merely present but too healthy? That census found:

    sim:    5,457 in-reach decisions, 5,386 too healthy, HP p50 352.4
    server: 4,496 in-reach decisions, 4,409 too healthy, HP p50 212.5

i.e. the sim's held champion has a minion in reach MORE often, but that
minion is much less likely to already be dying. This module is the fuller
characterisation that census was missing: the complete HP distribution (not
a median that can hide a bimodal or outlier-heavy shape -- exactly the shape
"21.5 attack-decisions per kill" turned out to have, see
`docs/JAX_FIDELITY_LEDGER.md`'s gate-3 row), a cheap first check of whether
the two champions are even standing in the same place, and a same-scenario
census of total enemy-minion mortality and what it is attributed to.

WHAT THIS DOES NOT DO
----------------------
This is read-only. It does not change `sim/`, and it does not re-litigate
`hp_band.py`'s own `in_band` predicate (`hp <= post_mitigation(champ_ad,
minion_armor)`) -- it recomputes the identical thing via
:func:`~lanerl_jax.parity.last_hit_oracle.post_mitigation`, the same function
`hp_band.py` itself imports.

IDENTITY, AND WHY MORTALITY EVENTS ARE ENGINE-LOCAL
------------------------------------------------------
Per `last_hit_drive.py`'s own methodology note: the sim's per-slot index is a
recycled array slot, the server's NetId is not. This module never compares a
sim identity to a server identity. Sim identity for a death event is
`spawn_seq` (`LaneState.spawn_seq`, monotonic, never reused -- `sim/state.py`);
server identity is the wire's own NetId. Both are used only to deduplicate one
death within one engine's own run, never across engines; only the reported
COUNTS are compared.

CAUSE ATTRIBUTION, AND HOW MUCH OF IT IS EXACT
-----------------------------------------------
Both drivers already know their own oracle's `d.attack` target for the
decision in which a minion disappears, and both expose an independent CS
counter (`state.cs` / wire `cs`). A death is attributed to "champion" only
when BOTH agree: the disappearing minion's identity is the one the oracle
attacked THIS decision, AND the champion's own CS counter incremented THIS
decision. That is exact, not inferred. Every other death is either "turret"
(the dying minion's last observed position was within a live blue top-lane
turret's `attack_range` plus a fixed collision-radius margin -- an
approximation: the true server test is a circle-vs-circle acquisition query
with its own annulus subtlety, `TURRET-001`, not distance-to-centre) or
"other" (residual). On this scenario the champion holds at lane fraction
0.553, past its own outermost blue turret's 0.388, so "turret" is expected
to be small; "other" is not further decomposed into minion-vs-minion vs.
anything else and must not be read as a mechanism, only as a residual size.

HOW TO READ THE OUTPUT
-----------------------
``python -m lanerl_jax.parity.archive.reach_hp_mortality`` for a human-readable
report; import the ``run_*`` functions for a follow-up analysis. Always quote
`print_provenance()`'s digest next to any number taken from this module
(`METH-001`).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ...obs.fog import visible_to
from ...sim.init import ALL_TURRETS, TOP_LANE_PATH, init_lane, lane_params
from ...sim.orders import OrderKind, Orders, apply_orders
from ...sim.profiles import PROFILES
from ...sim.state import Kind, Team, TurretTier
from ...sim.step import step_decision
from ...sim.targeting import MinionType
from ..hp_band import BandSample, leveled_ad_of
from ..last_hit_drive import (APPROACH_WAYPOINTS, DECISIONS_600S,
                             WIRE_MINION_TYPE, _advance_approach,
                             gate3_route_inputs)
from ..last_hit_oracle import ChampView, MinionView, decide, post_mitigation

__all__ = [
    "MortalityEvent", "PositionStats", "SimReachRun", "ServerReachRun",
    "run_sim_reach", "run_server_reach", "hp_distribution",
    "hp_distribution_by_type", "mortality_summary", "mortality_by_turret_tier",
    "position_summary",
]

_SIM_MINION_TYPE_NAME = {
    MinionType.MELEE: "melee",
    MinionType.CASTER: "caster",
    MinionType.CANNON: "cannon",
    MinionType.SUPER: "super",
}

#: The 5 blue turrets within 900 units of TOP_LANE_PATH (see
#: `sim.init.ALL_TURRETS`'s own docstring for the cross-reference); the other
#: 7 blue turrets are mid/bot lane and provably inert here.
_BLUE_TOP_LANE_TURRET_TIERS = (
    TurretTier.OUTER, TurretTier.INHIBITOR, TurretTier.INNER, TurretTier.NEXUS,
)
#: Model name (from `data.patch.TURRET_MODELS`) that carries each blue
#: top-lane tier's `attack_range` -- "Order" is blue, matching the model list
#: comment in `data/patch.py` (OUTER, INNER, INHIBITOR, NEXUS in that order).
_BLUE_TURRET_MODEL_FOR_TIER = {
    TurretTier.OUTER: "OrderTurretNormal",
    TurretTier.INNER: "OrderTurretNormal2",
    TurretTier.INHIBITOR: "OrderTurretDragon",
    TurretTier.NEXUS: "OrderTurretAngel",
}
#: Fixed margin added to a turret's `attack_range` for the coarse "died near
#: a live blue turret" proximity test, covering the widest lane-minion
#: collision radius seen in `COLL-003` (cannon, 55.7437) rounded up.
_TURRET_PROXIMITY_MARGIN = 56.0


_TURRET_TIER_NAME = {
    TurretTier.OUTER: "outer", TurretTier.INNER: "inner",
    TurretTier.INHIBITOR: "inhibitor", TurretTier.NEXUS: "nexus",
}


def _blue_top_lane_turrets(patch) -> List[Tuple[float, float, float, str]]:
    """``(x, y, attack_range, tier_name)`` for the 5 live blue turrets near
    the top lane.

    Shared, unmodified, between the sim and server drivers below: these
    positions are themselves measured off the real server
    (`sim.init.ALL_TURRETS`'s own docstring), so there is no engine-specific
    version to keep in sync.
    """
    out = []
    for team, tx, ty, _hp, tier in ALL_TURRETS:
        if team != Team.BLUE or tier not in _BLUE_TOP_LANE_TURRET_TIERS:
            continue
        model = _BLUE_TURRET_MODEL_FOR_TIER[tier]
        stats = patch.turrets.get(model)
        if stats is None:
            continue
        out.append((tx, ty, float(stats.attack_range), _TURRET_TIER_NAME[tier]))
    return out


def _turret_cause(x: float, y: float,
                  turrets: List[Tuple[float, float, float, str]]) -> Optional[str]:
    """Nearest live blue turret's tier name if ``(x, y)`` is within its
    ``attack_range + margin``, else ``None``. Ties go to the first match in
    ``turrets``' own (nexus/inhib/inner/outer) order -- irrelevant in
    practice, since the 5 turrets' ranges do not overlap on this map."""
    for tx, ty, rng, tier in turrets:
        reach = rng + _TURRET_PROXIMITY_MARGIN
        if (x - tx) ** 2 + (y - ty) ** 2 <= reach * reach:
            return tier
    return None


@dataclass(slots=True, frozen=True)
class MortalityEvent:
    """One enemy (red) lane-minion death, engine-local identity only."""

    t_ms: float
    x: float
    y: float
    minion_type: str
    #: "champion" (exact, cross-checked against the CS counter), "turret"
    #: (approximate proximity test) or "other" (residual).
    cause: str
    identity: int = -1
    #: which blue turret tier, only set when ``cause == "turret"``.
    turret_tier: Optional[str] = None


@dataclass(slots=True)
class PositionStats:
    """Champion position summary over every post-handover decision."""

    n: int = 0
    _sx: float = 0.0
    _sy: float = 0.0
    min_x: float = float("inf")
    max_x: float = float("-inf")
    min_y: float = float("inf")
    max_y: float = float("-inf")

    def add(self, x: float, y: float) -> None:
        self.n += 1
        self._sx += x
        self._sy += y
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

    @property
    def mean_x(self) -> float:
        return self._sx / self.n if self.n else float("nan")

    @property
    def mean_y(self) -> float:
        return self._sy / self.n if self.n else float("nan")


@dataclass(slots=True, frozen=True)
class SimReachRun:
    samples: List[BandSample]
    mortality: List[MortalityEvent]
    position: PositionStats
    cs: int = 0
    decisions: int = 0
    approach_decisions: int = 0
    attacks: int = 0
    moves: int = 0
    holds: int = 0
    deaths: int = 0


@dataclass(slots=True, frozen=True)
class ServerReachRun:
    samples: List[BandSample]
    mortality: List[MortalityEvent]
    position: PositionStats
    log_path: Optional[Path] = None
    cs: int = 0
    decisions: int = 0
    approach_decisions: int = 0
    attacks: int = 0
    moves: int = 0
    holds: int = 0
    deaths: int = 0


def run_sim_reach(
    decisions: int = DECISIONS_600S,
    seed: int = 0,
    *,
    route_table=None,
    terrain=None,
    table_disabled: bool = False,
) -> SimReachRun:
    """Drive the gate-3 approach + oracle policy in the sim, instrumenting
    every enemy minion in reach (like `hp_band.run_sim_band`) PLUS every
    enemy-minion death and the champion's own position, in one pass.
    """
    from ...data.patch import load_patch

    patch = load_patch()
    base_ad = patch.champion.base_ad
    ad_per_level = patch.champion.ad_per_level
    blue_turrets = _blue_top_lane_turrets(patch)

    route_table, terrain = gate3_route_inputs(
        route_table=route_table, terrain=terrain, table_disabled=table_disabled)
    params_tbl = lane_params()
    params_np = {k: np.asarray(v) for k, v in params_tbl.items()}
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    state = init_lane(seed=seed)

    @jax.jit
    def _step(state, order_kind, order_x, order_y, order_target):
        orders = Orders(
            kind=jnp.array([order_kind, OrderKind.NOOP], dtype=jnp.int8),
            x=jnp.array([order_x, 0.0], dtype=state.x.dtype),
            y=jnp.array([order_y, 0.0], dtype=state.y.dtype),
            target=jnp.array([order_target, -1], dtype=jnp.int8),
        )
        return step_decision(apply_orders(state, orders, params_tbl,
                                          route_table=route_table,
                                          terrain=terrain), params_tbl,
                             lane_path=path)

    wp_idx = 0
    prev_alive = True
    samples: List[BandSample] = []
    mortality: List[MortalityEvent] = []
    position = PositionStats()
    approach_decisions = attacks = moves = holds = deaths = 0

    for _ in range(decisions):
        x0 = float(state.x[0])
        y0 = float(state.y[0])
        champ_alive = bool(state.alive[0])
        if prev_alive and not champ_alive:
            deaths += 1
        respawned = champ_alive and not prev_alive
        prev_alive = champ_alive
        wp_idx = _advance_approach(x0, y0, wp_idx, respawned)

        kind = np.asarray(state.kind)
        team = np.asarray(state.team)
        alive = np.asarray(state.alive)
        x = np.asarray(state.x)
        y = np.asarray(state.y)
        hp = np.asarray(state.hp)
        model = np.asarray(state.model)
        spawn_seq = np.asarray(state.spawn_seq)
        t_ms = float(np.asarray(state.t_ms))
        level0 = int(np.asarray(state.level)[0])
        cs_before = int(np.asarray(state.cs)[0])

        red_mask = (kind == Kind.LANE_MINION) & (team == Team.RED) & alive
        pre_live: Dict[int, Tuple[float, float, float, str]] = {
            int(spawn_seq[i]): (t_ms, float(x[i]), float(y[i]),
                                _SIM_MINION_TYPE_NAME[PROFILES[int(model[i])][1]])
            for i in np.flatnonzero(red_mask)
        }

        attacked_seq: Optional[int] = None
        if wp_idx < len(APPROACH_WAYPOINTS):
            approach_decisions += 1
            tx, ty = APPROACH_WAYPOINTS[wp_idx]
            state = _step(state, OrderKind.MOVE, tx, ty, -1)
        else:
            position.add(x0, y0)
            champ_ad = leveled_ad_of(base_ad, ad_per_level, level0)
            champ = ChampView(
                x=x0, y=y0, attack_damage=champ_ad,
                attack_range=float(params_np["attack_range"][model[0]]),
            )
            vis = np.asarray(visible_to(
                Team.BLUE, state.x, state.y, state.kind, state.team, state.alive))
            enemy = np.flatnonzero(red_mask & vis)
            for i in enemy:
                mx, my = float(x[i]), float(y[i])
                reach = champ.attack_range + float(params_np["collision_radius"][model[i]])
                if (mx - x0) ** 2 + (my - y0) ** 2 > reach * reach:
                    continue
                m_hp = float(hp[i])
                m_armor = float(params_np["armor"][model[i]])
                dmg = post_mitigation(champ_ad, m_armor)
                samples.append(BandSample(
                    t_ms=t_ms, hp=m_hp, armor=m_armor, in_band=m_hp <= dmg,
                    champ_ad=champ_ad, champ_level=level0, champ_x=x0, champ_y=y0,
                    minion_x=mx, minion_y=my, minion_id=int(spawn_seq[i]),
                    minion_type=_SIM_MINION_TYPE_NAME[PROFILES[int(model[i])][1]]))

            minions = [
                MinionView(
                    uid=int(i), x=float(x[i]), y=float(y[i]), hp=float(hp[i]),
                    armor=float(params_np["armor"][model[i]]),
                    collision_radius=float(params_np["collision_radius"][model[i]]),
                )
                for i in enemy
            ]
            d = decide(champ, minions, lethal_epsilon=0.0)
            if d.attack is not None:
                attacks += 1
                attacked_seq = int(spawn_seq[d.attack])
                state = _step(state, OrderKind.ATTACK, 0.0, 0.0, d.attack)
            elif d.move is not None:
                moves += 1
                mx2, my2 = d.move
                state = _step(state, OrderKind.MOVE, mx2, my2, -1)
            else:
                holds += 1
                state = _step(state, OrderKind.NOOP, 0.0, 0.0, -1)

        kind2 = np.asarray(state.kind)
        team2 = np.asarray(state.team)
        alive2 = np.asarray(state.alive)
        spawn_seq2 = np.asarray(state.spawn_seq)
        post_live_ids = set(
            int(s) for s in spawn_seq2[(kind2 == Kind.LANE_MINION)
                                       & (team2 == Team.RED) & alive2])
        cs_delta = int(np.asarray(state.cs)[0]) - cs_before
        for sid, (lt, lx, ly, ltype) in pre_live.items():
            if sid in post_live_ids:
                continue
            tier = None
            if attacked_seq is not None and sid == attacked_seq and cs_delta >= 1:
                cause = "champion"
            elif (tier := _turret_cause(lx, ly, blue_turrets)) is not None:
                cause = "turret"
            else:
                cause = "other"
            mortality.append(MortalityEvent(t_ms=lt, x=lx, y=ly,
                                            minion_type=ltype, cause=cause,
                                            identity=sid, turret_tier=tier))

    cs = int(np.asarray(state.cs)[0])
    return SimReachRun(samples=samples, mortality=mortality, position=position,
                       cs=cs, decisions=decisions,
                       approach_decisions=approach_decisions, attacks=attacks,
                       moves=moves, holds=holds, deaths=deaths)


def run_server_reach(
    decisions: int = DECISIONS_600S,
    port_base: int = 44500,
    bot_seed: int = 4242,
    tag: str = "reach_mortality",
    log_dir: Optional[Path] = None,
    autobuy: bool = False,
) -> ServerReachRun:
    """Same approach + oracle policy as `run_sim_reach`, on the real server."""
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    from ...data.patch import load_patch

    patch = load_patch()
    blue_turrets = _blue_top_lane_turrets(patch)
    log_dir = Path(log_dir) if log_dir is not None else Path(
        tempfile.mkdtemp(prefix=f"{tag}_"))

    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
            toponly=True, bot_teams="none", bot_seed=bot_seed, step_ticks=2,
            extra_env={"LANERL_AUTOBUY": "1" if autobuy else "0"}),
        log_dir=log_dir,
        ports=PortAllocator(base=port_base).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    wp_idx = 0
    prev_alive = True
    samples: List[BandSample] = []
    mortality: List[MortalityEvent] = []
    position = PositionStats()
    log_path: Optional[Path] = None
    approach_decisions = attacks = moves = holds = deaths = 0
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        log_path = Path(env.handles[0].log_path)
        for i in range(decisions):
            obs = env.last_obs[0]
            if obs is None:
                raise RuntimeError(f"no observation at decision {i}")
            units = obs.get("u", [])
            blue = next(
                u for u in units if u.get("k") == "Champion" and u.get("tm") == 100)
            bx, by = float(blue["x"]), float(blue["y"])
            champ_alive = float(blue.get("hp", 1)) > 0
            if prev_alive and not champ_alive:
                deaths += 1
            respawned = champ_alive and not prev_alive
            prev_alive = champ_alive
            wp_idx = _advance_approach(bx, by, wp_idx, respawned)

            t_ms = float(obs.get("t", 0.0))
            cs_before = int(blue.get("cs", 0))
            # Full census, NOT `vb`-gated: obs carries every live unit with a
            # per-team visibility flag ("vb"/"vr" -- see `lanerl_train.
            # protocols`'s wire-field table), so a red minion missing from
            # `units` next decision is a real removal, not a fog-out.
            pre_live: Dict[int, Tuple[float, float, float, str]] = {
                int(u["id"]): (t_ms, float(u["x"]), float(u["y"]),
                               WIRE_MINION_TYPE.get(int(u.get("mt", 0)), "melee"))
                for u in units
                if u.get("k") == "LaneMinion" and u.get("tm") == 200
            }

            attacked_id: Optional[int] = None
            if wp_idx < len(APPROACH_WAYPOINTS):
                approach_decisions += 1
                tx, ty = APPROACH_WAYPOINTS[wp_idx]
                env.step([{"blue": {"t": "move", "x": tx, "y": ty}}])
            else:
                position.add(bx, by)
                champ_ad = float(blue["ad"])
                champ_rng = float(blue["rng"])

                minions = []
                for u in units:
                    if u.get("k") != "LaneMinion" or u.get("tm") != 200:
                        continue
                    if not u.get("vb", 0):        # visible to blue only
                        continue
                    key = WIRE_MINION_TYPE.get(int(u.get("mt", 0)), "melee")
                    stat = patch.minions[f"{key}_red"]
                    mx, my = float(u["x"]), float(u["y"])
                    reach = champ_rng + float(stat.collision_radius)
                    m_hp = float(u["hp"])
                    if (mx - bx) ** 2 + (my - by) ** 2 <= reach * reach:
                        dmg = post_mitigation(champ_ad, float(stat.armor))
                        samples.append(BandSample(
                            t_ms=t_ms, hp=m_hp, armor=float(stat.armor),
                            in_band=m_hp <= dmg, champ_ad=champ_ad,
                            champ_level=int(blue.get("lvl", 1)), champ_x=bx,
                            champ_y=by, minion_x=mx, minion_y=my,
                            minion_id=int(u["id"]), minion_type=key))
                    minions.append(MinionView(
                        uid=int(u["id"]), x=mx, y=my, hp=m_hp,
                        armor=float(stat.armor),
                        collision_radius=float(stat.collision_radius)))

                champ = ChampView(x=bx, y=by, attack_damage=champ_ad,
                                  attack_range=champ_rng)
                d = decide(champ, minions, lethal_epsilon=0.0)
                if d.attack is not None:
                    attacks += 1
                    attacked_id = d.attack
                    act = {"t": "attack", "id": d.attack}
                elif d.move is not None:
                    moves += 1
                    mx2, my2 = d.move
                    act = {"t": "move", "x": mx2, "y": my2}
                else:
                    holds += 1
                    act = {"t": "noop"}
                env.step([{"blue": act}])

            obs2 = env.last_obs[0]
            units2 = obs2.get("u", []) if obs2 is not None else []
            blue2 = next(
                (u for u in units2 if u.get("k") == "Champion" and u.get("tm") == 100),
                None)
            post_live_ids = {
                int(u["id"]) for u in units2
                if u.get("k") == "LaneMinion" and u.get("tm") == 200
            }
            cs_after = int(blue2.get("cs", 0)) if blue2 is not None else cs_before
            cs_delta = cs_after - cs_before
            for nid, (lt, lx, ly, ltype) in pre_live.items():
                if nid in post_live_ids:
                    continue
                tier = None
                if attacked_id is not None and nid == attacked_id and cs_delta >= 1:
                    cause = "champion"
                elif (tier := _turret_cause(lx, ly, blue_turrets)) is not None:
                    cause = "turret"
                else:
                    cause = "other"
                mortality.append(MortalityEvent(t_ms=lt, x=lx, y=ly,
                                                minion_type=ltype, cause=cause,
                                                identity=nid, turret_tier=tier))

        obs = env.last_obs[0]
        blue = next(u for u in obs["u"] if u.get("k") == "Champion" and u.get("tm") == 100)
        cs = int(blue.get("cs", 0))
    finally:
        env.close()

    return ServerReachRun(samples=samples, mortality=mortality, position=position,
                          log_path=log_path, cs=cs, decisions=decisions,
                          approach_decisions=approach_decisions, attacks=attacks,
                          moves=moves, holds=holds, deaths=deaths)


def hp_distribution(name: str, samples: List[BandSample]) -> str:
    """Full distribution of "in reach, above lethal" HP -- NOT a median.

    `docs/JAX_FIDELITY_LEDGER.md`'s gate-3 row records exactly one median
    hiding a 45-decision outlier ("21.5 attack-decisions per kill"); this
    function exists so that mistake cannot repeat here.
    """
    above = [s.hp for s in samples if not s.in_band]
    n_total = len(samples)
    n_above = len(above)
    if n_total == 0:
        return f"{name}: no in-reach samples"
    hp = np.array(above) if above else np.array([np.nan])
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    pcts = np.percentile(hp, qs) if above else [float("nan")] * len(qs)
    # Coarse 10-bin histogram over the observed range, printed as counts --
    # cheap bimodality check without pulling in a plotting dependency.
    hist_line = ""
    if above:
        lo, hi = float(hp.min()), float(hp.max())
        if hi > lo:
            edges = np.linspace(lo, hi, 11)
            counts, _ = np.histogram(hp, bins=edges)
            hist_line = " ".join(f"{c}" for c in counts)
        else:
            hist_line = f"all at {lo:.1f}"
    lines = [
        f"{name}: {n_total} in-reach decisions, {n_above} too healthy "
        f"({100 * n_above / n_total:.1f}%), {n_total - n_above} in-band",
        f"  above-lethal HP: mean={hp.mean():.1f} std={hp.std():.1f}",
        "  quantiles (" + ",".join(f"p{q}" for q in qs) + "): "
        + " ".join(f"{p:.1f}" for p in pcts),
        f"  10-bin histogram, {float(hp.min()):.1f}..{float(hp.max()):.1f}: {hist_line}"
        if above else "  (no above-lethal samples)",
    ]
    return "\n".join(lines)


def hp_distribution_by_type(name: str, samples: List[BandSample]) -> str:
    """Same as :func:`hp_distribution`, split by minion type, plus a
    "near max HP" fraction -- a cheap fresh-spawn proxy, since a minion at or
    near the highest HP this engine's own sample set ever saw for its type is
    a minion that has taken no damage yet, not one mid-fight."""
    by_type: Dict[str, List[float]] = {}
    for s in samples:
        if s.in_band:
            continue
        by_type.setdefault(s.minion_type, []).append(s.hp)
    lines = [f"{name}, by type (above-lethal only):"]
    for t, vals in sorted(by_type.items()):
        arr = np.array(vals)
        near_max = float((arr >= 0.95 * arr.max()).mean())
        lines.append(
            f"  {t}: n={len(arr)} mean={arr.mean():.1f} p10={np.percentile(arr,10):.1f} "
            f"p50={np.percentile(arr,50):.1f} p90={np.percentile(arr,90):.1f} "
            f"max={arr.max():.1f} frac_near_max={near_max:.2f}")
    return "\n".join(lines)


def mortality_by_turret_tier(name: str, events: List[MortalityEvent]) -> str:
    turret_events = [e for e in events if e.cause == "turret"]
    if not turret_events:
        return f"{name}: no turret-attributed deaths"
    by_tier: Dict[str, int] = {}
    for e in turret_events:
        by_tier[e.turret_tier or "?"] = by_tier.get(e.turret_tier or "?", 0) + 1
    return (f"{name}: {len(turret_events)} turret-attributed deaths by tier: "
           + ", ".join(f"{k}={v}" for k, v in sorted(by_tier.items())))


def position_summary(name: str, pos: PositionStats) -> str:
    if pos.n == 0:
        return f"{name}: no post-handover decisions"
    return (f"{name}: n={pos.n} mean=({pos.mean_x:.1f},{pos.mean_y:.1f}) "
           f"x=[{pos.min_x:.1f},{pos.max_x:.1f}] y=[{pos.min_y:.1f},{pos.max_y:.1f}]")


def mortality_summary(name: str, events: List[MortalityEvent]) -> str:
    n = len(events)
    if n == 0:
        return f"{name}: 0 enemy-minion deaths"
    by_cause: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for e in events:
        by_cause[e.cause] = by_cause.get(e.cause, 0) + 1
        by_type[e.minion_type] = by_type.get(e.minion_type, 0) + 1
    cause_str = ", ".join(f"{k}={v} ({100*v/n:.1f}%)" for k, v in sorted(by_cause.items()))
    type_str = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
    return f"{name}: {n} enemy-minion deaths -- cause: {cause_str} -- type: {type_str}"


def _main() -> None:
    import argparse

    from ..provenance import print_provenance

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=DECISIONS_600S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot-seed", type=int, default=4242)
    ap.add_argument("--skip-server", action="store_true")
    ap.add_argument("--table-disabled", action="store_true")
    ap.add_argument("--autobuy", action="store_true")
    ap.add_argument("--log-dir", type=str, default=None)
    args = ap.parse_args()

    print_provenance()

    print("Running sim...")
    sim = run_sim_reach(decisions=args.decisions, seed=args.seed,
                        table_disabled=args.table_disabled)
    print(f"sim gate numbers: cs={sim.cs} approach_decisions={sim.approach_decisions} "
          f"attacks={sim.attacks} moves={sim.moves} holds={sim.holds} "
          f"deaths={sim.deaths}")
    print(hp_distribution("sim", sim.samples))
    print(hp_distribution_by_type("sim", sim.samples))
    print(position_summary("sim champion position", sim.position))
    print(mortality_summary("sim", sim.mortality))
    print(mortality_by_turret_tier("sim", sim.mortality))

    if not args.skip_server:
        print("Running server (boots a real process; several minutes)...")
        server = run_server_reach(decisions=args.decisions, bot_seed=args.bot_seed,
                                  autobuy=args.autobuy, log_dir=args.log_dir)
        print(f"server gate numbers: cs={server.cs} "
              f"approach_decisions={server.approach_decisions} "
              f"attacks={server.attacks} moves={server.moves} "
              f"holds={server.holds} deaths={server.deaths}")
        print(hp_distribution("server", server.samples))
        print(hp_distribution_by_type("server", server.samples))
        print(position_summary("server champion position", server.position))
        print(mortality_summary("server", server.mortality))
        print(mortality_by_turret_tier("server", server.mortality))
        print(f"server log: {server.log_path}")


if __name__ == "__main__":
    _main()
