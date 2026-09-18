"""J1 gate 3 / lane equilibrium: is the champion standing inside its own wave?

WHY THIS EXISTS
----------------
``lanerl_jax/parity/tests/test_last_hit_gate.py`` measures the sim's
champion dying 5 times to the server's 0-1 under an identical, non-chasing,
hold-position oracle policy. A direct, source-derived test of the release
rule (:mod:`lanerl_jax.sim.minion_ai`, ``test_a_minion_holding_the_idle_
champion_is_NOT_displaced_by_a_fresh_minion``) confirmed the sim's
target-acquisition/incumbency logic already matches ``LaneMinionAI.cs``
exactly -- a minion that has validly acquired the champion is not released
by mere proximity of a fresh candidate, on the server either. So a pile of
minions on the champion is not, by itself, a targeting bug: it is what the
model produces once those minions have nothing else worth switching to
nearby. That reframes the death gap as a question about WHERE the champion
ends up relative to its own wave, not about targeting rules -- which is
what this module measures, in both engines, over the identical scripted
scenario :mod:`lanerl_jax.parity.last_hit_drive` already drives.

WHAT IT FOUND
--------------
Measured 2026-09-16, 18,000 decisions, same seed/config as the gate::

                                sim      server    ratio
    0 allies within 1500 u    32.3%     19.4%      1.7x
    nearest-ally > 3000 u     13.1%      7.0%      1.9x
    mean nearest-ally dist     1263       874      +45%
    median nearest-ally dist    280       306      ~equal

**The medians agreeing while the tails diverge is the informative part.**
The champion's TYPICAL position relative to its own wave is right in both
engines -- when he has an ally nearby at all, it is about as close in the
sim as on the server. What differs is the TAIL: a meaningfully larger share
of sim decisions find him with no ally within 1500 units, or none within
3000. That reads as this port's lane equilibrium swinging wider than the
server's (waves separating further, more often) rather than as the
champion being parked in a structurally different spot -- both drivers
walk him to the identical ``APPROACH_WAYPOINTS`` coordinate, so the
DESTINATION is not in question, only how often the wave has moved away
from it by the time he needs it.

**What this does and does not establish.** That the sim's champion is
measurably more often isolated from its own wave, while the (separately,
directly, source-verified) release rule cannot free a minion locked onto
him without a nearby ally to switch to, is *consistent with* isolation
driving the excess deaths -- more isolated decisions is more opportunity
for an unrecoverable lock-on to accumulate. It does not, by itself, prove
causation: this module does not trace any specific death back to a specific
isolated stretch, and does not rule out other contributors. Read it as a
real, measured tail effect with a plausible causal story attached, not as
a closed case.

**Handoff, not a next step for this module.** The mechanism behind a wider
equilibrium is very likely lane-equilibrium/collision separation --
``sim/collision.py`` applies one push-apart per unit per tick from a
pre-tick snapshot where the server resolves collisions sequentially,
several pushes per unit per tick, each visible to the next (see that
module's own booked-approximation note). That is gate 1's territory (the
sequential-collision port is in flight there); this module's job stops at
making the tail effect measurable and reproducible, not at diagnosing its
root cause.

HOW TO READ THE OUTPUT
-----------------------
Run as a script for a human-readable report; import the ``run_*`` functions
for a test or a follow-up analysis. ``python -m lanerl_jax.parity.isolation``.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .last_hit_drive import DECISIONS_600S, APPROACH_WAYPOINTS, WIRE_MINION_TYPE, _advance_approach
from .last_hit_oracle import ChampView, MinionView, decide

__all__ = ["IsolationRun", "run_sim_isolation", "run_server_isolation", "summarize"]

#: "nearby" radius for the allies/enemies-within count, chosen to be a few
#: multiples of a minion's own acquisition range (600) -- wide enough to ask
#: "is there a wave here at all" without being so wide it always says yes.
NEARBY_RADIUS = 1500.0
#: a second, wider radius for the "meaningfully isolated" tail check.
FAR_RADIUS = 3000.0


@dataclass(slots=True, frozen=True)
class IsolationRun:
    """Per-post-handover-decision samples. Same shape and meaning on both
    engines, so :func:`summarize` reads identically for either.
    """

    #: distance to the nearest LIVE ALLIED (same-team) minion, or ``inf`` if
    #: no allied minion is alive anywhere on the map that decision.
    nearest_ally: List[float]
    #: count of allied minions within :data:`NEARBY_RADIUS`.
    allies_nearby: List[int]
    #: count of enemy minions within :data:`NEARBY_RADIUS`.
    enemies_nearby: List[int]


def run_sim_isolation(decisions: int = DECISIONS_600S, seed: int = 0) -> IsolationRun:
    """Drive the same approach + oracle policy as
    :func:`lanerl_jax.parity.last_hit_drive.run_oracle_in_sim`, recording the
    champion's distance to its own (blue) wave each post-handover decision
    instead of counting attacks. Does not change the policy's behaviour.
    """
    import jax
    import jax.numpy as jnp

    from ..obs.fog import visible_to
    from ..sim.init import TOP_LANE_PATH, init_lane, lane_params
    from ..sim.orders import OrderKind, Orders, apply_orders
    from ..sim.state import Kind, Team
    from ..sim.step import step_decision

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
        return step_decision(apply_orders(state, orders, params_tbl), params_tbl, lane_path=path)

    wp_idx = 0
    prev_alive = True
    nearest_ally: List[float] = []
    allies_nearby: List[int] = []
    enemies_nearby: List[int] = []

    for _ in range(decisions):
        x0 = float(state.x[0])
        y0 = float(state.y[0])
        champ_alive = bool(state.alive[0])
        respawned = champ_alive and not prev_alive
        prev_alive = champ_alive
        wp_idx = _advance_approach(x0, y0, wp_idx, respawned)

        if wp_idx < len(APPROACH_WAYPOINTS):
            tx, ty = APPROACH_WAYPOINTS[wp_idx]
            state = _step(state, OrderKind.MOVE, tx, ty, -1)
            continue

        kind = np.asarray(state.kind)
        team = np.asarray(state.team)
        alive = np.asarray(state.alive)
        x = np.asarray(state.x)
        y = np.asarray(state.y)
        model = np.asarray(state.model)
        hp = np.asarray(state.hp)

        ally = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.BLUE) & alive)
        if len(ally) > 0:
            d = np.sqrt((x[ally] - x0) ** 2 + (y[ally] - y0) ** 2)
            nearest_ally.append(float(d.min()))
            allies_nearby.append(int((d <= NEARBY_RADIUS).sum()))
        else:
            nearest_ally.append(float("inf"))
            allies_nearby.append(0)

        enemy_all = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.RED) & alive)
        if len(enemy_all) > 0:
            de = np.sqrt((x[enemy_all] - x0) ** 2 + (y[enemy_all] - y0) ** 2)
            enemies_nearby.append(int((de <= NEARBY_RADIUS).sum()))
        else:
            enemies_nearby.append(0)

        champ = ChampView(
            x=x0, y=y0,
            attack_damage=float(params_np["attack_damage"][model[0]]),
            attack_range=float(params_np["attack_range"][model[0]]),
        )
        vis = np.asarray(visible_to(
            Team.BLUE, state.x, state.y, state.kind, state.team, state.alive))
        enemy_vis = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.RED)
                                   & alive & vis)
        minions = [
            MinionView(
                uid=int(j), x=float(x[j]), y=float(y[j]), hp=float(hp[j]),
                armor=float(params_np["armor"][model[j]]),
                collision_radius=float(params_np["collision_radius"][model[j]]),
            )
            for j in enemy_vis
        ]
        d = decide(champ, minions, lethal_epsilon=0.0)
        if d.attack is not None:
            state = _step(state, OrderKind.ATTACK, 0.0, 0.0, d.attack)
        else:
            state = _step(state, OrderKind.NOOP, 0.0, 0.0, -1)

    return IsolationRun(nearest_ally=nearest_ally, allies_nearby=allies_nearby,
                        enemies_nearby=enemies_nearby)


def run_server_isolation(
    decisions: int = DECISIONS_600S,
    port_base: int = 44950,
    bot_seed: int = 4242,
    autobuy: bool = False,
    log_dir: Optional[Path] = None,
) -> IsolationRun:
    """Same approach + oracle policy as
    :func:`lanerl_jax.parity.last_hit_drive.run_oracle_on_server`, recording
    the champion's distance to its own wave each post-handover decision.
    ``autobuy`` defaults off for the same reason as
    :func:`~lanerl_jax.parity.hp_band.run_server_band`: this measures wave
    position, which a free starting item does not touch, so it should not be
    a silent confound either way.
    """
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    from ..data.patch import load_patch

    patch = load_patch()
    log_dir = Path(log_dir) if log_dir is not None else Path(
        tempfile.mkdtemp(prefix="isolation_"))

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
    nearest_ally: List[float] = []
    allies_nearby: List[int] = []
    enemies_nearby: List[int] = []
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        for i in range(decisions):
            obs = env.last_obs[0]
            if obs is None:
                raise RuntimeError(f"no observation at decision {i}")
            units = obs.get("u", [])
            blue = next(
                u for u in units if u.get("k") == "Champion" and u.get("tm") == 100)
            bx, by = float(blue["x"]), float(blue["y"])
            champ_alive = float(blue.get("hp", 1)) > 0
            respawned = champ_alive and not prev_alive
            prev_alive = champ_alive
            wp_idx = _advance_approach(bx, by, wp_idx, respawned)

            if wp_idx < len(APPROACH_WAYPOINTS):
                tx, ty = APPROACH_WAYPOINTS[wp_idx]
                env.step([{"blue": {"t": "move", "x": tx, "y": ty}}])
                continue

            ax = [float(u["x"]) for u in units
                 if u.get("k") == "LaneMinion" and u.get("tm") == 100]
            ay = [float(u["y"]) for u in units
                 if u.get("k") == "LaneMinion" and u.get("tm") == 100]
            if ax:
                d = np.sqrt((np.array(ax) - bx) ** 2 + (np.array(ay) - by) ** 2)
                nearest_ally.append(float(d.min()))
                allies_nearby.append(int((d <= NEARBY_RADIUS).sum()))
            else:
                nearest_ally.append(float("inf"))
                allies_nearby.append(0)

            ex = [float(u["x"]) for u in units
                 if u.get("k") == "LaneMinion" and u.get("tm") == 200]
            ey = [float(u["y"]) for u in units
                 if u.get("k") == "LaneMinion" and u.get("tm") == 200]
            if ex:
                de = np.sqrt((np.array(ex) - bx) ** 2 + (np.array(ey) - by) ** 2)
                enemies_nearby.append(int((de <= NEARBY_RADIUS).sum()))
            else:
                enemies_nearby.append(0)

            champ_ad = float(blue["ad"])
            champ_rng = float(blue["rng"])
            minions = []
            for u in units:
                if u.get("k") != "LaneMinion" or u.get("tm") != 200:
                    continue
                if not u.get("vb", 0):
                    continue
                key = WIRE_MINION_TYPE.get(int(u.get("mt", 0)), "melee")
                stat = patch.minions[f"{key}_red"]
                minions.append(MinionView(
                    uid=int(u["id"]), x=float(u["x"]), y=float(u["y"]),
                    hp=float(u["hp"]), armor=float(stat.armor),
                    collision_radius=float(stat.collision_radius)))

            champ = ChampView(x=bx, y=by, attack_damage=champ_ad, attack_range=champ_rng)
            d = decide(champ, minions, lethal_epsilon=0.0)
            if d.attack is not None:
                env.step([{"blue": {"t": "attack", "id": d.attack}}])
            else:
                env.step([{"blue": {"t": "noop"}}])
    finally:
        env.close()

    return IsolationRun(nearest_ally=nearest_ally, allies_nearby=allies_nearby,
                        enemies_nearby=enemies_nearby)


def summarize(name: str, run: IsolationRun) -> str:
    na = np.array(run.nearest_ally)
    fin = na[np.isfinite(na)]
    allies = np.array(run.allies_nearby)
    enemies = np.array(run.enemies_nearby)
    lines = [
        f"{name}: {len(na)} post-handover decisions, "
        f"no-ally-alive-anywhere frac={np.mean(~np.isfinite(na)):.3f}",
        f"  nearest-ally distance (when >=1 ally alive): n={len(fin)} "
        f"mean={fin.mean():.1f} median={np.median(fin):.1f} "
        f"frac>{NEARBY_RADIUS:.0f}={np.mean(fin > NEARBY_RADIUS):.3f} "
        f"frac>{FAR_RADIUS:.0f}={np.mean(fin > FAR_RADIUS):.3f}",
        f"  mean allies within {NEARBY_RADIUS:.0f}: {allies.mean():.2f}  "
        f"mean enemies within {NEARBY_RADIUS:.0f}: {enemies.mean():.2f}  "
        f"frac decisions with 0 allies within {NEARBY_RADIUS:.0f}: "
        f"{np.mean(allies == 0):.3f}",
    ]
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=DECISIONS_600S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot-seed", type=int, default=4242)
    ap.add_argument("--skip-server", action="store_true")
    ap.add_argument("--autobuy", action="store_true")
    ap.add_argument("--log-dir", type=str, default=None)
    args = ap.parse_args()

    print("Running sim...")
    sim = run_sim_isolation(decisions=args.decisions, seed=args.seed)
    print(summarize("sim", sim))

    if not args.skip_server:
        print("Running server (boots a real process; several minutes)...")
        server = run_server_isolation(decisions=args.decisions, bot_seed=args.bot_seed,
                                      autobuy=args.autobuy, log_dir=args.log_dir)
        print(summarize("server", server))


if __name__ == "__main__":
    _main()
