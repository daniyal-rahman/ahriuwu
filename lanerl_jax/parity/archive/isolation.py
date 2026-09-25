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
Re-measured 2026-09-18, 18,000 decisions, same seed/config as the gate, with
BOTH sides re-run the same day. This supersedes the 2026-09-16 table, which was
stale on two counts: it drove the sim on the raw two-point path while the gate
ran routed, and it predates the sequential-collision port::

                                    sim      server     ratio
    post-handover decisions       11,600    14,803      sim has 21.6% less
    no ally alive ANYWHERE         11.4%     10.2%      1.12x
    0 allies within 1500 u         23.8%     19.4%      1.23x  (was 1.7x)
    nearest-ally > 3000 u           9.8%      7.0%      1.40x  (was 1.9x)
    nearest-ally > 1500 u          14.0%     10.2%      1.37x
    mean nearest-ally dist         1143.5     874.4     +31%   (was +45%)
    median nearest-ally dist        395.6     305.9     +29%   (was ~equal)
    mean allies within 1500          3.93      4.28     -8%
    mean enemies within 1500         4.12      4.35     -5%

**Three things changed, and one of them reverses this module's own reading.**

1. *The tail effect is real but roughly half as large.* 1.7x -> 1.23x and
   1.9x -> 1.40x. Routing plus sequential collision closed a lot of it. It did
   not close all of it: the sim's champion is still measurably more isolated.

2. *The medians no longer agree, so "typical position is right, only the tail
   is wrong" is no longer true.* The old headline rested on sim 280 against
   server 306. Under the configuration the gate actually runs, it is 395.6
   against 305.9 -- the whole distribution is shifted, not just its tail. Note
   which direction that is: the RAW path's median was the one that agreed.
   A two-point order cuts straight through terrain, so the old agreement may
   well have been a wrong path landing on a right-looking number, which is
   why a diagnostic must run the same mode as the gate it explains.

3. *The wave is NOT dying more often in the sim.* 11.4% of the sim's in-lane
   decisions have no allied minion alive anywhere on the map, against the
   server's 10.2%. That is close enough to rule out wave survival as the
   story, and it is worth stating as a negative because the sim's raw 11.4%
   looks alarming until the baseline is put next to it.

**The larger effect is not in this table at all.** The sim gets 11,600
post-handover decisions to the server's 14,803 because it spends 6,400 of
18,000 walking in against the server's 3,197 (and dies twice to the server's
once). Roughly 22% less time in lane dwarfs a 1.2x isolation ratio as an
explanation for missing CS, and it points at PATH-001's waypoint emission --
local routes carry 6-9 waypoints where the server carries 3. See the
cross-gate chain in ``docs/JAX_FIDELITY_LEDGER.md``.

**What this does and does not establish.** That the sim's champion is
measurably more often isolated from its own wave, while the (separately,
directly, source-verified) release rule cannot free a minion locked onto
him without a nearby ally to switch to, is *consistent with* isolation
contributing to the excess deaths. It does not, by itself, prove causation:
this module does not trace any specific death back to a specific isolated
stretch, and does not rule out other contributors.

**The 2026-09-16 handoff was taken up, and it was about half right.** That
note predicted the wider equilibrium came from collision separation --
``sim/collision.py`` then applied one push-apart per unit per tick from a
pre-tick snapshot, where the server resolves sequentially, several pushes per
unit per tick each visible to the next -- and handed it to gate 1, where the
sequential port was in flight. It has since landed. The tail ratios roughly
halved (1.7x -> 1.23x, 1.9x -> 1.40x), which is consistent with that having
been a real contributor, though this re-measurement changed routing at the
same time and so cannot apportion the improvement between the two.

What remains is no longer mainly an equilibrium question. The residual
isolation is modest, the medians have separated, and the dominant term is
lane TIME rather than lane POSITION: the sim simply gets 22% fewer
post-handover decisions. That belongs to PATH-001's waypoint emission, not
here.

HOW TO READ THE OUTPUT
-----------------------
Run as a script for a human-readable report; import the ``run_*`` functions
for a test or a follow-up analysis. ``python -m lanerl_jax.parity.archive.isolation``.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .last_hit_drive import (DECISIONS_600S, APPROACH_WAYPOINTS, WIRE_MINION_TYPE,
                             _advance_approach, gate3_route_inputs)
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


def run_sim_isolation(decisions: int = DECISIONS_600S, seed: int = 0, *,
                      route_table=None, terrain=None,
                      table_disabled: bool = False) -> IsolationRun:
    """Drive the same approach + oracle policy as
    :func:`lanerl_jax.parity.last_hit_drive.run_oracle_in_sim`, recording the
    champion's distance to its own (blue) wave each post-handover decision
    instead of counting attacks. Does not change the policy's behaviour.

    Routing defaults to production v2 through the SAME
    :func:`~lanerl_jax.parity.last_hit_drive.gate3_route_inputs` the gate uses.
    That matters more here than anywhere else: this module exists to explain
    the gate's death gap, and until now it ran the two-point raw path while
    the gate itself ran routed, so it was explaining a run that no longer
    happened.  Where the champion ends up relative to its own wave is
    downstream of how it walks, so a raw-path isolation figure is not evidence
    about a routed gate -- it is the booked PATH-006 mismatch.  Pass
    ``table_disabled=True`` for the named raw ablation.
    """
    import jax
    import jax.numpy as jnp

    from ...obs.fog import visible_to
    from ...sim.init import TOP_LANE_PATH, init_lane, lane_params
    from ...sim.orders import OrderKind, Orders, apply_orders
    from ...sim.state import Kind, Team
    from ...sim.step import step_decision

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
                                         terrain=terrain),
                             params_tbl, lane_path=path)

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

    from ...data.patch import load_patch

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
    ap.add_argument("--table-disabled", action="store_true",
                    help="run the sim side on the PATH-001 two-point raw path. "
                         "An ablation, not a baseline: the gate runs routed, so "
                         "a raw figure here does not explain the gate.")
    args = ap.parse_args()

    print(f"Running sim ({'RAW two-point ablation' if args.table_disabled else 'routed v2, same as the gate'})...")
    sim = run_sim_isolation(decisions=args.decisions, seed=args.seed,
                            table_disabled=args.table_disabled)
    print(summarize("sim", sim))

    if not args.skip_server:
        print("Running server (boots a real process; several minutes)...")
        server = run_server_isolation(decisions=args.decisions, bot_seed=args.bot_seed,
                                      autobuy=args.autobuy, log_dir=args.log_dir)
        print(summarize("server", server))


if __name__ == "__main__":
    _main()
