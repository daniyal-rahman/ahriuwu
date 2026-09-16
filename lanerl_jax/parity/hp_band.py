"""J1 gate 3: instrument the one-shot HP band, in both engines.

WHY THIS EXISTS
----------------
``docs/JAX_REWRITE_PLAN.md``'s J1 status (2026-09-16) frames the remaining
gate 3 gap like this: the tick reorder moved attack *opportunities* 109 -> 163
without moving CS, the server gets 535, so "the remaining gap is how often a
killable minion appears in reach ... i.e. minion HP trajectories, not
last-hitting." That is a hypothesis about *where* the missing opportunities
come from, not a measurement -- nobody had yet looked at the HP band itself.
This module is that measurement: for both engines, over the identical scripted
scenario :mod:`lanerl_jax.parity.last_hit_drive` already drives, record every
enemy minion's HP while it sits within Garen's reach, and how much of that
time is inside the one-shot band
``hp <= post_mitigation(garen_ad, minion_armor)``.

WHAT IT FOUND (read before trusting "minion HP trajectories")
----------------------------------------------------------------
Three things, in the order they were checked -- and the headline result is
that the ORIGINAL FRAMING WAS WRONG, not merely incomplete. The "163 vs 535
attack opportunities" comparison this module was built to explain does not
reproduce on the current codebase at all: traced by commit timestamp, that
535 figure was measured 37 minutes before a fix (``d463533``) that stopped
the champion from being scripted to stand inside the enemy turret's attack
range, and nobody re-measured the server baseline afterward. Freshly
measured, on the corrected position, three times, on both a contended and an
uncontended node: **sim gets MORE attacks and MORE CS than the server**, the
opposite direction (sim cs=9/attacks=473, server cs=4/attacks=86 -- see
``lanerl_jax/parity/tests/test_last_hit_gate.py`` for the full numbers and
the commit-timestamp evidence). ``docs/JAX_REWRITE_PLAN.md``'s J1 status
needs correcting to match.

**Champion AD not level-scaling was real, but small here.** ``post_mitigation``
takes Garen's *current* attack damage, and it is supposed to grow with level
(``Stats.LevelUp``, ``Stats.cs:270-271``). The sim baked it into a static
per-profile value instead, ignoring ``state.level`` entirely -- fixed in
``sim/step.py``/``sim/profiles.py`` (test:
``lanerl_jax/sim/tests/test_champion_level_scaling.py``). Effect on THIS
scenario: the sim's in-band count moved from 468 to 473 out of 8218 samples --
under 1%, because the sim's champion barely levels here (reaching only 1..5,
against the server's 1..8) since it keeps dying and losing its proximity-XP
window. A real, cited, fixed bug; not the story.

**The dominant, still-open factor is deaths, not the band.** The sim's
champion dies 5 times to the server's ~0-1 (see the module docstring of
``lanerl_jax/parity/tests/test_last_hit_gate.py`` for why the server's own
count sits right at a 0/1 boundary and should not be trusted to one decimal).
Each death costs a full fountain-to-lane walk the champion cannot farm during
(``approach_decisions`` 6998 vs 3197 out of 18,000). ``LANERL_AUTOBUY`` (the
server's free +80 HP / +1.2 HP/s regen item, bought before the first
observation frame) and ``enable_call_for_help=True`` (the sim's documented
minion-pile-up-without-release behaviour,
``lanerl_jax.sim.targeting.call_for_help_map``'s own docstring) were both
tried as explanations and both come back negative or backwards -- see
``last_hit_drive.run_oracle_in_sim``'s inline comments for the citations and
numbers. A collision-under-separation hypothesis (the sim applies one
push-apart per unit per tick where the server applies several, sequentially --
``sim/collision.py``'s own booked approximation) was also checked by
comparing mean nearest-neighbour distance among live minions: sim 191.4,
server 224.0 (``frac<100`` units apart: 0.405 vs 0.399) -- real but modest,
not the scale of difference (2.4x in-reach rate, 5x death rate) it would need
to fully explain this.

This module measures BOTH: the band-occupancy fraction each engine actually
sees (which conflates threshold and trajectory), and, in the sim only, what
the band-occupancy would have been under the OLD, unlevel-scaled AD -- computed
offline from the same recorded trajectory, so the fix's own (small) effect
size stays visible after the bug it measured is gone. See
:func:`leveled_ad_of` and :class:`SimBandRun`'s ``leveled_*`` fields.

HOW TO READ THE OUTPUT
-----------------------
Run as a script for a human-readable report; import the ``run_*`` functions
for a test or a follow-up analysis. ``python -m lanerl_jax.parity.hp_band``.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..obs.fog import visible_to
from ..sim.combat import growth_sum, post_mitigation_damage
from ..sim.init import RUNE_AD_BONUS, TOP_LANE_PATH, init_lane, lane_params
from ..sim.orders import OrderKind, Orders, apply_orders
from ..sim.state import Kind, Team
from ..sim.step import step_decision
from .last_hit_drive import (APPROACH_WAYPOINTS, DECISIONS_600S,
                             WIRE_MINION_TYPE, _advance_approach)
from .last_hit_oracle import ChampView, MinionView, decide, post_mitigation

__all__ = [
    "BandSample", "SimBandRun", "ServerBandRun",
    "run_sim_band", "run_server_band", "summarize", "leveled_ad_of",
]


@dataclass(slots=True, frozen=True)
class BandSample:
    """One (decision, enemy minion in Garen's reach) observation."""

    t_ms: float
    hp: float
    armor: float
    #: whether this minion is inside the one-shot band AT THIS ENGINE'S OWN
    #: current attack damage (flat, level-1, for the sim; real and leveled,
    #: for the server).
    in_band: bool
    champ_ad: float
    champ_level: int


@dataclass(slots=True, frozen=True)
class SimBandRun:
    samples: List[BandSample]
    #: `samples[i].in_band` is now the POST-FIX number (level-scaled AD, what
    #: `tick()` actually deals). This pair is the PRE-FIX number instead --
    #: what `in_band` would have been under the old flat level-1(+rune) AD
    #: that was the sim's entire episode before champion AD was wired to
    #: `state.level` -- kept so the fix's effect size stays visible after the
    #: bug it measured is gone. Same length and order as `samples`.
    leveled_in_band: List[bool]
    leveled_ad: List[float]
    #: same counters as `last_hit_drive.SimRun`, gathered in the same pass so
    #: a single run reports both the gate numbers and the band distribution --
    #: two server boots for the same 600 s episode is not a thing to do twice.
    cs: int = 0
    decisions: int = 0
    approach_decisions: int = 0
    attacks: int = 0
    moves: int = 0
    holds: int = 0
    deaths: int = 0


@dataclass(slots=True, frozen=True)
class ServerBandRun:
    samples: List[BandSample]
    log_path: Optional[Path] = None
    cs: int = 0
    decisions: int = 0
    approach_decisions: int = 0
    attacks: int = 0
    moves: int = 0
    holds: int = 0
    deaths: int = 0


def leveled_ad_of(base_ad: float, ad_per_level: float, level: int) -> float:
    """Garen's attack damage at ``level`` under the server's own growth curve,
    plus the rune/mastery flat bonus every training config grants
    (:data:`lanerl_jax.sim.init.RUNE_AD_BONUS` -- see that module for how it
    was measured off a real dump). ``Stats.LevelUp`` -- see the module
    docstring -- grows ``AttackDamage`` through the same non-linear curve as
    every other per-level stat, ``combat.stat_at_level``.
    """
    return float(base_ad + ad_per_level * growth_sum(level) + RUNE_AD_BONUS)


def run_sim_band(decisions: int = DECISIONS_600S, seed: int = 0) -> SimBandRun:
    """Drive the same approach + oracle policy as
    :func:`lanerl_jax.parity.last_hit_drive.run_oracle_in_sim`, instrumenting
    every enemy minion within Garen's reach post-handover instead of only
    counting attacks. Does NOT change the policy's behaviour (still decides
    off the sim's real, flat, current ``attack_damage``) -- this is a read,
    not a fix.
    """
    from ..data.patch import load_patch

    patch = load_patch()
    base_ad = patch.champion.base_ad
    ad_per_level = patch.champion.ad_per_level

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
        # See last_hit_drive.run_oracle_in_sim's matching comment: enabling
        # call-for-help here was tried and made this gate dramatically worse
        # (an active, attacking champion recruits fresh aggressors onto
        # itself every time it lands a hit -- CHAMPION_ATTACKING_MINION,
        # priority 5, beats any minion's own 6-9), unlike the passive
        # StandInWave scenario docs/CALL_FOR_HELP_SWITCH_RATE.md part 6
        # measured it helping. Left off, matching the `step_decision`
        # default and last_hit_drive.py's own instrumented decision.
        return step_decision(apply_orders(state, orders), params_tbl,
                             lane_path=path)

    wp_idx = 0
    prev_alive = True
    samples: List[BandSample] = []
    leveled_band: List[bool] = []
    leveled_ad_list: List[float] = []
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

        if wp_idx < len(APPROACH_WAYPOINTS):
            approach_decisions += 1
            tx, ty = APPROACH_WAYPOINTS[wp_idx]
            state = _step(state, OrderKind.MOVE, tx, ty, -1)
            continue

        kind = np.asarray(state.kind)
        team = np.asarray(state.team)
        alive = np.asarray(state.alive)
        x = np.asarray(state.x)
        y = np.asarray(state.y)
        hp = np.asarray(state.hp)
        model = np.asarray(state.model)
        level0 = int(np.asarray(state.level)[0])
        t_ms = float(np.asarray(state.t_ms))

        # POST-FIX: `sim/step.py`'s `tick()` now level-scales champion AD the
        # same way `Stats.LevelUp` does (see that module and `profiles.py`'s
        # `ad_per_level` column), so this is what the sim's real combat and
        # the oracle's own decision now both use -- `champ_ad_flat` below is
        # kept only to report the PRE-FIX number this instrument originally
        # measured, not to drive anything.
        champ_ad_leveled = leveled_ad_of(base_ad, ad_per_level, level0)
        champ_ad_flat = float(params_np["attack_damage"][model[0]])
        champ = ChampView(
            x=x0, y=y0, attack_damage=champ_ad_leveled,
            attack_range=float(params_np["attack_range"][model[0]]),
        )

        vis = np.asarray(visible_to(
            Team.BLUE, state.x, state.y, state.kind, state.team, state.alive))
        enemy = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.RED)
                               & alive & vis)
        for i in enemy:
            mx, my = float(x[i]), float(y[i])
            reach = champ.attack_range + float(params_np["collision_radius"][model[i]])
            if (mx - x0) ** 2 + (my - y0) ** 2 > reach * reach:
                continue
            m_hp = float(hp[i])
            m_armor = float(params_np["armor"][model[i]])
            dmg_leveled = post_mitigation(champ_ad_leveled, m_armor)
            dmg_flat = post_mitigation(champ_ad_flat, m_armor)
            samples.append(BandSample(
                t_ms=t_ms, hp=m_hp, armor=m_armor,
                in_band=m_hp <= dmg_leveled, champ_ad=champ_ad_leveled,
                champ_level=level0))
            leveled_band.append(m_hp <= dmg_flat)
            leveled_ad_list.append(champ_ad_flat)

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
            state = _step(state, OrderKind.ATTACK, 0.0, 0.0, d.attack)
        elif d.move is not None:
            moves += 1
            mx2, my2 = d.move
            state = _step(state, OrderKind.MOVE, mx2, my2, -1)
        else:
            holds += 1
            state = _step(state, OrderKind.NOOP, 0.0, 0.0, -1)

    cs = int(np.asarray(state.cs)[0])
    return SimBandRun(samples=samples, leveled_in_band=leveled_band,
                      leveled_ad=leveled_ad_list, cs=cs, decisions=decisions,
                      approach_decisions=approach_decisions, attacks=attacks,
                      moves=moves, holds=holds, deaths=deaths)


def run_server_band(
    decisions: int = DECISIONS_600S,
    port_base: int = 44300,
    bot_seed: int = 4242,
    tag: str = "hp_band",
    log_dir: Optional[Path] = None,
    autobuy: bool = False,
) -> ServerBandRun:
    """Same approach + oracle policy as
    :func:`lanerl_jax.parity.last_hit_drive.run_oracle_on_server`,
    instrumenting every enemy minion within Garen's reach post-handover. Reads
    the server's own real, leveled ``ad``/``lvl`` fields straight off the wire
    -- see the module docstring on why re-deriving either is the mistake this
    project already paid for once.

    ``autobuy`` defaults to **off** here (unlike
    :func:`~lanerl_jax.parity.last_hit_drive.run_oracle_on_server`, which
    defaults it on to leave old callers unchanged): this instrument exists to
    read the HP band the last-hit MECHANIC produces, and ``LanerlHooks.
    AutoBuyUndriven`` buying Doran's Shield for free at boot (+80 max HP,
    +1.2 HP/s regen -- see ``run_oracle_on_server``'s docstring for the
    citations) changes how long the champion survives standing in the band,
    not how the band itself works. Pass ``True`` to measure the confound
    itself.
    """
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    from ..data.patch import load_patch

    patch = load_patch()
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

            if wp_idx < len(APPROACH_WAYPOINTS):
                approach_decisions += 1
                tx, ty = APPROACH_WAYPOINTS[wp_idx]
                env.step([{"blue": {"t": "move", "x": tx, "y": ty}}])
                continue

            champ_ad = float(blue["ad"])
            champ_rng = float(blue["rng"])
            champ_level = int(blue.get("lvl", 1))
            t_ms = float(obs.get("t", 0.0))

            minions = []
            for u in units:
                if u.get("k") != "LaneMinion" or u.get("tm") != 200:
                    continue
                if not u.get("vb", 0):
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
                        champ_level=champ_level))
                minions.append(MinionView(
                    uid=int(u["id"]), x=mx, y=my, hp=m_hp,
                    armor=float(stat.armor),
                    collision_radius=float(stat.collision_radius)))

            champ = ChampView(x=bx, y=by, attack_damage=champ_ad,
                              attack_range=champ_rng)
            d = decide(champ, minions, lethal_epsilon=0.0)
            if d.attack is not None:
                attacks += 1
                act = {"t": "attack", "id": d.attack}
            elif d.move is not None:
                moves += 1
                mx2, my2 = d.move
                act = {"t": "move", "x": mx2, "y": my2}
            else:
                holds += 1
                act = {"t": "noop"}
            env.step([{"blue": act}])

        obs = env.last_obs[0]
        blue = next(u for u in obs["u"] if u.get("k") == "Champion" and u.get("tm") == 100)
        cs = int(blue.get("cs", 0))
    finally:
        env.close()

    return ServerBandRun(samples=samples, log_path=log_path, cs=cs,
                         decisions=decisions, approach_decisions=approach_decisions,
                         attacks=attacks, moves=moves, holds=holds, deaths=deaths)


def summarize(name: str, samples: List[BandSample],
             leveled_in_band: Optional[List[bool]] = None) -> str:
    n = len(samples)
    if n == 0:
        return f"{name}: no in-reach samples"
    hp = np.array([s.hp for s in samples])
    band = np.array([s.in_band for s in samples])
    ad = np.array([s.champ_ad for s in samples])
    lvl = np.array([s.champ_level for s in samples])
    lines = [
        f"{name}: {n} in-reach decision-minion samples, "
        f"{int(band.sum())} in-band ({100 * band.mean():.1f}%)",
        f"  hp in reach: mean={hp.mean():.1f} median={np.median(hp):.1f} "
        f"p10={np.percentile(hp, 10):.1f} p90={np.percentile(hp, 90):.1f}",
        f"  champ ad while in reach: min={ad.min():.1f} max={ad.max():.1f} "
        f"(level {lvl.min()}..{lvl.max()})",
    ]
    if leveled_in_band is not None:
        lb = np.array(leveled_in_band)
        lines.append(
            f"  PRE-FIX (flat, unlevel-scaled ad) would show: {int(lb.sum())} "
            f"in-band ({100 * lb.mean():.1f}%) -- vs {100 * band.mean():.1f}% now")
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=int, default=DECISIONS_600S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot-seed", type=int, default=4242)
    ap.add_argument("--skip-server", action="store_true")
    ap.add_argument("--autobuy", action="store_true",
                    help="leave LANERL_AUTOBUY on for the server boot "
                         "(default: off, see run_server_band)")
    ap.add_argument("--log-dir", type=str, default=None,
                    help="where the server's own log lands -- pass a path "
                         "under the shared repo (e.g. lanerl_jax/runs/...) "
                         "when running via slurm, or the default tempdir "
                         "lands on the compute node's LOCAL disk and is not "
                         "readable from the login node afterward")
    args = ap.parse_args()

    print("Running sim...")
    sim = run_sim_band(decisions=args.decisions, seed=args.seed)
    print(f"sim gate numbers: cs={sim.cs} approach_decisions={sim.approach_decisions} "
          f"attacks={sim.attacks} moves={sim.moves} holds={sim.holds} "
          f"deaths={sim.deaths}")
    print(summarize("sim (flat, actual)", sim.samples, sim.leveled_in_band))

    if not args.skip_server:
        print("Running server (boots a real process; several minutes)...")
        server = run_server_band(decisions=args.decisions, bot_seed=args.bot_seed,
                                 autobuy=args.autobuy, log_dir=args.log_dir)
        print(f"server gate numbers: cs={server.cs} "
              f"approach_decisions={server.approach_decisions} "
              f"attacks={server.attacks} moves={server.moves} "
              f"holds={server.holds} deaths={server.deaths}")
        print(summarize("server (real, leveled)", server.samples))
        print(f"server log: {server.log_path}")


if __name__ == "__main__":
    _main()
