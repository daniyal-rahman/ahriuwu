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

WHAT IT FOUND (current canonical fixture, 2026-09-17)
--------------------------------------------------------
The old 13-CS/200-attack result is not the source-faithful baseline: it
predates canonical call-for-help. The server broadcasts aggro on every landed
hit, so the gate now takes ``step_decision``'s default
``enable_call_for_help=True``; the historical OFF ablation is not parity
evidence.

On a fresh 18,000-decision comparison with that mechanism enabled, the sim
has **72** lethal decision frames in **7** windows (mean 10.29 frames, max
11), while the server has **86** frames in only **4** windows (mean 21.50,
max 56). The sim scores CS=7 to the server's CS=4, with one death each. Thus
the counter is not duplicating orders or representing excess pathing exposure:
it reports exact decision-frame eligibility. The sim has fewer total eligible
frames but more independent HP crossings, and currently converts each short
crossing into a last hit. The open mechanism is minion HP crossover cadence
(including minion attack/missile timing), which needs Tier-1 attribution;
there is no source-supported behavioural tweak to make here yet.

Champion AD level scaling remains source-required and is correctly modelled.
On this current sim trajectory, the offline old-flat-AD control produces 42
in-band frames versus 72 with live level-scaled AD (levels 1..8). That is an
important calibration control, not a reason to disable scaling: the server
wire likewise reports level 1..8 and AD 78.1..101.5 during its windows.

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

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..obs.fog import visible_to
from ..sim.combat import growth_sum, post_mitigation_damage
from ..sim.config import SimConfig
from ..sim.init import RUNE_AD_BONUS, init_lane
from ..sim.orders import OrderKind, Orders
from ..sim.profiles import PROFILES
from ..sim.state import Kind, Team
from ..sim.step import env_step
from ..sim.targeting import MinionType
from .last_hit_drive import (APPROACH_WAYPOINTS, DECISIONS_600S,
                             WIRE_MINION_TYPE, _advance_approach,
                             gate3_route_inputs)
from .last_hit_oracle import ChampView, MinionView, decide, post_mitigation

__all__ = [
    "BandSample", "BandWindows", "SimBandRun", "ServerBandRun",
    "band_windows", "run_sim_band", "run_server_band", "summarize",
    "leveled_ad_of",
]


# This is a diagnostic label, not an extra simulation classification.  The
# profile id is already the authoritative per-slot type after a spawn; keeping
# the label beside the sample lets a causal report distinguish a melee hit from
# a caster/cannon missile without pretending a recycled JAX slot is a server
# NetId.
_SIM_MINION_TYPE_NAME = {
    MinionType.MELEE: "melee",
    MinionType.CASTER: "caster",
    MinionType.CANNON: "cannon",
    MinionType.SUPER: "super",
}


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
    champ_x: float
    champ_y: float
    minion_x: float
    minion_y: float
    #: Engine-local identity, used only to connect adjacent samples in one
    #: run.  It is a recycled sim slot on JAX and a NetId on the server, so it
    #: must never be compared across engines.
    minion_id: int = -1
    #: Patch minion key (``melee``, ``caster``, ``cannon`` or ``super``).
    minion_type: str = ""


@dataclass(slots=True, frozen=True)
class BandWindows:
    """Decision-level occupancy of the one-shot band.

    ``BandSample`` is deliberately one row per *in-reach minion*, whereas an
    oracle ATTACK decision is one row per *decision* whenever any such minion
    is lethal.  This summary makes that distinction explicit.  A window ends
    whenever a decision time is missing from the in-band set, so two unrelated
    lethal minions separated by even one non-lethal decision do not look like
    one long opportunity.
    """

    #: Unique decision frames with at least one in-range, one-shot minion.
    in_band_decisions: int
    #: Contiguous runs of such decision frames at the 30 Hz driver cadence.
    windows: int
    #: Mean/max length of a contiguous run, in decision frames.
    mean_frames: float
    max_frames: int


def band_windows(samples: List[BandSample], *, max_step_ms: float = 34.0) -> BandWindows:
    """Summarize the decision-frame lethal windows represented by ``samples``.

    The server clock serializes 30 Hz decision times as alternating 33/34 ms
    integer timestamps.  ``34`` is therefore the inclusive continuity bound;
    a 66/67 ms gap proves that an intervening decision was not in the band.
    This routine intentionally does not infer target identity from sim slots
    or server NetIds -- neither is common across engines and identity is not
    needed to answer whether the oracle had an ATTACK choice on a frame.
    """
    in_band_times = sorted({s.t_ms for s in samples if s.in_band})
    if not in_band_times:
        return BandWindows(0, 0, 0.0, 0)
    lengths: List[int] = []
    run = 1
    previous = in_band_times[0]
    for t_ms in in_band_times[1:]:
        if t_ms - previous <= max_step_ms:
            run += 1
        else:
            lengths.append(run)
            run = 1
        previous = t_ms
    lengths.append(run)
    return BandWindows(len(in_band_times), len(lengths),
                       float(np.mean(lengths)), max(lengths))


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
    #: Optional pre-action state stream used only to align free-running minion
    #: target/AA histories with a diagnostic server run.
    state_capture_path: Optional[Path] = None


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
    #: Optional pre-action wire observations captured for causal window work.
    observation_path: Optional[Path] = None


def leveled_ad_of(base_ad: float, ad_per_level: float, level: int) -> float:
    """Garen's attack damage at ``level`` under the server's own growth curve,
    plus the rune/mastery flat bonus every training config grants
    (:data:`lanerl_jax.sim.init.RUNE_AD_BONUS` -- see that module for how it
    was measured off a real dump). ``Stats.LevelUp`` -- see the module
    docstring -- grows ``AttackDamage`` through the same non-linear curve as
    every other per-level stat, ``combat.stat_at_level``.
    """
    return float(base_ad + ad_per_level * growth_sum(level) + RUNE_AD_BONUS)


def run_sim_band(
    decisions: int = DECISIONS_600S,
    seed: int = 0,
    state_capture_path: Optional[Path] = None,
    *,
    route_table=None,
    terrain=None,
    table_disabled: bool = False,
) -> SimBandRun:
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

    route_table, terrain = gate3_route_inputs(
        route_table=route_table, terrain=terrain, table_disabled=table_disabled)
    # `STRUCT-003`: one step configuration -- TOP lane waves, the tick's
    # INLINE terrain repair, call for help on, routed unless table_disabled.
    sim = SimConfig.scripted(route_table=route_table, terrain=terrain)
    params_tbl = sim.params
    params_np = {k: np.asarray(v) for k, v in params_tbl.items()}
    state = init_lane(seed=seed)

    @jax.jit
    def _step(state, order_kind, order_x, order_y, order_target):
        orders = Orders(
            kind=jnp.array([order_kind, OrderKind.NOOP], dtype=jnp.int8),
            x=jnp.array([order_x, 0.0], dtype=state.x.dtype),
            y=jnp.array([order_y, 0.0], dtype=state.y.dtype),
            target=jnp.array([order_target, -1], dtype=jnp.int8),
        )
        # Omit `enable_call_for_help` deliberately: the source broadcasts
        # aggro on every landed hit and `step_decision`'s canonical default
        # reproduces it. An earlier OFF ablation reduced some active-oracle
        # deaths, but the server cannot disable this mechanism, so it is not
        # valid gate evidence.
        return env_step(state, orders, sim)

    wp_idx = 0
    prev_alive = True
    samples: List[BandSample] = []
    leveled_band: List[bool] = []
    leveled_ad_list: List[float] = []
    approach_decisions = attacks = moves = holds = deaths = 0
    state_fh = None
    if state_capture_path is not None:
        state_capture_path = Path(state_capture_path)
        state_capture_path.parent.mkdir(parents=True, exist_ok=True)
        state_fh = state_capture_path.open("w")

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
        target = np.asarray(state.target)
        spawn_seq = np.asarray(state.spawn_seq)
        aa_cooldown = np.asarray(state.aa_cooldown)
        aa_windup = np.asarray(state.aa_windup)
        is_attacking = np.asarray(state.is_attacking)
        ai_timer = np.asarray(state.ai_timer)
        target_priority = np.asarray(state.target_priority)
        level0 = int(np.asarray(state.level)[0])
        t_ms = float(np.asarray(state.t_ms))

        if state_fh is not None:
            # Capture the engine's actual pre-action state.  ``slot`` is
            # recycled, whereas ``spawn_seq`` identifies a concrete spawned
            # minion within this run; ``target_spawn_seq`` makes target
            # transitions comparable without leaking slot allocation order.
            rows = []
            for i in np.flatnonzero((kind == Kind.LANE_MINION) & alive):
                target_slot = int(target[i])
                rows.append({
                    "slot": int(i), "spawn_seq": int(spawn_seq[i]),
                    "team": int(team[i]),
                    "type": _SIM_MINION_TYPE_NAME[PROFILES[int(model[i])][1]],
                    "hp": float(hp[i]), "x": float(x[i]), "y": float(y[i]),
                    "target_slot": target_slot,
                    "target_spawn_seq": (
                        int(spawn_seq[target_slot]) if target_slot >= 0 else -1),
                    "aa_cooldown": float(aa_cooldown[i]),
                    "aa_windup": float(aa_windup[i]),
                    "is_attacking": bool(is_attacking[i]),
                    "ai_timer": float(ai_timer[i]),
                    "target_priority": int(target_priority[i]),
                })
            state_fh.write(json.dumps({
                "t": t_ms,
                "champion": {
                    "x": x0, "y": y0, "target_slot": int(target[0]),
                    "target_spawn_seq": (
                        int(spawn_seq[int(target[0])]) if int(target[0]) >= 0 else -1),
                },
                "minions": rows,
            }, separators=(",", ":")) + "\n")

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
                champ_level=level0, champ_x=x0, champ_y=y0,
                minion_x=mx, minion_y=my, minion_id=int(i),
                minion_type=_SIM_MINION_TYPE_NAME[PROFILES[int(model[i])][1]]))
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

    if state_fh is not None:
        state_fh.close()
    cs = int(np.asarray(state.cs)[0])
    return SimBandRun(samples=samples, leveled_in_band=leveled_band,
                      leveled_ad=leveled_ad_list, cs=cs, decisions=decisions,
                      approach_decisions=approach_decisions, attacks=attacks,
                      moves=moves, holds=holds, deaths=deaths,
                      state_capture_path=state_capture_path)


def run_server_band(
    decisions: int = DECISIONS_600S,
    port_base: int = 44300,
    bot_seed: int = 4242,
    tag: str = "hp_band",
    log_dir: Optional[Path] = None,
    autobuy: bool = False,
    diagnostic_internals: bool = False,
    observation_path: Optional[Path] = None,
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

    extra_env = {"LANERL_AUTOBUY": "1" if autobuy else "0"}
    if diagnostic_internals:
        # The canonical state hash stays untouched; this opt-in stream exposes
        # the target/AA/AI/missile facts needed to attribute a lethal crossing.
        extra_env.update({"LANERL_STATE_DUMP": "1",
                          "LANERL_STATE_DUMP_FULL": "1",
                          "LANERL_STATE_DUMP_INTERNALS": "1"})
    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
            toponly=True, bot_teams="none", bot_seed=bot_seed, step_ticks=2,
            extra_env=extra_env),
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
    obs_fh = None
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        log_path = Path(env.handles[0].log_path)
        if observation_path is not None:
            observation_path = Path(observation_path)
            observation_path.parent.mkdir(parents=True, exist_ok=True)
            obs_fh = observation_path.open("w")
        for i in range(decisions):
            obs = env.last_obs[0]
            if obs is None:
                raise RuntimeError(f"no observation at decision {i}")
            if obs_fh is not None:
                # Pre-action, matching the window the oracle itself evaluates.
                obs_fh.write(json.dumps(obs, separators=(",", ":")) + "\n")
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
                        champ_level=champ_level, champ_x=bx, champ_y=by,
                        minion_x=mx, minion_y=my, minion_id=int(u["id"]),
                        minion_type=key))
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
        if obs_fh is not None:
            obs_fh.close()
        env.close()

    return ServerBandRun(samples=samples, log_path=log_path, cs=cs,
                         decisions=decisions, approach_decisions=approach_decisions,
                         attacks=attacks, moves=moves, holds=holds, deaths=deaths,
                         observation_path=observation_path)


def summarize(name: str, samples: List[BandSample],
             leveled_in_band: Optional[List[bool]] = None) -> str:
    n = len(samples)
    if n == 0:
        return f"{name}: no in-reach samples"
    hp = np.array([s.hp for s in samples])
    band = np.array([s.in_band for s in samples])
    ad = np.array([s.champ_ad for s in samples])
    lvl = np.array([s.champ_level for s in samples])
    windows = band_windows(samples)
    lines = [
        f"{name}: {n} in-reach decision-minion samples, "
        f"{int(band.sum())} in-band ({100 * band.mean():.1f}%)",
        f"  hp in reach: mean={hp.mean():.1f} median={np.median(hp):.1f} "
        f"p10={np.percentile(hp, 10):.1f} p90={np.percentile(hp, 90):.1f}",
        f"  champ ad while in reach: min={ad.min():.1f} max={ad.max():.1f} "
        f"(level {lvl.min()}..{lvl.max()})",
        f"  one-shot windows: {windows.in_band_decisions} decision frames in "
        f"{windows.windows} runs (mean={windows.mean_frames:.2f}, "
        f"max={windows.max_frames})",
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
    ap.add_argument("--table-disabled", action="store_true",
                    help="run the PATH-006 raw two-point Move ablation; the "
                    "default loads the production routed Gate-3 artifact")
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
    sim = run_sim_band(decisions=args.decisions, seed=args.seed,
                       table_disabled=args.table_disabled)
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
