"""After N in-process resets the world must still be the world.

WHY THIS EXISTS, AND WHY IT IS GENERAL
--------------------------------------
Three separate bugs found in one day were the same bug: state that survived a
reset it should not have survived. Each was found individually, by staring at
a different symptom, and each produced a smooth, believable, completely fake
CS curve in between.

    1. Spell.CastCancelCheck skipped FinishCasting when the owner died, so
       _castingSpell was never cleared. CanMove() requires it to be null, so a
       champion that died mid-cast could never move again -- across deaths,
       respawns, and all 51 resets of the process.
    2. The same on the channel side: a reset landing mid-recall left
       ChannelSpell set, and Recall is not CanMoveWhileChanneling.
    3. RestoreBuildings revives buildings that are dead but PRESENT. A turret
       the engine had REMOVED from the ObjectManager was never enumerated, so
       it was simply gone: buildings=32 at the first reset, 28 by the end. The
       wave then has nothing to stop it and BOTH champions' CS falls.

Finding those one at a time cost a day and produced three wrong explanations
along the way. This test asserts the INVARIANT instead: a server that has been
reset N times must look like a server that has just booted.

The reference is a second instance that is never reset, booted alongside --
the same trick ``test_episode_reset_stats`` uses, because "a fresh process" is
exactly what ``collect_demos`` records and therefore what the BC prior is
cloned from. A "fix" that broke episode 1 in the same way would satisfy an
episode-1-vs-episode-2 comparison and fail this one.

WHAT IS ASSERTED, and which bug each clause would have caught
-------------------------------------------------------------
* turret/building count            -- bug 3, directly
* champion combat stats            -- the rune/mastery page regression
* the champion still RESPONDS to a move order  -- bugs 1 and 2, via their only
  externally visible effect: the wire carries no _castingSpell or ChannelSpell,
  but a champion that cannot move is a champion that does not move when told to

SCOPE, MEASURED RATHER THAN ASSUMED
-----------------------------------
The driving loop casts and recalls, and puts the last decision before each
reset on a live recall channel. That was INTENDED to reproduce bug 2, and it
does not: with the ChannelSpell clear in LanerlEpisode.ResetChampion
deliberately commented out and the server rebuilt, this test still passed.

So what it actually covers today:

    covered      building/structure count      (bug 3, the map degradation)
    covered      champion combat stats         (the rune/mastery page)
    covered      champion responds to orders   (the SYMPTOM of bugs 1 and 2,
                                                 if something triggers them)
    NOT covered  the triggers of bugs 1 and 2

Two consequences worth stating plainly. First, do not cite this as coverage
for the freeze bugs. Second, it means the mechanism I assumed for bug 2 --
"a reset landed mid-recall" -- is UNPROVEN: something in ResetChampion ahead
of that clause (UpdateMoveOrder(Stop), CancelAutoAttack, or Respawn) may
already end a channel. The fix there is a defensive clear that stops the
symptom persisting across resets either way, but the real trigger for the
stuck recall observed on a live anchor instance is still unknown.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping

import pytest

from lanerl_train import paths
from lanerl_train.ports import PortAllocator
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

pytestmark = pytest.mark.skipif(
    not paths.server_available(),
    reason="vendored server build not available (or LANERL_SKIP_SERVER_TESTS=1)",
)

#: index 0 is played and reset repeatedly; index 1 is never touched.
N = 2

#: Resets to put the played instance through. The real failures appeared
#: gradually -- the building count fell over tens of episodes -- so a handful
#: of resets would have passed happily while the run rotted.
RESETS = int(os.environ.get("LANERL_RESET_INVARIANCE_N", "40"))

#: Game time between resets. Long enough for the champion to leave the
#: fountain, take damage and cast, which is what gives the reset real state to
#: undo; short enough that 40 cycles is minutes.
DECISIONS_BETWEEN = 90

COMBAT_STATS = ("ad", "mhp", "ar", "mr")
TEAMS = (100, 200)
BUILDING_KINDS = ("Turret", "LaneTurret", "Nexus", "Inhibitor", "ObjBuilding")


def _champs(obs: Mapping) -> Dict[int, Mapping]:
    return {u["tm"]: u for u in obs["u"] if u.get("k") == "Champion"}


def _buildings(obs: Mapping) -> int:
    return sum(1 for u in obs["u"] if any(k in str(u.get("k", "")) for k in BUILDING_KINDS))


def _boot(tmp_path: Path, base_port: int) -> VecLaneEnv:
    env = VecLaneEnv(
        N,
        spec=ServerLaunchSpec(toponly=True, bot_teams="none"),
        log_dir=tmp_path / "logs",
        ports=PortAllocator(base=base_port).allocate(N),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    return env


def _move_order(obs: Mapping, team: int, dx: float, dy: float):
    ch = _champs(obs).get(team)
    if ch is None:
        return {"t": "noop"}
    return {"t": "move", "x": float(ch["x"]) + dx, "y": float(ch["y"]) + dy}


@pytest.mark.slow
def test_a_server_reset_n_times_still_looks_like_a_fresh_one(tmp_path):
    env = _boot(tmp_path, base_port=47100)
    try:
        assert all(env.alive), f"an instance failed to boot: {env.alive}"
        fresh = env.last_obs[1]
        assert fresh is not None, "the reference instance produced no observation"
        ref_buildings = _buildings(fresh)
        ref_stats = {t: {k: float(u[k]) for k in COMBAT_STATS}
                     for t, u in _champs(fresh).items() if all(u.get(k) is not None
                                                               for k in COMBAT_STATS)}
        assert ref_buildings > 0, "reference sees no buildings; the probe is vacuous"
        assert set(ref_stats) == set(TEAMS), f"reference champions: {sorted(ref_stats)}"

        for cycle in range(RESETS):
            for i in range(DECISIONS_BETWEEN):
                lines = [None] * N
                cur = env.last_obs[0]
                if cur is not None:
                    # The champions must CAST and RECALL, not just walk. Both
                    # freeze bugs are triggered by a reset (or a death) landing
                    # while a spell or channel is live, so a move-only probe
                    # would pass happily on the broken build -- which is worse
                    # than no test, because it would be cited as coverage.
                    if i == DECISIONS_BETWEEN - 1:
                        # land the reset DIRECTLY on a live recall channel:
                        # this is bug 2's exact trigger.
                        blue = {"t": "recall"}
                        red = {"t": "recall"}
                    elif i % 17 == 0:
                        blue = {"t": "cast", "slot": 2, "id": 0}   # E, a real cast time
                        red = {"t": "cast", "slot": 2, "id": 0}
                    else:
                        blue = _move_order(cur, 100, 400.0, 400.0)
                        red = _move_order(cur, 200, -400.0, -400.0)
                    lines[0] = {"blue": blue, "red": red}
                env.step(lines)
            res = env.reset_episodes([0])
            assert all(res.alive), (
                f"instance died during reset {cycle + 1}/{RESETS}: {res.died}"
            )

        played = env.last_obs[0]
        assert played is not None, (
            f"instance 0 stopped producing observations after {RESETS} resets"
        )

        # -- bug 3: the map must still be the whole map ---------------------
        got = _buildings(played)
        assert got == ref_buildings, (
            f"after {RESETS} resets the played instance sees {got} buildings "
            f"against {ref_buildings} on a never-reset one. A destroyed "
            f"structure the engine removed from the ObjectManager cannot be "
            f"revived, so every later episode is played on a different map -- "
            f"the wave has nothing to stop it and BOTH champions' CS falls."
        )

        # -- the rune/mastery page must still be on ------------------------
        for team, want in ref_stats.items():
            have = _champs(played).get(team)
            assert have is not None, f"team {team} is missing after {RESETS} resets"
            for stat, v in want.items():
                assert float(have[stat]) == pytest.approx(v, rel=1e-6), (
                    f"after {RESETS} resets team {team} has {stat}="
                    f"{float(have[stat])} against {v} on a fresh server"
                )

        # -- bugs 1 and 2: it must still DO WHAT IT IS TOLD -----------------
        # The wire carries no _castingSpell or ChannelSpell, so the only
        # visible signature of both freeze bugs is a champion that does not
        # move when ordered to. That is exactly what an anchor instance did
        # for 18,001 decisions while the trainer sent it 13,935 move orders.
        for team in TEAMS:
            start = _champs(env.last_obs[0])[team]
            x0, y0 = float(start["x"]), float(start["y"])
            for _ in range(60):
                cur = env.last_obs[0]
                lines = [None] * N
                if cur is not None:
                    lines[0] = {"blue": _move_order(cur, 100, 600.0, 600.0),
                                "red": _move_order(cur, 200, -600.0, -600.0)}
                env.step(lines)
            end = _champs(env.last_obs[0])[team]
            moved = ((float(end["x"]) - x0) ** 2 + (float(end["y"]) - y0) ** 2) ** 0.5
            assert moved > 100.0, (
                f"after {RESETS} resets team {team} moved {moved:.0f} units in 60 "
                f"decisions of move orders. A champion that cannot move is the "
                f"signature of _castingSpell or ChannelSpell surviving a reset "
                f"(CanMove() requires both to be null) -- it stands in its "
                f"fountain at level 1 on full hp for the whole episode while "
                f"every order is silently dropped."
            )
    finally:
        env.close()
