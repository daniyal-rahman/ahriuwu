"""An in-process episode reset must hand back the champion it started with.

THE BUG THIS EXISTS FOR
-----------------------
``LanerlEpisode.ResetChampion`` zeroed every stat bonus (it has to: that is how
items and buffs come off) and then called ``Stats.LoadStats(CharData)``, which
restores *base* values only.  Nothing put back the rune page and the
mastery/talent page that ``Champion.OnAdded`` applies exactly once, at spawn,
from ``lanerl/cfg/garen1v1.json``.  So from episode 2 onwards -- for the whole
life of the process -- the champion played at Garen's raw ``Garen.json``
numbers::

    episode 1   ad 78.14   mhp 672   armor 36.54   mr 44.16
    episode 2   ad 57.88   mhp 616   armor 27.54   mr 32.10

That is not a cosmetic drift, it is two different games.  ``collect_demos.py``
spawns a **fresh process per game**, so every BC demonstration is an episode 1;
RL training resets in-process, so almost every RL step is an episode 2+.  The
prior was cloned in a game the policy never plays.  Reproduced on the wire and
in ``runs/rl-bc4-0912/actor0_logs/instance000.log`` (``mhp=672`` in episode 1,
``mhp=616`` in all eleven episodes after it).

The same boundary lost the *items*: the reset strips the champion's inventory
but ``LanerlHooks.AutoBuyUndriven`` kept its per-champion build index, so
episode 2 resumed the build path against a champion that owned nothing --
measured at ``mhp=754 gold=0`` (Doran's Shield bought) in episode 1 against
``mhp=616 gold=115`` in episode 2.

WHAT IS ASSERTED
----------------
``ad``, ``mhp``, ``ar`` and ``mr`` on the **first frame of episode 2** equal
those on the **first frame of episode 1**, read off the control channel that
training itself reads (``lanerl_rl.frame`` / ``LanerlControl.BuildObservation``).

Two things make it a hard test rather than a tautology:

* a second, never-reset instance is booted alongside, and its first frame is
  the reference.  A fresh process is exactly what ``collect_demos`` records, so
  "episode 2 == a fresh process" is the BC/RL contract stated directly.  A
  "fix" that stripped the page from episode 1 too would satisfy an
  episode1 == episode2 check and fails this one.
* the episode-1 numbers are checked against Garen's own ``Garen.json`` base
  stats, read from the same vendored Content tree the server loaded.  If the
  page is not on at all, the guard fires before the equality ever runs.

Run it deliberately (it boots real servers)::

    pytest lanerl_train/tests/test_episode_reset_stats.py -m slow -s
"""

from __future__ import annotations

import json
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

#: Two instances: index 0 is played and reset, index 1 is never touched and is
#: the fresh-process reference.
N = 2

#: Long enough that the champion has left the fountain, taken minion damage and
#: had a level-up, so the reset has real state to undo -- and short enough that
#: the test is seconds rather than minutes.  4 s of game time at 30 Hz.
DECISIONS_BEFORE_RESET = 120

#: The four the bug moved.  ``hp`` and ``gold`` are deliberately NOT here: the
#: post-reset observation arrives one tick later into its episode than the
#: first one does (``LanerlControl`` rewinds its step counter), so both can
#: legitimately differ by a regen/ambient-gold tick.  Everything below is a
#: derived total that no amount of elapsed time changes at level 1.
COMBAT_STATS = ("ad", "mhp", "ar", "mr")

TEAMS = (100, 200)


def champions(obs: Mapping) -> Dict[int, Mapping]:
    return {u["tm"]: u for u in obs["u"] if u["k"] == "Champion"}


def combat(obs: Mapping) -> Dict[int, Dict[str, float]]:
    """``{team: {stat: value}}`` for the four stats the page moves."""
    out: Dict[int, Dict[str, float]] = {}
    for team, unit in champions(obs).items():
        missing = [k for k in COMBAT_STATS if unit.get(k) is None]
        assert not missing, (
            f"the control channel did not emit {missing} for team {team}. "
            f"LanerlControl.BuildObservation must carry ad/ap/ar/mr; without them "
            f"this invariant cannot be checked at all."
        )
        out[team] = {k: float(unit[k]) for k in COMBAT_STATS}
    assert set(out) == set(TEAMS), f"expected both champions, got teams {sorted(out)}"
    return out


def base_stats_of_the_configured_champion() -> Dict[str, float]:
    """Garen's raw ``Garen.json`` numbers -- i.e. what a page-less champion has.

    Read from the vendored Content tree rather than written down here, because
    a copied constant that silently stops matching the server is the exact
    failure mode this whole test is about (``constants.garen_attack_damage``
    held a hand-copied 57.88 for months while the server played 78.14).
    """
    cfg = json.loads(paths.default_game_config().read_text())
    names = {p["champion"] for p in cfg["players"]}
    assert len(names) == 1, f"this test assumes a mirror match, config has {names}"
    champ = names.pop()

    root = paths.vendor_root()
    candidates = [
        root / f"LoLServer/Content/LeagueSandbox-Default/Stats/{champ}/{champ}.json",
        root / f"Content/LeagueSandbox-Default/Stats/{champ}/{champ}.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())["Values"]["Data"]
            return {
                "ad": float(data["BaseDamage"]),
                "mhp": float(data["BaseHP"]),
                "ar": float(data["Armor"]),
                "mr": float(data["SpellBlock"]),
            }
    raise AssertionError(
        f"no stat file for {champ} under {[str(p) for p in candidates]}; the server "
        f"loaded one, so this test is looking in the wrong place"
    )


def boot(tmp_path: Path, base_port: int, autobuy: bool) -> VecLaneEnv:
    spec = ServerLaunchSpec(
        toponly=True,
        bot_teams="none",  # both champions control-driven, like a training actor
        extra_env={"LANERL_AUTOBUY": "1" if autobuy else "0"},
    )
    env = VecLaneEnv(
        N,
        spec=spec,
        log_dir=tmp_path / f"logs_autobuy{int(autobuy)}",
        ports=PortAllocator(base=base_port).allocate(N),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    return env


def assert_same(what: str, first: Mapping, second: Mapping) -> None:
    for team in TEAMS:
        for stat in COMBAT_STATS:
            a, b = first[team][stat], second[team][stat]
            assert a == pytest.approx(b, abs=0.01), (
                f"{what}: team {team} {stat} is {b}, was {a}. "
                f"episode1={first[team]} other={second[team]}"
            )


def assert_report_says_the_page_came_back(log: Path) -> None:
    """``LANERL_RESET`` carries ``pages=<restored> no_page=<missing>``.

    Asserted as well as the wire values because it is the only signal a long
    training run has, without a test, that the champion it is training on is
    still the champion it started with.
    """
    assert log.exists(), f"no server log to read at {log}"
    lines = [ln for ln in log.read_text(errors="replace").splitlines() if "LANERL_RESET " in ln]
    assert lines, f"the server logged no LANERL_RESET line at all ({log})"
    last = lines[-1]
    fields = dict(
        tok.split("=", 1) for tok in last.split() if "=" in tok and not tok.startswith("LANERL")
    )
    assert fields.get("no_page") == "0", f"a champion was reset without its page: {last}"
    assert int(fields.get("pages", "0")) == N, (
        f"expected the page restored on both champions, got: {last}"
    )


def _check(tmp_path: Path, base_port: int, autobuy: bool) -> None:
    env = boot(tmp_path, base_port, autobuy)
    log = Path(env.handles[0].log_path)
    try:
        assert all(env.alive), f"an instance failed to boot: {env.alive}"
        ep1 = combat(env.last_obs[0])
        fresh = combat(env.last_obs[1])

        # 0. the page is actually on in episode 1.  Without this the equality
        #    below would be satisfied by a server that never applies runes.
        base = base_stats_of_the_configured_champion()
        for team in TEAMS:
            for stat in COMBAT_STATS:
                assert ep1[team][stat] > base[stat], (
                    f"team {team} {stat} is {ep1[team][stat]} on the FIRST frame of the "
                    f"first episode, which is at or below the champion's base {base[stat]}. "
                    f"The rune/mastery page from {paths.default_game_config()} is not "
                    f"being applied at spawn at all -- nothing after this means anything."
                )

        # 1. two fresh processes agree (sanity on the reference itself).
        assert_same("two freshly booted servers disagree", ep1, fresh)

        # 2. play, then reset instance 0 in process.
        for _ in range(DECISIONS_BEFORE_RESET):
            res = env.step([None] * N)
            assert all(res.alive), f"an instance died while playing: {res.died}"
        res = env.reset_episodes([0])
        assert all(res.alive), f"an instance died during the reset: {res.died}"
        ep2 = combat(res.obs[0])

        # 3. THE INVARIANT.
        assert_same(
            "episode 2 is not the game episode 1 was: the in-process reset lost the "
            "rune/mastery page (and/or the starting items)",
            ep1,
            ep2,
        )

        # 4. ...and it is the same game a fresh process plays, which is the game
        #    every BC demonstration was recorded in.
        assert_same(
            "episode 2 of a reset server is not the game a fresh process plays, so "
            "the BC prior and RL training are on different champions",
            fresh,
            ep2,
        )

        # 5. once is luck.  The second reset must not drift either.
        for _ in range(DECISIONS_BEFORE_RESET):
            res = env.step([None] * N)
            assert all(res.alive), f"an instance died while playing: {res.died}"
        res = env.reset_episodes([0])
        assert all(res.alive), f"an instance died during the second reset: {res.died}"
        assert_same("episode 3 drifted from episode 1", ep1, combat(res.obs[0]))

        # 6. the instance that was never reset must be untouched throughout --
        #    proving the reset is per-instance and that nothing here is a
        #    process-wide side effect.
        assert_same("the never-reset instance changed", fresh, combat(env.last_obs[1]))
    finally:
        env.close()

    assert_report_says_the_page_came_back(log)


@pytest.mark.slow
def test_reset_restores_the_rune_and_mastery_page(tmp_path):
    """The page alone, with the auto-shopper off so nothing else can move."""
    _check(tmp_path, base_port=23600, autobuy=False)


@pytest.mark.slow
def test_reset_restores_page_and_items_in_the_production_config(tmp_path):
    """The configuration training actually runs: ``LANERL_AUTOBUY`` on.

    Items are stat modifiers too, and the reset strips them, so the build index
    ``AutoBuyUndriven`` keeps has to be rewound with them or episode 2 starts
    with a different inventory than episode 1 -- which shows up here as a
    different ``mhp``.
    """
    _check(tmp_path, base_port=23700, autobuy=True)
