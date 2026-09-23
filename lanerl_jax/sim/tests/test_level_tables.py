"""`STRUCT-005`: every level-indexed table has ONE convention -- row ``L`` is
the value for a champion at level ``L`` (1..18), row 0 unused -- and each is
pinned, at every level 1-18, to the C# expression that reads it.

The expected values are rebuilt here from the raw Content JSON with the
server's own list-building loops (`Package.cs:124-153`) and read expressions,
not through ``lanerl_jax.data.patch``, so a shared loader bug cannot make both
sides agree. The respawn off-by-one (`eaa2e77`) was a table built level-1 and
read as if level-indexed; these tests would have failed on it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.profiles import LEVEL_ROWS, build_profile_tables
from lanerl_jax.sim.rewards import level_for_xp
from lanerl_jax.sim.spells import RANKS_BY_LEVEL, SKILL_ORDER

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available")

LEVELS = range(1, 19)
_VENDOR_LIB = CONTENT_ROOT.parents[1] / "GameServerLib"


def _values(name: str) -> dict:
    with open(CONTENT_ROOT / "Maps" / "Map1" / f"{name}.json") as fh:
        return json.load(fh)["Values"]


def _cs_exp_curve() -> list:
    """``MapData.ExpCurve``: ``for (i = 2; i <= EXP.Count + 1; i++)
    ExpCurve.Add(EXP["Level{i}"])`` (`Package.cs:124-131`)."""
    exp = _values("ExpCurve")["EXP"]
    return [float(exp[f"Level{i}"]) for i in range(2, len(exp) + 2)]


def _cs_death_times() -> list:
    """``MapData.DeathTimes``: ``for (i = 1; i < TimeDeadPerLevel.Count; i++)``
    adding ``Level0{i}``/``Level{i}`` (`Package.cs:141-153`)."""
    t = _values("DeathTimes")["TimeDeadPerLevel"]
    return [float(t[f"Level0{i}" if i <= 9 else f"Level{i}"])
            for i in range(1, len(t))]


@pytest.fixture(scope="module")
def tables():
    return {k: np.asarray(v) for k, v in build_profile_tables(load_patch()).items()}


@pytest.mark.parametrize("name", ["xp_to_reach_level", "death_times",
                                  "champion_kill_exp"])
def test_every_level_table_has_one_row_per_level(tables, name):
    assert tables[name].shape == (LEVEL_ROWS,) == (19,)
    assert len(RANKS_BY_LEVEL) == LEVEL_ROWS


def test_xp_to_reach_level_is_the_level_up_threshold_at_every_level(tables):
    """``Champion.AddExperience`` (`Champion.cs:330`) loops
    ``while (Experience >= ExpCurve[Level - 1] && LevelUp())``, so a champion
    at level L-1 becomes L at ``ExpCurve[L - 2]``; level 1 needs nothing."""
    cs = _cs_exp_curve()
    f32 = np.float32
    for L in LEVELS:
        want = 0.0 if L == 1 else cs[(L - 1) - 1]
        assert tables["xp_to_reach_level"][L] == f32(want), L
    # ...and the reader agrees with the C# loop on both sides of every edge.
    curve = jnp.asarray(tables["xp_to_reach_level"])
    for L in LEVELS:
        at = float(tables["xp_to_reach_level"][L])
        assert int(level_for_xp(jnp.asarray([at], jnp.float32), curve)[0]) == L
        if L > 1:
            below = float(np.nextafter(f32(at), f32(0)))
            assert int(level_for_xp(jnp.asarray([below], jnp.float32),
                                    curve)[0]) == L - 1
    # The cap: `LevelUp` refuses past MaxLevel 18 however much XP there is.
    assert int(level_for_xp(jnp.asarray([1e9], jnp.float32), curve)[0]) == 18


def test_the_legacy_xp_curve_view_is_gone(tables):
    """One level-table convention: `parity/inject.py` reads
    ``xp_to_reach_level`` too, so the 18-row ``xp_curve`` view is deleted."""
    assert "xp_curve" not in tables


def test_death_times_is_the_death_timer_at_every_level(tables):
    """``RespawnTimer = MapData.DeathTimes[Stats.Level] * 1000``
    (`Champion.cs:400`) -- so row L is ``DeathTimes[L]``, JSON Level(L+1)."""
    cs = _cs_death_times()
    for L in LEVELS:
        assert tables["death_times"][L] == np.float32(cs[L]), L
    assert tables["death_times"][1] == np.float32(10.0), "Level02, not Level01"


def test_champion_kill_exp_is_the_kill_formula_base_at_every_level(tables):
    """``EXP = ExpCurve[Stats.Level - 1] * BaseExpMultiple`` (`Champion.cs:444`)."""
    cs = _cs_exp_curve()
    mult = float(_values("ExpCurve")["ExpGrantedOnDeath"]["BaseExpMultiple"])
    for L in LEVELS:
        assert tables["champion_kill_exp"][L] == np.float32(cs[L - 1] * mult), L


def _cs_skill_order() -> list:
    src = (_VENDOR_LIB / "Lanerl" / "LanerlConfig.cs").read_text()
    m = re.search(r"public int\[\] SkillOrder = \{([^}]*)\}", src)
    return [int(v) for v in m.group(1).split(",")]


def _cs_spells_up_levels() -> list:
    src = (_VENDOR_LIB / "Content" / "CharData.cs").read_text()
    m = re.search(r"SpellsUpLevels \{[^=]*=\s*\{(.*?)\};", src, re.S)
    return [[int(v) for v in row.split(",")]
            for row in re.findall(r"new\[\] \{([^}]*)\}", m.group(1))]


def test_ranks_by_level_is_autolevel_at_every_level():
    """``LanerlBot.AutoLevel`` (`LanerlBot.cs:285-321`), replayed: one skill
    point per level (`Champion.LevelUp`, `Champion.cs:345-348`), each spent on
    the first ``SkillOrder`` entry at or after the points already spent whose
    spell is below rank 5 and passes ``CanLevelUpSpell``
    (``SpellsUpLevels[slot][rank] <= Level``, `ObjAIBase.cs:367-370`). Garen's
    Content sets no ``SpellsUpLevels``, so the ``CharData`` defaults apply."""
    garen = json.loads((CONTENT_ROOT / "Stats" / "Garen" / "Garen.json").read_text())
    assert not any(k.startswith("SpellsUpLevels")
                   for k in garen["Values"]["Data"]), "Garen overrides the gate"
    order = _cs_skill_order()
    assert tuple(order) == SKILL_ORDER
    up = _cs_spells_up_levels()
    ranks = [0, 0, 0, 0]
    for L in LEVELS:
        points = 1
        while points:
            spent = sum(ranks)
            chosen = next((s for s in order[spent:]
                           if ranks[s] < 5 and up[s][ranks[s]] <= L), None)
            if chosen is None:
                break
            ranks[chosen] += 1
            points -= 1
        assert tuple(RANKS_BY_LEVEL[L]) == tuple(ranks), L


@pytest.mark.parametrize("level", [1, 8, 18])
def test_the_respawn_reader_uses_the_level_itself(level):
    """The reader half: a champion killed at ``level`` gets
    ``DeathTimes[level]`` seconds, minus the tick it died on."""
    from lanerl_jax.sim.init import init_lane, lane_params
    from lanerl_jax.sim.step import TICK_MS, tick

    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    s = s.replace(level=s.level.at[0].set(level),
                  xp=s.xp.at[0].set(params["xp_to_reach_level"][level]),
                  hp=s.hp.at[0].set(-1e4))
    out = jax.jit(lambda st: tick(st, params))(s)
    assert not bool(out.alive[0])
    want = np.float32(np.float32(_cs_death_times()[level] * 1000.0)
                      - np.float32(TICK_MS))
    assert np.float32(out.respawn_ms[0]) == want
