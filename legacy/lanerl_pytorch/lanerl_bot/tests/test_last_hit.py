"""The last-hit predicate: correctness against known HP/damage values, and parity
with the C# implementation the anchor actually plays."""
from __future__ import annotations

import math
import re

import pytest

from lanerl_bot import build, content
from lanerl_bot.damage import (
    LastHitModel,
    garen_e_tick_damage,
    garen_q_damage,
    is_last_hit,
    post_mitigation,
    windup_seconds,
)
from pathlib import Path
from lanerl_bot.tests.conftest import requires_server

_VENDOR = str(Path(__file__).resolve().parents[3] / "lanerl-vendor")


# ----------------------------------------------------------------------------
# the numbers the predicate is built on
# ----------------------------------------------------------------------------
def test_content_matches_the_running_server():
    """Guard the constants the rest of the tests reason about."""
    garen = content.champion("Garen")
    assert garen.base_damage == pytest.approx(57.88, abs=0.01)
    assert garen.attack_range == pytest.approx(125.0)
    assert garen.is_melee

    melee = content.unit("Blue_Minion_Basic")
    caster = content.unit("Blue_Minion_Wizard")
    cannon = content.unit("Blue_Minion_MechCannon")
    assert (melee.base_hp, melee.armor) == (455.0, 0.0)
    assert (caster.base_hp, caster.armor) == (290.0, 0.0)
    assert (cannon.base_hp, cannon.armor) == (700.0, 15.0)
    assert melee.gold_on_death == 20.0
    assert cannon.gold_on_death == 35.0


def test_post_mitigation_matches_the_engine_formula():
    # damage * 100/(100+armor)
    assert post_mitigation(100, 0) == pytest.approx(100.0)
    assert post_mitigation(100, 100) == pytest.approx(50.0)
    assert post_mitigation(115, 15) == pytest.approx(100.0)
    # negative resist is mirrored (2 - pct), not left to blow up as resist -> -100
    assert post_mitigation(100, -50) == pytest.approx(100 * (2 - 100 / 50))
    assert post_mitigation(100, -30) == pytest.approx(100 * (2 - 100 / 70))
    assert post_mitigation(0, 30) == 0.0
    assert post_mitigation(-5, 30) == 0.0


def test_garen_windup_is_a_third_of_a_second():
    """1.6s swing * (0.300 global - 0.0916667 Garen) = 0.3333s."""
    g = content.champion("Garen")
    w = windup_seconds(g.attack_delay_offset_pct, g.attack_delay_cast_offset_pct)
    assert w == pytest.approx(1 / 3, abs=0.002)
    # attack speed divides it
    w2 = windup_seconds(g.attack_delay_offset_pct, g.attack_delay_cast_offset_pct, 2.0)
    assert w2 == pytest.approx(w / 2, abs=1e-6)


# ----------------------------------------------------------------------------
# the predicate itself, on known values
# ----------------------------------------------------------------------------
def test_level_1_damage_against_each_minion_type():
    m = LastHitModel(level=1)
    assert m.attack_damage == pytest.approx(57.88, abs=0.01)
    # melee and caster have 0 armor -> full damage
    assert m.damage_vs("Blue_Minion_Basic") == pytest.approx(57.88, abs=0.01)
    assert m.damage_vs("Blue_Minion_Wizard") == pytest.approx(57.88, abs=0.01)
    # cannon has 15 armor -> 57.88 * 100/115
    assert m.damage_vs("Blue_Minion_MechCannon") == pytest.approx(57.88 * 100 / 115, abs=0.01)


@pytest.mark.parametrize(
    "hp,expected",
    [
        (0.0, False),    # already dead: not a last hit
        (-1.0, False),
        (1.0, True),
        (57.0, True),    # just inside one auto
        (57.88, True),   # exactly one auto
        (58.5, False),   # one point too healthy
        (200.0, False),
        (455.0, False),  # full melee minion
    ],
)
def test_last_hit_boundary_on_a_zero_armour_minion(hp, expected):
    m = LastHitModel(level=1)
    assert m.can_last_hit("Blue_Minion_Basic", hp=hp) is expected


def test_last_hit_boundary_shifts_with_armour():
    """The cannon's 15 armour moves the threshold down by ~13%."""
    m = LastHitModel(level=1)
    dmg = m.damage_vs("Blue_Minion_MechCannon")
    assert m.can_last_hit("Blue_Minion_MechCannon", hp=dmg - 0.5)
    assert not m.can_last_hit("Blue_Minion_MechCannon", hp=dmg + 0.5)
    # and a minion that a zero-armour check would have called killable is not
    assert 50.0 < dmg < 57.88
    assert not m.can_last_hit("Blue_Minion_MechCannon", hp=57.0)


def test_incoming_damage_lets_the_bot_commit_earlier():
    """A minion still above one auto now, but not once the wave's hits land."""
    m = LastHitModel(level=1)
    hp = 90.0
    assert not m.can_last_hit("Blue_Minion_Basic", hp=hp)
    assert m.can_last_hit("Blue_Minion_Basic", hp=hp, incoming_damage=40.0)


def test_declining_is_off_by_default_and_can_be_turned_on():
    """The asymmetry that is worth about a quarter of CS@10.

    A minion the wave is about to finish: by default we swing anyway (an auto is
    nearly free, a lost CS is not); with decline_when_wave_kills we skip it.
    """
    m = LastHitModel(level=1)
    kw = dict(minion_name="Blue_Minion_Basic", hp=30.0, incoming_damage=50.0)
    assert m.can_last_hit(**kw) is True
    assert m.can_last_hit(**kw, decline_when_wave_kills=True) is False


def test_attack_cooldown_pushes_the_landing_later():
    """Regeneration during a long wait can flip a borderline call."""
    # a unit with regen: the siege minion has none on this server, so use the model
    # directly to keep the test about the rule rather than about content
    assert is_last_hit(target_hp=57.0, my_damage=57.88, time_to_land=0.333,
                       regen_per_second=0.0)
    assert not is_last_hit(target_hp=57.0, my_damage=57.88, time_to_land=2.0,
                           regen_per_second=5.0)


def test_autos_to_kill_a_full_minion():
    m = LastHitModel(level=1)
    assert m.autos_to_kill("Blue_Minion_Basic") == math.ceil(455 / 57.88)
    assert m.autos_to_kill("Blue_Minion_Wizard") == math.ceil(290 / 57.88)


def test_damage_grows_with_level():
    d1 = LastHitModel(level=1).damage_vs("Blue_Minion_Basic")
    d6 = LastHitModel(level=6).damage_vs("Blue_Minion_Basic")
    d11 = LastHitModel(level=11).damage_vs("Blue_Minion_Basic")
    assert d1 < d6 < d11
    # a level-11 Garen should one-shot a caster minion
    assert LastHitModel(level=11).can_last_hit("Blue_Minion_Wizard", hp=d11 - 1)


# ----------------------------------------------------------------------------
# ability damage, taken from this server's scripts
# ----------------------------------------------------------------------------
def test_q_execute_beats_a_plain_auto():
    ad = LastHitModel(level=1).attack_damage
    q = garen_q_damage(ad, rank=1, target_armor=0.0)
    assert q == pytest.approx(30 + 1.4 * ad, abs=0.01)
    assert q > ad  # which is why Q is worth spending on a cannon


def test_e_max_first_is_justified_by_the_actual_spell_data():
    """The reason GAREN_SKILL_ORDER maxes E: its marginal rank is worth more.

    Compared per rank against a wave, not per cast against one target.
    """
    ad = LastHitModel(level=1).attack_damage
    # Q's marginal rank is a flat +25 on one target
    q1 = garen_q_damage(ad, 1, 0.0)
    q2 = garen_q_damage(ad, 2, 0.0)
    assert q2 - q1 == pytest.approx(25.0, abs=0.01)

    # E's marginal rank, over the 6 ticks its 3s buff lands, against one minion
    e1 = garen_e_tick_damage(ad, 1, 0.0) * 6
    e2 = garen_e_tick_damage(ad, 2, 0.0) * 6
    assert e2 - e1 > q2 - q1
    # and E hits every minion in 330 units, so a 6-minion wave multiplies that again
    assert (e2 - e1) * 6 > (q2 - q1) * 10

    # cooldowns back it up: E's drops with rank, Q's never does
    assert build.q_cooldowns() == [8.0] * 5
    assert build.e_cooldowns() == [13.0, 12.0, 11.0, 10.0, 9.0]


def test_skill_order_is_well_formed_and_maxes_e_first():
    order = build.GAREN_SKILL_ORDER
    assert len(order) == 18
    counts = {s: order.count(s) for s in (build.Q, build.W, build.E, build.R)}
    assert counts == {build.Q: 5, build.W: 5, build.E: 5, build.R: 3}
    # R only at 6/11/16, which is also what the content data gates it to
    assert [i + 1 for i, s in enumerate(order) if s == build.R] == [6, 11, 16]
    # E is maxed before Q
    e_max = max(i for i, s in enumerate(order) if s == build.E)
    q_max = max(i for i, s in enumerate(order) if s == build.Q)
    assert e_max < q_max
    assert build.ranks_at(8)[build.E] == 5


def test_build_path_items_exist_and_start_is_affordable():
    from pathlib import Path

    items = Path(_VENDOR + "/LoLServer/Content/"
                 "LeagueSandbox-Default/Items")
    for item_id in build.GAREN_BUILD_PATH:
        assert (items / str(item_id) / f"{item_id}.json").exists(), item_id
    # Doran's Shield must be affordable out of the 475 starting gold
    import json
    doran = json.loads((items / "1054/1054.json").read_text())["Values"]["Data"]
    assert float(doran["Price"]) <= content.STARTING_GOLD


# ----------------------------------------------------------------------------
# parity with the C# model the anchor actually plays
# ----------------------------------------------------------------------------
POSTMIT_RE = re.compile(r"LANERL_SELFTEST postmit d=(\S+) r=(\S+) out=(\S+)")
LASTHIT_RE = re.compile(
    r"LANERL_SELFTEST lasthit hp=(\S+) dmg=(\S+) inc=(\S+) ttl=(\S+) regen=(\S+) "
    r"decline=(\d) out=(\d)"
)


@pytest.mark.slow
@requires_server
def test_python_and_csharp_agree_on_mitigation(selftest_output):
    rows = POSTMIT_RE.findall(selftest_output)
    assert rows, "server printed no mitigation rows"
    for d, r, out in rows:
        assert post_mitigation(float(d), float(r)) == pytest.approx(float(out), rel=1e-5)


@pytest.mark.slow
@requires_server
def test_python_and_csharp_agree_on_the_last_hit_rule(selftest_output):
    rows = LASTHIT_RE.findall(selftest_output)
    assert rows, "server printed no last-hit rows"
    for hp, dmg, inc, ttl, regen, decline, out in rows:
        got = is_last_hit(
            target_hp=float(hp), my_damage=float(dmg), incoming_damage=float(inc),
            time_to_land=float(ttl), regen_per_second=float(regen),
            decline_when_wave_kills=decline == "1",
        )
        assert got is (out == "1"), (hp, dmg, inc, decline, out)
