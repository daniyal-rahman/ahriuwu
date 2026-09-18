"""Guards for the guard.

`METH-001` exists because a measurement can silently run the wrong code. A
fingerprint that fails to move is worth *less* than no fingerprint, because it
converts "I did not check" into "I checked and it was fine" -- so the digest's
sensitivity is the property under test here, not its stability.
"""
from __future__ import annotations

import pytest

from lanerl_jax.parity import provenance as prov


def test_the_digest_is_stable_when_nothing_changes():
    """A digest that moves on its own makes every before/after pair look valid."""
    first, per_first = prov.module_digest()
    second, per_second = prov.module_digest()
    assert first == second
    assert per_first == per_second
    assert len(first) == 12


@pytest.mark.parametrize("attr", ["MASTERY_AD_PER_LEVEL_BONUS",
                                 "MASTERY_HP_FLAT_BONUS",
                                 "RUNE_AD_BONUS"])
def test_changing_a_module_level_constant_moves_the_digest(attr):
    """**The case a bytecode-only digest misses.**

    These are module globals, looked up by name at call time, so they appear in
    no function's ``co_consts``. A fidelity fix is very often exactly one of
    them -- ``STAT-002`` is ``3.5 -> 4.05`` and nothing else -- so a digest
    blind to them would sit still through the change it most needs to catch.
    """
    import lanerl_jax.sim.init as init

    before, _ = prov.module_digest()
    original = getattr(init, attr)
    try:
        setattr(init, attr, original + 1.0)
        after, _ = prov.module_digest()
    finally:
        setattr(init, attr, original)
    assert after != before, f"{attr} changed and the digest did not move"
    restored, _ = prov.module_digest()
    assert restored == before, "the digest did not come back"


def test_changing_a_functions_bytecode_moves_the_digest():
    import lanerl_jax.sim.combat as combat

    before, _ = prov.module_digest()
    original = combat.growth_sum
    try:
        def growth_sum(level, xp=None):      # different code object entirely
            return 0.0
        growth_sum.__module__ = "lanerl_jax.sim.combat"
        combat.growth_sum = growth_sum
        after, _ = prov.module_digest()
    finally:
        combat.growth_sum = original
    assert after != before


def test_git_never_reports_absence_by_returning_an_empty_string():
    """`git` on the compute node fails *silently*: the dubious-ownership check
    and the worktree's absolute `gitdir:` pointer both exit non-zero with empty
    stdout, which a naive caller reads as "no git here". Whatever happens, this
    must return something a human can act on.
    """
    head = prov._git("rev-parse", "--short", "HEAD")
    assert head, "empty string is the one forbidden answer"
    assert head.startswith("<") or len(head) >= 7, head


def test_provenance_reports_the_package_that_was_actually_imported():
    """Not a configured path -- the one Python resolved. A compute node reading
    a different tree is exactly the failure this is for."""
    import lanerl_jax

    p = prov.provenance()
    assert p["package_path"] == str(
        __import__("pathlib").Path(lanerl_jax.__file__).parent)
    for key in ("bytecode_digest", "per_module", "constants", "git_head",
                "git_dirty", "host", "python"):
        assert key in p, key
    assert set(p["per_module"]) == set(prov.MODULES)


def test_the_named_constants_are_all_present():
    """A renamed constant silently dropping out of the report would quietly
    shrink the guard. `<absent>` is reported rather than raised, so assert
    here that none currently is."""
    values = prov.constant_values()
    assert values, "no constants configured"
    missing = [k for k, v in values.items() if v == "<absent>"]
    assert not missing, f"CONSTANTS names a constant that no longer exists: {missing}"


def test_format_states_the_rule_it_exists_to_enforce():
    text = prov.format_provenance()
    assert "void" in text and "digest" in text
