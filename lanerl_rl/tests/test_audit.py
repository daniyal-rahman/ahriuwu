"""The leak audit must pass -- and must be able to fail."""

from __future__ import annotations

import numpy as np

from lanerl_rl import audit
from lanerl_rl import constants as C
from lanerl_rl.frame import ApproxFogModel
from lanerl_rl.obs import ObservationBuilder


def test_audit_passes():
    findings = audit.run_audit(verbose=False)
    assert findings == [], "\n".join(str(f) for f in findings)


def test_audit_main_returns_zero(capsys):
    assert audit.main([]) == 0
    out = capsys.readouterr().out
    assert "AUDIT PASSED" in out


# --------------------------------------------------------------------------
# Negative controls: an audit that cannot fail is worthless.
# --------------------------------------------------------------------------


class _LeakyBuilder(ObservationBuilder):
    """Deliberately leaks the enemy's gold into a reserved global slot."""

    def _build_global_vec(self, frame, self_u, enemy_u, visible, ax, ay):
        g = super()._build_global_vec(frame, self_u, enemy_u, visible, ax, ay)
        if enemy_u is not None:
            g[C.G_RESERVED.start] = float(enemy_u.gold or 0.0) / C.NORM_GOLD
        return g


class _FogLeakyBuilder(ObservationBuilder):
    """Deliberately writes the enemy's LIVE position into a valid slot.

    This is the exact bug the fog checks exist to catch: reading the raw frame
    on the actor path instead of the fog-gated memory.
    """

    def build(self, frame):
        self._leak_frame = frame
        return super().build(frame)

    def _slot_entities(self, t_ms, ax, ay, self_id, visible):
        import math

        from lanerl_rl.obs import _SlotEntity

        slots = super()._slot_entities(t_ms, ax, ay, self_id, visible)
        for u in self._leak_frame.units.values():
            if u.etype == "champion" and u.team == self.enemy_team:
                cs_, cn_ = self.transform.point(u.x, u.y)
                slots[C.SLOT_ENEMY_CHAMP[0]] = _SlotEntity(
                    uid=u.id,
                    etype="champion",
                    team_rel="enemy",
                    s=cs_,
                    n=cn_,
                    dist=math.hypot(cs_ - ax, cn_ - ay),
                    visible=True,
                    on_screen=True,
                    hp_known=True,
                    staleness=0.0,
                    age_s=0.0,
                    hp_frac=u.hp / u.mhp,
                    mhp=u.mhp,
                    hp_d_short=0.0,
                    hp_d_long=0.0,
                    vs=0.0,
                    vn=0.0,
                    heading=None,
                    reach_radius=0.0,
                )
        return slots


class _CooldownLeakyBuilder(ObservationBuilder):
    """Deliberately writes the enemy's LIVE remaining cooldowns into the globals.

    The exact bug the two cooldown differential checks exist to catch: reading
    the enemy's cooldown *value* off the server instead of inferring a cast
    from something a player could actually see.
    """

    def build(self, frame):
        self._leak_frame = frame
        return super().build(frame)

    def _build_global_vec(self, frame, self_u, enemy_u, visible, ax, ay):
        g = super()._build_global_vec(frame, self_u, enemy_u, visible, ax, ay)
        if enemy_u is not None and enemy_u.cooldowns is not None:
            for i, cd in enumerate(enemy_u.cooldowns[:4]):
                g[C.G_ENEMY_ABILITY_CD_EST.start + i] = 0.0 if cd is None else float(cd) / 160.0
        return g


class _DeadIntelBuilder(ObservationBuilder):
    """Enemy ability book wired to a constant -- the leak checks go vacuous."""

    def _build_global_vec(self, frame, self_u, enemy_u, visible, ax, ay):
        g = super()._build_global_vec(frame, self_u, enemy_u, visible, ax, ay)
        g[C.G_ENEMY_ABILITY_SINCE_CAST] = 0.0
        g[C.G_ENEMY_ABILITY_CD_EST] = 0.0
        g[C.G_ENEMY_ABILITY_UNKNOWN] = 0.0
        return g


def _patched(monkeypatch, cls):
    monkeypatch.setattr(audit, "ObservationBuilder", cls)


def test_audit_detects_an_economy_leak(monkeypatch):
    _patched(monkeypatch, _LeakyBuilder)
    findings = audit.check_enemy_economy_leak()
    assert findings, "the economy differential check failed to notice a real leak"
    assert any("global_vec" in f.message for f in findings)


def test_audit_detects_a_fog_leak(monkeypatch):
    _patched(monkeypatch, _FogLeakyBuilder)
    findings = audit.check_fog_leak() + audit.check_all_slots_never_alias_fogged()
    assert findings, "the fog differential check failed to notice a real leak"


def test_audit_detects_a_fogged_cooldown_leak(monkeypatch):
    """Writing the enemy's live cooldowns into the globals must be caught."""
    _patched(monkeypatch, _CooldownLeakyBuilder)
    findings = audit.check_enemy_cooldown_leak()
    assert findings, "the fogged-cooldown check failed to notice a real leak"
    assert any("global_vec" in f.message for f in findings)


def test_audit_detects_a_visible_cooldown_value_leak(monkeypatch):
    """Reading the VALUE off a visible enemy is a leak even though seeing them is not."""
    _patched(monkeypatch, _CooldownLeakyBuilder)
    findings = audit.check_visible_enemy_cooldown_value_leak()
    assert findings, "the visible-cooldown check failed to notice a value read"
    assert any("jittered below the cast threshold" in f.message for f in findings)


def test_cooldown_check_rejects_a_dead_ability_book(monkeypatch):
    """A leak probe that cannot fire is worthless; the positive control catches that."""
    _patched(monkeypatch, _DeadIntelBuilder)
    findings = audit.check_visible_enemy_cooldown_value_leak()
    assert any("vacuous" in f.message for f in findings), [str(f) for f in findings]


def test_audit_detects_a_bad_field_name(monkeypatch):
    monkeypatch.setattr(
        C, "GLOBAL_FIELD_NAMES", list(C.GLOBAL_FIELD_NAMES) + ["enemy_gold_norm"]
    )
    findings = audit.check_field_names()
    assert any("enemy_gold_norm" in f.message for f in findings)


def test_audit_detects_a_static_privileged_read(monkeypatch, tmp_path):
    """Point the static checker at a module whose actor path reads .gold."""
    leaky = tmp_path / "leaky_obs.py"
    leaky.write_text(
        "def _build_global_vec(self, frame, self_u, enemy_u, visible, ax, ay):\n"
        "    return enemy_u.gold\n"
        "def _slot_entities(self, t_ms, ax, ay, self_id, visible):\n"
        "    return list(frame.units.values())\n"
    )

    import inspect as _inspect

    monkeypatch.setattr(audit, "ACTOR_PATH_FUNCTIONS", ("_build_global_vec", "_slot_entities"))
    monkeypatch.setattr(_inspect, "getsourcefile", lambda _m: str(leaky))
    monkeypatch.setattr(audit, "inspect", _inspect)

    findings = audit.check_actor_path_attrs()
    msgs = " ".join(f.message for f in findings)
    assert ".gold" in msgs, msgs
    assert "frame.units" in msgs, msgs


def test_privileged_arrays_really_are_privileged():
    """The critic input must change when the actor input cannot."""
    from lanerl_rl.scenarios import top_lane_scenario

    fog = ApproxFogModel(warn=False)
    a = ObservationBuilder(C.TEAM_BLUE, fog_model=fog).build(top_lane_scenario(red_gold=100.0))
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=fog).build(top_lane_scenario(red_gold=9000.0))
    assert np.array_equal(a.entities, b.entities)
    assert np.array_equal(a.global_vec, b.global_vec)
    assert not np.array_equal(a.priv_vec, b.priv_vec)
