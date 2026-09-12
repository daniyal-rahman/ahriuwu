"""The leak audit must pass -- and must be able to fail."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lanerl_rl import audit
from lanerl_rl import constants as C
from lanerl_rl import frame as frame_module
from lanerl_rl.frame import ApproxFogModel, UnknownWireField, decode_frame
from lanerl_rl.obs import ObservationBuilder
from lanerl_rl.scenarios import encode_frame, top_lane_scenario


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


#: Where a leak probe writes its poison, now that there is no reserved padding.
#:
#: ``G_RESERVED`` -- eight permanently-zero globals -- was deleted along with
#: the other 206 dead inputs, and it was the obvious place to smuggle a value
#: into the actor observation without disturbing anything else. There is no
#: such place any more, which is the point of the layout, so a probe has to
#: overwrite a REAL field. Which one is irrelevant: every differential check
#: compares the whole actor array between a clean and a poisoned frame, so any
#: index that moves with enemy-private state is a finding.
_LEAK_SLOT = C.G_CLOCK_NORM


class _LeakyBuilder(ObservationBuilder):
    """Deliberately leaks the enemy's gold into an actor global."""

    def _build_global_vec(self, frame, self_u, enemy_u, visible, ax, ay):
        g = super()._build_global_vec(frame, self_u, enemy_u, visible, ax, ay)
        if enemy_u is not None:
            g[_LEAK_SLOT] = float(enemy_u.gold or 0.0) / C.NORM_GOLD
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
            # Straight into the witnessed-cast block. The estimate block this
            # used to poison (G_ENEMY_ABILITY_CD_EST) was deleted, and writing
            # the raw value here is the SAME bug in the field that remains:
            # "how long since I watched him cast it" replaced by "how many
            # seconds the server says are left on it".
            for i, cd in enumerate(enemy_u.cooldowns[:4]):
                g[C.G_ENEMY_ABILITY_SINCE_CAST.start + i] = (
                    0.0 if cd is None else float(cd) / 160.0
                )
        return g


class _DeadIntelBuilder(ObservationBuilder):
    """Enemy ability book wired to a constant -- the leak checks go vacuous."""

    def _build_global_vec(self, frame, self_u, enemy_u, visible, ax, ay):
        g = super()._build_global_vec(frame, self_u, enemy_u, visible, ax, ay)
        g[C.G_ENEMY_ABILITY_SINCE_CAST] = 0.0
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


# --------------------------------------------------------------------------
# Schema completeness: the check that makes a NEW server field fail by default
# --------------------------------------------------------------------------

#: A miniature emitter, shaped like the real one (escaped quotes, a looped key,
#: an apostrophe in a comment) so the parser is exercised the way it is in
#: production rather than on a toy.
_FAKE_EMITTER = r'''
        private string BuildObservation(Game game)
        {
            var sb = new StringBuilder(8192);
            sb.Append("{\"t\":").Append(((int)game.GameTime).ToString());
            sb.Append(",\"u\":[");
            foreach (var kv in game.ObjectManager.GetObjects())
            {
                // the champion's own row -- note the apostrophe
                sb.Append("{\"id\":").Append(kv.Key)
                  .Append(",\"k\":\"").Append(au.GetType().Name).Append('"')
                  .Append(",\"tm\":").Append((int)au.Team)
                  .Append(",\"x\":").Append((int)au.Position.X)
                  .Append(",\"y\":").Append((int)au.Position.Y)
                  .Append(",\"hp\":").Append((int)au.Stats.CurrentHealth)
                  .Append(",\"mhp\":").Append((int)au.Stats.HealthPoints.Total);
                for (byte sl = 0; sl < 4; sl++)
                {
                    sb.Append(",\"cd").Append(sl).Append("\":").Append(0);
                }
                sb.Append('}');
            }
            sb.Append("]}");
            return sb.ToString();
        }
'''


def _emitted(source):
    return audit.emitted_wire_keys(source)


def test_the_emitter_parser_survives_apostrophes_in_comments():
    """An apostrophe in prose opens a C# char literal and can eat the method.

    That failure mode reports FEWER keys than the server emits -- i.e. it makes
    the completeness check pass while seeing nothing -- so it gets its own test.
    """
    keys, unresolved = _emitted(_FAKE_EMITTER)
    assert unresolved == set()
    assert {"t", "u", "id", "k", "tm", "x", "y", "hp", "mhp"} <= keys
    assert {"cd0", "cd1", "cd2", "cd3"} <= keys


def test_the_real_emitter_parses(monkeypatch):
    """The check is worthless if it cannot read the file it is about."""
    path = audit.control_source_path()
    if path is None:
        pytest.fail(
            "LanerlControl.cs was not found, so the schema-completeness check "
            "cannot prove anything on this machine"
        )
    keys, unresolved = audit.emitted_wire_keys(path.read_text())
    assert unresolved == set(), unresolved
    # Exactly the set the wire-format docstring claims, and nothing else.
    assert keys == {
        "t", "u", "id", "k", "tm", "x", "y", "hp", "mhp", "vb", "vr",
        "gold", "xp", "lvl", "rc", "tgt", "atk", "mo",
        "cd0", "cd1", "cd2", "cd3",
        # added 2026-09-11: cs so cs_at_10 can be recorded at all, and
        # demo/slot for behaviour-cloning labels. Still an EXACT match --
        # a new emitted key must fail here until it is classified.
        "cs", "demo", "slot",
        # added 2026-09-12: the champion's REAL combat stats. Python used to
        # re-derive attack damage from a hand-copied base and level curve; that
        # copy read 57.88 at level 1, then 73.14 once someone modelled the rune
        # page, against the server's true 78.14 -- the rest being a mastery
        # page nobody had modelled. A re-derived server quantity is wrong by
        # however much of the server you forgot, so these are emitted instead.
        "ad", "ap", "ar", "mr", "as", "rng",
        # and the champion's SPELL RANKS. Python used to model these from an
        # assumed skill order (Q@1,W@2,E@3,R@6) while the server levels
        # Q,E,E,W,E,R -- so at champion level 2-3 the action mask forbade the
        # one ability the champion had and offered one it did not own.
        "sl",
    }, sorted(keys)


def test_an_unclassified_new_server_field_fails_the_audit(monkeypatch):
    """The whole point: a field nobody has classified must FAIL, not pass."""
    leaky = _FAKE_EMITTER.replace(
        '.Append(",\\"mhp\\":")', '.Append(",\\"soul\\":").Append(au.Soul).Append(",\\"mhp\\":")'
    )
    assert "soul" in leaky
    keys, _ = _emitted(leaky)
    assert "soul" in keys

    monkeypatch.setattr(audit, "emitted_wire_keys", lambda source=None: (keys, set()))
    monkeypatch.setattr(audit, "control_source_path", lambda: audit.Path(__file__))
    findings = audit.check_wire_schema_covers_the_emitter()
    assert any("'soul'" in f.message for f in findings), [str(f) for f in findings]


def test_an_unresolvable_looped_key_fails_the_audit(monkeypatch):
    """A key built from an expression the parser cannot resolve is not a pass."""
    keys, unresolved = _emitted(
        _FAKE_EMITTER.replace('",\\"cd"', '",\\"zz"')
    )
    assert unresolved == {"zz<EXPR>"}, (keys, unresolved)

    monkeypatch.setattr(audit, "emitted_wire_keys", lambda source=None: (keys, unresolved))
    monkeypatch.setattr(audit, "control_source_path", lambda: audit.Path(__file__))
    findings = audit.check_wire_schema_covers_the_emitter()
    assert any("zz<EXPR>" in f.message for f in findings), [str(f) for f in findings]


def test_a_broken_parse_fails_rather_than_passing(monkeypatch):
    """Finding no keys means the parser broke, not that the server emits none."""
    monkeypatch.setattr(audit, "control_source_path", lambda: audit.Path(__file__))
    monkeypatch.setattr(audit, "emitted_wire_keys", lambda source=None: (set(), set()))
    findings = audit.check_wire_schema_covers_the_emitter()
    assert any("no emitted keys at all" in f.message for f in findings)


def test_a_partial_parse_fails_rather_than_passing(monkeypatch):
    """Missing a key we know is there discredits the whole parse."""
    monkeypatch.setattr(audit, "control_source_path", lambda: audit.Path(__file__))
    monkeypatch.setattr(audit, "emitted_wire_keys", lambda source=None: ({"t", "u"}, set()))
    findings = audit.check_wire_schema_covers_the_emitter()
    assert any("missed keys that are certainly there" in f.message for f in findings)


def test_a_missing_emitter_source_is_a_finding_not_a_skip(monkeypatch):
    monkeypatch.setattr(audit, "control_source_path", lambda: None)
    findings = audit.check_wire_schema_covers_the_emitter()
    assert any("cannot find LanerlControl.cs" in f.message for f in findings)


def _patched_registry(monkeypatch, **replacements):
    reg = dict(frame_module.WIRE_FIELDS)
    for key, changes in replacements.items():
        if changes is None:
            reg.pop(key, None)
        elif key in reg:
            reg[key] = dataclasses.replace(reg[key], **changes)
        else:
            reg[key] = frame_module.WireField(key=key, **changes)
    monkeypatch.setattr(frame_module, "WIRE_FIELDS", reg)
    monkeypatch.setattr(audit, "WIRE_FIELDS", reg)
    return reg


def test_decoder_check_catches_a_false_unconsumed_claim(monkeypatch):
    """'Nothing reads it' is the strongest safety claim here, so it is verified."""
    _patched_registry(monkeypatch, gold={"disposition": "unconsumed", "decoded": False})
    findings = audit.check_wire_schema_matches_the_decoder()
    assert any("'gold'" in f.message and "NOT consumed" in f.message for f in findings), [
        str(f) for f in findings
    ]


def test_decoder_check_catches_a_stale_decoded_claim(monkeypatch):
    """A registry entry for a key nothing reads is an excuse with no subject."""
    _patched_registry(
        monkeypatch,
        zz={
            "scope": "unit",
            "disposition": "critic",
            "emitted": False,
            "decoded": True,
            "actor_invariant": True,
            "reason": "test fixture",
        },
    )
    findings = audit.check_wire_schema_matches_the_decoder()
    assert any("'zz'" in f.message and "never reads it" in f.message for f in findings), [
        str(f) for f in findings
    ]


def test_registry_check_catches_an_undeclared_actor_visible_field(monkeypatch):
    _patched_registry(monkeypatch, tgt={"actor_invariant": False})
    findings = audit.check_actor_visible_wire_fields_are_declared()
    assert any("'tgt'" in f.message for f in findings), [str(f) for f in findings]


# -- the generated differential probe ---------------------------------------


class _LevelLeakyBuilder(ObservationBuilder):
    """Leaks the ENEMY's level -- a field only the generated probe covers."""

    def build(self, frame):
        self._leak_frame = frame
        return super().build(frame)

    def _build_global_vec(self, frame, self_u, enemy_u, visible, ax, ay):
        g = super()._build_global_vec(frame, self_u, enemy_u, visible, ax, ay)
        if enemy_u is not None:
            g[_LEAK_SLOT] = float(enemy_u.lvl or 0) / 18.0
        return g


def test_the_generated_probe_catches_a_leak_no_handwritten_probe_covers(monkeypatch):
    """``lvl`` has no bespoke check; the registry-driven one must still catch it."""
    _patched(monkeypatch, _LevelLeakyBuilder)
    findings = audit.check_wire_fields_do_not_leak_to_the_actor()
    assert any("'lvl'" in f.message for f in findings), [str(f) for f in findings]


def test_the_generated_probe_reports_itself_vacuous(monkeypatch):
    """A field claimed decoded that nothing reads makes its own probe empty."""
    _patched_registry(monkeypatch, tgt={"disposition": "critic", "decoded": True})
    findings = audit.check_wire_fields_do_not_leak_to_the_actor()
    assert any("'tgt'" in f.message and "vacuous" in f.message for f in findings), [
        str(f) for f in findings
    ]


def test_the_generated_probe_rejects_a_false_unconsumed_claim(monkeypatch):
    _patched_registry(monkeypatch, gold={"disposition": "unconsumed", "decoded": False})
    findings = audit.check_wire_fields_do_not_leak_to_the_actor()
    assert any(
        "'gold'" in f.message and "registered as unconsumed" in f.message for f in findings
    ), [str(f) for f in findings]


# -- the runtime half: decode_frame is strict -------------------------------


def test_decode_frame_rejects_an_unknown_unit_field():
    raw = encode_frame(top_lane_scenario())
    raw["u"][0]["soul"] = 3
    with pytest.raises(UnknownWireField, match="soul"):
        decode_frame(raw)


def test_decode_frame_rejects_an_unknown_record_field():
    raw = encode_frame(top_lane_scenario())
    raw["fog_override"] = 1
    with pytest.raises(UnknownWireField, match="fog_override"):
        decode_frame(raw)


def test_the_unknown_field_escape_hatch_warns_and_does_not_stick(monkeypatch):
    """The hatch is for replaying an old dump; it must not bless the key set."""
    raw = encode_frame(top_lane_scenario())
    raw["u"][0]["soul"] = 3
    monkeypatch.setenv("LANERL_ALLOW_UNKNOWN_WIRE_FIELDS", "1")
    with pytest.warns(RuntimeWarning, match="soul"):
        decode_frame(raw)
    monkeypatch.delenv("LANERL_ALLOW_UNKNOWN_WIRE_FIELDS")
    with pytest.raises(UnknownWireField, match="soul"):
        decode_frame(raw)


def test_every_registered_wire_field_states_a_reason():
    for key, f in frame_module.WIRE_FIELDS.items():
        assert f.reason.strip(), f"{key} has no reason"
        assert (f.disposition == "unconsumed") != f.decoded, key


def test_privileged_arrays_really_are_privileged():
    """The critic input must change when the actor input cannot."""
    from lanerl_rl.scenarios import top_lane_scenario

    fog = ApproxFogModel(warn=False)
    a = ObservationBuilder(C.TEAM_BLUE, fog_model=fog).build(top_lane_scenario(red_gold=100.0))
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=fog).build(top_lane_scenario(red_gold=9000.0))
    assert np.array_equal(a.entities, b.entities)
    assert np.array_equal(a.global_vec, b.global_vec)
    assert not np.array_equal(a.priv_vec, b.priv_vec)
