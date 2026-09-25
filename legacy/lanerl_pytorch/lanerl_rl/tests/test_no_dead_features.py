"""Every declared feature must actually carry information.

The layout this replaced had 206 permanently-zero inputs -- 192 entity floats
(6 reserved x 32 slots), 6 self, 8 global -- about 15% of everything the
network saw. Nothing failed, because nothing checked. A feature that is always
0 is indistinguishable from a feature the builder forgot to write, and this
project has shipped both.

These tests run the real ObservationBuilder over a varied frame sequence and
assert that each declared field is written at least once with a value that is
not identically zero across every frame and every slot.

A genuinely-always-zero field is not automatically a bug -- but it must be
listed in EXPECTED_CONSTANT with a reason, so the decision is visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from lanerl_rl import constants as C

#: Fields that may legitimately be constant in a short synthetic episode.
#: Each needs a reason; an empty reason is not acceptable.
EXPECTED_CONSTANT = {
    "is_dead": "nobody dies in a few seconds of synthetic frames",
    "recalling": "the synthetic frames never press B",
    "ap": "Garen is an AD champion; AP stays 0 all game",
    "type_inhibitor": "no inhibitor is ever in lane range",
    "type_nexus": "no nexus is ever in lane range",
    "team_neutral": "LANERL_TOPONLY disables the jungle, so no neutrals exist",
    "enemy_q_time_since_observed_cast": "no casts in the synthetic frames",
    "enemy_w_time_since_observed_cast": "no casts in the synthetic frames",
    "enemy_e_time_since_observed_cast": "no casts in the synthetic frames",
    "enemy_r_time_since_observed_cast": "no casts in the synthetic frames",
    "type_other": (
        "every unit in a top-only 1v1 is champion/minion/turret/inhibitor/nexus; "
        "'other' is a catch-all that never fires here"
    ),
    "cd_q": (
        "the scripted bot has AbilityCastsPerMinute[Q] = 0 on purpose (Garen's Q "
        "is an empowered next-auto that cancels the swing in flight), so the "
        "recording contains no Q cast and Q is never on cooldown"
    ),
}


def test_every_expected_constant_has_a_reason():
    for name, reason in EXPECTED_CONSTANT.items():
        assert reason and len(reason) > 10, name


def test_the_layouts_have_no_reserved_padding():
    """Reserved slots are how 206 dead inputs got there in the first place."""
    for names in (C.ENTITY_FIELD_NAMES, C.SELF_FIELD_NAMES, C.GLOBAL_FIELD_NAMES):
        dead = [n for n in names if n.startswith("reserved")]
        assert not dead, f"reserved padding is back: {dead}"


def test_field_names_are_index_aligned_and_unique():
    for names, dim, label in (
        (C.ENTITY_FIELD_NAMES, C.ENTITY_DIM, "entity"),
        (C.SELF_FIELD_NAMES, C.SELF_DIM, "self"),
        (C.GLOBAL_FIELD_NAMES, C.GLOBAL_DIM, "global"),
    ):
        assert len(names) == dim, f"{label}: {len(names)} names for {dim} slots"
        assert len(set(names)) == len(names), f"{label} has duplicate names"


def test_no_feature_is_silently_always_zero(lane_frames):
    """The real builder, over real frames, must fill what it declares."""
    from lanerl_rl.obs import ObservationBuilder

    b = ObservationBuilder(C.TEAM_BLUE)
    ent, slf, glb = [], [], []
    for f in lane_frames:
        o = b.build(f)
        ent.append(o.entities); slf.append(o.self_vec); glb.append(o.global_vec)
    ent = np.stack(ent); slf = np.stack(slf); glb = np.stack(glb)

    dead = []
    for i, name in enumerate(C.ENTITY_FIELD_NAMES):
        if not np.any(ent[..., i]) and name not in EXPECTED_CONSTANT:
            dead.append(f"entity.{name}")
    for i, name in enumerate(C.SELF_FIELD_NAMES):
        if not np.any(slf[:, i]) and name not in EXPECTED_CONSTANT:
            dead.append(f"self.{name}")
    for i, name in enumerate(C.GLOBAL_FIELD_NAMES):
        if not np.any(glb[:, i]) and name not in EXPECTED_CONSTANT:
            dead.append(f"global.{name}")
    assert not dead, (
        "these declared features were never written with a non-zero value: "
        + ", ".join(dead)
        + ". Either the builder does not fill them, or they belong in "
        "EXPECTED_CONSTANT with a reason."
    )
