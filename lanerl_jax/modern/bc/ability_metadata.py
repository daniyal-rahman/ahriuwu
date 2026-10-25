"""Saved-client ability evidence inventory, not a simulator/default-value table."""
from __future__ import annotations


def value(field):
    if isinstance(field, dict):
        return field.get("value") if field.get("available", True) else None
    return field


def ability_evidence(catalog, slot):
    entry = value(catalog["slots"][slot])
    roots = [sp for sp in catalog["spells"] if value(sp.get("slot")) == slot
             and any(p.get("entry") == entry for p in sp.get("slot", {}).get("provenance", []))]
    if len(roots) != 1:
        raise ValueError(f"slot {slot}: expected one explicit root, found {len(roots)}")
    root = roots[0]
    ranges = value(root.get("configured_range_rank_table"))
    # Raw cast-range sentinels are not projectile distance or executable dash range.
    sentinel = ranges is not None and any(float(x) >= 5000 for x in ranges)
    return dict(name=value(root.get("name")), raw_mode=value(root.get("cast_mode")),
                raw_cooldown_by_rank=value(root.get("base_cooldown_seconds_by_rank")),
                raw_configured_ranges=ranges, range_has_sentinel=sentinel,
                qualified_effect_range=None, qualified_policy_mode=None,
                provenance=root.get("slot", {}).get("provenance", []),
                qualification="raw evidence only; policy mode/effect range require semantic verification")
