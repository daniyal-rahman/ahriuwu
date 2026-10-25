from lanerl_jax.modern.bc.ability_metadata import ability_evidence


def test_inventory_never_converts_missing_metadata_into_defaults_or_sentinel_range():
    catalog = dict(slots=[dict(value='root')], spells=[dict(
        name=dict(value='spell'), slot=dict(value=0, provenance=[dict(entry='root')]),
        cast_mode=dict(value=None, available=False),
        base_cooldown_seconds_by_rank=dict(value=None, available=False),
        configured_range_rank_table=dict(value=[25000.] * 6))])
    result = ability_evidence(catalog, 0)
    assert result['raw_mode'] is None and result['raw_cooldown_by_rank'] is None
    assert result['range_has_sentinel'] and result['qualified_effect_range'] is None
    assert result['qualified_policy_mode'] is None
