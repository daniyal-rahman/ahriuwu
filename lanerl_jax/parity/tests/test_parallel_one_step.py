from dataclasses import asdict

from lanerl_jax.parity.archive.one_step import FieldStats, OneStepResult, merge_one_step_results
from lanerl_jax.parity.archive.parallel_one_step import partition_pair_ranges


def test_partition_pair_ranges_is_balanced_contiguous_and_complete():
    ranges = partition_pair_ranges(10, 3)
    assert ranges == [(0, 4), (4, 7), (7, 10)]
    assert partition_pair_ranges(2, 8) == [(0, 1), (1, 2)]
    assert partition_pair_ranges(0, 4) == []


def test_merge_one_step_results_adds_all_counters_and_preserves_error_order():
    a = OneStepResult(
        n_ticks=2, n_ticks_skipped=1, n_units_seen=4,
        death_confusion={"LaneMinion": {
            "both_alive": 2, "sim_only_alive": 1,
            "server_only_alive": 0, "both_dead": 0,
        }},
        spawn_mismatches={"LaneMinion": 1},
        recovery_counts={"exact": 3},
        per_tick_worst_pos_error=[(20, 2.0), (10, 1.0)],
    )
    a.fields[("LaneMinion", "hp")] = FieldStats(
        "hp", "LaneMinion", n_total=2, n_exact=1, errors=[1.0])
    b = OneStepResult(
        n_ticks=3, n_units_seen=6,
        death_confusion={"LaneMinion": {
            "both_alive": 3, "sim_only_alive": 0,
            "server_only_alive": 1, "both_dead": 1,
        }},
        n_spawn_ticks={"LaneMinion": 2},
        recovery_counts={"exact": 5, "proxy": 2},
        per_tick_worst_pos_error=[(30, 3.0)],
    )
    b.fields[("LaneMinion", "hp")] = FieldStats(
        "hp", "LaneMinion", n_total=3, n_exact=2, errors=[2.0])

    merged = merge_one_step_results([a, b])

    assert merged.n_ticks == 5
    assert merged.n_ticks_skipped == 1
    assert merged.n_units_seen == 10
    assert asdict(merged.fields[("LaneMinion", "hp")]) == {
        "name": "hp", "kind": "LaneMinion", "n_total": 5,
        "n_exact": 3, "errors": [1.0, 2.0],
    }
    assert merged.death_confusion["LaneMinion"] == {
        "both_alive": 5, "sim_only_alive": 1,
        "server_only_alive": 1, "both_dead": 1,
    }
    assert merged.spawn_mismatches == {"LaneMinion": 1}
    assert merged.n_spawn_ticks == {"LaneMinion": 2}
    assert merged.recovery_counts == {"exact": 8, "proxy": 2}
    assert merged.per_tick_worst_pos_error == [
        (10, 1.0), (20, 2.0), (30, 3.0)]
