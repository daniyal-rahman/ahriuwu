"""Pure aggregation checks for the gate-3 HP-band diagnostic."""

from lanerl_jax.parity.archive.hp_band import BandSample, band_windows


def _sample(t_ms: float, *, in_band: bool = True) -> BandSample:
    return BandSample(
        t_ms=t_ms, hp=10.0, armor=0.0, in_band=in_band,
        champ_ad=50.0, champ_level=1,
        champ_x=0.0, champ_y=0.0, minion_x=10.0, minion_y=0.0,
    )


def test_band_windows_counts_decision_frames_not_minion_rows():
    """Two lethal minions on one frame are one oracle ATTACK opportunity."""
    summary = band_windows([
        _sample(100), _sample(100), _sample(133), _sample(167),
        _sample(233), _sample(267), _sample(300, in_band=False),
    ])

    assert summary.in_band_decisions == 5
    assert summary.windows == 2
    assert summary.mean_frames == 2.5
    assert summary.max_frames == 3


def test_band_windows_accepts_the_wire_clock_34ms_boundary():
    summary = band_windows([_sample(16), _sample(50), _sample(84)])

    assert summary.windows == 1
    assert summary.max_frames == 3
