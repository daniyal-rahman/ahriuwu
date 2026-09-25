"""Reported CS uses champion-episodes, not update batches, as samples."""
import math

import pytest

from lanerl_jax.train.run_train import episode_cs_mean


def test_cs_weights_the_number_of_completed_champion_episodes():
    # One update ends two champion-episodes at 10 CS, another ends six at
    # 30 CS. An intervening update has no sample. The answer is 25, not 20.
    assert episode_cs_mean([10, math.nan, 30], [2, 0, 6]) == pytest.approx(25)


def test_no_completed_episode_has_no_cs_estimate():
    assert math.isnan(episode_cs_mean([math.nan], [0]))


def test_nonfinite_sample_is_not_silently_dropped():
    assert math.isnan(episode_cs_mean([10, math.nan], [2, 2]))
