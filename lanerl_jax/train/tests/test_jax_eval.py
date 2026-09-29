"""Evaluation stops at terminal state instead of resetting away its totals."""
from types import SimpleNamespace

import numpy as np
import pytest

from lanerl_jax.train.jax_eval import resolve_task, run_episode


def test_mirror_recording_preserves_red_action():
    from lanerl_jax.train.jax_eval import replay_actions
    assert np.stack(replay_actions([[1, 12, 23], [2, 34, 45]]), -1).tolist() == [[1, 12, 23], [2, 34, 45]]
    assert np.stack(replay_actions([[1, 12, 23]]), -1).tolist() == [[1, 12, 23], [0, 0, 0]]


def test_evaluation_honors_training_click_handling():
    from lanerl_jax.train.jax_eval import click_handling
    assert click_handling({'config': {'collector': {'unwalkable_click': 'noop'}}}) == 'noop'
    assert click_handling({'config': {'vec': {'unwalkable_click': 'noop'}}}) == 'noop'
    assert click_handling({}) == 'resolve'


def test_terminal_transition_recorded_without_reset_or_extra_action():
    class Collector:
        states = 'before'
        last_orders = None
        calls = 0

        def observe(self):
            return 'blue-observation', None

        def step(self, action):
            self.calls += 1
            assert action.tolist() == [[2, 3, 4]]
            self.states = 'terminal-with-CS'
            self.last_orders = 'resolved-screen-click'
            return np.array([True])

        def restart_done(self, _):
            raise AssertionError('evaluation must not reset')

    c, records = Collector(), []
    def choose(obs):
        assert obs == 'blue-observation'
        return [[2, 3, 4]]
    n = run_episode(c, choose, lambda before, action, orders: records.append((before, orders)), 5)
    assert n == c.calls == 1
    assert c.states == 'terminal-with-CS'
    assert records == [('before', 'resolved-screen-click')]


def test_nonterminating_collector_fails_at_bound():
    c = SimpleNamespace(states=None, last_orders=None,
        observe=lambda:(None, None), step=lambda action:np.array([False]))
    with pytest.raises(RuntimeError, match='decision bound'):
        run_episode(c, lambda obs:[[0, 0, 0]], lambda *args:None, 2)


def test_default_task_is_checkpoint_task_and_explicit_overrides_recorded():
    manifest = {'config': {'collector': {'episode_s':600, 'step_ticks':6,
        'start_near_wave':True, 'route_artifact':'route-path'}}}
    args = SimpleNamespace(seconds=None, step_ticks=None, start_near_wave=None, route_artifact=None)
    assert resolve_task(args, manifest) == dict(episode_s=600., step_ticks=6,
                                               start_near_wave=True, route_artifact='route-path')
    args.seconds, args.start_near_wave = 1., False
    assert resolve_task(args, manifest)['episode_s'] == 1.


def test_resolved_route_path_takes_precedence_over_legacy_path_repr():
    args = SimpleNamespace(seconds=None, step_ticks=None, start_near_wave=None, route_artifact=None)
    manifest = {'config': {'collector': {'route_artifact': "PosixPath('/legacy')"}},
                'sim': {'resolved': {'route_artifact': '/actual/routes'}}}
    assert resolve_task(args, manifest)['route_artifact'] == '/actual/routes'
