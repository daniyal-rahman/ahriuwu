"""Diagnostic counts must distinguish attacking minions from holding allies."""
import numpy as np

from lanerl_jax.replay import summarize


def test_replay_counts_the_actual_decoded_target_and_living_time():
    d = dict(
        x=np.tile([0, 1000, 100, 200], (4, 1)), y=np.zeros((4, 4)),
        kind=np.tile([1, 1, 3, 2], (4, 1)),
        team=np.tile([0, 1, 0, 1], (4, 1)),
        alive=np.ones((4, 4), bool),
        cs=np.tile([2, 0], (4, 1)), deaths=np.zeros((4, 4)),
        order_target=np.array([[3, 0], [2, 0], [1, 0], [-1, -1]]),
        order_kind=np.array([[2, 2], [2, 2], [2, 2], [1, 0]]),
        aa_target=np.full((4, 4), -1, dtype=int),
        spawn_seq=np.zeros((4, 4), dtype=int),
        is_attacking=np.zeros((4, 4), dtype=bool),
    )
    d['alive'][-1, 0] = False
    blue, red = summarize(d)
    assert blue['cs'] == 2
    assert blue['alive_fraction'] == .75
    assert blue['near_enemy_minion_fraction'] == 1.0
    assert blue['attack_enemy_minion_fraction'] == .25
    assert blue['attack_enemy_champion_fraction'] == .25
    assert blue['attack_own_turret_fraction'] == .25
    assert blue['attack_ally_fraction'] == .25
    assert red['median_nearest_minion_when_alive'] is None
    assert blue['windup_target_switches'] == 0
    assert blue['enemy_minion_windup_frames'] == 0
    assert blue['minimum_nearest_minion_when_alive'] == 200


def test_player_restart_uses_first_recorded_game_time():
    """A replay beginning after setup must not wait through setup on restart."""
    from pathlib import Path
    import shutil
    import subprocess
    import pytest

    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required to execute the actual player restart handler')
    html = Path(__file__).parents[1].joinpath('replay_player.html').read_text()
    handler = next(line for line in html.splitlines() if line.startswith('play.onclick='))
    script = '''const frames=[[120],[600]], play={};
let playing=false,index=1,clock=600,last=0;
''' + handler + '''
play.onclick();
if(index!==0 || clock!==120 || !playing) throw Error('restart did not use recorded start');
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
