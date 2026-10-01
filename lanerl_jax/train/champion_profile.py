"""Modern champion selection shared by server collection and checkpoint loading."""
import json
from pathlib import Path


def names(value):
    if value is None:
        return None
    pair = tuple(value.split(',')) if isinstance(value, str) else tuple(value)
    if len(pair) != 2 or any(n not in ('Garen', 'Jax') for n in pair):
        raise ValueError('modern champions must be BLUE,RED, each Garen or Jax')
    return pair


def self_dim(pair):
    from ..obs.builder import SELF_DIM, MODERN_SELF_DIM
    return MODERN_SELF_DIM if names(pair) else SELF_DIM


def game_config(pair, server_dir, out):
    pair = names(pair)
    if pair is None:
        return None
    if server_dir is None:
        raise ValueError('modern champions require an explicit isolated --server-dir')
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / 'lanerl/cfg/modern_garen_jax_26_19.json').read_text())
    config['_comment'] = 'Explicit 26.19 bare-champion profile; no runes, talents, or autobuy.'
    for player, name in zip(config['players'], pair):
        player['champion'] = name
    config['gameInfo']['CONTENT_PATH'] = str(Path(server_dir).resolve().parents[1] / 'Content')
    path = Path(out).resolve() / 'modern-game.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + '\n')
    return path


def checkpoint_profile(config):
    return names(config.get('collector', {}).get('modern_champions'))


def validate_checkpoint(config, pair):
    pair = names(pair)
    if checkpoint_profile(config) != pair:
        raise ValueError('checkpoint champion profile differs from requested collector profile')
    if config.get('train', {}).get('policy', {}).get('self_dim', 16) != self_dim(pair):
        raise ValueError('checkpoint observation width differs from champion profile')
