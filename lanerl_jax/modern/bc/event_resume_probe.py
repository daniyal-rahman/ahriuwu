"""Separate checkpoint fidelity from GPU continuation numerical variation."""
import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .event_batches import event_batch, event_windows, load_event_games
from .event_loss import make_event_loss
from .event_policy import EventPolicy, EventPolicyConfig


def differences(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    pairs = [(np.asarray(x, np.float64), np.asarray(y, np.float64))
             for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))]
    maximum = max(float(np.max(np.abs(x-y))) for x, y in pairs)
    squared = sum(float(np.sum((x-y)**2)) for x, y in pairs)
    scale = sum(float(np.sum(x*x)) for x, _ in pairs)
    return dict(max_abs=maximum, relative_l2=float(np.sqrt(squared/max(scale, 1e-30))))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--donor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert jax.default_backend() == 'gpu'
    a.output.mkdir(parents=True, exist_ok=False)
    checkpoint = serialization.msgpack_restore((a.donor/'control/checkpoint-16.msgpack').read_bytes())
    contract = json.loads(checkpoint['contract_json'])
    cfg = EventPolicyConfig(**contract['config'])
    policy = EventPolicy(cfg)
    loss = make_event_loss(policy)
    games = load_event_games(a.dataset, split='train', champion=contract['champion'])
    window, burn = contract['window'], contract['burn']
    windows = np.asarray([(gi, s) for gi, s in event_windows(games, window)
                          if games[gi]['arrays']['valid'][s+burn:s+window].any()], np.int32)

    def batch_at(step):
        rng = np.random.default_rng(np.random.SeedSequence([contract['seed'], step]))
        items = windows[rng.integers(len(windows), size=contract['batch_size'])]
        return jax.tree.map(jnp.asarray, event_batch(games, list(map(tuple, items)), window, burn))

    warmup = min(contract['warmup'], max(contract['steps']//10, 1))
    schedule = optax.warmup_cosine_decay_schedule(0., contract['lr'], warmup, contract['steps'],
                                                 end_value=.1*contract['lr'])
    tx = optax.chain(optax.clip_by_global_norm(.5), optax.adam(schedule, eps=1e-5))
    params = jax.device_put(checkpoint['params'])
    state = jax.device_put(serialization.from_state_dict(tx.init(params), checkpoint['opt_state']))
    roundtrip = serialization.msgpack_restore(serialization.to_bytes(dict(params=params, opt_state=state)))
    assert differences(checkpoint['params'], roundtrip['params'])['max_abs'] == 0
    assert differences(checkpoint['opt_state'], roundtrip['opt_state'])['max_abs'] == 0
    counts = [int(x) for x in jax.tree.leaves(state) if np.asarray(x).dtype.kind in 'iu']
    assert counts and all(x == 16 for x in counts), counts

    @jax.jit
    def update(params, state, batch):
        (value, info), grad = jax.value_and_grad(loss, has_aux=True)(params, batch)
        updates, state = tx.update(grad, state, params)
        return optax.apply_updates(params, updates), state, value

    replicas = []
    curves = []
    for _ in range(2):
        pp, ss = params, state
        curve = []
        for step in range(16, contract['steps']):
            pp, ss, value = update(pp, ss, batch_at(step))
            curve.append(float(value))
        replicas.append(pp)
        curves.append(curve)
    control = serialization.msgpack_restore((a.donor/'control/latest.msgpack').read_bytes())
    resumed = serialization.msgpack_restore((a.donor/'resumed/latest.msgpack').read_bytes())
    assert control['step'] == resumed['step'] == contract['steps'] == 32
    comparisons = dict(replicas=differences(replicas[0], replicas[1]),
        control_vs_replica=differences(control['params'], replicas[0]),
        old_resumed_vs_replica=differences(resumed['params'], replicas[0]),
        old_control_vs_resumed=differences(control['params'], resumed['params']))
    evaluate = jax.jit(loss)
    final_losses = [float(evaluate(x, batch_at(31))[0]) for x in
                    (control['params'], resumed['params'], *replicas)]
    passed = all(x['max_abs'] < 1e-3 and x['relative_l2'] < 1e-4 for x in comparisons.values())
    passed &= max(final_losses)-min(final_losses) < 1e-3
    report = dict(passed=bool(passed), checkpoint_roundtrip_exact=True, optimizer_counts=counts,
        comparisons=comparisons, final_losses=final_losses, curves=curves,
        tolerances=dict(max_abs=1e-3, relative_l2=1e-4, loss_spread=1e-3),
        limits='one checkpoint/two continuation replicas; bounded numerical equivalence, not bitwise determinism')
    (a.output/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
