"""Real-gradient fixed-window v2 wiring check, executed under Slurm on a GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .event_batches import event_batch, load_event_games
from .event_loss import INPUTS, make_event_loss
from .event_policy import EventPolicy, EventPolicyConfig
from .events import BUTTON_INDEX as B


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    if a.steps < 1:
        p.error('steps must be positive')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('this gradient check requires a Slurm GPU allocation')
    a.output.mkdir(parents=True, exist_ok=False)
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    games = load_event_games(a.dataset, split='train')
    items = []
    for champion, mode in [('Ahri', 2), ('Garen', 1)]:
        options = []
        for gi, g in enumerate(games):
            if g['row']['champion'] != champion:
                continue
            x = g['arrays']
            positions = np.flatnonzero(x['valid'] & x['aim_valid'] & (x['aim_mode'] == mode))
            for t in positions:
                cast = B['q'] <= x['button'][t] <= B['r']
                options.append((int(cast), gi, max(0, int(t) - 32)))
        if not options:
            raise RuntimeError('fixture lacks a valid ' + champion + ' target')
        _, gi, start = max(options)
        items.append((gi, start))
    batch = jax.tree.map(jnp.asarray, event_batch(games, items, window=64, burn=8))
    policy = EventPolicy(EventPolicyConfig())
    init_args = [batch[k][:, 0] for k in INPUTS]
    started = time.monotonic()
    params = policy.init(jax.random.key(0), *init_args, policy.initial_carry((len(items),)),
                         target_button=batch['button'][:, 0])
    n_params = sum(x.size for x in jax.tree.leaves(params))
    loss = make_event_loss(policy)
    tx = optax.chain(optax.clip_by_global_norm(.5), optax.adam(3e-4, eps=1e-5))
    state = tx.init(params)
    initial_params = params

    @jax.jit
    def update(params, state):
        (value, info), grad = jax.value_and_grad(loss, has_aux=True)(params, batch)
        updates, state = tx.update(grad, state, params)
        info['grad_norm'] = optax.global_norm(grad)
        return optax.apply_updates(params, updates), state, info

    evaluate = jax.jit(loss)
    initial, initial_info = evaluate(params, batch)
    initial = float(initial)
    assert np.isfinite(initial)
    rows = []
    step = 0
    for step in range(1, a.steps + 1):
        params, state, info = update(params, state)
        values = {k: float(v) for k, v in info.items()}
        if not all(np.isfinite(list(values.values()))):
            raise FloatingPointError(values)
        if step == 1 or step % 25 == 0:
            rows.append(dict(step=step, **values))
            print(json.dumps(rows[-1]), flush=True)
        if stopping:
            break
    final, final_info = evaluate(params, batch)
    delta = float(optax.global_norm(jax.tree.map(lambda x, y: x - y, params, initial_params)))
    payload = dict(params=params, opt_state=state, step=step, config=policy.cfg._asdict())
    temp = a.output / 'checkpoint.tmp'
    temp.write_bytes(serialization.to_bytes(payload))
    temp.replace(a.output / 'checkpoint.msgpack')
    # This is a wiring/memorization gate, not a held-out quality or leakage result.
    passed = not stopping and float(final) < .7 * initial and delta > 0 and rows[0]['grad_norm'] > 0
    report = dict(passed=passed, status='interrupted' if stopping else 'complete', steps=step,
                  initial_loss=initial, final_loss=float(final), parameter_delta=delta, parameters=n_params,
                  initial={k: float(v) for k, v in initial_info.items()},
                  final={k: float(v) for k, v in final_info.items()}, curve=rows,
                  windows=[dict(game=games[g]['row']['game'], start=s) for g, s in items],
                  seconds=time.monotonic()-started, job=os.environ.get('SLURM_JOB_ID'),
                  limits='two fixed training windows; no generalization, calibration or leakage claim')
    (a.output / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(143 if stopping else 1)


if __name__ == '__main__':
    main()
