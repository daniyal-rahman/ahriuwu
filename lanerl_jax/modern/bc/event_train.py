"""Bounded replay-only v2 learner; immutable dataset, exact update budget and resume."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .event_batches import event_batch, event_windows, load_event_games, ahri_angle_targets, drop_previous_event_history
from .event_auxiliary import future_targets
from .event_next import next_event_targets
from .event_eval import evaluate_games
from .event_loss import INPUTS, make_event_loss
from .event_policy import EventPolicy, EventPolicyConfig


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(serialization.to_bytes(value))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--champion', default='Ahri')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--burn', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--conditional-aim', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--opponent-relative-angle', action='store_true')
    p.add_argument('--own-cast-history', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--previous-event-history', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--smoothing-radius', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--sample-eval', action='store_true')
    p.add_argument('--ahri-angle', action='store_true')
    p.add_argument('--cast-aim-weight', type=float, default=1.)
    p.add_argument('--next-event-horizon', type=int, default=0)
    p.add_argument('--auxiliary-weight', type=float, default=0.)
    p.add_argument('--previous-event-dropout', type=float, default=0.)
    p.add_argument('--history', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-limit', type=int, default=0)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    if a.steps < 2 or min(a.batch_size, a.save_every, a.window) <= 0 or not 0 <= a.burn < a.window:
        p.error('invalid update/window/batch budget')
    if not np.isfinite(a.cast_aim_weight) or a.cast_aim_weight <= 0:
        p.error('cast aim weight must be finite and positive')
    if not np.isfinite(a.previous_event_dropout) or not 0 <= a.previous_event_dropout <= 1:
        p.error('previous-event dropout must be in [0,1]')
    if a.previous_event_dropout and not a.previous_event_history:
        p.error('dropout requires enabled previous-event history')
    if not np.isfinite(a.auxiliary_weight) or a.auxiliary_weight < 0:
        p.error('auxiliary weight must be finite and nonnegative')
    if a.next_event_horizon < 0 or (a.next_event_horizon and a.smoothing_radius):
        p.error('next-event prediction requires nonnegative horizon and no cast smoothing')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('training requires a Slurm GPU allocation')
    a.output.mkdir(parents=True, exist_ok=False)
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    manifest_hash = hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest()
    cfg = EventPolicyConfig(conditional_aim=a.conditional_aim, history=a.history,
                            own_cast_history=a.own_cast_history, previous_event_history=a.previous_event_history,
                            next_event_horizon=a.next_event_horizon,
                            opponent_relative_angle=a.opponent_relative_angle)
    smoothing = {0: (1.,), 1: (.5, 1., .5), 2: (.2, .5, 1., .5, .2)}[a.smoothing_radius]
    # Sampling is stateless by global update, so resume repeats no completed draw.
    contract = dict(config=cfg._asdict(), steps=a.steps, batch_size=a.batch_size, window=a.window,
                    burn=a.burn, seed=a.seed, lr=a.lr, warmup=a.warmup, dataset_sha256=manifest_hash,
                    champion=a.champion, smoothing=list(smoothing), co_move_weight=0., auxiliary_weight=a.auxiliary_weight,
                    ahri_angle=a.ahri_angle, cast_aim_weight=a.cast_aim_weight)
    if a.previous_event_dropout:
        contract['previous_event_dropout'] = a.previous_event_dropout
    contract_json = json.dumps(contract, sort_keys=True)
    (a.output/'config.json').write_text(json.dumps(contract, indent=2))
    started = time.monotonic()
    games = load_event_games(a.dataset, split='train', champion=a.champion, smoothing=smoothing)
    if a.ahri_angle:
        ahri_angle_targets(games)
    if a.auxiliary_weight:
        for game in games:
            game['arrays'].update(future_targets(game['arrays']))
    if a.next_event_horizon:
        for game in games:
            arrays = game['arrays']
            arrays.update(next_event_targets(arrays['button'], arrays['valid'], arrays['t'], a.next_event_horizon))
    if not games:
        raise ValueError('no training perspectives')
    windows = event_windows(games, a.window)
    eligible = []
    for gi, start in windows:
        valid = games[gi]['arrays']['valid'][start+a.burn:start+a.window]
        if valid.any():
            eligible.append((gi, start))
    windows = np.asarray(eligible, np.int32)
    if not len(windows):
        raise ValueError('no supervised windows')

    def batch_at(step):
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, step]))
        items = windows[rng.integers(len(windows), size=a.batch_size)]
        batch = event_batch(games, list(map(tuple, items)), a.window, a.burn)
        history_rng = np.random.default_rng(np.random.SeedSequence([a.seed, step, 137]))
        batch = drop_previous_event_history(batch, history_rng, a.previous_event_dropout)
        return jax.tree.map(jnp.asarray, batch)

    policy = EventPolicy(cfg)
    batch = batch_at(0)
    params = policy.init(jax.random.key(a.seed), *(batch[k][:, 0] for k in INPUTS),
                         policy.initial_carry((a.batch_size,)), target_button=batch['button'][:, 0])
    warmup = min(a.warmup, max(a.steps//10, 1))
    schedule = optax.warmup_cosine_decay_schedule(0., a.lr, warmup, a.steps, end_value=.1*a.lr)
    tx = optax.chain(optax.clip_by_global_norm(.5), optax.adam(schedule, eps=1e-5))
    state = tx.init(params)
    step = 0
    if a.resume:
        saved = serialization.msgpack_restore(a.resume.read_bytes())
        if saved['contract_json'] != contract_json:
            raise ValueError('resume changes model, dataset, sampling or total budget')
        params = serialization.from_state_dict(params, saved['params'])
        state = serialization.from_state_dict(state, saved['opt_state'])
        step = int(saved['step'])
    loss = make_event_loss(policy, cast_aim_weight=a.cast_aim_weight, auxiliary_weight=a.auxiliary_weight)

    @jax.jit
    def update(params, state, batch):
        (value, info), grad = jax.value_and_grad(loss, has_aux=True)(params, batch)
        updates, state = tx.update(grad, state, params)
        info['grad_norm'] = optax.global_norm(grad)
        return optax.apply_updates(params, updates), state, info

    metrics = a.output/'metrics.jsonl'
    times = []
    while step < a.steps and not stopping:
        batch = batch_at(step)
        tick = time.monotonic()
        params, state, info = update(params, state, batch)
        values = {k: float(v) for k, v in info.items()}
        if not all(np.isfinite(list(values.values()))):
            raise FloatingPointError(values)
        times.append(time.monotonic()-tick)
        step += 1
        row = dict(step=step, seconds=time.monotonic()-started, update_seconds=times[-1], **values)
        with metrics.open('a') as f:
            f.write(json.dumps(row)+'\n')
        if step == 1 or step % 25 == 0:
            print(json.dumps(row), flush=True)
        if step % a.save_every == 0:
            save(a.output/('checkpoint-'+str(step)+'.msgpack'),
                 dict(params=params, opt_state=state, step=step, contract_json=contract_json))
    checkpoint = dict(params=params, opt_state=state, step=step, contract_json=contract_json)
    save(a.output/'latest.msgpack', checkpoint)
    result = dict(status='interrupted' if stopping else 'complete', steps=step,
                  parameters=sum(v.size for v in jax.tree.leaves(params)), train_games=len(games),
                  train_windows=len(windows), seconds=time.monotonic()-started,
                  median_update_seconds=float(np.median(times[1:])) if len(times)>1 else None,
                  limits='bounded baseline, no automatic checkpoint promotion; no calibration fitted on validation')
    # Persist the training result before any potentially interrupted evaluation.
    (a.output/'results.json').write_text(json.dumps(result, indent=2))
    if a.evaluate and not stopping:
        del games, batch
        games = load_event_games(a.dataset, split='val', champion=a.champion)
        if a.ahri_angle:
            ahri_angle_targets(games)
        games = sorted(games, key=lambda g: g['row']['base'])
        if a.eval_limit:
            games = games[:a.eval_limit]
        reports = evaluate_games(params, policy, games)
        (a.output/'validation.json').write_text(json.dumps(reports, indent=2))
        if a.sample_eval:
            sampled = evaluate_games(params, policy, games, seed=a.seed)
            (a.output/'validation-sampled.json').write_text(json.dumps(sampled, indent=2))
    print(json.dumps(result), flush=True)
    if stopping:
        raise SystemExit(143)


if __name__ == '__main__':
    main()
