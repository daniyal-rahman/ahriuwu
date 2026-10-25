"""Stateful replay evaluation with actual-button and oracle-button target readouts."""
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .event_batches import event_batch
from .event_loss import INPUTS
from .event_metrics import sequence_metrics
from .event_targets import pointer_candidates, emission_logits


def make_predict(policy, *, sampled=False, temperature=1.):
    @jax.jit
    def predict(params, carry, batch, bias, key):
        inputs = tuple(jnp.swapaxes(batch[k], 0, 1) for k in INPUTS)
        teacher = jnp.swapaxes(batch['button'], 0, 1)

        def step(state, values):
            carry, key = state
            x, human_button = values[:-1], values[-1]
            hidden, carry, tokens = policy.apply(params, *x, carry, method=policy.encode)
            logits = policy.apply(params, hidden, method=policy.buttons)/temperature + bias
            logits = emission_logits(logits, x[0], x[1], x[5])
            if sampled:
                key, draw = jax.random.split(key)
                button = jax.random.categorical(draw, logits, axis=-1)
            else:
                button = logits.argmax(-1)
            actual = policy.apply(params, hidden, tokens, x[1], x[7], button, positions=x[0][..., :, 1:3],
                                  eligible=pointer_candidates(x[0], x[1], x[5], button), method=policy.targets)
            teacher = policy.apply(params, hidden, tokens, x[1], x[7], human_button, positions=x[0][..., :, 1:3],
                                  eligible=pointer_candidates(x[0], x[1], x[5], human_button), method=policy.targets)
            summary = dict(probabilities=jax.nn.softmax(logits), button=button)
            for prefix, heads in [('actual', actual), ('teacher', teacher)]:
                for name, head in zip(('pointer', 'grid', 'angle'), heads):
                    choice = head.argmax(-1)
                    if name == 'pointer':
                        selected = button if prefix == 'actual' else human_button
                        available = pointer_candidates(x[0], x[1], x[5], selected).any(-1)
                        choice = jnp.where(available, choice, -1)
                    summary[prefix+'_'+name] = choice
            return (carry, key), summary
        state, summary = jax.lax.scan(step, (carry, key), inputs + (teacher,))
        return state, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), summary)
    return predict


def predict_game(params, policy, predict, game, *, chunk=128, bias=None, seed=None, reset_every=None):
    if reset_every is not None and (reset_every < chunk or reset_every % chunk):
        raise ValueError('memory reset interval must be a positive multiple of chunk length')
    if bias is None:
        bias = np.zeros(9, np.float32)
    carry = policy.initial_carry((1,))
    identity = str(seed) + ':' + game['row']['game']
    game_seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
    key = jax.random.key(game_seed)
    summaries = []
    n = len(game['arrays']['button'])
    for start in range(0, n, chunk):
        if reset_every is not None and start and start % reset_every == 0:
            carry = policy.initial_carry((1,))
        batch = jax.tree.map(jnp.asarray, event_batch([game], [(0, start)], window=chunk, burn=0))
        (carry, key), summary = predict(params, carry, batch, jnp.asarray(bias), key)
        summaries.append(jax.tree.map(lambda x: np.asarray(x)[0, :min(chunk, n-start)], summary))
    return {k: np.concatenate([s[k] for s in summaries]) for k in summaries[0]}


def evaluate_games(params, policy, games, *, chunk=128, bias=None, seed=None, prediction_dir=None, temperature=1., reset_every=None):
    predict = make_predict(policy, sampled=seed is not None, temperature=temperature)
    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for game in games:
        predictions = predict_game(params, policy, predict, game, chunk=chunk, bias=bias, seed=seed, reset_every=reset_every)
        if prediction_dir is not None:
            arrays = game['arrays']
            np.savez_compressed(prediction_dir/(game['row']['game']+'.npz'), **predictions,
                                t=arrays['t'], valid=arrays['valid'], label_button=arrays['button'])
        reports.append(dict(game=game['row']['game'], base=game['row']['base'],
                            metrics=sequence_metrics(game['arrays'], predictions)))
    return reports
