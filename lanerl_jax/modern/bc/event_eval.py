"""Stateful replay evaluation with actual-button and oracle-button target readouts."""
import jax
import jax.numpy as jnp
import numpy as np

from .event_batches import event_batch
from .event_loss import INPUTS
from .event_metrics import sequence_metrics


def make_predict(policy):
    @jax.jit
    def predict(params, carry, batch, bias):
        inputs = tuple(jnp.swapaxes(batch[k], 0, 1) for k in INPUTS)
        teacher = jnp.swapaxes(batch['button'], 0, 1)

        def step(carry, values):
            x, human_button = values[:-1], values[-1]
            hidden, carry, tokens = policy.apply(params, *x, carry, method=policy.encode)
            logits = policy.apply(params, hidden, method=policy.buttons) + bias
            button = logits.argmax(-1)
            actual = policy.apply(params, hidden, tokens, x[1], x[7], button, method=policy.targets)
            teacher = policy.apply(params, hidden, tokens, x[1], x[7], human_button, method=policy.targets)
            summary = dict(probabilities=jax.nn.softmax(logits), button=button)
            for prefix, heads in [('actual', actual), ('teacher', teacher)]:
                for name, head in zip(('pointer', 'grid', 'angle'), heads):
                    summary[prefix+'_'+name] = head.argmax(-1)
            return carry, summary
        carry, summary = jax.lax.scan(step, carry, inputs + (teacher,))
        return carry, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), summary)
    return predict


def predict_game(params, policy, predict, game, *, chunk=128, bias=None):
    if bias is None:
        bias = np.zeros(9, np.float32)
    carry = policy.initial_carry((1,))
    summaries = []
    n = len(game['arrays']['button'])
    for start in range(0, n, chunk):
        batch = jax.tree.map(jnp.asarray, event_batch([game], [(0, start)], window=chunk, burn=0))
        carry, summary = predict(params, carry, batch, jnp.asarray(bias))
        summaries.append(jax.tree.map(lambda x: np.asarray(x)[0, :min(chunk, n-start)], summary))
    return {k: np.concatenate([s[k] for s in summaries]) for k in summaries[0]}


def evaluate_games(params, policy, games, *, chunk=128, bias=None):
    predict = make_predict(policy)
    reports = []
    for game in games:
        predictions = predict_game(params, policy, predict, game, chunk=chunk, bias=bias)
        reports.append(dict(game=game['row']['game'], base=game['row']['base'],
                            metrics=sequence_metrics(game['arrays'], predictions)))
    return reports
