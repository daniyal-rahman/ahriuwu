"""Next-event targets and causal per-frame emission; BC.md section15."""
import numpy as np
import jax
import jax.numpy as jnp

from .events import BUTTONS


def next_event_targets(button, valid, t, horizon=20):
    """Include current receipts; mask right-censored intervals and never cross gaps."""
    button, valid, t = np.asarray(button), np.asarray(valid, bool), np.asarray(t)
    if horizon < 1 or button.ndim != 1 or valid.shape != button.shape or t.shape != button.shape:
        raise ValueError('next-event targets need aligned vectors and positive horizon')
    n = len(button)
    event = np.zeros(n, np.int32)
    delay = np.zeros(n, np.int32)
    known = np.zeros(n, bool)
    observed = np.zeros(n, bool)
    end, upcoming = n, n
    for i in range(n-1, -1, -1):
        if not valid[i]:
            end, upcoming = i, n
            continue
        if i+1 < n and (not valid[i+1] or not np.isclose(t[i+1]-t[i], .1, atol=1e-5, rtol=0)):
            end, upcoming = i+1, n
        if button[i] != 0:
            upcoming = i
        if upcoming < end and upcoming-i <= horizon:
            event[i], delay[i] = button[upcoming], upcoming-i
            known[i], observed[i] = True, True
        elif i+horizon < end:
            known[i] = True
    return dict(next_button=event, next_delay=delay, next_valid=known, next_observed=observed)


def immediate_logits(event_logits, delay_logits):
    """P(action now)=P(next type)*P(delay=0|type); remaining mass is continue."""
    event = jax.nn.softmax(event_logits, -1)
    due = jax.nn.softmax(delay_logits, -1)[..., 0]
    emitted = event[..., 1:]*due
    continuation = jnp.maximum(1-emitted.sum(-1, keepdims=True), 0.)
    probability = jnp.concatenate([continuation, emitted], -1)
    return jnp.log(jnp.maximum(probability, 1e-30))
