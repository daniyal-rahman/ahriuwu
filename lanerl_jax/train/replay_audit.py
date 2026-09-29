"""E36: frozen teacher/initial-policy comparisons on the actor's full history.

Used only by jax_eval --compare-checkpoint. Shadow actions are never executed;
this measures behavior on the recorded trajectory, not counterfactual outcomes.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from lanerl_jax.parity.policy_driver import load_params
from lanerl_jax.train.scripted_policy import scripted_act
from lanerl_jax.train.ppo import factored_log_prob
from lanerl_jax.train.vec_train import _relative_reward, VecConfig


class ReplayComparison:
    def __init__(self, checkpoint, policy, params, out):
        self.initial_policy, self.initial_params, _ = load_params(str(checkpoint))
        self.carries = [None, None]
        self.file = Path(out) / 'learning_audit.jsonl'
        self.file.write_text('')
        def build(p):
            @jax.jit
            def f(weights, obs, carry, teacher, actual):
                logits, carry = p.apply(weights, obs.entities, obs.entity_pad_mask,
                    obs.self_vec, obs.global_vec, carry)
                heads = (logits.button, logits.screen_x, logits.screen_y)
                return carry, dict(value=logits.value,
                    mode=jnp.stack([jnp.argmax(x, -1) for x in heads], -1),
                    button_prob=jax.nn.softmax(logits.button),
                    teacher_logp=factored_log_prob(heads, teacher),
                    actual_logp=factored_log_prob(heads, actual))
            return f
        self.models = [(build(self.initial_policy), self.initial_policy, self.initial_params),
                       (build(policy), policy, params)]
        self.teacher = jax.jit(jax.vmap(lambda o: jnp.stack(scripted_act(o, None))))
        self.reward = jax.jit(lambda a,b: _relative_reward(a,b,VecConfig())[0])

    def observe(self, obs, actual):
        self.teacher_action = self.teacher(obs)
        self.latest = {}
        for i, (f, p, weights) in enumerate(self.models):
            if self.carries[i] is None:
                self.carries[i] = p.initial_carry((len(actual),))
            self.carries[i], row = f(weights, obs, self.carries[i], self.teacher_action, actual)
            self.latest[('initial','final')[i]] = jax.tree.map(lambda x:np.asarray(x).tolist(),row)
        self.latest['teacher'] = np.asarray(self.teacher_action).tolist()
        self.latest['self_observation'] = np.asarray(obs.self_vec).tolist()
        self.latest['global_observation'] = np.asarray(obs.global_vec).tolist()

    def record(self, collector, before, actual, orders):
        teacher_orders = collector._decode(before, self.teacher_action[None], collector._model)
        mode = jnp.asarray(self.latest['initial']['mode'], jnp.int32)
        initial_orders = collector._decode(before, mode[None], collector._model)
        row = dict(self.latest, t_ms=float(before.t_ms[0]), actual=np.asarray(actual).tolist())
        for name, order in [('executed',orders),('teacher_orders',teacher_orders),('initial_mode_orders',initial_orders)]:
            row[name] = {k:np.asarray(getattr(order,k))[0].tolist() for k in ('kind','target','x','y')}
        row['reward'] = np.asarray(self.reward(jax.tree.map(lambda x:x[0],before),
            jax.tree.map(lambda x:x[0],collector.states))).tolist()
        with self.file.open('a') as f:
            f.write(json.dumps(row)+'\n')
