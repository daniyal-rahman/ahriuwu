"""Real compilation of memory attribution for both learner layouts; run through Slurm."""
import contextlib
from collections import namedtuple
import io
import json
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp

from lanerl_jax.modern.rl.learner import make_learner
from lanerl_jax.modern.rl.policy import LanePolicy, PolicyConfig
from lanerl_jax.modern.rl.ppo import PPOConfig
from ops.modern.training_bench import diagnose_components


class ComponentTests(unittest.TestCase):
    def test_both_learner_layouts_compile_without_real_rollout(self):
        for core in ("mlp", "gru"):
            with self.subTest(core=core):
                pcfg = PolicyConfig(core=core, d_model=8, n_heads=2, n_layers=1, ffn_dim=8,
                                    ctx_dim=8, core_dim=8, mlp_hidden=8, mlp_layers=1)
                policy = LanePolicy(pcfg)
                ppo = PPOConfig()
                shape = (8, 2) if core == "gru" else (16,)
                b = dict(entities=jnp.zeros(shape + (pcfg.n_slots, pcfg.entity_dim)),
                         mask=jnp.zeros(shape + (pcfg.n_slots,), bool), self=jnp.zeros(shape + (pcfg.self_dim,)),
                         global_=jnp.zeros(shape + (pcfg.global_dim,)),
                         action=tuple(jnp.zeros(shape, jnp.int32) for _ in range(3)),
                         log_prob=jnp.zeros(shape), uses_screen=jnp.ones(shape), value=jnp.zeros(shape),
                         adv=jnp.ones(shape), returns=jnp.ones(shape))
                b['global'] = b.pop('global_')
                args = (b['entities'], b['mask'], b['self'], b['global'])
                if core == "gru":
                    b.update(done=jnp.zeros(shape, bool), carry0=policy.initial_carry((8,)))
                    args += (policy.initial_carry(shape),)
                params = policy.init(jax.random.key(0), *args)
                _, loss = make_learner(policy, ppo)
                built = dict(loss=loss, rollout=lambda _: ((), (), b))
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    diagnose_components(built, namedtuple('Runner', 'params')(params),
                                        SimpleNamespace(n_minibatches=4, ppo=ppo))
                rows = [json.loads(x) for x in out.getvalue().splitlines()]
                self.assertEqual([x['component'] for x in rows],
                                 ['post_kl_full_forward', 'minibatch_forward', 'minibatch_value_and_grad'])
                self.assertTrue(all(x['compiled_memory']['argument_size_in_bytes'] > 0 for x in rows))


if __name__ == '__main__':
    unittest.main()
