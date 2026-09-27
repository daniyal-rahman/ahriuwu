"""Probe (one-off): scale of the trunk output that feeds the GRU vs the MLP's h."""
import sys, numpy as np, jax, jax.numpy as jnp, flax.linen as nn
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
d = np.load(sys.argv[1]); T, N = d["action"].shape[:2]
rng = np.random.default_rng(0); idx = (rng.integers(0, T, 64), rng.integers(0, N, 64))
ent, mask, sv, gv = (jnp.asarray(d[k][idx]) for k in ("entities", "mask", "self", "global"))
ent = ent.astype(jnp.float32)
print("obs scales: entities std %.2f max %.1f | self std %.2f max %.1f | global std %.2f max %.1f" % (
    float(ent.std()), float(jnp.abs(ent).max()), float(sv.std()), float(jnp.abs(sv).max()), float(gv.std()), float(jnp.abs(gv).max())))
p = LanePolicy(PolicyConfig(core="gru"))
c0 = p.initial_carry((64,))
params = p.init(jax.random.key(0), ent, mask, sv, gv, c0)
_, inter = p.apply(params, ent, mask, sv, gv, c0, capture_intermediates=lambda mdl, name: name == "__call__")
# trunk output = input to core_gru: find Dense feeding the GRU (last Dense before core_gru in the flat list)
flat = {"/".join(map(str, k)): v for k, v in jax.tree_util.tree_flatten_with_path(inter)[0]}
for k, v in flat.items():
    if "Dense_9" in k or "core_gru" in k:
        v = jnp.asarray(v); print(k, tuple(v.shape), "std %.3f mean|x| %.3f cross-batch std %.3f" % (float(v.std()), float(jnp.abs(v).mean()), float(v.std(0).mean())))
