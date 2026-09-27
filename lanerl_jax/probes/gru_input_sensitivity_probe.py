"""Probe (one-off): do the GRU policy's logits depend on the observation at all?
Std of the button/x logits across a batch of DIFFERENT observations, mlp vs gru,
at init; plus the GRU's own output std and the Dense(core_dim) pre-activation std."""
import sys, numpy as np, jax, jax.numpy as jnp
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
d = np.load(sys.argv[1]); T, N = d["action"].shape[:2]
rng = np.random.default_rng(0); idx = (rng.integers(0, T, 64), rng.integers(0, N, 64))
ent, mask, sv, gv = (jnp.asarray(d[k][idx]) for k in ("entities", "mask", "self", "global"))
ent = ent.astype(jnp.float32)
for core in ("mlp", "gru"):
    p = LanePolicy(PolicyConfig(core=core))
    extra = [p.initial_carry((64,))] if core == "gru" else []
    params = p.init(jax.random.key(0), ent, mask, sv, gv, *extra)
    out = p.apply(params, ent, mask, sv, gv, *extra)
    lg = out[0] if core == "gru" else out
    print(core, "button logit std across batch %.4f  x logit std %.4f  value std %.4f" % (
        float(lg.button.std(0).mean()), float(lg.screen_x.std(0).mean()), float(lg.value.std())))
    if core == "gru":
        new_c = out[1]
        print("  gru output: mean |h| %.4f, std across batch %.4f" % (float(jnp.abs(new_c).mean()), float(new_c.std(0).mean())))
        g = params["params"]["core_gru"]
        print("  gru param keys:", {k: {kk: tuple(vv.shape) for kk, vv in v.items()} if isinstance(v, dict) else tuple(v.shape) for k, v in g.items()})
        # feed the same obs twice with different carries: does the carry matter?
        c2 = jax.random.normal(jax.random.key(1), (64, p.cfg.core_dim))
        lg2, _ = p.apply(params, ent, mask, sv, gv, c2)
        print("  button logit change from carry noise %.4f" % float(jnp.abs(lg2.button - lg.button).mean()))
