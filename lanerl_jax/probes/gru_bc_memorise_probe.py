"""Probe (one-off): can the GRU policy memorise ONE fixed batch of scripted
demos under the exact bc_diag loss? The MLP fits the JAX mirror demos to 98%
attack-click accuracy in 30 epochs; the GRU sits at the majority class after
100. If the GRU cannot even memorise a fixed batch, the fault is in the GRU
path (gradient flow / wiring), not in data or schedule."""
import sys, time, numpy as np, jax, jax.numpy as jnp, optax
from lanerl_jax.train.policy import LanePolicy, PolicyConfig

d = np.load(sys.argv[1]); core = sys.argv[2]; L = 32; B = 16; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 150
T = d["action"].shape[0]
rng = np.random.default_rng(0)
# pick windows with attack steps in them
w = [(int(s0), int(n)) for s0 in rng.integers(0, T - L, 400) for n in [int(rng.integers(0, d["action"].shape[1]))]
     if (d["action"][s0:s0+L, n, 0] == 2).any()][:B]
sl = lambda x: jnp.asarray(np.stack([x[s0:s0+L, n] for s0, n in w]))
ent, mask, sv, gv, act, dn = sl(d["entities"]).astype(jnp.float32), sl(d["mask"]), sl(d["self"]), sl(d["global"]), sl(d["action"]).astype(jnp.int32), sl(d["done"])
print("attack fraction in batch", float((act[..., 0] == 2).mean()))
variant = core; core = "gru" if core.startswith("gru") else core
policy = LanePolicy(PolicyConfig(core=core, core_norm="norm" in variant, core_residual="res" in variant))
carry0 = policy.initial_carry((1,)) if core == "gru" else None
params = policy.init(jax.random.key(0), ent[:1, 0], mask[:1, 0], sv[:1, 0], gv[:1, 0], *([carry0] if carry0 is not None else []))
tx = optax.adam(3e-4); opt = tx.init(params)

def heads_loss(logits, action):
    lp = [jax.nn.log_softmax(l, axis=-1) for l in (logits.button, logits.screen_x, logits.screen_y)]
    nll = -sum(jnp.take_along_axis(l, action[..., i:i+1], axis=-1)[..., 0] for i, l in enumerate(lp))
    hit = jnp.stack([jnp.argmax(l, -1) == action[..., i] for i, l in enumerate(lp)], -1)
    is_atk = (action[..., 0] == 2)
    atk_btn = (hit[..., 0] & is_atk).sum() / is_atk.sum()
    return nll.mean(), (hit.reshape(-1, 3).mean(0), atk_btn)

def loss_fn(params):
    if core == "mlp":
        return heads_loss(policy.apply(params, ent, mask, sv, gv), act)
    tm = lambda x: jnp.swapaxes(x, 0, 1)
    d_prev = jnp.concatenate([jnp.zeros_like(dn[:, :1]), dn[:, :-1]], axis=1)
    def step(carry, xs):
        e_, m_, s_, g_, dp = xs
        carry = jnp.where(dp[:, None], 0.0, carry)
        logits, carry = policy.apply(params, e_, m_, s_, g_, carry)
        return carry.astype(jnp.float32), logits
    _, logits = jax.lax.scan(step, jnp.zeros((B, policy.cfg.core_dim), jnp.float32), (tm(ent), tm(mask), tm(sv), tm(gv), tm(d_prev)))
    return heads_loss(jax.tree.map(tm, logits), act)

@jax.jit
def update(params, opt):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    gn = optax.global_norm(g)
    trunk_gn = optax.global_norm({k: v for k, v in g["params"].items() if "core_gru" not in k and "head" not in k.lower()})
    upd, opt = tx.update(g, opt, params)
    return optax.apply_updates(params, upd), opt, l, aux, gn, trunk_gn

t0 = time.time()
for i in range(steps):
    params, opt, l, (acc, atk), gn, tgn = update(params, opt)
    if i % 25 == 0 or i == steps - 1:
        print(f"step {i} loss {float(l):.3f} acc {np.asarray(acc).round(3).tolist()} atk_btn {float(atk):.3f} grad {float(gn):.3f} trunk_grad {float(tgn):.3f} {time.time()-t0:.0f}s", flush=True)
print("param groups:", list(params["params"].keys()))
