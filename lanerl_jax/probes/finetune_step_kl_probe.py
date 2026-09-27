"""Probe (one-off): how far does ONE Adam step move a sharp BC prior, and which
loss term does it? E12b applied exactly one minibatch step per update because
the KL stop tripped on the second; its logged KL (applied steps only) was 0.
Here: load the E12a checkpoint (actor = DAgger-2 clone), take 640 recorded
observations, and for each loss term take one Adam step (lr as given, eps 1e-5,
grad clip 0.5) from a fresh optimiser; report KL(old||new) and entropy change."""
import sys, numpy as np, jax, jax.numpy as jnp, optax
from flax.serialization import from_state_dict, msgpack_restore
from lanerl_jax.train.server_train import load_checkpoint_policy

ckpt, demos = sys.argv[1], sys.argv[2]
lrs = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["5e-5"])]
policy, params = load_checkpoint_policy(ckpt)
d = np.load(demos); T, N = d["action"].shape[:2]
rng = np.random.default_rng(0); idx = (rng.integers(0, T, 640), rng.integers(0, N, 640))
ent, mask, sv, gv = (jnp.asarray(d[k][idx]) for k in ("entities", "mask", "self", "global")); ent = ent.astype(jnp.float32)
act = jnp.asarray(d["action"][idx]).astype(jnp.int32)

def heads(p):
    return policy.apply(p, ent, mask, sv, gv)
def logp_all(lg):
    return [jax.nn.log_softmax(l, -1) for l in (lg.button, lg.screen_x, lg.screen_y)]
def entropy(lg):
    return sum(-(jnp.exp(l) * l).sum(-1) for l in logp_all(lg)).mean()
def kl(lg_old, lg_new):
    return sum((jnp.exp(o) * (o - n)).sum(-1) for o, n in zip(logp_all(lg_old), logp_all(lg_new))).mean()
lg0 = heads(params)
adv = jnp.asarray(rng.standard_normal(640), jnp.float32)          # normalised advantages: unit noise
ret = lg0.value + jnp.asarray(rng.standard_normal(640), jnp.float32) * 0.5   # value targets ~ current value + noise

def pg_loss(p):  # PPO surrogate at ratio 1 == -adv * logp(a)
    lp = logp_all(heads(p)); l = sum(jnp.take_along_axis(x, act[:, i:i+1], -1)[:, 0] for i, x in enumerate(lp))
    return -(adv * l).mean()
def ent_loss(p): return -(0.01 / 3) * entropy(heads(p))
def val_loss(p): return 0.5 * ((heads(p).value - ret) ** 2).mean()
def all_loss(p): return pg_loss(p) + ent_loss(p) + val_loss(p)
def val_head_only(p):  # value loss, gradient masked to the value head (what E12a did)
    return val_loss(p)

def one_step(loss, lr, mask_fn=None):
    g = jax.grad(loss)(params)
    if mask_fn: g = mask_fn(g)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5)); st = tx.init(params)
    upd, _ = tx.update(g, st, params); new = optax.apply_updates(params, upd)
    lg1 = heads(new)
    return float(kl(lg0, lg1)), float(entropy(lg1) - entropy(lg0)), float(optax.global_norm(g))
def only_value_head(g):
    return jax.tree_util.tree_map_with_path(lambda path, x: x if any(getattr(k, "key", None) == "value_head" for k in path) else jnp.zeros_like(x), g)

print(f"prior entropy {float(entropy(lg0)):.3f}  (uniform would be {np.log(8)+np.log(96)+np.log(54):.2f})")
for lr in lrs:
    print(f"--- lr {lr:g}")
    for name, fn, mk in (("pg (unit-noise adv)", pg_loss, None), ("entropy 0.01/3", ent_loss, None),
                         ("value 0.5 (shared trunk)", val_loss, None), ("value, head only", val_head_only, only_value_head),
                         ("all three", all_loss, None)):
        k, de, gn = one_step(fn, lr, mk)
        print(f"  {name:28s} KL after 1 step {k:.4f}  entropy change {de:+.3f}  grad norm {gn:.2f}")
