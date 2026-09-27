"""Probe: WHAT changed inside the network between the prior and each
fine-tuned checkpoint, on the same recorded observations. Step-size damage
scatters (all layers drift, click mass spreads to many cells); a systematic
term shifts (one head moves in a consistent direction, e.g. attack -> move).

    python -m lanerl_jax.probes.prior_drift_probe demos.npz prior.msgpack name=ckpt ...
"""
import sys, numpy as np, jax, jax.numpy as jnp
from flax.serialization import from_state_dict, msgpack_restore
from lanerl_jax.train.server_train import load_checkpoint_policy, ScreenCellTable
from lanerl_jax.train.reward import LANE_AXIS

demos = sys.argv[1]; prior_path = sys.argv[2]; others = [a.split("=", 1) for a in sys.argv[3:]]
policy, prior = load_checkpoint_policy(prior_path)
d = np.load(demos); T, N = d["action"].shape[:2]
rng = np.random.default_rng(0); idx = (rng.integers(0, T, 640), rng.integers(0, N, 640))
ent, mask, sv, gv = (jnp.asarray(d[k][idx]) for k in ("entities", "mask", "self", "global")); ent = ent.astype(jnp.float32)

def heads(p):
    out = policy.apply(p, ent, mask, sv, gv, *([policy.initial_carry((640,))] if policy.cfg.core == "gru" else []))
    return out if hasattr(out, "button") else out[0]
def trunk_feats(p):
    _, inter = policy.apply(p, ent, mask, sv, gv, capture_intermediates=lambda m, n: n == "__call__")
    flat = {"/".join(map(str, k)): v for k, v in jax.tree_util.tree_flatten_with_path(inter)[0]}
    # the last Dense before the heads: the largest (640, 512) activation
    cands = [v for k, v in flat.items() if hasattr(v, "shape") and v.shape == (640, 512)]
    return jnp.asarray(cands[-1])
def probs(lg): return [jax.nn.softmax(l, -1) for l in (lg.button, lg.screen_x, lg.screen_y)]
def kl(p, q): return float(jnp.sum(p * (jnp.log(p + 1e-12) - jnp.log(q + 1e-12)), -1).mean())
def entropy(p): return float(-(p * jnp.log(p + 1e-12)).sum(-1).mean())

lg0 = heads(prior); P0 = probs(lg0); f0 = trunk_feats(prior)
btn_names = ["noop", "move", "attack_move", "q", "w", "e", "r", "recall"]
print(f"prior: entropy button {entropy(P0[0]):.2f} x {entropy(P0[1]):.2f} y {entropy(P0[2]):.2f}; P(attack_move) {float(P0[0][:,2].mean()):.3f} P(move) {float(P0[0][:,1].mean()):.3f}")
print(f"{'checkpoint':14s} {'KL(prior||ckpt)':>15s} {'ent_btn':>7s} {'ent_x':>6s} {'ent_y':>6s} {'P(atk)':>6s} {'P(move)':>7s} {'argmax_click_same':>17s} {'mass_within_3cells':>18s} {'trunk_cos':>9s} {'w_drift trunk/heads':>19s}")
for name, path in others:
    _, params = load_checkpoint_policy(path)
    lg = heads(params); P = probs(lg); f = trunk_feats(params)
    kl_total = kl(P0[0], P[0]) + kl(P0[1], P[1]) + kl(P0[2], P[2])
    ax0, ay0 = jnp.argmax(P0[1], -1), jnp.argmax(P0[2], -1); ax, ay = jnp.argmax(P[1], -1), jnp.argmax(P[2], -1)
    same = float(((ax0 == ax) & (ay0 == ay)).mean())
    # mass of the ckpt's click distribution within 3 cells of the prior's argmax click
    ix = jnp.arange(96)[None, :]; iy = jnp.arange(54)[None, :]
    near_x = (jnp.abs(ix - ax0[:, None]) <= 3).astype(jnp.float32); near_y = (jnp.abs(iy - ay0[:, None]) <= 3).astype(jnp.float32)
    mass_near = float(((P[1] * near_x).sum(-1) * (P[2] * near_y).sum(-1)).mean())
    cos = float((jnp.sum(f0 * f, -1) / (jnp.linalg.norm(f0, axis=-1) * jnp.linalg.norm(f, axis=-1) + 1e-9)).mean())
    # relative weight drift per group
    def rel(a, b): return float(jnp.linalg.norm(b - a) / (jnp.linalg.norm(a) + 1e-9))
    drift = {}
    for (k1, v1), (k2, v2) in zip(jax.tree_util.tree_flatten_with_path(prior)[0], jax.tree_util.tree_flatten_with_path(params)[0]):
        key = "/".join(str(getattr(p, "key", p)) for p in k1)
        grp = "heads" if any(h in key for h in ("Dense_10", "Dense_11", "Dense_12", "value_head")) else "trunk"
        drift.setdefault(grp, []).append(rel(v1, v2))
    print(f"{name:14s} {kl_total:15.3f} {entropy(P[0]):7.2f} {entropy(P[1]):6.2f} {entropy(P[2]):6.2f} {float(P[0][:,2].mean()):6.3f} {float(P[0][:,1].mean()):7.3f} {same:17.2f} {mass_near:18.2f} {cos:9.3f} {np.mean(drift.get('trunk',[0])):9.4f}/{np.mean(drift.get('heads',[0])):.4f}")
