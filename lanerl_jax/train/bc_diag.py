"""Behaviour-cloning DIAGNOSTIC: can the policy network represent the scripted
last-hitter from the observation alone?

Supervised cross-entropy on the three heads (button, screen x, screen y) over
(observation, action) pairs recorded by `server_train --scripted ... --record-npz`.
Writes a checkpoint + manifest that `server_train --eval-episodes` / the
launcher can evaluate. This is NOT a training prior for the no-prior gate; it
answers "is the observation sufficient?" (representability) and seeds the
"does PPO preserve a strong policy?" learner test.

    python -m lanerl_jax.train.bc_diag demos.npz --out lanerl_jax/runs/BC --core mlp --epochs 30
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
from flax.serialization import to_bytes
from .policy import LanePolicy, PolicyConfig
from .run_manifest import RunDir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("demos", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--core", choices=("mlp", "gru"), default="mlp")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=2048, help="mlp: samples per step; gru: sequences (agents) x seq_len windows")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=float, default=0.1, help="fraction of agent rows held out for validation")
    a = p.parse_args()
    d = np.load(a.demos)
    ent, mask, sv, gv, act, done = (d[k] for k in ("entities", "mask", "self", "global", "action", "done"))
    T, N = act.shape[:2]
    ent = ent.astype(np.float32)
    n_val = max(1, int(round(N * a.holdout)))
    val_rows, tr_rows = np.arange(N)[:n_val], np.arange(N)[n_val:]
    policy = LanePolicy(PolicyConfig(core=a.core))
    run = RunDir(a.out, f"bc-{a.core}", {"train": {"policy": policy.cfg._asdict()}, "demos": str(a.demos),
                                        "shape": {"T": int(T), "N": int(N)}, "epochs": a.epochs, "lr": a.lr,
                                        "purpose": "representability diagnostic (BC of the scripted last-hitter); not a gate prior"})
    key = jax.random.key(a.seed)
    carry0 = policy.initial_carry((1,)) if a.core == "gru" else None
    init_args = (ent[:1, 0], mask[:1, 0], sv[:1, 0], gv[:1, 0]) + ((carry0,) if carry0 is not None else ())
    params = policy.init(key, *init_args)
    tx = optax.adam(a.lr); opt = tx.init(params)

    def heads_loss(logits, action):
        lp = [jax.nn.log_softmax(l, axis=-1) for l in (logits.button, logits.screen_x, logits.screen_y)]
        nll = -sum(jnp.take_along_axis(l, action[..., i:i+1], axis=-1)[..., 0] for i, l in enumerate(lp))
        acc = jnp.stack([jnp.argmax(l, -1) == action[..., i] for i, l in enumerate(lp)], -1)
        return nll.mean(), acc.mean(0)

    if a.core == "mlp":
        def loss_fn(params, e, m, s, g, act):
            logits = policy.apply(params, e, m, s, g)
            return heads_loss(logits, act)
    else:
        def loss_fn(params, e, m, s, g, act, dn):
            # e,m,s,g,act: [B, L, ...]; scan over L with carry reset after terminals
            def step(carry, xs):
                e_, m_, s_, g_, d_prev = xs
                carry = jnp.where(d_prev[:, None], 0.0, carry)
                logits, carry = policy.apply(params, e_, m_, s_, g_, carry)
                return carry.astype(jnp.float32), logits
            tm = lambda x: jnp.swapaxes(x, 0, 1)
            d_prev = jnp.concatenate([jnp.zeros_like(dn[:, :1]), dn[:, :-1]], axis=1)
            _, logits = jax.lax.scan(step, jnp.zeros((e.shape[0], policy.cfg.core_dim), jnp.float32),
                                     (tm(e), tm(m), tm(s), tm(g), tm(d_prev)))
            return heads_loss(jax.tree.map(tm, logits), act)

    @jax.jit
    def update(params, opt, *batch):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *batch)
        upd, opt = tx.update(grads, opt, params)
        return optax.apply_updates(params, upd), opt, loss, acc
    eval_loss = jax.jit(loss_fn)

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for epoch in range(a.epochs):
        if a.core == "mlp":
            idx = np.array([(t, n) for t in range(T) for n in tr_rows]); rng.shuffle(idx)
            losses, accs = [], []
            for i in range(0, len(idx), a.batch):
                b = idx[i:i+a.batch]; sel = (b[:, 0], b[:, 1])
                params, opt, l, acc = update(params, opt, ent[sel], mask[sel], sv[sel], gv[sel], act[sel].astype(np.int32))
                losses.append(float(l)); accs.append(np.asarray(acc))
            vsel = (np.repeat(np.arange(T), n_val), np.tile(val_rows, T))
            vl, vacc = eval_loss(params, ent[vsel], mask[vsel], sv[vsel], gv[vsel], act[vsel].astype(np.int32))
        else:
            starts = np.arange(0, T - a.seq_len + 1, a.seq_len)
            wins = np.array([(s0, n) for s0 in starts for n in tr_rows]); rng.shuffle(wins)
            per = max(1, a.batch // a.seq_len); losses, accs = [], []
            for i in range(0, len(wins), per):
                w = wins[i:i+per]
                sl = lambda x: np.stack([x[s0:s0+a.seq_len, n] for s0, n in w])
                params, opt, l, acc = update(params, opt, sl(ent), sl(mask), sl(sv), sl(gv), sl(act).astype(np.int32), sl(done))
                losses.append(float(l)); accs.append(np.asarray(acc))
            vw = np.array([(s0, n) for s0 in starts for n in val_rows])
            sl = lambda x: np.stack([x[s0:s0+a.seq_len, n] for s0, n in vw])
            vl, vacc = eval_loss(params, sl(ent), sl(mask), sl(sv), sl(gv), sl(act).astype(np.int32), sl(done))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "train_acc_button_x_y": np.mean(accs, 0).round(3).tolist(),
               "val_loss": float(vl), "val_acc_button_x_y": np.asarray(vacc).round(3).tolist(), "wall_s": round(time.time() - t0, 1)}
        run.log(row); print(json.dumps(row), flush=True)
    step = a.epochs * T * len(tr_rows)
    run.save(step, a.epochs, {"params": params, "opt_state": opt, "step": step})
    run.set_results(status="complete", final=row)
    print(run.path / "ckpt_latest.msgpack")


if __name__ == "__main__":
    main()
