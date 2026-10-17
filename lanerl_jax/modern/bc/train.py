"""Supervised training of the modern policy on a converted replay dataset (behaviour cloning).

    python -m lanerl_jax.modern.bc.train DATASET [--epochs 30] [--out /mnt/nfs/checkpoints/lanerl-jax/modern-bc]

The dataset is `dataset.py`'s output: ``plan.json`` (the game list and split) and ``games/<game>.npz``. Every
game is loaded into host memory (entities as float16) and cut into windows of ``--window`` decisions at a
random offset per epoch; a batch is ``--batch`` windows, each run through the policy's GRU from a zero carry
(the first ``--burn-in`` decisions of a window warm the state and carry no loss). The loss per decision is

    label_valid * (w[button] * CE(button) + click_valid * (CE(screen_x) + CE(screen_y)))

with the click target plain on a unit and a Gaussian over neighbouring cells on the ground
(`targets.click_cross_entropy`, ``--sigma``). Buttons the RL trainer masks (`train.DEFAULT_BUTTONS_OFF`)
are masked here too; the data never labels them. Two augmentations against the domain gap: with
``--enemy-id-dropout`` a window's enemy-champion one-hot is zeroed (an unknown opponent), with
``--stat-dropout`` its patched combat stats are held at the window's first decision's values (no item
growth within the window: closer to the world, where nothing is bought).

Validation every epoch on the held-out games: button NLL and accuracy, per-button recall and precision,
click NLL, exact-cell and cell-distance accuracy, split by opponent class and by whether the opponent's kit
exists. Checkpoints are the RL trainer's layout (`rl/run_dir.RunDir`, ``{"params", "opt_state", "step"}``
with the policy config at ``manifest.config.train.policy``), so `modern/train.py --init-from` reads them.
SIGTERM/SIGINT/SIGUSR1 save a checkpoint and stop.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .. import champions as K
from ..actions import MODERN_BUTTON_INDEX, MODERN_BUTTONS
from ..obs import MODERN_WORLD_SELF_DIM
from ..rl.policy import LanePolicy, PolicyConfig
from ..train import DEFAULT_BUTTONS_OFF
from .targets import gaussian_target

__all__ = ["Game", "load_games", "windows", "make_loss", "evaluate", "main"]

_K = len(K.KITS)
ENEMY_ID_COLUMNS = slice(16 + _K, 16 + 2 * _K)
#: Self columns patched from the replay's replicated stats (`convert._self_columns`).
STAT_COLUMNS = np.asarray([10, 11, 12, 13, 16 + 2 * _K + 8, 16 + 2 * _K + 9])     # ad, ap, armor, mr, ms, range


class Game(NamedTuple):
    name: str
    split: str
    group: str                  # "melee" / "ranged"
    kit_known: bool
    entities: np.ndarray        # (T, 32, 20) float16
    pad: np.ndarray             # (T, 32) bool
    self_vec: np.ndarray        # (T, 56) float32
    global_vec: np.ndarray      # (T, 6) float32
    button: np.ndarray          # (T,) int32
    sx: np.ndarray
    sy: np.ndarray
    valid: np.ndarray           # (T,) bool
    click: np.ndarray           # (T,) bool
    on_unit: np.ndarray         # (T,) bool


def load_games(dataset: Path, split: str, limit: int | None = None) -> list[Game]:
    plan = json.loads((dataset / "plan.json").read_text())
    out = []
    for g in plan["games"]:
        if g["split"] != split:
            continue
        f = dataset / "games" / f"{g['game']}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        out.append(Game(g["game"], split, g["opponent_class"], g.get("opponent_kit") is not None,
                        d["entities"].astype(np.float16), d["entity_pad_mask"], d["self_vec"], d["global_vec"],
                        d["button"].astype(np.int32), d["screen_x"].astype(np.int32), d["screen_y"].astype(np.int32),
                        d["label_valid"], d["click_valid"] & d["label_valid"], d["click_on_unit"]))
        if limit and len(out) >= limit:
            break
    return out


def windows(games: list[Game], window: int, rng: np.random.Generator | None):
    """(game index, start) of every window of `window` decisions; a random offset per game when `rng` is
    given, else from 0. A game's tail shorter than the window is padded by `batch_of`."""
    out = []
    for i, g in enumerate(games):
        T = len(g.button)
        off = int(rng.integers(0, window)) if rng is not None else 0
        out.extend((i, s) for s in range(off, T, window))
        if off:
            out.append((i, 0))                             # the head before the offset, padded
    return out


def batch_of(games: list[Game], items, window: int, burn_in: int, rng: np.random.Generator | None,
             enemy_id_dropout: float = 0.0, stat_dropout: float = 0.0):
    """Stack windows into a batch dict of arrays [N, window, ...]; `weight` is the loss weight per decision."""
    n = len(items)
    ent = np.zeros((n, window, 32, 20), np.float32)
    pad = np.ones((n, window, 32), bool)
    sv = np.zeros((n, window, MODERN_WORLD_SELF_DIM), np.float32)
    gv = np.zeros((n, window, 6), np.float32)
    button = np.zeros((n, window), np.int32)
    sx = np.zeros((n, window), np.int32)
    sy = np.zeros((n, window), np.int32)
    weight = np.zeros((n, window), np.float32)
    click = np.zeros((n, window), np.float32)
    on_unit = np.zeros((n, window), bool)
    for j, (gi, s) in enumerate(items):
        g = games[gi]
        e = min(s + window, len(g.button))
        L = e - s
        ent[j, :L] = g.entities[s:e]
        pad[j, :L] = g.pad[s:e]
        sv[j, :L] = g.self_vec[s:e]
        gv[j, :L] = g.global_vec[s:e]
        button[j, :L] = g.button[s:e]
        sx[j, :L] = g.sx[s:e]
        sy[j, :L] = g.sy[s:e]
        w = g.valid[s:e].astype(np.float32)
        w[:burn_in] = 0.0
        weight[j, :L] = w
        click[j, :L] = g.click[s:e]
        on_unit[j, :L] = g.on_unit[s:e]
        if rng is not None and enemy_id_dropout > 0 and rng.random() < enemy_id_dropout:
            sv[j, :, ENEMY_ID_COLUMNS] = 0.0
        if rng is not None and stat_dropout > 0 and rng.random() < stat_dropout:
            sv[j, :, STAT_COLUMNS] = sv[j, :1, STAT_COLUMNS]
    return dict(entities=ent, mask=pad, self=sv, global_=gv, button=button, sx=sx, sy=sy, weight=weight,
                click=click, on_unit=on_unit)


def make_forward(policy: LanePolicy):
    def forward(params, batch):
        tm = lambda x: jnp.swapaxes(x, 0, 1)                                    # noqa: E731
        ent, mask, sv, gv = (tm(batch[k]) for k in ("entities", "mask", "self", "global_"))
        n = ent.shape[1]

        def step(carry, xs):
            e, m, s, g = xs
            logits, carry = policy.apply(params, e, m, s, g, carry)
            return carry, logits
        _, logits = jax.lax.scan(step, policy.initial_carry((n,)), (ent, mask, sv, gv))
        return jax.tree.map(tm, logits)
    return forward


def _per_decision(logits, batch, sigma: float, button_weight):
    """Per-decision terms (N, T): button CE, click CE, button correct, click exact, click cell distance."""
    lb = jax.nn.log_softmax(logits.button, -1)
    ce_b = -jnp.take_along_axis(lb, batch["button"][..., None], -1)[..., 0]
    sig = jnp.where(batch["on_unit"], 0.0, sigma)

    def ce(lg, idx):
        one_hot = (jnp.arange(lg.shape[-1]) == idx[..., None]).astype(jnp.float32)
        target = jnp.where((sig > 0)[..., None], gaussian_target(idx, lg.shape[-1], max(sigma, 1e-6)), one_hot)
        return -jnp.sum(target * jax.nn.log_softmax(lg, -1), -1)
    ce_c = ce(logits.screen_x, batch["sx"]) + ce(logits.screen_y, batch["sy"])
    pred_b = jnp.argmax(logits.button, -1)
    px, py = jnp.argmax(logits.screen_x, -1), jnp.argmax(logits.screen_y, -1)
    dist = jnp.abs(px - batch["sx"]) + jnp.abs(py - batch["sy"])
    return dict(ce_b=ce_b * button_weight[batch["button"]], ce_c=ce_c, correct=(pred_b == batch["button"]),
                exact=(dist == 0), dist=dist.astype(jnp.float32), pred_b=pred_b)


def make_loss(policy: LanePolicy, sigma: float, button_weight):
    forward = make_forward(policy)
    bw = jnp.asarray(button_weight, jnp.float32)

    def loss(params, batch):
        terms = _per_decision(forward(params, batch), batch, sigma, bw)
        w, c = batch["weight"], batch["weight"] * batch["click"]
        n_w, n_c = jnp.maximum(w.sum(), 1.0), jnp.maximum(c.sum(), 1.0)
        total = (jnp.sum(w * terms["ce_b"]) + jnp.sum(c * terms["ce_c"])) / n_w
        info = dict(loss=total, button_nll=jnp.sum(w * terms["ce_b"]) / n_w,
                    click_nll=jnp.sum(c * terms["ce_c"]) / n_c,
                    button_acc=jnp.sum(w * terms["correct"]) / n_w, click_exact=jnp.sum(c * terms["exact"]) / n_c,
                    click_dist=jnp.sum(c * terms["dist"]) / n_c)
        return total, info
    loss.forward = forward
    loss.per_decision = lambda params, batch: _per_decision(forward(params, batch), batch, sigma, bw)
    return loss


def evaluate(per_decision, params, games: list[Game], window: int, burn_in: int, batch: int):
    """Validation metrics over whole games (windows from offset 0), overall and by group."""
    nb = len(MODERN_BUTTONS)
    acc = {}

    def bucket(name):
        return acc.setdefault(name, dict(n=0.0, ce_b=0.0, correct=0.0, n_c=0.0, ce_c=0.0, exact=0.0, dist=0.0,
                                         conf=np.zeros((nb, nb))))
    items = windows(games, window, None)
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b = batch_of(games, chunk, window, burn_in, None)
        b = {k: jnp.asarray(v) for k, v in b.items()}
        t = jax.device_get(per_decision(params, b))
        w, c = np.asarray(b["weight"]), np.asarray(b["weight"] * b["click"])
        for j, (gi, _) in enumerate(chunk):
            g = games[gi]
            for name in ("all", g.group, "kit_known" if g.kit_known else "kit_unknown"):
                a = bucket(name)
                a["n"] += w[j].sum(); a["ce_b"] += (w[j] * t["ce_b"][j]).sum()
                a["correct"] += (w[j] * t["correct"][j]).sum()
                a["n_c"] += c[j].sum(); a["ce_c"] += (c[j] * t["ce_c"][j]).sum()
                a["exact"] += (c[j] * t["exact"][j]).sum(); a["dist"] += (c[j] * t["dist"][j]).sum()
                m = w[j] > 0
                np.add.at(a["conf"], (np.asarray(b["button"][j])[m], t["pred_b"][j][m]), 1.0)
    out = {}
    for name, a in acc.items():
        conf = a["conf"]
        recall = {MODERN_BUTTONS[k]: float(conf[k, k] / conf[k].sum()) for k in range(nb) if conf[k].sum() > 0}
        precision = {MODERN_BUTTONS[k]: float(conf[k, k] / conf[:, k].sum()) for k in range(nb)
                     if conf[:, k].sum() > 0}
        share = {MODERN_BUTTONS[k]: float(conf[k].sum() / max(conf.sum(), 1)) for k in range(nb) if conf[k].sum() > 0}
        out[name] = dict(decisions=int(a["n"]), button_nll=a["ce_b"] / max(a["n"], 1),
                         button_acc=a["correct"] / max(a["n"], 1), majority_acc=max(share.values()) if share else 0.0,
                         clicks=int(a["n_c"]), click_nll=a["ce_c"] / max(a["n_c"], 1),
                         click_exact=a["exact"] / max(a["n_c"], 1), click_dist=a["dist"] / max(a["n_c"], 1),
                         recall=recall, precision=precision, share=share)
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", type=Path)
    p.add_argument("--out", type=Path, default=Path("/mnt/nfs/checkpoints/lanerl-jax/modern-bc"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--burn-in", type=int, default=32)
    p.add_argument("--batch", type=int, default=128, help="windows per optimiser step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-floor", type=float, default=3e-5, help="cosine decay to this")
    p.add_argument("--warmup", type=int, default=200, help="optimiser steps of linear LR warm-up")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian cells for ground clicks; 0 = plain")
    p.add_argument("--button-weights", default="", help="name=weight,... (default 1 for every button)")
    p.add_argument("--enemy-id-dropout", type=float, default=0.3)
    p.add_argument("--stat-dropout", type=float, default=0.0)
    p.add_argument("--core", default="gru", choices=("gru",))
    p.add_argument("--no-core-norm", action="store_true")
    p.add_argument("--no-core-residual", action="store_true")
    p.add_argument("--buttons-off", default=",".join(DEFAULT_BUTTONS_OFF))
    p.add_argument("--max-games", type=int, default=0, help="limit games per split (smoke tests)")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--tag", default="modern-bc")
    p.add_argument("--notes", default="")
    return p


def main(argv=None):
    from flax.serialization import from_state_dict, msgpack_restore

    from ..jax_cache import enable_compile_cache
    from ..rl.run_dir import RunDir
    a = build_parser().parse_args(argv)
    stopping: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    enable_compile_cache()
    t0 = time.perf_counter()
    train = load_games(a.dataset, "train", a.max_games or None)
    val = load_games(a.dataset, "val", a.max_games or None)
    n_train = sum(int(g.valid.sum()) for g in train)
    print(f"loaded {len(train)} train games ({n_train} decisions to learn from), {len(val)} val games "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    bw = np.ones(len(MODERN_BUTTONS), np.float32)
    for kv in filter(None, a.button_weights.split(",")):
        name, v = kv.split("=")
        bw[MODERN_BUTTON_INDEX[name]] = float(v)
    buttons_off = tuple(b for b in a.buttons_off.split(",") if b)
    off = tuple(sorted(MODERN_BUTTON_INDEX[b] for b in buttons_off))
    pcfg = PolicyConfig(core=a.core, core_norm=not a.no_core_norm, core_residual=not a.no_core_residual)
    policy = LanePolicy(pcfg, off)
    loss = make_loss(policy, a.sigma, bw)
    rng = np.random.default_rng(a.seed)
    steps_per_epoch = (len(windows(train, a.window, rng)) + a.batch - 1) // a.batch
    total_steps = steps_per_epoch * a.epochs
    schedule = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(a.warmup, max(total_steps // 10, 1)), total_steps,
                                                  end_value=a.lr_floor)
    tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), optax.adam(schedule, eps=1e-5))
    b0 = batch_of(train, windows(train, a.window, None)[:1], a.window, a.burn_in, None)
    params = policy.init(jax.random.key(a.seed), jnp.asarray(b0["entities"][0, 0]), jnp.asarray(b0["mask"][0, 0]),
                         jnp.asarray(b0["self"][0, 0]), jnp.asarray(b0["global_"][0, 0]), policy.initial_carry(()))
    opt_state = tx.init(params)
    step = 0
    if a.resume is not None:
        saved = msgpack_restore(a.resume.read_bytes())
        params = from_state_dict(params, saved["params"])
        opt_state = from_state_dict(opt_state, saved["opt_state"])
        step = int(saved["step"])
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
    run = RunDir(a.out, a.tag, {
        "train": {"policy": pcfg._asdict()}, "bc": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "dataset": {"path": str(a.dataset), "train_games": len(train), "val_games": len(val),
                    "train_decisions": n_train},
        "buttons_off": buttons_off, "button_weights": bw.tolist(), "n_params": n_params,
        "observation": "modern-world-v1 (converted replays)", "action": "modern-world-v2 (button, sx, sy)"},
        notes=a.notes)
    print(f"run dir {run.path}; {n_params} params; {steps_per_epoch} steps/epoch", flush=True)

    @jax.jit
    def train_step(params, opt_state, batch):
        (total, info), grads = jax.value_and_grad(loss, has_aux=True)(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        info["grad_norm"] = optax.global_norm(grads)
        return optax.apply_updates(params, updates), opt_state, info
    per_decision = jax.jit(loss.per_decision)
    payload = lambda: {"params": params, "opt_state": opt_state, "step": jnp.asarray(step, jnp.int32)}  # noqa: E731

    best = None
    status = "finished"
    for epoch in range(a.epochs):
        items = windows(train, a.window, rng)
        rng.shuffle(items)
        t1 = time.perf_counter()
        sums, n_steps = {}, 0
        for i in range(0, len(items), a.batch):
            if stopping:
                break
            b = batch_of(train, items[i:i + a.batch], a.window, a.burn_in, rng, a.enemy_id_dropout, a.stat_dropout)
            params, opt_state, info = train_step(params, opt_state, {k: jnp.asarray(v) for k, v in b.items()})
            info = jax.device_get(info)
            for k, v in info.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n_steps += 1
            step += 1
        if stopping:
            status = "preempted"
            break
        dt = time.perf_counter() - t1
        row = {f"train_{k}": v / max(n_steps, 1) for k, v in sums.items()}
        ev = evaluate(per_decision, params, val, a.window, a.burn_in, a.eval_batch) if val else {}
        row.update(epoch=epoch + 1, step=step, lr=float(schedule(step)), epoch_s=round(dt, 1),
                   dec_per_s=round(n_steps * a.batch * a.window / dt), val=ev)
        run.log(row)
        v = ev.get("all", {})
        print(f"epoch {epoch + 1:>3} train nll {row['train_button_nll']:.3f} acc {row['train_button_acc']:.3f} "
              f"click {row['train_click_nll']:.3f} | val nll {v.get('button_nll', float('nan')):.3f} "
              f"acc {v.get('button_acc', float('nan')):.3f} (maj {v.get('majority_acc', float('nan')):.3f}) "
              f"click {v.get('click_nll', float('nan')):.3f} exact {v.get('click_exact', float('nan')):.3f} "
              f"dist {v.get('click_dist', float('nan')):.2f} | {row['dec_per_s']} dec/s {dt:.0f}s", flush=True)
        if not np.isfinite(row["train_loss"]):
            run.save(step, epoch + 1, payload(), latest=False)
            run.set_results(status="diverged", epoch=epoch + 1)
            run.close()
            return
        score = v.get("button_nll", row["train_button_nll"])
        if best is None or score < best:
            best = score
            run.save(step, epoch + 1, payload())
            run.set_results(best_epoch=epoch + 1, best_val_button_nll=best, val=ev)
    if status == "preempted":
        run.save(step, 0, payload())
    run.set_results(status=status, steps=step, **({"signal": int(stopping[0])} if stopping else {}))
    run.close()
    print(f"{status}; run dir {run.path}", flush=True)


if __name__ == "__main__":
    main()
