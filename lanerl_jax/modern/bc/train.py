"""Supervised training of the modern policy on a converted replay dataset (behaviour cloning).

    python -m lanerl_jax.modern.bc.train DATASET [--epochs 30] [--out /mnt/nfs/checkpoints/lanerl-jax/modern-bc]

The dataset is `dataset.py`'s output: ``plan.json`` (the game list and split) and ``games/<game>.npz``. Every
game is loaded into host memory (entities as float16; with ``--cache DIR`` unpacked once to one file per game there and
memory-mapped instead, for datasets larger than host memory) and cut into windows of ``--window`` decisions at a
random offset per epoch; a batch is ``--batch`` windows, each run through the policy's GRU from a zero carry
(the first ``--burn-in`` decisions of a window warm the state and carry no loss). The loss per decision is

    label_valid * (w[button] * CE(button) + click_valid * (CE(screen_x) + CE(screen_y)))

with the click target plain on a unit and a Gaussian over neighbouring cells on the ground
(`targets.click_cross_entropy`, ``--sigma``). With ``--cast-time-sigma`` (seconds) a cast's button target is
spread in time (`timing_targets`): a Gaussian over the decisions within ``--cast-time-window`` of the press,
only onto decisions labelled noop or move whose observation shows that spell ready, normalised so the
press keeps one unit of mass in total (the cast rate stays calibrated; a near miss is no longer a full miss). Buttons the RL trainer masks (`train.DEFAULT_BUTTONS_OFF`)
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
import shutil
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
from .. import obs as OB
from ..rl.policy import LanePolicy, make_policy_config
from ..train import DEFAULT_BUTTONS_OFF
from .convert import _self_columns
from .targets import gaussian_target
from .sources import SOURCE_CONTRACT, TARGET_CONTRACT, require_replay_only

__all__ = ["Game", "load_games", "windows", "make_loss", "evaluate", "main"]


def enemy_id_columns(profile: str = OB.PROFILE):
    """Self columns that name the enemy champion: the kit one-hot (v1) or the enemy index (v2); zeroed by
    ``--enemy-id-dropout`` (an unknown opponent; under v2 the descriptor block stays)."""
    col = _self_columns(profile)
    if profile in OB.TABLE_PROFILES:
        return np.asarray([col["enemy_id"]])
    return np.asarray([col[f"enemy_{x.NAME}"] for x in K.KITS])


def stat_columns(profile: str = OB.PROFILE):
    """Self columns patched from the replay's replicated stats (`convert._self_columns`): ad, ap, armor, mr, ms, range."""
    col = _self_columns(profile)
    return np.asarray([col[n] for n in ("ad", "ap", "armor", "mr", "ms", "range")])


def dataset_profile(dataset: Path) -> str:
    """The observation profile a converted dataset was built with (its manifest, else plan.json, else v1)."""
    manifest = dataset / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text()).get("profiles", {}).get("obs", OB.PROFILE)
    return json.loads((dataset / "plan.json").read_text()).get("profile", OB.PROFILE)


class Game(NamedTuple):
    name: str
    split: str
    group: str                  # "melee" / "ranged"
    kit_known: bool
    entities: np.ndarray        # (T, 32, 20) float16
    pad: np.ndarray             # (T, 32) bool
    self_vec: np.ndarray        # (T, obs.self_dim(profile)) float32
    global_vec: np.ndarray      # (T, 6) float32
    button: np.ndarray          # (T,) int32
    sx: np.ndarray
    sy: np.ndarray
    valid: np.ndarray           # (T,) bool
    click: np.ndarray           # (T,) bool
    on_unit: np.ndarray         # (T,) bool
    held: np.ndarray            # (T,) bool: label filled from the active order (datasets without it: False)
    own_m: np.ndarray | None = None   # (T,) float16: target mass on the label's button (`timing_targets`)
    alt_b: np.ndarray | None = None   # (T,) int8: a nearby press's button, with mass alt_m
    alt_m: np.ndarray | None = None
    observed_self: np.ndarray | None = None
    observed_global: np.ndarray | None = None
    champion: str = "unknown"
    role: str = "unknown"
    excluded_targets: int = 0


#: Buttons whose press is spread in time by ``--cast-time-sigma``, and the self column that says it is ready.
TIMED = {"q": "cd_q", "w": "cd_w", "e": "cd_e", "r": "cd_r", "summoner_d": "summ_d", "summoner_f": "summ_f",
         "recall": None, "ward": None}
_SPREAD_ONTO = ("noop", "move")


def timing_targets(g: Game, sigma_s: float, window_s: float, profile: str, dt: float = 0.1) -> Game:
    """``g`` with soft button targets: each event press (not held) of a `TIMED` button keeps mass
    exp(-d^2 / 2 sigma^2) at its own decision (d = 0) and gives the same to every decision within
    ``window_s`` labelled noop or move, valid, not already carrying another press's mass, and (for a spell)
    whose observation shows that spell ready; the masses are normalised to 1. Right after a press the spell
    shows on cooldown, so the spread falls mostly before it (the observation is lagged one decision)."""
    col = _self_columns(profile)
    T = len(g.button)
    own = np.ones(T, np.float32)
    alt_b = np.zeros(T, np.int32)
    alt_m = np.zeros(T, np.float32)
    k = int(round(window_s / dt))
    kernel = np.exp(-0.5 * (np.arange(-k, k + 1) * dt / sigma_s) ** 2)
    onto = np.isin(g.button, [MODERN_BUTTON_INDEX[b] for b in _SPREAD_ONTO]) & g.valid
    for name, cd_col in TIMED.items():
        b = MODERN_BUTTON_INDEX[name]
        ready = g.self_vec[:, col[cd_col]] <= 0 if cd_col else np.ones(T, bool)
        for t in np.flatnonzero((g.button == b) & g.valid & ~g.held):
            lo, hi = max(t - k, 0), min(t + k + 1, T)
            idx = np.arange(lo, hi)
            ok = (idx == t) | (onto[lo:hi] & ready[lo:hi] & (alt_m[lo:hi] == 0))
            w = kernel[idx - t + k] * ok
            w = w / w.sum()
            nb = idx[ok & (idx != t)]
            alt_b[nb] = b
            alt_m[nb] = w[nb - lo]
            own[nb] = 1.0 - alt_m[nb]
            own[t] = w[t - lo]
    return g._replace(own_m=own.astype(np.float16), alt_b=alt_b.astype(np.int8), alt_m=alt_m.astype(np.float16))


class PackedEntities:
    """(T, 32, 20) float16 entity rows stored without the padding rows (zeros; about 7 of the 32 rows are
    units), so a whole dataset's cache fits on local disk. Slicing returns the dense rows."""

    def __init__(self, rows, offsets, pad):
        self.rows, self.offsets, self.pad = rows, offsets, pad

    def __len__(self):
        return len(self.pad)

    def __getitem__(self, sl: slice) -> np.ndarray:
        s, e, _ = sl.indices(len(self.pad))
        out = np.zeros((e - s, *self.pad.shape[1:], self.rows.shape[1]), np.float16)
        out[~np.asarray(self.pad[s:e])] = self.rows[self.offsets[s]:self.offsets[e]]
        return out


_CACHED = ("entity_pad_mask", "self_vec", "global_vec", "button", "screen_x", "screen_y",
           "label_valid", "click_valid", "click_on_unit", "held")


def game_arrays(f: Path) -> dict:
    """A converted game's arrays, with the label arrays of ``<game>.labels.npz`` (`dataset.relabel`) where
    it exists."""
    meta = json.loads(f.with_suffix(".json").read_text())
    require_replay_only("BC game arrays", meta)
    with np.load(f, allow_pickle=False) as d:
        a = {k: d[k] for k in d.files}
    side = f.with_name(f.stem + ".labels.npz")
    if side.exists():
        require_replay_only("BC label sidecar", meta.get("relabel"))
        with np.load(side, allow_pickle=False) as s:
            a.update({k: s[k] for k in s.files})
    return a


def _arrays(f: Path, cache: Path | None) -> dict:
    """A game's arrays: from the .npz, or memory-mapped from ``cache/<game>.bin`` (written on first use).
    One file and one mapping per game, all fields views into it: a mapping per field ran a 58k-game dataset
    past the process's open-file and memory-map limits."""
    if cache is None:
        return game_arrays(f)
    data, layout = cache / f"{f.stem}.bin", cache / f"{f.stem}.json"
    side = f.with_name(f.stem + ".labels.npz")
    require_replay_only("BC cache game", json.loads(f.with_suffix(".json").read_text()))
    def signature(path):
        return [str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size] if path.exists() else None
    stamp = [SOURCE_CONTRACT, signature(f), signature(f.with_suffix(".json")), signature(side)]
    cached = json.loads(layout.read_text()) if layout.exists() else {}
    if cached.get("source") != stamp or not data.exists():
        d = game_arrays(f)
        units = ~d["entity_pad_mask"]
        parts = {k: d[k] for k in _CACHED if k in d}
        parts["entity_rows"] = d["entities"].astype(np.float16)[units]
        parts["entity_offsets"] = np.concatenate([[0], np.cumsum(units.sum(1))]).astype(np.int64)
        spec, end = {}, 0
        tmp = data.with_suffix(".bin.tmp")
        with open(tmp, "wb") as fh:
            for k, a in parts.items():
                a = np.ascontiguousarray(a)
                end = -(-end // 64) * 64
                spec[k] = (end, a.dtype.str, list(a.shape))
                fh.seek(end)
                fh.write(a.tobytes())
                end += a.nbytes
            fh.truncate(max(end, 1))
        tmp.rename(data)
        layout.write_text(json.dumps(dict(source=stamp, fields=spec)))              # last: its presence marks the game complete
    spec = json.loads(layout.read_text())["fields"]
    mm = np.memmap(data, np.uint8, "r")
    a = {k: np.ndarray(tuple(shape), np.dtype(dt), buffer=mm, offset=off) for k, (off, dt, shape) in spec.items()}
    a["entities"] = PackedEntities(a.pop("entity_rows"), np.array(a.pop("entity_offsets")), a["entity_pad_mask"])
    return a


def _raise_open_files():
    """Each mapped game holds a file descriptor (``mmap`` keeps one): lift the soft limit to the hard one."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def target_supervision_mask(button, metadata):
    """Older arrays cannot distinguish slot4/5/12 origin fallbacks from qualified targets."""
    if metadata.get("label_target_contract") == TARGET_CONTRACT:
        return np.ones(np.shape(button), bool)
    unknown = [MODERN_BUTTON_INDEX[n] for n in ("summoner_d", "summoner_f", "ward")]
    return ~np.isin(button, unknown)


def load_games(dataset: Path, split: str, limit: int | None = None, cache: Path | None = None) -> list[Game]:
    require_replay_only("BC training data load", dataset)
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("BC training requires a finalized dataset manifest")
    manifest = json.loads(manifest_path.read_text())
    require_replay_only("BC finalized manifest", manifest)
    if manifest.get("missing"):
        raise ValueError("BC training refused: planned games are missing")
    from concurrent.futures import ThreadPoolExecutor
    plan = json.loads((dataset / "plan.json").read_text())
    picked = [g for g in plan["games"] if g["split"] == split and (dataset / "games" / f"{g['game']}.npz").exists()]
    picked = picked[:limit] if limit else picked
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        _raise_open_files()
    with ThreadPoolExecutor(8) as pool:      # zlib releases the GIL: unpacking a large dataset is I/O + inflate bound
        arrays = list(pool.map(lambda g: _arrays(dataset / "games" / f"{g['game']}.npz", cache), picked))
    out = []
    for g, d in zip(picked, arrays):
        meta = json.loads((dataset / "games" / f"{g['game']}.json").read_text())
        available = meta["availability"]
        target_ok = target_supervision_mask(d["button"], meta)
        valid = np.asarray(d["label_valid"]) & target_ok
        out.append(Game(g["game"], split, g["opponent_class"], g.get("opponent_kit") is not None,
                        d["entities"] if cache else d["entities"].astype(np.float16), d["entity_pad_mask"], d["self_vec"],
                        d["global_vec"], d["button"].astype(np.int32, copy=False),
                        d["screen_x"].astype(np.int32, copy=False), d["screen_y"].astype(np.int32, copy=False),
                        valid, d["click_valid"] & valid, d["click_on_unit"],
                        np.asarray(d["held"]) if "held" in d else np.zeros(len(d["button"]), bool),
                        observed_self=np.asarray(available["self_fraction"]) > 0,
                        observed_global=np.asarray(available["global_fraction"]) > 0,
                        champion=meta["champion"], role=meta["role"],
                        excluded_targets=int((np.asarray(d["label_valid"]) & ~target_ok).sum())))
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
             enemy_id_dropout: float = 0.0, stat_dropout: float = 0.0, hold_weight: float = 1.0,
             profile: str | None = None):
    """Stack windows into a batch dict of arrays [N, window, ...]; `weight` is the loss weight per decision
    (`hold_weight` for held labels), `held` marks them. ``profile`` defaults to the games' own layout."""
    n = len(items)
    if profile is None:
        profile = OB.profile_of(games[0].self_vec.shape[1])
    enemy_cols, stat_cols = enemy_id_columns(profile), stat_columns(profile)
    ent = np.zeros((n, window, 32, 20), np.float32)
    pad = np.ones((n, window, 32), bool)
    sv = np.zeros((n, window, OB.self_dim(profile)), np.float32)
    gv = np.zeros((n, window, 6), np.float32)
    button = np.zeros((n, window), np.int32)
    sx = np.zeros((n, window), np.int32)
    sy = np.zeros((n, window), np.int32)
    weight = np.zeros((n, window), np.float32)
    click = np.zeros((n, window), np.float32)
    on_unit = np.zeros((n, window), bool)
    held = np.zeros((n, window), bool)
    own_m = np.ones((n, window), np.float32)
    alt_b = np.zeros((n, window), np.int32)
    alt_m = np.zeros((n, window), np.float32)
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
        w = g.valid[s:e].astype(np.float32) * np.where(g.held[s:e], hold_weight, 1.0).astype(np.float32)
        w[:burn_in] = 0.0
        held[j, :L] = g.held[s:e]
        weight[j, :L] = w
        click[j, :L] = g.click[s:e]
        on_unit[j, :L] = g.on_unit[s:e]
        if g.own_m is not None:
            own_m[j, :L] = g.own_m[s:e]
            alt_b[j, :L] = g.alt_b[s:e]
            alt_m[j, :L] = g.alt_m[s:e]
        if rng is not None and enemy_id_dropout > 0 and rng.random() < enemy_id_dropout:
            sv[j, :, enemy_cols] = 0.0
        if rng is not None and stat_dropout > 0 and rng.random() < stat_dropout:
            sv[j, :, stat_cols] = sv[j, :1, stat_cols]
    return dict(entities=ent, mask=pad, self=sv, global_=gv, button=button, sx=sx, sy=sy, weight=weight,
                click=click, on_unit=on_unit, held=held, own_m=own_m, alt_b=alt_b, alt_m=alt_m)


def make_forward(policy: LanePolicy):
    def forward(params, batch):
        tm = lambda x: jnp.swapaxes(x, 0, 1)                                    # noqa: E731
        ent, mask, sv, gv = (tm(batch[k]) for k in ("entities", "mask", "self", "global_"))
        n = ent.shape[1]

        # Rematerialise each step in the backward pass: without this the scan keeps every step's
        # encoder activations alive (batch 128 x window 256 needed ~20 GB on the 5080).
        @jax.checkpoint
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
    if "own_m" in batch:      # timing-spread targets (`timing_targets`); own_m = 1, alt_m = 0 elsewhere
        ce_alt = -jnp.take_along_axis(lb, batch["alt_b"][..., None], -1)[..., 0]
        ce_b = batch["own_m"] * ce_b + batch["alt_m"] * ce_alt
    sig = jnp.where(batch["on_unit"], 0.0, sigma)

    def ce(lg, idx):
        one_hot = (jnp.arange(lg.shape[-1]) == idx[..., None]).astype(jnp.float32)
        target = jnp.where((sig > 0)[..., None], gaussian_target(idx, lg.shape[-1], max(sigma, 1e-6)), one_hot)
        return -jnp.sum(target * jax.nn.log_softmax(lg, -1), -1)
    ce_c = ce(logits.screen_x, batch["sx"]) + ce(logits.screen_y, batch["sy"])
    pred_b = jnp.argmax(logits.button, -1)
    px, py = jnp.argmax(logits.screen_x, -1), jnp.argmax(logits.screen_y, -1)
    dist = jnp.abs(px - batch["sx"]) + jnp.abs(py - batch["sy"])
    return dict(ce_b=ce_b * button_weight[batch["button"]], ce_raw=ce_b, ce_c=ce_c, correct=(pred_b == batch["button"]),
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


def _dilate(x: np.ndarray, k: int) -> np.ndarray:
    """(N, T) bool: true within k decisions of a true entry, per row."""
    c = np.cumsum(np.pad(x, ((0, 0), (k + 1, k))).astype(np.int32), 1)
    return (c[:, 2 * k + 1:] - c[:, :-2 * k - 1]) > 0


TOLERANCE = 5      # decisions (0.5 s): a top button within this of an event press counts in recall_tol / precision_tol


def evaluate(per_decision, params, games: list[Game], window: int, burn_in: int, batch: int):
    """Validation metrics over whole games (windows from offset 0), overall and by group. Button NLL by class
    is unweighted (``--button-weights`` scales only the training loss). ``recall_tol``: share of event presses
    with the model's top button equal to it within `TOLERANCE` decisions (in the same window);
    ``precision_tol``: share of the model's top-button picks with an event press of it within `TOLERANCE`."""
    nb = len(MODERN_BUTTONS)
    acc = {}

    def bucket(name):
        return acc.setdefault(name, dict(n=0.0, ce_b=0.0, correct=0.0, n_c=0.0, ce_c=0.0, exact=0.0, dist=0.0,
                                         conf=np.zeros((nb, nb)), ce_by=np.zeros(nb), n_by=np.zeros(nb),
                                         click_by=np.zeros(nb), joint_by=np.zeros(nb),
                                         n_ev=0.0, ce_ev=0.0, n_hc=0.0, ce_hc=0.0, dist_hc=0.0,
                                         tol_ev=np.zeros(nb), tol_hit=np.zeros(nb), tol_pred=np.zeros(nb),
                                         tol_near=np.zeros(nb)))
    items = windows(games, window, None)
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b = batch_of(games, chunk, window, burn_in, None)
        b = {k: jnp.asarray(v) for k, v in b.items()}
        t = jax.device_get(per_decision(params, b))
        w, c = np.asarray(b["weight"]), np.asarray(b["weight"] * b["click"])
        held = np.asarray(b["held"])
        bt_all, pred_all, scored = np.asarray(b["button"]), np.asarray(t["pred_b"]), w > 0
        event_all = scored & ~held
        tol = {}
        for k in np.unique(bt_all[event_all]):
            ev_k = event_all & (bt_all == k)
            pr_k = scored & (pred_all == k)
            tol[k] = ((ev_k & _dilate(pr_k, TOLERANCE)).sum(1), ev_k.sum(1),
                      (pr_k & _dilate(ev_k, TOLERANCE)).sum(1), pr_k.sum(1))
        for j, (gi, _) in enumerate(chunk):
            g = games[gi]
            for name in ("all", g.group, "kit_known" if g.kit_known else "kit_unknown",
                         "champion:" + g.champion, "role:" + g.role):
                a = bucket(name)
                a["n"] += w[j].sum(); a["ce_b"] += (w[j] * t["ce_b"][j]).sum()
                a["correct"] += (w[j] * t["correct"][j]).sum()
                a["n_c"] += c[j].sum(); a["ce_c"] += (c[j] * t["ce_c"][j]).sum()
                a["exact"] += (c[j] * t["exact"][j]).sum(); a["dist"] += (c[j] * t["dist"][j]).sum()
                m = w[j] > 0
                bt = np.asarray(b["button"][j])
                cm = np.asarray(b["click"][j]) > 0
                np.add.at(a["click_by"], bt[m & cm], 1.)
                np.add.at(a["joint_by"], bt[m & cm],
                          (t["correct"][j] & t["exact"][j])[m & cm].astype(float))
                np.add.at(a["conf"], (bt[m], t["pred_b"][j][m]), 1.0)
                np.add.at(a["ce_by"], bt[m], t["ce_raw"][j][m]); np.add.at(a["n_by"], bt[m], 1.0)
                for k, (hit, n_ev, near, n_pred) in tol.items():
                    a["tol_hit"][k] += hit[j]; a["tol_ev"][k] += n_ev[j]
                    a["tol_near"][k] += near[j]; a["tol_pred"][k] += n_pred[j]
                ev = m & ~held[j]
                a["n_ev"] += ev.sum(); a["ce_ev"] += t["ce_b"][j][ev].sum()
                hc = (c[j] > 0) & held[j]
                a["n_hc"] += hc.sum(); a["ce_hc"] += t["ce_c"][j][hc].sum(); a["dist_hc"] += t["dist"][j][hc].sum()
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
                         recall=recall, precision=precision, share=share,
                         button_count_by={MODERN_BUTTONS[k]: int(a["n_by"][k]) for k in range(nb)
                                          if a["n_by"][k] > 0},
                         joint_button_click_exact_by={MODERN_BUTTONS[k]: float(a["joint_by"][k] / a["click_by"][k])
                                                      for k in range(nb) if a["click_by"][k] > 0},
                         button_nll_by={MODERN_BUTTONS[k]: float(a["ce_by"][k] / a["n_by"][k]) for k in range(nb)
                                        if a["n_by"][k] > 0},
                         recall_tol={MODERN_BUTTONS[k]: float(a["tol_hit"][k] / a["tol_ev"][k]) for k in range(nb)
                                     if a["tol_ev"][k] > 0},
                         precision_tol={MODERN_BUTTONS[k]: float(a["tol_near"][k] / a["tol_pred"][k]) for k in range(nb)
                                        if a["tol_pred"][k] > 0},
                         event_decisions=int(a["n_ev"]), event_button_nll=a["ce_ev"] / max(a["n_ev"], 1),
                         held_clicks=int(a["n_hc"]), held_click_nll=a["ce_hc"] / max(a["n_hc"], 1),
                         held_click_dist=a["dist_hc"] / max(a["n_hc"], 1))
    return jax.tree.map(lambda v: v.item() if isinstance(v, np.generic) else v, out)


def cast_regressions(baseline, current, max_increase=0.05, min_events=100):
    """Compare unweighted QWER NLL on the same fixed validation decisions."""
    failures = {}
    for button in ("q", "w", "e", "r"):
        count = baseline.get("button_count_by", {}).get(button, 0)
        if count < min_events:
            continue
        reference = baseline["button_nll_by"][button]
        value = current.get("button_nll_by", {}).get(button, float("nan"))
        if current.get("button_count_by", {}).get(button) != count:
            failures[button] = dict(reason="support_changed", baseline_count=count)
        elif not np.isfinite(reference) or not np.isfinite(value) or value > reference + max_increase:
            failures[button] = dict(reason="button_nll", baseline=float(reference), current=float(value),
                                    count=count)
    return failures


class CastGuard:
    """Do not compare a fresh model against its untrained uniform class prior."""
    def __init__(self, baseline, step, start_step=0, max_increase=0.05, min_events=100):
        self.reference = baseline if step >= start_step else None
        self.reference_step = step if self.reference is not None else None
        self.start_step, self.max_increase, self.min_events = start_step, max_increase, min_events

    def check(self, current, step):
        if self.reference is None:
            if step < self.start_step:
                return {}
            self.reference, self.reference_step = current, step
        return cast_regressions(self.reference, current, self.max_increase, self.min_events)


def validation_subset(games: list[Game], limit: int) -> list[Game]:
    """Fixed held-out sample spread across champion/role groups before repeats."""
    if not limit or len(games) <= limit:
        return games
    groups = {}
    for g in sorted(games, key=lambda g: g.name):
        groups.setdefault((g.champion, g.role), []).append(g)
    selected = []
    for depth in range(max(map(len, groups.values()))):
        for key in sorted(groups):
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                if len(selected) == limit:
                    return selected
    return selected


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
    p.add_argument("--cast-time-sigma", type=float, default=0.0,
                   help="seconds: spread cast presses in time onto ready noop/move decisions (timing_targets); 0 = off")
    p.add_argument("--cast-time-window", type=float, default=0.5, help="seconds either side of a press")
    p.add_argument("--button-weights", default="", help="name=weight,... (default 1 for every button)")
    p.add_argument("--enemy-id-dropout", type=float, default=0.3)
    p.add_argument("--stat-dropout", type=float, default=0.0)
    p.add_argument("--hold-weight", type=float, default=0.3, help="loss weight of held (non-event) labels")
    p.add_argument("--core", default="gru", choices=("gru",))
    p.add_argument("--no-core-norm", action="store_true")
    p.add_argument("--no-core-residual", action="store_true")
    p.add_argument("--buttons-off", default=",".join(DEFAULT_BUTTONS_OFF))
    p.add_argument("--max-games", type=int, default=0, help="limit games per split (smoke tests)")
    p.add_argument("--cache", type=Path, default=None, help="unpack games here (one file per game) and memory-map them")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=0, help="checkpoint and sampled held-out evaluation every N steps; 0 = epoch only")
    p.add_argument("--eval-max-games", type=int, default=512, help="fixed champion/role sample for step evaluations; 0 = all")
    p.add_argument("--early-stop-patience", type=int, default=0, help="stop after N sampled evaluations without improvement; 0 = disabled")
    p.add_argument("--early-stop-min-delta", type=float, default=0.01, help="minimum decrease in held-out event button NLL + click NLL")
    p.add_argument("--max-cast-nll-regression", type=float, default=0.05,
                   help="maximum per-QWER NLL increase from sampled baseline; used with --eval-every")
    p.add_argument("--cast-guard-start-step", type=int, default=0,
                   help="bootstrap fresh training before fixing the cast reference; resumes can use 0")
    p.add_argument("--min-cast-events", type=int, default=100, help="minimum fixed-sample support for cast regression checks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--tag", default="modern-bc")
    p.add_argument("--notes", default="")
    return p


def main(argv=None):
    from flax.serialization import from_state_dict, msgpack_restore

    from ..jax_cache import enable_compile_cache
    from ..rl.run_dir import RunDir
    parser = build_parser()
    a = parser.parse_args(argv)
    if min(a.eval_every, a.eval_max_games, a.early_stop_patience, a.early_stop_min_delta, a.max_cast_nll_regression, a.cast_guard_start_step) < 0:
        parser.error("evaluation and early-stop settings must be nonnegative")
    if not np.isfinite(a.max_cast_nll_regression):
        parser.error("--max-cast-nll-regression must be finite")
    if a.min_cast_events < 1:
        parser.error("--min-cast-events must be positive")
    if a.early_stop_patience and not a.eval_every:
        parser.error("--early-stop-patience requires --eval-every")
    stopping: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    enable_compile_cache()
    t0 = time.perf_counter()
    train = load_games(a.dataset, "train", a.max_games or None, a.cache)
    val = load_games(a.dataset, "val", a.max_games or None, a.cache)
    n_train = sum(int(g.valid.sum()) for g in train)
    if not train or not val or n_train == 0:
        raise ValueError("BC training needs nonempty train/validation splits and valid training labels")
    if a.cast_time_sigma > 0:      # training targets only: validation scores the presses where they happened
        profile0 = dataset_profile(a.dataset)
        train = [timing_targets(g, a.cast_time_sigma, a.cast_time_window, profile0) for g in train]
    print(f"loaded {len(train)} train games ({n_train} decisions to learn from), {len(val)} val games "
          f"in {time.perf_counter() - t0:.0f}s", flush=True)
    bw = np.ones(len(MODERN_BUTTONS), np.float32)
    for kv in filter(None, a.button_weights.split(",")):
        name, v = kv.split("=")
        bw[MODERN_BUTTON_INDEX[name]] = float(v)
    buttons_off = tuple(b for b in a.buttons_off.split(",") if b)
    off = tuple(sorted(MODERN_BUTTON_INDEX[b] for b in buttons_off))
    profile = dataset_profile(a.dataset)
    if train and train[0].self_vec.shape[1] != OB.self_dim(profile):
        raise ValueError(f"dataset self width {train[0].self_vec.shape[1]} is not profile {profile}'s")
    unavailable = {}
    if profile == OB.REPLAY_PROFILE:
        unavailable = dict(unsupported_self=tuple(np.flatnonzero(~np.logical_or.reduce([g.observed_self for g in train])).tolist()),
                           unsupported_global=tuple(np.flatnonzero(~np.logical_or.reduce([g.observed_global for g in train])).tolist()))
    pcfg = make_policy_config(profile, core=a.core, core_norm=not a.no_core_norm,
                              core_residual=not a.no_core_residual, **unavailable)
    policy = LanePolicy(pcfg, off)
    loss = make_loss(policy, a.sigma, bw)
    rng = np.random.default_rng(a.seed)
    steps_per_epoch = (len(windows(train, a.window, rng)) + a.batch - 1) // a.batch
    total_steps = steps_per_epoch * a.epochs
    schedule = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(a.warmup, max(total_steps // 10, 1)), total_steps,
                                                  end_value=a.lr_floor)
    tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), optax.adam(schedule, eps=1e-5))
    b0 = batch_of(train, windows(train, a.window, None)[:1], a.window, a.burn_in, None, profile=profile)
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
                    "train_decisions": n_train, "label_target_contract": TARGET_CONTRACT,
                    "excluded_train_targets": sum(g.excluded_targets for g in train),
                    "excluded_val_targets": sum(g.excluded_targets for g in val)},
        "buttons_off": buttons_off, "button_weights": bw.tolist(), "n_params": n_params,
        "observation": f"{profile} (converted replays)", "action": "modern-world-v2 (button, sx, sy)"},
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
    partial_best, stale = float("inf"), 0
    sampled_val = validation_subset(val, a.eval_max_games) if a.eval_every else []
    if sampled_val:
        initial = run.save(step, 0, payload())
        shutil.copyfile(initial, run.path / "best_partial.msgpack")
        baseline = evaluate(per_decision, params, sampled_val, a.window, a.burn_in, a.eval_batch)
        run.log(dict(kind="baseline_validation", step=step, games=[g.name for g in sampled_val], val=baseline))
        partial_best = baseline["all"]["event_button_nll"] + baseline["all"]["click_nll"]
        cast_guard = CastGuard(baseline["all"], step, a.cast_guard_start_step,
                               a.max_cast_nll_regression, a.min_cast_events)
        run.set_results(baseline_val=baseline, partial_monitor="event_button_nll+click_nll",
                        best_partial_step=step, best_partial_score=partial_best,
                        cast_guard=dict(max_nll_increase=a.max_cast_nll_regression, min_events=a.min_cast_events,
                                        start_step=a.cast_guard_start_step, reference_step=cast_guard.reference_step))
    resume_steps = step
    for epoch in range(a.epochs):
        items = windows(train, a.window, rng)
        rng.shuffle(items)
        skipped = min(resume_steps, (len(items) + a.batch - 1) // a.batch)
        resume_steps -= skipped
        if skipped:
            # Replay augmentation draws without loading batches, retaining the original shuffle stream.
            draws = min(skipped * a.batch, len(items)) * ((a.enemy_id_dropout > 0) + (a.stat_dropout > 0))
            rng.random(draws)
            run.log(dict(kind="resume_cursor", epoch=epoch + 1, skipped_batches=skipped, step=step))
        if skipped * a.batch >= len(items):
            continue
        t1 = time.perf_counter()
        sums, n_steps = {}, 0
        for i in range(skipped * a.batch, len(items), a.batch):
            if stopping:
                break
            b = batch_of(train, items[i:i + a.batch], a.window, a.burn_in, rng, a.enemy_id_dropout, a.stat_dropout,
                         a.hold_weight, profile=profile)
            params, opt_state, info = train_step(params, opt_state, {k: jnp.asarray(v) for k, v in b.items()})
            info = jax.device_get(info)
            for k, v in info.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n_steps += 1
            step += 1
            if not all(np.isfinite(float(v)) for v in info.values()):
                run.save(step, epoch + 1, payload(), latest=False)
                status = "diverged"
                break
            if a.eval_every and step % a.eval_every == 0:
                saved = run.save(step, epoch + 1, payload())
                ev = evaluate(per_decision, params, sampled_val, a.window, a.burn_in, a.eval_batch)
                v = ev["all"]
                score = v["event_button_nll"] + v["click_nll"]
                cast_failures = cast_guard.check(v, step)
                improved = not cast_failures and np.isfinite(score) and score < partial_best - a.early_stop_min_delta
                stale = 0 if improved else stale + 1
                if improved:
                    partial_best = score
                    shutil.copyfile(saved, run.path / "best_partial.msgpack")
                    run.set_results(best_partial_step=step, best_partial_score=score, best_partial_val=ev)
                run.log(dict(kind="partial_validation", epoch=epoch + 1, step=step, score=score,
                             improved=bool(improved), stale=stale, cast_regressions=cast_failures,
                             cast_guard_reference_step=cast_guard.reference_step, val=ev))
                print(f"step {step} held-out event+click NLL {score:.3f}; best {partial_best:.3f}; "
                      f"stale {stale}; cast recall {v['recall']}; rejected casts {list(cast_failures)}", flush=True)
                if not np.isfinite(score):
                    status = "diverged"
                    break
                if a.early_stop_patience and stale >= a.early_stop_patience:
                    status = "early_stopped"
                    break
        if stopping or status != "finished":
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
        cast_failures = {}
        if sampled_val:
            guard_ev = ev if sampled_val is val else evaluate(
                per_decision, params, sampled_val, a.window, a.burn_in, a.eval_batch)
            cast_failures = cast_guard.check(guard_ev["all"], step)
            run.log(dict(kind="epoch_cast_guard", step=step, cast_regressions=cast_failures))
        if not cast_failures and (best is None or score < best):
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
