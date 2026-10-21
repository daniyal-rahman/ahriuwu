"""Where a BC checkpoint misses the human's casts: per button, on held-out games.

    python -m lanerl_jax.modern.bc.misses CKPT DATASET [--games 1000] [--out misses.json]

For every event decision (not held) whose label is a cast button, it reports the model's probability of the
human's button, what the model's top button was instead, the spell's cooldown fraction in the observation
(``cd_*`` > 0: the observation said not ready), and timing tolerance: the model's best probability of that
button within +-k decisions, and whether its top button was that button anywhere within +-5 decisions
(10 decisions a second). Rate calibration compares the summed probability of a button over all decisions
with the number of human presses: a ratio near 1 with low recall means the model knows how often but
not when. Decisions are windowed as in `train.evaluate` (zero carry per window, burn-in unscored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..actions import MODERN_BUTTON_INDEX, MODERN_BUTTONS
from ..rl.policy import LanePolicy
from .convert import _self_columns
from .gate import _load, _policy_config
from .train import batch_of, dataset_profile, load_games, make_loss, windows

CASTS = ("q", "w", "e", "r", "summoner_d", "summoner_f", "recall", "ward", "stop")
CD_COLUMN = {"q": "cd_q", "w": "cd_w", "e": "cd_e", "r": "cd_r", "summoner_d": "summ_d", "summoner_f": "summ_f"}
SHIFTS = (0, 1, 2, 3, 5, 10)


def game_probs(forward, params, games, window, burn_in, batch):
    """Per game: (probs (T, buttons) float32, scored (T,) bool), windows as in `train.evaluate`."""
    T = {i: len(g.button) for i, g in enumerate(games)}
    probs = {i: np.zeros((T[i], len(MODERN_BUTTONS)), np.float32) for i in T}
    scored = {i: np.zeros(T[i], bool) for i in T}
    items = windows(games, window, None)
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        b = batch_of(games, chunk, window, burn_in, None)
        p = np.asarray(jax.device_get(jax.nn.softmax(forward(params, {x: jnp.asarray(v) for x, v in b.items()}).button, -1)))
        for j, (gi, s) in enumerate(chunk):
            e = min(s + window, T[gi])
            probs[gi][s:e] = p[j, :e - s]
            scored[gi][s:e] = b["weight"][j, :e - s] > 0
    return [(probs[i], scored[i]) for i in range(len(games))]


def analyse(games, outputs, profile):
    col = _self_columns(profile)
    nb = len(MODERN_BUTTONS)
    acc = {b: dict(events=0, p=[], top=np.zeros(nb), cd=[], best={k: [] for k in SHIFTS}, hit5=0,
                   pred=0, pred_near=0, p_sum=0.0, presses=0, p_sum_notready=0.0, n_notready=0, n_ready=0,
                   presses_ready=0) for b in CASTS}
    for g, (probs, scored) in zip(games, outputs):
        T = len(g.button)
        event = scored & ~g.held
        top = probs.argmax(-1)
        for b in CASTS:
            i = MODERN_BUTTON_INDEX[b]
            a = acc[b]
            pb = probs[:, i]
            human = event & (g.button == i)
            a["p_sum"] += float(pb[scored].sum()); a["presses"] += int((scored & (g.button == i)).sum())
            if b in CD_COLUMN:
                cd = g.self_vec[:, col[CD_COLUMN[b]]]
                ready = cd <= 0
                a["p_sum_notready"] += float(pb[scored & ~ready].sum()); a["n_notready"] += int((scored & ~ready).sum())
                a["n_ready"] += int((scored & ready).sum()); a["presses_ready"] += int((human & ready).sum())
            ev = np.flatnonzero(human)
            a["events"] += len(ev)
            a["p"] += pb[ev].tolist()
            np.add.at(a["top"], top[ev], 1.0)
            if b in CD_COLUMN:
                a["cd"] += g.self_vec[ev, col[CD_COLUMN[b]]].tolist()
            for k in SHIFTS:
                a["best"][k] += [float(pb[max(t - k, 0):t + k + 1].max()) for t in ev]
            a["hit5"] += sum(bool((top[max(t - 5, 0):t + 6] == i).any()) for t in ev)
            mine = np.flatnonzero(scored & (top == i))
            near = np.zeros(T, bool)
            for t in ev:
                near[max(t - 5, 0):t + 6] = True
            a["pred"] += len(mine); a["pred_near"] += int(near[mine].sum())
    out = {}
    for b, a in acc.items():
        p = np.asarray(a["p"])
        if not len(p):
            continue
        r = dict(events=a["events"], recall=float(a["top"][MODERN_BUTTON_INDEX[b]] / a["events"]),
                 p_true=dict(median=float(np.median(p)), p10=float(np.quantile(p, .1)), p90=float(np.quantile(p, .9)),
                             below_1pct=float((p < .01).mean()), below_0_1pct=float((p < .001).mean()),
                             above_50pct=float((p > .5).mean())),
                 top_instead={MODERN_BUTTONS[k]: round(float(v / a["events"]), 4) for k, v in enumerate(a["top"]) if v},
                 best_p_within={f"+-{k}": float(np.median(a["best"][k])) for k in SHIFTS},
                 best_p_within_above_50pct={f"+-{k}": float((np.asarray(a["best"][k]) > .5).mean()) for k in SHIFTS},
                 recall_within_5=a["hit5"] / a["events"],
                 precision=a["pred_near"] / a["pred"] if a["pred"] else None, top_predictions=a["pred"],
                 rate_ratio=a["p_sum"] / max(a["presses"], 1))
        if a["cd"]:
            cd = np.asarray(a["cd"])
            r["obs_cooldown_at_press"] = dict(share_not_ready=float((cd > 0).mean()), share_above_half=float((cd > .5).mean()),
                                              share_full=float((cd >= .999).mean()),
                                              p_true_median_ready=float(np.median(p[cd <= 0])) if (cd <= 0).any() else None,
                                              p_true_median_not_ready=float(np.median(p[cd > 0])) if (cd > 0).any() else None)
            r["ready_decisions"] = a["n_ready"]
            r["press_rate_when_ready"] = a["presses_ready"] / max(a["n_ready"], 1)
            r["mean_p_when_not_ready"] = a["p_sum_notready"] / max(a["n_notready"], 1)
        out[b] = r
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("dataset", type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--burn-in", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    profile = dataset_profile(a.dataset)
    games = load_games(a.dataset, a.split, a.games)
    manifest = json.loads((a.checkpoint.parent / "manifest.json").read_text())["config"]
    off = tuple(sorted(MODERN_BUTTON_INDEX[b] for b in manifest.get("buttons_off", ())))
    policy = LanePolicy(_policy_config(a.checkpoint), off)
    b0 = batch_of(games, windows(games, a.window, None)[:1], a.window, a.burn_in, None)
    template = policy.init(jax.random.key(0), *(jnp.asarray(b0[k][0, 0]) for k in ("entities", "mask", "self", "global_")),
                           policy.initial_carry(()))
    params = _load(a.checkpoint, template)
    forward = jax.jit(make_loss(policy, 1.0, np.ones(len(MODERN_BUTTONS), np.float32)).forward)
    outputs = game_probs(forward, params, games, a.window, a.burn_in, a.batch)
    by_champion_kit = {}
    from .. import champions as K
    for name, sel in (("own_kit", lambda g: g.name.split(".")[-1] in K.BY_NAME),
                      ("table_only", lambda g: g.name.split(".")[-1] not in K.BY_NAME)):
        idx = [i for i, g in enumerate(games) if sel(g)]
        if idx:
            by_champion_kit[name] = dict(games=len(idx), **{b: v["recall"] for b, v in analyse(
                [games[i] for i in idx], [outputs[i] for i in idx], profile).items()})
    result = dict(checkpoint=str(a.checkpoint), dataset=str(a.dataset), split=a.split, games=len(games),
                  buttons=analyse(games, outputs, profile), recall_by_kit=by_champion_kit)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
