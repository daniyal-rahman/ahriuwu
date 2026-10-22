"""Behaviour-cloning data from a decoded replay: the modern policy's observations and the replay champion's
actions, 10 decisions a second over the lane window.

    python -m lanerl_jax.modern.bc.convert OUT_DIR GAME_DIR [GAME_DIR ...] [--lag 1 | --lag 1-2]

v4 builds inputs directly from replay fields and pinned client geometry, with
approved heuristic fog. Missing values are NaN and the policy consumes explicit
finite masks. Legacy datasets remain rejected. See docs/modern/BC.md section 14.

Output per game: ``<game>.npz`` (arrays below, leading axis = decision) and ``<game>.json`` (provenance:
dataset, decoder commit, converter commit; counts; label checks).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .. import champions as K
from .. import obs as OB
from ..actions import MODERN_BUTTON_INDEX, MODERN_BUTTONS
from ..train import LANE_HALF_WIDTH
from . import fog
from .labels import Labels, label_actions, cast_rules
from .sources import SOURCE_CONTRACT, require_replay_only
from . import observation as RO
from .replay import (BLUE, STAT_SCALE, ReplayGame, decision_times, lane_distance,
                     read_game)
from ..data import champion_table as CT

__all__ = ["LANE_WINDOW_S", "OBSERVATION_FIELDS", "BCGame", "convert_game", "received_spell_state", "parse_lag"]

#: Game clock 0:30 (the first wave leaves the base) to 14:00.
LANE_WINDOW_S = (30.0, 840.0)
#: A decision is in scope when the champion is alive, within the trainer's lane corridor (`LANE_HALF_WIDTH`
#: around the top lane, extended to the fountains) and no champion but the two laners is within
#: `THIRD_CHAMPION_RADIUS`.
THIRD_CHAMPION_RADIUS = 2000.0
_TOP = 2
#: The lane each replay role is converted in (0 bot, 1 mid, 2 top), and its index in ``obs.ROLES``.
ROLE_LANE = {"top": 2, "mid": 1, "bot": 0, "support": 0}
#: The arrays that are the policy's view of the game, which `BCGame.lagged` delays.
OBSERVATION_FIELDS = ("entities", "entity_pad_mask", "self_vec", "global_vec", "champion_stats")


def parse_lag(text: str):
    """``"1"`` -> 1; ``"1-2"`` -> (1, 2), a range of decisions drawn per decision."""
    lo, _, hi = text.partition("-")
    return (int(lo), int(hi)) if hi else int(lo)


class BCGame(NamedTuple):
    entities: np.ndarray        # (T, 32, 20) float32
    entity_pad_mask: np.ndarray  # (T, 32) bool, True = empty row
    self_vec: np.ndarray        # (T, 56) float32
    global_vec: np.ndarray      # (T, 6) float32
    champion_stats: np.ndarray  # (T, 2, 11) float32, me then the enemy (zero while it has no row)
    button: np.ndarray          # (T,) int32, `actions.MODERN_BUTTONS`
    screen_x: np.ndarray        # (T,) int32
    screen_y: np.ndarray        # (T,) int32
    uses_screen: np.ndarray     # (T,) float32, the trainer's `actions.screen_usage`
    click_valid: np.ndarray     # (T,) bool: the click loss applies
    click_on_unit: np.ndarray   # (T,) bool: the click is on a unit (plain target), else ground (sloped)
    label_valid: np.ndarray     # (T,) bool: in scope and representable, the decisions to learn from
    held: np.ndarray            # (T,) bool: the label is the order still active, not an event (`labels.py`)
    in_scope: np.ndarray        # (T,) bool
    obs_index: np.ndarray       # (T,) int32: the decision whose observation is shown (itself until `lagged`)
    t: np.ndarray               # (T,) float64 game seconds
    meta: dict

    def lagged(self, lag, seed: int = 0) -> "BCGame":
        """Each action paired with the observation of `lag` decisions earlier: a reaction time. `lag` is a
        count of decisions, or ``(lo, hi)`` to draw one per decision uniformly from lo..hi (seeded).
        Decisions with no observation that old are not learned from."""
        T = len(self.t)
        lo, hi = lag if isinstance(lag, tuple) else (lag, lag)
        k = np.random.default_rng(seed).integers(lo, hi + 1, T) if hi > lo else np.full(T, lo)
        src = np.arange(T) - k
        delayed = {f: getattr(self, f)[np.maximum(src, 0)] for f in OBSERVATION_FIELDS}
        meta = dict(self.meta, observation_lag=dict(decisions=[int(lo), int(hi)], seed=seed,
                                                    seconds_per_decision=0.1))
        if "availability" in meta:
            meta["availability"] = dict(meta["availability"],
                self_fraction=np.isfinite(delayed["self_vec"]).mean(0).tolist(),
                global_fraction=np.isfinite(delayed["global_vec"]).mean(0).tolist())
        return self._replace(**delayed, obs_index=np.maximum(src, 0).astype(np.int32),
                             label_valid=self.label_valid & (src >= 0), meta=meta)

    def save(self, out: Path, name: str | None = None):
        out.mkdir(parents=True, exist_ok=True)
        game = name or self.meta["provenance"]["game"]
        np.savez_compressed(out / f"{game}.npz", **{k: v for k, v in self._asdict().items() if k != "meta"})
        (out / f"{game}.json").write_text(json.dumps(self.meta, indent=1))


def _converter_commit(repo: Path | None = None) -> str:
    """Record checkout state or the pinned commit shipped with an isolated archive."""
    repo = repo or Path(__file__).resolve().parents[3]
    git_path = repo / ".git"
    if not git_path.exists():
        commit = (repo / ".bc-source-commit").read_text().strip()
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Invalid archive .bc-source-commit; expected full Git SHA")
        return commit[:8]
    git_dir = git_path
    if git_path.is_file():
        git_dir = Path(git_path.read_text().split("gitdir:")[1].strip())
        if not git_dir.is_absolute():
            git_dir = repo / git_dir
        if not git_dir.exists():
            git_dir = Path(str(git_dir).replace("/srv/nfs/", "/mnt/nfs/", 1))
    git = ["git", f"--git-dir={git_dir}", f"--work-tree={repo}"]
    head = subprocess.run(git + ["rev-parse", "--short=8", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain", "--", str(repo / "lanerl_jax")],
                           capture_output=True, text=True, check=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def received_spell_state(t, events, casts, n_slots=4):
    """Received timers only; NaN means unknown, never ready or a kit-table estimate.

    A received cast invalidates the previous timer until a countdown for the same
    slot arrives. Messages at the cast's timestamp take precedence. No future
    message is used to backfill an earlier decision. The second result counts
    unknown decision/slot values, rather than table fallbacks.
    """
    t = np.asarray(t, float)
    cooldown = np.full((len(t), n_slots), np.nan)
    for slot in range(n_slots):
        updates = [(float(tc), 0, np.nan) for tc, sl in casts if sl == slot]
        updates += [(float(te), 1, float(rem)) for te, sl, rem, _dur, _mode in events if sl == slot]
        updates.sort(key=lambda row: (row[0], row[1]))
        for i, (te, _kind, rem) in enumerate(updates):
            end = updates[i + 1][0] if i + 1 < len(updates) else np.inf
            sel = (t >= te) & (t < end)
            if np.isfinite(rem) and rem >= 0:
                cooldown[sel, slot] = np.maximum(te + rem - t[sel], 0.0)
    return cooldown, int(np.isnan(cooldown).sum())


def _fill(a):
    """Causal forward fill; leading unknown positions remain unknown."""
    a = np.asarray(a).copy()
    last = np.full(a.shape[1:], np.nan)
    for k, row in enumerate(a):
        if np.all(np.isfinite(row)):
            last = row
        a[k] = last
    return a


def _scope(g: ReplayGame, me_xy, corridor, extra_champions: bool = False):
    """(in scope, in the lane corridor, a third champion near) per decision; a third champion only takes a
    decision out of scope when the observation cannot show it (no ``extra_champions``)."""
    i_me = g.index(g.me)
    others = [i for i, u in enumerate(g.units) if u.family == "champion" and u.id not in (g.me, g.opponent)]
    d = np.hypot(g.x[:, others] - me_xy[:, :1], g.y[:, others] - me_xy[:, 1:])
    crowded = np.any(g.alive[:, others] & (np.nan_to_num(d, nan=np.inf) < THIRD_CHAMPION_RADIUS), axis=1)
    in_lane = lane_distance(me_xy[:, 0], me_xy[:, 1], corridor) < LANE_HALF_WIDTH
    return g.alive[:, i_me] & in_lane & (extra_champions | ~crowded), in_lane, crowded


def _self_names(profile: str = OB.PROFILE) -> list[str]:
    """Names of the self-vector columns of a profile (`obs.build_modern_observation`)."""
    head = ["s", "n", "hp", "level", "gold", "cs", "cd_q", "cd_w", "cd_e", "cd_r", "ad", "ap", "armor", "mr",
            "dead", "recalling"]
    tail = ["mana", "shield", "summ_d", "summ_f", "next_level", "quest", "quest_done", "in_combat", "ms",
            "range", "skill_points", "trinket"]
    if profile in OB.TABLE_PROFILES:
        names = ["own_id", "enemy_id", "tier", *head, *tail, "rank_q", "rank_w", "rank_e", "rank_r",
                 *CT.side_columns("own"), *CT.side_columns("enemy")]
        names += [f"role_{r}" for r in OB.ROLES] if profile in ("modern-world-v3", OB.REPLAY_PROFILE) else []
    else:
        names = [*head, *(f"own_{x.NAME}" for x in K.KITS), *(f"enemy_{x.NAME}" for x in K.KITS), *tail]
    assert len(names) == OB.self_dim(profile)
    return names


def _self_columns(profile: str = OB.PROFILE):
    """Column index of each named self feature (`obs.build_modern_observation`)."""
    return {n: i for i, n in enumerate(_self_names(profile))}


def convert_game(game_dir: Path, window=LANE_WINDOW_S, champion: str = "Garen", hold: bool = True,
                 profile: str = OB.REPLAY_PROFILE, tier: int = 0, role: str = "top",
                 extra_champions: bool = False) -> BCGame:
    """Replay fields with explicit NaN availability and approved heuristic fog."""
    if profile != OB.REPLAY_PROFILE:
        require_replay_only("BC conversion")
    lane = ROLE_LANE[role]
    g = read_game(game_dir, decision_times(*window), champion, lane)
    T = len(g.t)
    me = g.index(g.me)
    team = g.units[me].team
    frame = RO.frames(g, lane)[team]
    me_xy = np.stack([g.x[:, me], g.y[:, me]], -1)
    casts = [(a.t, a.slot) for a in g.actions if a.kind == "cast" and 0 <= a.slot < 6]
    timers, unknown = received_spell_state(g.t, g.cooldown_events, casts, n_slots=6)
    entities, pad, sv, gv, slots, visible = RO.build(
        g, frame, _self_columns(profile), role, tier, extra_champions, timers)
    label_view = np.zeros_like(visible)
    for k in range(T):
        label_view[k, slots[k, slots[k] >= 0]] = True
    label_view[:, me] = np.isfinite(me_xy).all(1)
    labels = _labels(g, champion, team, frame, np.nan_to_num(_fill(me_xy)), hold, label_view)
    scope, in_lane, crowded = _scope(g, me_xy, RO.corridor(lane), extra_champions)
    scope &= np.isfinite(me_xy).all(1) & np.isfinite(g.hp[:, me])
    valid = scope & labels.representable
    actions = [g.actions[i] if i >= 0 else None for i in labels.source]
    attack = np.asarray([a is not None and a.kind == "attack" for a in actions]) & valid
    rcast = valid & (labels.button == MODERN_BUTTON_INDEX["r"])
    target_seen = np.asarray([a is not None and a.target is not None and
                            any(g.units[int(u)].id == a.target for u in slots[k] if u >= 0)
                            for k, a in enumerate(actions)])
    meta = dict(source_contract=SOURCE_CONTRACT,
        provenance=dict(g.provenance, converter_commit=_converter_commit(), obs_profile=profile,
                        action_profile="modern-world-v2", fog=fog.provenance(),
                        geometry=dict(source=RO.geometry()["source"], sha256=RO.geometry()["source_sha256"])),
        champion=champion, role=role, lane=lane, extra_champions=extra_champions,
        opponent=g.opponent_name, opponent_kit=g.opponent_name if g.opponent_name in K.BY_NAME else None,
        team="blue" if team == BLUE else "red", summoners=list(g.summoners), skill_tier=tier,
        window_s=list(window), decisions=T,
        counts=dict(in_scope=int(scope.sum()), label_valid=int(valid.sum()), dead=int((~g.alive[:, me]).sum()),
                    out_of_lane=int((~in_lane).sum()), third_champion=int(crowded.sum()),
                    not_representable=int((~labels.representable).sum()), clicks_clamped=int(labels.clamped.sum()),
                    click_valid=int((labels.click_valid & valid).sum()), superseded=labels.n_superseded,
                    repeat_attacks=labels.n_repeat_attacks, held=int((labels.held & valid).sum()),
                    held_on_unit=int((labels.held & labels.click_on_unit & valid).sum()),
                    cooldown_conflicts=0, unknown_timer_values=unknown,
                    buttons={b: int(np.sum((labels.button == i) & valid)) for i,b in enumerate(MODERN_BUTTONS)}),
        label_check=dict(orders_checked=False, attacks=int(attack.sum()),
                         attacks_target_in_obs=int((attack & target_seen).sum()),
                         r_casts=int(rcast.sum()), r_unit_casts=int((rcast & labels.click_on_unit).sum()),
                         r_target_in_obs=int((rcast & labels.click_on_unit & target_seen).sum())),
        availability=dict(encoding="NaN means unavailable; finite means observed or declared representation",
                          self_fraction=np.isfinite(sv).mean(0).tolist(),
                          global_fraction=np.isfinite(gv).mean(0).tolist(),
                          self_columns=_self_names(profile)),
        not_from_replay=["approved heuristic fog", "canonical camera/layout/scaling", "pinned client geometry"],
        observation_lag=dict(decisions=[0, 0]))
    stats = (g.champion_stats / np.asarray(STAT_SCALE)).astype(np.float32)
    stats[:, 1] = np.where(gv[:, 1:2] > 0, stats[:, 1], np.nan)
    return BCGame(entities, pad, sv, gv, stats, labels.button, labels.screen_x, labels.screen_y,
                  labels.uses_screen, labels.click_valid, labels.click_on_unit, valid, labels.held,
                  scope, np.arange(T, dtype=np.int32), g.t, meta)


def _labels(g: ReplayGame, champion: str, me_team: int, frame, me_xy, hold: bool, visible=None):
    enemies = {u.id for u in g.units if u.team != me_team}
    unit_xy = {u.id: np.stack([g.x[:, i], g.y[:, i]], -1) for i, u in enumerate(g.units)}
    if visible is not None:
        unit_xy = {u.id: np.where(visible[:, i, None], unit_xy[u.id], np.nan) for i, u in enumerate(g.units)}
    return label_actions(g.t, g.actions, me_xy, unit_xy, np.asarray(frame.axis), np.asarray(frame.normal),
                         enemies, g.summoner_slot, hold=hold, rules=cast_rules(champion), me=g.me)


def relabel_game(game_dir: Path, champion: str, role: str = "top", hold: bool = True,
                 window=LANE_WINDOW_S) -> Labels:
    """The labels `convert_game` gives the game now, without building observations (labels depend only on
    the replay's actions and positions and the lane frame)."""
    require_replay_only("Standalone relabeling is unsupported for v4; reconvert with fog and availability")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("games", type=Path, nargs="+")
    parser.add_argument("--lag", type=parse_lag, default=1, help="observation lag in decisions, N or LO-HI")
    parser.add_argument("--seed", type=int, default=0, help="for a lag range")
    parser.add_argument("--no-hold", action="store_true", help="label click events only, not held orders")
    args = parser.parse_args()
    for game_dir in args.games:
        bc = convert_game(game_dir, hold=not args.no_hold).lagged(args.lag, args.seed)
        bc.save(args.out)
        print(game_dir.name, json.dumps(bc.meta["counts"]), flush=True)


if __name__ == "__main__":
    main()
