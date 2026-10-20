"""Build a modern-world behaviour-cloning dataset from decoded top-lane replays of one champion (Garen by default).

    python -m lanerl_jax.modern.bc.dataset plan OUT [--lag 1]      select games, split, vocabulary
    python -m lanerl_jax.modern.bc.dataset convert OUT TASK TASKS  convert every TASKS-th game
    python -m lanerl_jax.modern.bc.dataset finalize OUT            manifest.json and statistics

(`slurm/modern_bc_dataset.sbatch` runs the three on the desktop.)

`plan` reads only export manifests and one stat per game: it keeps games of at
least `MIN_GAME_S` (the lane window plus two minutes), names the lane opponent,
classes it melee or ranged (`stat.mCombatType` as replicated), builds the
champion vocabulary (Garen, then opponents alphabetically; `opponent_kit` names the modern kit where one exists), and holds out `VAL_SHARE` of the games of each class, chosen
by a hash of the game id so the split never depends on what else is present.

`convert` writes ``OUT/games/<game>.npz|json`` (`convert.BCGame.save`), with the
plan's observation lag; a game already written is skipped, so a task can be
rerun. `finalize` writes ``OUT/manifest.json``: datasets and the decoder commit
each was exported with, the converter commit, lag, vocabulary, stat layout,
the split, per-game counts and the totals.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .replay import STAT_NAMES, STAT_SCALE, DECODER_SCRIPTS, decoder_commit
from ..actions import MODERN_BUTTONS as BUTTONS_WITH_STOP
from .convert import LANE_WINDOW_S, convert_game, parse_lag
from .. import obs as OB
from ..data import champion_table as CT

__all__ = ["DATASETS", "MIN_GAME_S", "VAL_SHARE", "plan", "convert", "finalize"]

DATASETS = (Path("/mnt/nfs/datasets/rofl_bc_2619_garen_vs_jax_native_20261014"),
            Path("/mnt/nfs/datasets/rofl_bc_2619_garen_top_native_20261015"))
#: Games shorter than this (game seconds) are left out: the lane window ends at 12:00
#: and a game that ends soon after was not laned normally.
MIN_GAME_S = 840.0
VAL_SHARE = 0.10
CHAMPION = "Garen"
_RANGED = 2      # stat.mCombatType


def _opponent(manifest, champion=CHAMPION):
    roster = manifest["initial_roster"]
    me = next(r for r in roster if r["name"] == champion)
    return next(r for r in roster if r["team"] != me["team"]
                and r["assigned_role"] == me["assigned_role"])


def _combat_type(export: Path, champion_id: int) -> int:
    if str(DECODER_SCRIPTS) not in sys.path:
        sys.path.append(str(DECODER_SCRIPTS))
    from event_training_loader import load_event_game
    native = load_event_game(export / "event-training")
    f = native["fields"]["stat.mCombatType"]
    tracks = np.asarray([t[0] if t[0] is not None else -1 for t in native["index"]["tracks"]])
    rows = np.flatnonzero((tracks[f["entity_track"]] == champion_id) & f["available"])
    return int(f["values"][rows[np.argmin(f["update_index"][rows])]])


def plan(out: Path, lag, hold: bool = True, champion: str = CHAMPION, datasets=DATASETS,
         profile: str = OB.PROFILE, tier: int = CT.ARCHIVE_TIER, champions=None) -> dict:
    """With ``champions`` (several clones in one dataset) every game enters once per listed top laner it
    has, as ``<game>.<champion>``, whoever the export's actor was (the decoder resolves both laners' casts,
    MULTICHAMP.md section 8); mirrors are skipped, and both sides of a game share its split."""
    games, skipped = [], []
    combat = {}      # champion -> stat.mCombatType at spawn, read once per champion
    datasets = tuple(Path(d) for d in datasets)
    for dataset in datasets:
        for game_dir in sorted((dataset / "games").iterdir()):
            export = game_dir / "export"
            manifest_path = export / "manifest.json"
            if not manifest_path.exists():
                skipped.append(dict(game=game_dir.name, reason="no export"))
                continue
            receipt_path = game_dir / "run-manifest.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            if receipt.get("status", "success") != "success" or not (export / "event-training" / "manifest.json").exists():
                # A worker that was preempted mid-export leaves manifest.json without the training tensors.
                skipped.append(dict(game=game_dir.name, reason="export incomplete", status=receipt.get("status")))
                continue
            manifest = json.loads(manifest_path.read_text())
            if champions:
                tops = [r["name"] for r in manifest["initial_roster"] if r["assigned_role"] == "TOP"]
                picks = [c for c in tops if champions == ["all"] or c in champions] \
                    if len(set(tops)) == len(tops) == 2 else []
            elif manifest.get("actor_champion", champion) == champion and any(
                    r["name"] == champion for r in manifest["initial_roster"]):
                picks = [champion]
            else:
                picks = []
            if not picks:
                skipped.append(dict(game=game_dir.name, reason=f"no {'/'.join(champions or [champion])}"))
                continue
            if manifest["end_seconds"] < MIN_GAME_S:
                skipped.append(dict(game=game_dir.name, reason="shorter than 14 min",
                                    seconds=manifest["end_seconds"]))
                continue
            for c in picks:
                opp = _opponent(manifest, c)
                if opp["name"] not in combat:
                    combat[opp["name"]] = _combat_type(export, opp["object_id"])
                games.append(dict(game=f"{game_dir.name}.{c}" if champions else game_dir.name,
                                  base=game_dir.name, champion=c, path=str(game_dir), dataset=str(dataset),
                                  opponent=opp["name"],
                                  opponent_class="ranged" if combat[opp["name"]] == _RANGED else "melee"))
    clones = sorted({g["champion"] for g in games}) if champions == ["all"] else list(champions or [champion])
    vocabulary = clones + sorted({g["opponent"] for g in games} - set(clones))
    # Hold out VAL_SHARE of each class: the games whose hash ranks lowest; a game held out for one side's
    # class holds out its other side too.
    val = set()
    for cls in ("melee", "ranged"):
        members = sorted((g for g in games if g["opponent_class"] == cls),
                         key=lambda g: hashlib.sha256(g["base"].encode()).hexdigest())
        val |= {g["base"] for g in members[:round(VAL_SHARE * len(members))]}
    for g in games:
        g["split"] = "val" if g["base"] in val else "train"
    from .. import champions as K
    for g in games:
        g["opponent_kit"] = g["opponent"] if g["opponent"] in K.BY_NAME else None
    result = dict(lag=lag, hold=hold, champion=champion, champions=champions, profile=profile, tier=tier, window_s=list(LANE_WINDOW_S),
                  min_game_s=MIN_GAME_S, val_share=VAL_SHARE, vocabulary=vocabulary,
                  datasets={str(d): decoder_commit(d) for d in datasets},
                  games=games, skipped=skipped)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(result, indent=1))
    return result


_LIBC = ctypes.CDLL("libc.so.6")


def _seed(game: str) -> int:
    return int(hashlib.sha256(game.encode()).hexdigest()[:8], 16)


def convert(out: Path, task: int, tasks: int):
    from ..jax_cache import enable_compile_cache
    enable_compile_cache()
    p = json.loads((out / "plan.json").read_text())
    lag = tuple(p["lag"]) if isinstance(p["lag"], list) else p["lag"]
    # Tasks take whole games, so both laners of one are converted from a single read (`replay._fields_of`).
    bases = list(dict.fromkeys(g.get("base", g["game"]) for g in p["games"]))
    mine = set(bases[task::tasks])
    for g in (g for g in p["games"] if g.get("base", g["game"]) in mine):
        target = out / "games" / f"{g['game']}.json"
        if target.exists():
            continue
        bc = convert_game(Path(g["path"]), champion=g.get("champion", p.get("champion", CHAMPION)), hold=p.get("hold", True),
                          profile=p.get("profile", OB.PROFILE), tier=p.get("tier", CT.ARCHIVE_TIER))
        bc = bc.lagged(lag, seed=_seed(g["game"]))
        bc.meta.update(split=g["split"], opponent_class=g["opponent_class"])
        bc.save(out / "games", name=g["game"])
        print(g["game"], g["split"], json.dumps(bc.meta["counts"]["buttons"]), flush=True)
        del bc
        _LIBC.malloc_trim(0)     # return each game's freed heap: without it RSS grows ~0.35 GB a game


def finalize(out: Path) -> dict:
    p = json.loads((out / "plan.json").read_text())
    rows, missing = [], []
    for g in p["games"]:
        path = out / "games" / f"{g['game']}.json"
        if not path.exists():
            missing.append(g["game"])
            continue
        meta = json.loads(path.read_text())
        rows.append(dict(game=g["game"], split=g["split"], opponent=g["opponent"],
                         opponent_class=g["opponent_class"], opponent_kit=g.get("opponent_kit"),
                         decoder_commit=meta["provenance"]["decoder_commit"],
                         cooldown_fields_read=meta["provenance"].get("cooldown_fields_read", False),
                         converter_commit=meta["provenance"]["converter_commit"],
                         counts=meta["counts"], label_check=meta["label_check"], decisions=meta["decisions"]))

    def totals(sel):
        c = Counter()
        for r in sel:
            c.update({k: v for k, v in r["counts"].items() if isinstance(v, int)})
            c.update({"button_" + b: n for b, n in r["counts"]["buttons"].items()})
            c.update({"lc_" + k: v for k, v in r["label_check"].items() if isinstance(v, int)})
        decisions = sum(r["decisions"] for r in sel)
        valid = c["label_valid"]
        return dict(games=len(sel), decisions=decisions, in_scope_share=c["in_scope"] / decisions,
                    label_valid=valid,
                    buttons={b: c["button_" + b] / valid for b in BUTTONS_WITH_STOP},
                    excluded=dict(dead=c["dead"], out_of_lane=c["out_of_lane"],
                                  third_champion=c["third_champion"],
                                  not_representable=c["not_representable"]),
                    attacks_resolved=f"{c['lc_attacks_resolved']}/{c['lc_attacks']}",
                    held=c["held"], held_on_unit=c["held_on_unit"],
                    stops=c["lc_stops"], stops_resolved=c["lc_stops_resolved"],
                    r_resolved=f"{c['lc_r_resolved']}/{c['lc_r_casts']}", click_valid=c["click_valid"],
                    cooldown_conflicts=c["cooldown_conflicts"])
    by_opponent = Counter(r["opponent"] for r in rows)
    commits = sorted({r["converter_commit"] for r in rows})
    manifest = dict(
        plan=dict((k, v) for k, v in p.items() if k not in ("games", "skipped")),
        converter_commit=commits, observation_lag=p["lag"], hold=p.get("hold", True), vocabulary=p["vocabulary"],
        stats=dict(names=list(STAT_NAMES), scale=list(STAT_SCALE)), buttons=list(BUTTONS_WITH_STOP),
        cooldown_fields_read=bool(rows) and all(r["cooldown_fields_read"] for r in rows),
        profiles=dict(obs=p.get("profile", OB.PROFILE), actions="modern-world-v2"), skill_tier=p.get("tier"),
        totals=dict(all=totals(rows), train=totals([r for r in rows if r["split"] == "train"]),
                    val=totals([r for r in rows if r["split"] == "val"])),
        opponents={o: _opponent_row([r for r in rows if r["opponent"] == o])
                   for o, _ in by_opponent.most_common()},
        split=dict(train=[r["game"] for r in rows if r["split"] == "train"],
                   val=[r["game"] for r in rows if r["split"] == "val"]),
        skipped=p["skipped"], missing=missing, games=rows)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _opponent_row(rows):
    return dict(games=len(rows), val=sum(r["split"] == "val" for r in rows),
                opponent_class=rows[0]["opponent_class"], kit=rows[0]["opponent_kit"],
                label_valid=sum(r["counts"]["label_valid"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("out", type=Path)
    a.add_argument("--lag", type=parse_lag, default=1)
    a.add_argument("--no-hold", action="store_true", help="label click events only, not held orders")
    a.add_argument("--champion", default=CHAMPION, help="the cloned champion (a registered modern kit)")
    a.add_argument("--champions", default="", help="comma list, or 'all' top laners: several clones in one dataset (v2 profile)")
    a.add_argument("--datasets", nargs="+", type=Path, default=list(DATASETS), help="exported batch directories")
    a.add_argument("--profile", default=OB.PROFILE, choices=OB.PROFILES, help="observation profile")
    a.add_argument("--tier", type=int, default=CT.ARCHIVE_TIER, help="skill tier recorded in v2 (8 = Master)")
    a = sub.add_parser("convert")
    a.add_argument("out", type=Path)
    a.add_argument("task", type=int)
    a.add_argument("tasks", type=int)
    a = sub.add_parser("finalize")
    a.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        p = plan(args.out, args.lag, hold=not args.no_hold, champion=args.champion, datasets=args.datasets,
                 profile=args.profile, tier=args.tier,
                 champions=[c for c in args.champions.split(",") if c] or None)
        print(len(p["games"]), "games,", len(p["skipped"]), "skipped,",
              Counter(g["split"] for g in p["games"]))
    elif args.command == "convert":
        convert(args.out, args.task, args.tasks)
    else:
        print(json.dumps(finalize(args.out)["totals"], indent=1))


if __name__ == "__main__":
    main()
