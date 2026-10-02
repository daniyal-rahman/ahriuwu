"""TOOL: plot E87 progress and every recorded 120s AFK training-arm evaluation.

Reads study-selected runs, never evaluates weights. Outputs PNGs and their frozen
point CSV; the CSV also records source paths and the snapshot time.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "E46": "Initial AFK farming", "E50": "LR 3e-5; personal reward",
    "E51": "LR 1e-4; personal reward", "E52": "Higher entropy bonus",
    "E53": "Shared actor / critic trunk", "E56": "One PPO epoch",
    "E57": "Four PPO epochs", "E60": "Own combat-state inputs",
    "E61": "Unchanged-input continuation", "E62": "Own state + death cost",
    "E63": "Death cost only", "E65": "Farming-focused reward",
    "E66": "Click proposals + death cost", "E67": "Staggered episode phases",
    "E75c": "Unchanged-input control", "E78": "CS-only reward",
    "E80": "CS-only continuation; cancelled", "E81": "Full-wave teaching",
    "E82": "CS-only PPO after teaching", "E87": "Long CS-only PPO after E82",
}


def read_rows(path):
    # Snapshot complete records while the active worker may be appending a row.
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(s) for s in lines if s.endswith("\n") and s.strip()]


def frozen_rows(path):
    return [r for r in read_rows(path) if r.get("frozen") and
            r.get("opponent") == "afk" and r.get("duration_s") == 120]


def mean(row, key):
    return float(np.mean([e[key] for e in row["episodes"] if e["team"] == 0]))


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/nfs/checkpoints/lanerl-jax"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "docs/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    arms = []
    for study_path in args.root.glob("E*/study.json"):
        study = json.loads(study_path.read_text())
        if "path" not in study:
            continue
        run = Path(study["path"])
        eid = study_path.parent.name.split("_")[0]
        if eid == "E83":
            initial = run / "initial/evaluations.jsonl"
            for arm in ("teaching", "ppo"):
                path = run / f"{arm}_final/evaluations.jsonl"
                rows = frozen_rows(initial) + frozen_rows(path)
                assert len(rows) == 2
                arms.append((f"E83 {arm}", f"Short-task {arm}: full-wave transfer", path, rows))
        else:
            path = run / "evaluations.jsonl"
            if path.exists() and (rows := frozen_rows(path)):
                arms.append((eid, LABELS.get(eid, study_path.parent.name), path, rows))
    arms.sort(key=lambda a: (int(''.join(c for c in a[0].split()[0] if c.isdigit())), a[0]))
    assert all(r["games"] == 64 and sum(e["team"] == 0 for e in r["episodes"]) == 64
               for _, _, _, rows in arms for r in rows)
    current = next(a for a in arms if a[0] == "E87")
    run = current[2].parent
    manifest = json.loads((run / "manifest.json").read_text())
    metrics = read_rows(run / "metrics.jsonl")
    checkpoints = sorted(manifest["checkpoints"], key=lambda c: c["update"])
    cu = np.array([c["update"] for c in checkpoints])
    ct = np.array([utc(c["utc"]) for c in checkpoints])
    start = ct[0]
    # Recorded checkpoint timestamps include evaluation / I/O time. Extend only
    # the short unsaved tail using logged update runtimes, not a guessed speed.
    if metrics[-1]["update"] > cu[-1]:
        ct = np.append(ct, ct[-1] + sum(m["update_s"] for m in metrics if m["update"] > cu[-1]))
        cu = np.append(cu, metrics[-1]["update"])
    hours = lambda u: (np.interp(u, cu, ct) - start) / 3600
    rows = current[3]
    x = hours([r["update"] for r in rows])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained")
    for ax, key, title, color in zip(axs[0], ["cs", "deaths"],
                                    ["Frozen CS / game", "Frozen deaths / game"],
                                    ["#16708d", "#b84b31"]):
        y = [mean(r, key) for r in rows]
        ax.plot(x, y, "o-", color=color, linewidth=2)
        ax.axhline(y[0], color="#68747c", linestyle="--", linewidth=1, label="Starting E82 policy")
        for xx, yy in zip(x, y):
            ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(0, 9 if xx else -16),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=title, ylim=(0, 18 if key == "cs" else 1))
        if key == "cs":
            ax.axhline(15.125, color="#30945d", linestyle=":", label="Declared improvement gate: 15.125")
            ax.legend(loc="lower right", fontsize=8)
    tx = hours([m["update"] for m in metrics])
    for ax, key, title in zip(axs[1], ["post_kl", "entropy"],
                              ["TRAIN sampled policy KL after update", "TRAIN action entropy"]):
        y = np.array([m[key] for m in metrics])
        ax.plot(tx, y, color="#9da4b3", linewidth=.4, alpha=.25)
        n = min(128, len(y))
        ax.plot(tx[n-1:], np.convolve(y, np.ones(n)/n, mode="valid"), color="#6c56a0", linewidth=2)
        ax.set(title=title, ylim=(0, None))
    for ax in axs.flat:
        ax.set(xlabel="Hours since initial checkpoint (startup excluded)", xlim=(-.04, tx[-1] + .12))
        ax.grid(alpha=.18)
    fig.suptitle("E87: longer CS-only PPO from the 14.125-CS E82 checkpoint\n"
                 f"Frozen: same 64 games, 120s each | last evaluated update {rows[-1]['update']:,} | "
                 f"training through {metrics[-1]['update']:,}", fontsize=13)
    fig.savefig(args.out / "E87_training_curve.png", dpi=170)
    plt.close(fig)

    ncols = 4
    nrows = (len(arms) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 2.4), layout="constrained")
    point_count = 0
    csv_rows = []
    for ax, (eid, label, path, rs) in zip(axes.flat, arms):
        stage = eid.startswith(("E81", "E83"))
        xx = list(range(len(rs))) if stage else [r["update"] for r in rs]
        yy = [mean(r, "cs") for r in rs]
        color = "#16708d" if eid in ("E46", "E67", "E78", "E81", "E82", "E87") else "#69758a"
        ax.plot(xx, yy, "o-", color=color, markersize=4, linewidth=1.7)
        ax.axhline(yy[0], color="#bac1c7", linestyle="--", linewidth=.8)
        ax.annotate(f"{yy[-1]:.2f}", (xx[-1], yy[-1]), xytext=(-4, 8),
                    textcoords="offset points", ha="right", color=color, fontweight="bold")
        ax.set(title=f"{eid}: {label}", ylim=(0, 18), yticks=[0, 6, 12, 18], ylabel="Frozen CS",
               xlabel="Teaching stage" if eid == "E81" else "Stage" if stage else "PPO update (local to run)")
        ax.title.set_fontsize(9)
        if stage:
            ax.set_xticks(xx, ["Start", "BC", "DAgger 1", "DAgger 2"] if eid == "E81" else ["Start", "Final"])
        ax.grid(alpha=.18)
        point_count += len(rs)
        for i, r in enumerate(rs):
            source = path.parent.parent / "initial/evaluations.jsonl" if eid.startswith("E83") and i == 0 else path
            csv_rows.append(dict(snapshot_utc=snapshot, arm=eid, stage=i, update=r["update"],
                                 games=r["games"], duration_s=r["duration_s"], cs=mean(r, "cs"),
                                 deaths=mean(r, "deaths"),
                                 tower_damage=mean(r, "tower_damage") if all("tower_damage" in e for e in r["episodes"] if e["team"] == 0) else None,
                                 source=str(source)))
    for ax in list(axes.flat)[len(arms):]:
        ax.axis("off")
    axes.flat[-1].text(0, .95, "Each panel is a separate training arm.\n"
                      "Same metric; rewards, inputs and budgets vary.\n"
                      "Dashed line = that arm's starting score.\n\n"
                      "E83 shows full-wave transfer, not local-task CS.\n"
                      "E80 was cancelled; E87 is still running.\n"
                      "Saved but unevaluated weights have no score.\n"
                      "One training seed per arm; no across-seed CI.\n\n"
                      f"Snapshot: {snapshot}", va="top", fontsize=9, linespacing=1.6)
    fig.suptitle(f"All recorded 120-second AFK training-arm checkpoint evaluations\n"
                 f"{len(arms)} arms, {point_count} points | 64 frozen games per point | "
                 "panels have different update scales; these are not one continuous run", fontsize=14)
    fig.savefig(args.out / "afk_checkpoint_history.png", dpi=170)
    plt.close(fig)
    with (args.out / "afk_checkpoint_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(dict(snapshot_utc=snapshot, arms=len(arms), points=point_count,
                          latest_training_update=metrics[-1]["update"],
                          latest_frozen_update=rows[-1]["update"], latest_frozen_cs=mean(rows[-1], "cs"),
                          outputs=[str(args.out / n) for n in ("E87_training_curve.png", "afk_checkpoint_history.png", "afk_checkpoint_history.csv")])))


if __name__ == "__main__":
    main()
