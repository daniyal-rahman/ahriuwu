"""Score an existing Phase-2 checkpoint through the DIRECTION acceptance metric.

Runs the trainer's own evaluate() / run_step() / DirectionReference path — not a
reimplementation — over the checkpoint's OWN recorded val games, so what it
prints is exactly what a training run prints at --val-interval. Use it to place a
finished checkpoint against the persistence bar without launching a run.

  PYTHONPATH=src python scripts/score_checkpoint_direction.py \
      --checkpoint data/phase2_bc_clicks/agent_finetune_latest.pt

Only the val packs are symlinked into --stage: the dataset index globs *.pt and
torch.loads each one just for frame_indices, so pointing it at the full corpus
would re-read ~26 GB over NFS for six games' worth of frames.
"""
import argparse
import os
import sys
import types
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import train_agent_finetune as T  # noqa: E402
from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM  # noqa: E402
from ahriuwu.data.replay_dataset import STATE_TARGETS  # noqa: E402
from ahriuwu.models import (  # noqa: E402
    DiffusionSchedule, PolicyHead, RewardHead, RunningRMS, StateHead,
)


def nfs_root() -> Path:
    """The same export is /srv/nfs on the login node and /mnt/nfs on the desktop."""
    for c in ("/srv/nfs", "/mnt/nfs"):
        if Path(c, "datasets").is_dir():
            return Path(c)
    raise SystemExit("no NFS export found at /srv/nfs or /mnt/nfs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents-dir", default=None)
    ap.add_argument("--labels-root", default=None)
    ap.add_argument("--stage", default=None, help="scratch dir of symlinked val packs")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--val-batches", type=int, default=40)
    ap.add_argument("--direction-perms", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    root = nfs_root()
    latents = Path(a.latents_dir or root / "datasets/replay_latents_v7_bc")
    labels = Path(a.labels_root or root / "datasets/lol_replays_16_9_772")
    stage = Path(a.stage or (os.environ.get("SCRATCH") or "/tmp") + "/dir_eval_stage")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    saved = ck["args"]
    saved = saved if isinstance(saved, dict) else vars(saved)
    vids = list(saved["val_matches_resolved"])
    print(f"checkpoint  : {a.checkpoint}")
    print(f"  epoch {ck.get('epoch')}, global_step {ck.get('global_step'):,}")
    print(f"  val games  : {vids}")

    # Stage ONLY the val packs: _index globs *.pt and torch.loads each one just
    # for frame_indices, so pointing it at all 125 would read ~26 GB over NFS.
    stage.mkdir(parents=True, exist_ok=True)
    for p in stage.glob("*.pt"):
        p.unlink()
    for v in vids:
        (stage / f"{v}.pt").symlink_to(latents / f"{v}.pt")

    args = types.SimpleNamespace(**saved)
    args.latents_dir, args.labels_root = str(stage), str(labels)
    args.dataset_cache = None          # 6 games; a rebuild is seconds
    args.manifest = None
    args.batch_size = a.batch_size
    args.val_batches = a.val_batches
    args.direction_metric = True
    args.direction_perms = a.direction_perms
    args.cursor_weight = getattr(args, "cursor_weight", 0.0)
    args.action_dropout = 0.0
    args.movement_mode = getattr(args, "movement_mode", "axis")
    args.movement_action_mode = getattr(args, "movement_action_mode", "held")
    args.video_loss_weight = 0.0
    device = a.device

    dataset = T.build_dataset(args)
    val_vids = set(vids) & {s["video_id"] for s in dataset.sequences}
    order = T.build_val_order(dataset, val_vids, args.batch_size, args.val_batches)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, order), batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=(device != "cpu"), drop_last=True)
    print(f"  val loader : {len(val_loader)} batches x {args.batch_size} x "
          f"seq {args.seq_len}")

    dynamics = T.build_dynamics(args, use_actions=True, device=device)
    T.load_state_dict_guarded(dynamics, ck["dynamics_state_dict"],
                              what=f"{a.checkpoint} dynamics")
    T.freeze_backbone_train_agent(dynamics)
    dynamics.eval()
    md = dynamics.model_dim
    reward_head = RewardHead(input_dim=md, hidden_dim=args.hidden_dim,
                             num_buckets=args.num_buckets,
                             mtp_length=args.mtp_length).to(device)
    reward_head.load_state_dict(ck["reward_head_state_dict"])
    policy_head = PolicyHead(
        input_dim=md, num_abilities=len(ABILITY_KEYS), hidden_dim=args.hidden_dim,
        mtp_length=args.mtp_length, movement_dim=MOVEMENT_DIM,
        movement_bins=args.movement_bins, movement_gate=args.movement_gate,
        movement_mode=args.movement_mode,
        cursor_head=args.cursor_weight > 0).to(device)
    policy_head.load_state_dict(ck["policy_head_state_dict"])
    state_head = None
    if "state_head_state_dict" in ck and args.aux_state_weight > 0:
        state_head = StateHead(input_dim=md, hidden_dim=args.hidden_dim,
                               num_targets=len(STATE_TARGETS)).to(device)
        state_head.load_state_dict(ck["state_head_state_dict"])

    schedule = DiffusionSchedule(device=device)
    rms = {"bc": RunningRMS(), "reward": RunningRMS(), "aux": RunningRMS(),
           "video": RunningRMS(), "cursor": RunningRMS()}
    for k, st in (ck.get("rms_state") or {}).items():
        if k in rms:
            rms[k].load_state_dict(st)

    dir_ref = T.DirectionReference(args.labels_root, val_vids)
    print(f"  dir ref    : {len(dir_ref)} games, y_scale="
          f"{sorted({round(g['y_scale'], 4) for g in dir_ref.games.values()})}, "
          f"first_click={sorted(g['first_click'] for g in dir_ref.games.values())}")

    # bf16 needs sm_80+; the login box is a GTX 1060 (sm_61), so fall back to
    # fp16 there rather than silently emulating.
    amp = torch.float32
    if device.startswith("cuda"):
        amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
            else torch.float16
    print(f"  amp        : {amp}")
    v = T.evaluate(val_loader, dynamics, reward_head, policy_head, schedule, args,
                   device, amp, rms, state_head=state_head,
                   max_batches=len(val_loader), dir_ref=dir_ref)
    print("\n" + "=" * 72)
    print(f"move_event_ce = {v['move_event_ce']:.4f}  acc={100*v['move_event_acc']:.2f}%  "
          f"n={int(v['move_event_n'])}   [bar {T.MOVE_CE_BLIND_TABLE:.4f}, "
          f"deployed ref {T.MOVE_CE_DEPLOYED:.4f}]")
    print(T._fmt_direction(v, args))
    print("=" * 72)
    for k in sorted(v):
        if k.startswith("dir_"):
            print(f"  {k:20s} {v[k]}")


if __name__ == "__main__":
    main()
