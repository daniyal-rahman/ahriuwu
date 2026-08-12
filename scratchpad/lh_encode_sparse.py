#!/usr/bin/env python3
"""Encode ONLY the frames needed around events, with the frozen v7 tokenizer.

Used to get v7 latents for matches that were never tokenized (the truly-unseen
held-out candidates). Writes the same {latents, frame_indices} format as
scripts/pretokenize_replay_v7.py so downstream code is identical, but with a
SPARSE (still strictly ascending) frame_indices.

Also has --verify-against: re-encode frames of a match that already has stored
latents and report cosine/MSE agreement, which proves this checkpoint is the one
that produced /srv/nfs/datasets/replay_latents_v7_bc.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "src")
from ahriuwu.models.transformer_tokenizer import TransformerTokenizer  # noqa: E402

FRAMES_ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"


def load_v7(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ck["model_config"].items() if k != "size_preset"}
    model = TransformerTokenizer(**cfg)
    sd = ck["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "rope" not in k.lower()]
    if bad or unexpected:
        raise RuntimeError(f"tokenizer load mismatch missing={bad[:8]} unexpected={unexpected[:8]}")
    return model.to(device).eval(), cfg, ck.get("global_step")


def load_frame(path, size=352):
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"cv2 could not read {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (size, size):
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).float().div_(255.0).permute(2, 0, 1)


class _DS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_frame(self.paths[i])


@torch.no_grad()
def encode_frames(model, paths, device, bs, workers):
    dl = DataLoader(_DS(paths), batch_size=bs, num_workers=workers, pin_memory=True)
    out = []
    for b in dl:
        b = b.to(device, non_blocking=True)
        lat = model.encode(b)["latent"]
        B = lat.shape[0]
        lat = lat.view(B, 16, 16, -1).permute(0, 3, 1, 2)
        out.append(lat.float().cpu().to(torch.float16))
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rollout_stage/transformer_tokenizer_latest.pt")
    ap.add_argument("--frames-root", default=FRAMES_ROOT)
    ap.add_argument("--out", default="/srv/nfs/projects/ahriuwu/scratchpad/lh_latents_unseen")
    ap.add_argument("--matches", nargs="+", default=[])
    ap.add_argument("--events", default="scratchpad/lh_events.npz")
    ap.add_argument("--pre", type=int, default=16, help="frames of history to keep per event")
    ap.add_argument("--post", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--verify-against", default=None,
                    help="match id with stored latents -> compare re-encode")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, cfg, gs = load_v7(args.checkpoint, args.device)
    print(f"v7 gs={gs} num_latents={cfg['num_latents']} latent_dim={cfg['latent_dim']}", flush=True)

    if args.verify_against:
        mid = args.verify_against
        ref = torch.load(f"/srv/nfs/datasets/replay_latents_v7_bc/{mid}.pt", weights_only=True)
        fi = ref["frame_indices"].numpy()
        pick = np.linspace(100, len(fi) - 100, 16).astype(int)
        paths = [f"{args.frames_root}/{mid}/{int(fi[i]):06d}.png" for i in pick]
        paths = [p if os.path.exists(p) else f"{args.frames_root}/{mid}/frames/{Path(p).name}"
                 for p in paths]
        got = encode_frames(model, paths, args.device, args.batch_size, args.num_workers).float()
        want = ref["latents"][pick].float()
        num = (got * want).sum((1, 2, 3))
        den = got.flatten(1).norm(dim=1) * want.flatten(1).norm(dim=1)
        cos = (num / den).numpy()
        rel = ((got - want).pow(2).mean((1, 2, 3)) / want.pow(2).mean((1, 2, 3))).numpy()
        print(f"VERIFY {mid}: cosine mean={cos.mean():.5f} min={cos.min():.5f}  "
              f"rel_mse mean={rel.mean():.5f} max={rel.max():.5f}")
        print("  -> SAME tokenizer" if cos.min() > 0.99 else "  -> DIFFERENT tokenizer, ABORT")
        return

    ev = np.load(args.events, allow_pickle=True)
    mid_a, frame_a = ev["mid"], ev["frame"]
    os.makedirs(args.out, exist_ok=True)
    for mid in args.matches:
        outp = os.path.join(args.out, f"{mid}.pt")
        if os.path.exists(outp):
            print(f"skip {mid} (exists)", flush=True)
            continue
        fr = frame_a[mid_a == mid]
        need = set()
        for f in fr:
            need.update(range(int(f) - args.pre, int(f) + args.post + 1))
        fdir = os.path.join(args.frames_root, mid, "frames")
        have = {int(p.stem) for p in Path(fdir).glob("*.png")}
        need = sorted(f for f in need if f in have)
        paths = [os.path.join(fdir, f"{f:06d}.png") for f in need]
        print(f"{mid}: {len(fr)} events -> {len(paths)} frames", flush=True)
        import time
        t0 = time.time()
        lat = encode_frames(model, paths, args.device, args.batch_size, args.num_workers)
        dt = time.time() - t0
        torch.save({"latents": lat,
                    "frame_indices": torch.tensor(need, dtype=torch.int32)}, outp + ".tmp")
        os.replace(outp + ".tmp", outp)
        print(f"  wrote {outp} {tuple(lat.shape)} in {dt:.0f}s ({len(paths)/dt:.1f} fps)", flush=True)


if __name__ == "__main__":
    main()
