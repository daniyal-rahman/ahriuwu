"""Convert packed .npz latents to raw .pt tensors for zero-overhead loading.

Also builds an index file (index.pt) containing frame_indices for all videos,
so the dataset can index sequences without loading full latent tensors.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed


def convert_one(npz_path: Path, out_dir: Path) -> tuple[str, torch.Tensor, bool]:
    video_id = npz_path.stem
    pt_path = out_dir / f"{video_id}.pt"
    data = np.load(npz_path)
    frame_indices = torch.from_numpy(data["frame_indices"].copy())
    if not pt_path.exists():
        torch.save({
            "latents": torch.from_numpy(data["latents"]),
            "frame_indices": frame_indices,
        }, pt_path)
        return video_id, frame_indices, True
    return video_id, frame_indices, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("*.npz"))
    print(f"Converting {len(npz_files)} files: {in_dir} -> {out_dir}")

    index = {}
    converted = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, f, out_dir): f for f in npz_files}
        for i, future in enumerate(as_completed(futures)):
            video_id, frame_indices, was_new = future.result()
            index[video_id] = frame_indices
            if was_new:
                converted += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(npz_files)} done ({converted} converted)")

    # Save index file
    torch.save(index, out_dir / "index.pt")
    print(f"Done. {converted} new, {len(npz_files) - converted} skipped.")
    print(f"Index saved: {len(index)} videos in {out_dir / 'index.pt'}")


if __name__ == "__main__":
    main()
