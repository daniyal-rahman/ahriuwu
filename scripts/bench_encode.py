#!/usr/bin/env python3
"""A/B the v7 encode throughput: batch size x torch.compile, on a fixed frame
set. The pretok showed 100% util but only 223/360 W → not compute-saturated, so
this finds the config that actually maxes the 5080. Prints f/s + peak VRAM +
avg power per config; pick the fastest that fits ~15 GB."""
import argparse, subprocess, tarfile, threading, time
from pathlib import Path
import torch
from pretokenize_replay_v7 import load_v7, encode_match


def sample_power(stop, out):
    while not stop.is_set():
        try:
            w = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=2).decode().split("\n")[0].strip()
            out.append(float(w))
        except Exception:
            pass
        time.sleep(0.25)


def run(ckpt, pngs, dev, size, bs, nw, compile_on):
    model, cfg, _ = load_v7(ckpt, dev)           # fresh model per config (compile is sticky)
    if compile_on:
        model.encode = torch.compile(model.encode, dynamic=True)
    # warmup (triggers compile) on a couple batches incl. a partial one
    _ = encode_match(model, pngs[:bs * 2 + 1], dev, bs, torch.bfloat16, size, nw)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stop = threading.Event(); pw = []
    th = threading.Thread(target=sample_power, args=(stop, pw), daemon=True); th.start()
    t0 = time.time()
    lat = encode_match(model, pngs, dev, bs, torch.bfloat16, size, nw)
    torch.cuda.synchronize()
    dt = time.time() - t0
    stop.set()
    peak = torch.cuda.max_memory_allocated() / 1e9
    fps = len(pngs) / dt
    avgw = sum(pw) / max(len(pw), 1)
    maxw = max(pw) if pw else 0
    del model
    torch.cuda.empty_cache()
    return fps, peak, avgw, maxw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt")
    ap.add_argument("--tar", default=None, help="a YT tar to pull frames from")
    ap.add_argument("--tars-dir", default="/mnt/nfs/datasets/yt_pretrain_garen")
    ap.add_argument("--n-frames", type=int, default=4000)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--workdir", default="/mnt/storage/data/ahriuwu/_bench_extract")
    args = ap.parse_args()
    dev = "cuda"

    _, cfg, _ = load_v7(args.checkpoint, dev)
    size = int(cfg.get("img_size", 352))
    tar = args.tar or sorted(Path(args.tars_dir).glob("*.tar"))[0]
    dest = Path(args.workdir); dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(dest)
    pngs = sorted(dest.glob("*.jpg"), key=lambda p: int(p.stem))[:args.n_frames]
    print(f"bench on {len(pngs)} frames from {Path(tar).name} @ {size}px, workers={args.num_workers}\n")

    configs = [(32, False), (48, False), (64, False), (48, True), (64, True), (96, True)]
    print(f"{'batch':>5} {'compile':>7} | {'f/s':>6} | {'peakVRAM':>9} | {'avgW':>6} {'maxW':>6}")
    best = None
    for bs, comp in configs:
        try:
            fps, peak, avgw, maxw = run(args.checkpoint, pngs, dev, size, bs, args.num_workers, comp)
            flag = "" if peak < 15.0 else "  <-- near VRAM cap"
            print(f"{bs:>5} {str(comp):>7} | {fps:6.0f} | {peak:7.1f}GB | {avgw:6.0f} {maxw:6.0f}{flag}", flush=True)
            if peak < 15.0 and (best is None or fps > best[0]):
                best = (fps, bs, comp)
        except RuntimeError as e:
            print(f"{bs:>5} {str(comp):>7} | OOM/ERR ({str(e)[:40]})", flush=True)
            torch.cuda.empty_cache()
    import shutil; shutil.rmtree(dest, ignore_errors=True)
    if best:
        print(f"\nBEST (fits <15GB): batch={best[1]} compile={best[2]} @ {best[0]:.0f} f/s "
              f"(baseline batch=32 no-compile) -> {best[0]/85:.2f}x")


if __name__ == "__main__":
    main()
