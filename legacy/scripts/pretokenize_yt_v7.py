#!/usr/bin/env python3
"""Pre-tokenize the YT corpus (tars of JPG frames) → v7 dim-32 latents for
UNLABELED world-model pretraining (DreamerV4 Sec 3.2: unlabeled video uses the
learned no-action embedding downstream — no action labels needed here).

Streams frames straight from the NFS tars so the 602 GB never all lands on disk,
and keeps the 5080 COMPUTE-bound rather than IO-stalled via a 3-stage pipeline:

    [background thread]  extract tar  ->  local scratch
    [dataloader workers] cv2 decode + resize  (this is what took the GPU from
                                               ~30-60% to ~full — see _FrameDS)
    [main / GPU]         tokenizer.encode -> fold to (N,32,16,16) -> save .pt

A bounded queue (``--prefetch-tars``) holds a few extracted tars ahead: it both
CAPS disk (only ~N tars extracted at once, deleted right after encode) and gives
the GPU a buffer so it never waits on extraction. The per-tar log prints
``qsize`` — if it stays > 0 the GPU is the bottleneck (good, compute-bound); if
it's usually 0 the extractor/decoder can't keep up (IO-bound → raise workers).

Output: ``<out>/<video_id>.pt = {latents:(N,32,16,16) f16, frame_indices:(N,) i32}``
— the exact packed format PackedLatentSequenceDataset reads (latents only).

Run (desktop 5080):
  PYTHONPATH=src python scripts/pretokenize_yt_v7.py --resume \
    --tars-dir /mnt/nfs/datasets/yt_pretrain_garen \
    --out /scratch/ahriuwu/dynamics_yt_latents_v7_dim32
"""
import argparse, glob, queue, shutil, tarfile, threading, time
from pathlib import Path
import torch

# reuse the frozen-tokenizer loader + the PARALLEL frame encoder (DataLoader with
# worker-process decode) from the replay pretok — same encode path, same format.
from pretokenize_replay_v7 import load_v7, encode_match


def jpgs_in(d):
    return sorted(Path(d).glob("*.jpg"), key=lambda p: int(p.stem))


def _extractor(tars, workdir, q, stop, resume, out):
    """Background producer: extract each tar to local scratch, hand the frame
    list to the queue. q.put blocks when the buffer is full → bounds disk."""
    for tp in tars:
        if stop.is_set():
            break
        vid = Path(tp).stem
        if resume and (Path(out) / f"{vid}.pt").exists():
            continue
        dest = Path(workdir) / vid
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tp) as t:
                t.extractall(dest)
        except Exception as e:
            print(f"extract FAIL {vid}: {e}", flush=True)
            shutil.rmtree(dest, ignore_errors=True)
            continue
        q.put((vid, jpgs_in(dest), dest))
    q.put(None)  # sentinel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars-dir", default="/mnt/nfs/datasets/yt_pretrain_garen")
    ap.add_argument("--checkpoint",
                    default="/mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt")
    ap.add_argument("--out", default="/scratch/ahriuwu/dynamics_yt_latents_v7_dim32")
    ap.add_argument("--workdir", default="/scratch/ahriuwu/_yt_extract")
    ap.add_argument("--batch-size", type=int, default=32)   # 64 OOMs v7 'large' on 16GB
    ap.add_argument("--num-workers", type=int, default=8)   # parallel decode = the throughput lever
    ap.add_argument("--prefetch-tars", type=int, default=2)  # extracted-ahead buffer (caps disk)
    ap.add_argument("--max-tars", type=int, default=0, help="0=all; >0 for a quick benchmark")
    ap.add_argument("--compile", action="store_true", help="torch.compile the encode (fuses kernels)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    dev, amp = "cuda", torch.bfloat16  # 5080 is bf16-native

    model, cfg, step = load_v7(args.checkpoint, dev)
    size = int(cfg.get("img_size", 352))
    if args.compile:
        # dynamic=True: each tar's final partial batch is a new shape — dynamic
        # avoids a recompile per unique last-batch-size across 453 tars.
        model.encode = torch.compile(model.encode, dynamic=True)
        print("torch.compile(encode, dynamic=True) enabled", flush=True)
    print(f"v7 tok step {step}: num_latents={cfg['num_latents']} latent_dim={cfg['latent_dim']} "
          f"-> dynamics dim {cfg['num_latents']*cfg['latent_dim']//256}", flush=True)

    tars = sorted(glob.glob(f"{args.tars_dir}/*.tar"))
    if args.max_tars:
        tars = tars[:args.max_tars]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    print(f"{len(tars)} tars | out={args.out} | batch={args.batch_size} workers={args.num_workers} "
          f"prefetch={args.prefetch_tars}", flush=True)

    q = queue.Queue(maxsize=args.prefetch_tars)
    stop = threading.Event()
    th = threading.Thread(target=_extractor, args=(tars, args.workdir, q, stop, args.resume, args.out),
                          daemon=True)
    th.start()

    t0 = time.time()
    total_frames = done = 0
    try:
        while True:
            item = q.get()
            if item is None:
                break
            vid, pngs, dest = item
            if not pngs:
                shutil.rmtree(dest, ignore_errors=True)
                continue
            t_enc = time.time()
            lat = encode_match(model, pngs, dev, args.batch_size, amp, size, args.num_workers)
            idxs = torch.tensor([int(p.stem) for p in pngs], dtype=torch.int32)
            outp = Path(args.out) / f"{vid}.pt"
            tmp = outp.with_suffix(".pt.tmp")
            torch.save({"latents": lat, "frame_indices": idxs}, tmp)
            tmp.replace(outp)                        # atomic
            shutil.rmtree(dest, ignore_errors=True)  # free disk immediately
            n = lat.shape[0]
            total_frames += n
            done += 1
            enc = time.time() - t_enc
            avg = total_frames / max(time.time() - t0, 1e-6)
            print(f"[{done}] {vid}: {tuple(lat.shape)} | {n} frames {enc:.1f}s "
                  f"({n/max(enc,1e-6):.0f} f/s) | qsize={q.qsize()} (>0=GPU-bound) | "
                  f"cumul {total_frames:,}f @ {avg:.0f} f/s", flush=True)
    finally:
        stop.set()
    dt = time.time() - t0
    print(f"DONE: {done} videos, {total_frames:,} frames in {dt/3600:.2f}h "
          f"({total_frames/max(dt,1e-6):.0f} f/s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
