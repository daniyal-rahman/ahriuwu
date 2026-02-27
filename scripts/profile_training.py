#!/usr/bin/env python3
"""Profile tokenizer training to find bottlenecks."""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ahriuwu.data.dataset import SingleFrameDataset
from ahriuwu.models import create_transformer_tokenizer, MAELoss


def main():
    device = "cuda"
    batch_size = 16
    num_workers = 6

    print("Loading dataset...")
    dataset = SingleFrameDataset("data/processed/frames")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    print("Creating model...")
    model = create_transformer_tokenizer("small", use_rope=True).to(device)
    criterion = MAELoss(lpips_weight=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda")

    dataloader_iter = iter(dataloader)

    # Warmup
    print("Warmup (5 steps)...")
    for _ in range(5):
        batch = next(dataloader_iter)
        frames = batch["frame"].to(device, non_blocking=True)
        with autocast("cuda"):
            out = model(frames, mask_ratio=0.5)
            losses = criterion(out["reconstruction"], frames)
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    print("Profiling (20 steps)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(20):
            with record_function("data_loading"):
                batch = next(dataloader_iter)
                frames = batch["frame"].to(device, non_blocking=True)

            with record_function("forward"):
                with autocast("cuda"):
                    out = model(frames, mask_ratio=0.5)

            with record_function("loss_computation"):
                with autocast("cuda"):
                    losses = criterion(out["reconstruction"], frames)

            with record_function("backward"):
                scaler.scale(losses["loss"]).backward()

            with record_function("optimizer_step"):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    print("\n" + "=" * 80)
    print("PROFILER RESULTS - Sorted by CUDA time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print("\n" + "=" * 80)
    print("PROFILER RESULTS - Sorted by CPU time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    print("\n" + "=" * 80)
    print("PROFILER RESULTS - By record_function regions")
    print("=" * 80)
    for key in ["data_loading", "forward", "loss_computation", "backward", "optimizer_step"]:
        events = [e for e in prof.key_averages() if e.key == key]
        if events:
            e = events[0]
            print(f"{key:20s}: CPU {e.cpu_time_total/1000:8.1f}ms  CUDA {e.cuda_time_total/1000:8.1f}ms")

    # Save trace for chrome://tracing
    prof.export_chrome_trace("profile_trace.json")
    print("\nTrace saved to profile_trace.json (open in chrome://tracing)")


if __name__ == "__main__":
    main()
