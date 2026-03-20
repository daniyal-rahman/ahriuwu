"""Quick VRAM test for different batch size configs."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/mnt/storage/ahriuwu/repo")

from ahriuwu.models.dynamics import create_dynamics


def test_config(batch_size, seq_len, gradient_checkpointing, shortcut_forcing=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = create_dynamics_model(
        model_size="small",
        latent_dim=48,
        num_kv_heads=4,
        num_register_tokens=8,
        soft_cap=50.0,
    ).cuda().bfloat16()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Simulate worst case: shortcut forcing step (3x activations)
    x = torch.randn(batch_size, seq_len, 48, 16, 16, device="cuda", dtype=torch.bfloat16)
    t = torch.rand(batch_size, device="cuda")

    if shortcut_forcing:
        # Bootstrap needs two forward passes
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out1 = model(x, t)
            loss1 = out1.mean()
            out2 = model(x, t)
            loss = loss1 + out2.mean()
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, t)
            loss = out.mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"B={batch_size:2d} T={seq_len:2d} GC={'on ' if gradient_checkpointing else 'off'} "
          f"SC={'on ' if shortcut_forcing else 'off'} -> Peak: {peak:.1f}GB / {total:.1f}GB "
          f"({'OK' if peak < total * 0.9 else 'TIGHT' if peak < total else 'OOM'})")

    del model, optimizer, x, t, loss
    torch.cuda.empty_cache()
    return peak


configs = [
    # (batch, seqlen, grad_ckpt, shortcut)
    (4, 32, False, True),   # B=4 short, GC off, worst case
    (2, 64, False, True),   # B=2 long, GC off, worst case
    (4, 32, True, True),    # B=4 short, GC on, worst case
    (2, 64, True, True),    # B=2 long, GC on, worst case
    (8, 32, True, True),    # B=8 short, GC on, worst case
    (4, 64, True, True),    # B=4 long, GC on, worst case
]

print(f"Testing VRAM usage (16GB RTX 5080)")
print("=" * 70)
for b, t, gc, sc in configs:
    try:
        test_config(b, t, gc, sc)
    except torch.cuda.OutOfMemoryError:
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"B={b:2d} T={t:2d} GC={'on ' if gc else 'off'} "
              f"SC={'on ' if sc else 'off'} -> OOM!")
        torch.cuda.empty_cache()
