"""Integration tests for dynamics model + diffusion pipeline.

Level 2: Does the real model architecture actually use tau conditioning?
Level 3: Does training actually cause tau-dependent behavior?
"""
import sys
sys.path.insert(0, "/mnt/storage/ahriuwu/repo")

import torch
import torch.nn as nn
from pathlib import Path
from ahriuwu.models import create_dynamics, DiffusionSchedule
from ahriuwu.models.diffusion import ShortcutForcing, x_prediction_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
schedule = DiffusionSchedule(device=device)

# Create a fresh (untrained) model
model = create_dynamics(
    size="small", latent_dim=48,
    num_kv_heads=4, num_register_tokens=8,
    soft_cap=50.0,
).to(device)

B, T, C, H, W = 2, 8, 48, 16, 16
z_input = torch.randn(B, T, C, H, W, device=device)
step_size = torch.ones(B, dtype=torch.long, device=device)

print("=" * 60)
print("TEST A: Does untrained model give different outputs for different tau?")
print("=" * 60)

model.eval()
outputs = {}
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for tau_val in [0.0, 0.1, 0.5, 0.9]:
        tau = torch.full((B, T), tau_val, device=device)
        out = model(z_input, tau, step_size=step_size)
        outputs[tau_val] = out.float().clone()
        print(f"  tau={tau_val:.1f}: mean={out.float().mean().item():.6f} std={out.float().std().item():.6f}")

# Check pairwise differences
diffs = []
for t1, t2 in [(0.0, 0.9), (0.1, 0.5), (0.5, 0.9)]:
    diff = (outputs[t1] - outputs[t2]).abs().mean().item()
    diffs.append(diff)
    print(f"  |out(tau={t1}) - out(tau={t2})| = {diff:.6f}")

if all(d < 1e-5 for d in diffs):
    print("  FAIL: Model outputs are IDENTICAL for all tau values!")
    print("  Tau conditioning is NOT reaching the model.")
else:
    print("  PASS: Model outputs differ with tau (conditioning works)")

print("\n" + "=" * 60)
print("TEST B: Does untrained model give different outputs for different step_size?")
print("=" * 60)

tau = torch.full((B, T), 0.5, device=device)
outputs_d = {}
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for d in [1, 2, 4, 16]:
        d_t = torch.full((B,), d, dtype=torch.long, device=device)
        out = model(z_input, tau, step_size=d_t)
        outputs_d[d] = out.float().clone()
        print(f"  d={d:2d}: mean={out.float().mean().item():.6f} std={out.float().std().item():.6f}")

diffs_d = []
for d1, d2 in [(1, 2), (1, 16), (2, 4)]:
    diff = (outputs_d[d1] - outputs_d[d2]).abs().mean().item()
    diffs_d.append(diff)
    print(f"  |out(d={d1}) - out(d={d2})| = {diff:.6f}")

if all(d < 1e-5 for d in diffs_d):
    print("  FAIL: Model outputs are IDENTICAL for all step sizes!")
else:
    print("  PASS: Model outputs differ with step_size")

print("\n" + "=" * 60)
print("TEST C: Do gradients flow through tau_embed?")
print("=" * 60)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()

z_0 = torch.randn(B, T, C, H, W, device=device)
tau = torch.full((B, T), 0.5, device=device)
z_tau, _ = schedule.add_noise(z_0, tau)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    z_pred = model(z_tau, tau, step_size=step_size)
    loss = x_prediction_loss(z_pred, z_0, tau, use_ramp_weight=True)

loss.backward()

tau_grad = model.tau_embed.weight.grad
step_grad = model.step_embed.weight.grad
print(f"  tau_embed grad norm: {tau_grad.norm().item():.6f}" if tau_grad is not None else "  tau_embed grad: None!")
print(f"  step_embed grad norm: {step_grad.norm().item():.6f}" if step_grad is not None else "  step_embed grad: None!")

if tau_grad is not None and tau_grad.norm().item() > 1e-8:
    print("  PASS: Gradients flow through tau embedding")
else:
    print("  FAIL: No gradients in tau embedding!")

if step_grad is not None and step_grad.norm().item() > 1e-8:
    print("  PASS: Gradients flow through step embedding")
else:
    print("  WARN: No gradients in step embedding (may be expected if d=1 only)")

print("\n" + "=" * 60)
print("TEST D: Single-batch overfit (100 steps)")
print("=" * 60)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler("cuda")

z_0 = torch.randn(B, T, C, H, W, device=device)
tau = torch.full((B, T), 0.3, device=device)  # Fixed tau
z_tau, _ = schedule.add_noise(z_0, tau)

for step in range(100):
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z_pred = model(z_tau, tau, step_size=step_size)
        loss = ((z_pred.float() - z_0.float()) ** 2).mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if step % 20 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.6f}")

if loss.item() < 0.01:
    print("  PASS: Model can overfit a single batch")
else:
    print(f"  WARN: Loss={loss.item():.4f} after 100 steps (expected <0.01)")

print("\n" + "=" * 60)
print("TEST E: After overfitting, does tau=0.9 beat tau=0.1?")
print("=" * 60)

model.eval()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for tau_val in [0.1, 0.5, 0.9]:
        tau_test = torch.full((B, T), tau_val, device=device)
        z_tau_test, _ = schedule.add_noise(z_0, tau_test)
        z_pred_test = model(z_tau_test, tau_test, step_size=step_size)
        mse = ((z_pred_test.float() - z_0.float()) ** 2).mean().item()
        max_val = z_0.abs().max().item()
        psnr = 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse, 1e-10))).item()
        print(f"  tau={tau_val:.1f}: MSE={mse:.6f} PSNR={psnr:.1f} dB")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
