"""Debug: check if trained model's output varies with tau.
Run while training job is active (uses a separate GPU allocation).
"""
import torch
import sys
from pathlib import Path

from ahriuwu.models import create_dynamics, DiffusionSchedule
from ahriuwu.data import PackedLatentSequenceDataset

device = "cuda"
schedule = DiffusionSchedule(device=device)

# Load the latest checkpoint
ckpt_dir = Path("/mnt/storage/ahriuwu-data/checkpoints")
ckpt_path = ckpt_dir / "dynamics_latest.pt"
if not ckpt_path.exists():
    # Find the latest run dir
    run_dirs = sorted(ckpt_dir.glob("run_*"))
    if run_dirs:
        ckpt_path = run_dirs[-1] / "dynamics_latest.pt"
    else:
        print(f"No checkpoint found in {ckpt_dir}")
        sys.exit(1)

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

model = create_dynamics(
    size="small", latent_dim=48,
    num_kv_heads=4, num_register_tokens=8,
    soft_cap=50.0,
).to(device)

# Load weights (handle compiled model prefix)
state_dict = ckpt["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

print(f"Loaded at epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')}")

# Load a batch
dataset = PackedLatentSequenceDataset(
    latents_dir="/opt/ahriuwu/latents_pt",
    sequence_length=32,
)
z_0 = dataset[0]["latents"].unsqueeze(0).to(device)
B, T = z_0.shape[:2]
step_size = torch.ones(B, dtype=torch.long, device=device)

print(f"\nz_0 shape: {z_0.shape}, range: [{z_0.min():.3f}, {z_0.max():.3f}], std: {z_0.std():.4f}")

# Test 1: Does output change with tau?
print("\n=== Test 1: Output variation with tau ===")
print(f"{'tau':>6} | {'z_tau mean':>10} | {'z_tau std':>9} | {'z_pred mean':>11} | {'z_pred std':>10} | {'MSE vs z_0':>10} | {'PSNR':>6}")

outputs = {}
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for tau_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]:
        tau = torch.full((B, T), tau_val, device=device)
        z_tau, _ = schedule.add_noise(z_0, tau)
        z_pred = model(z_tau, tau, step_size=step_size)

        mse = ((z_pred.float() - z_0.float()) ** 2).mean().item()
        max_val = z_0.abs().max().item()
        psnr = 10 * torch.log10(torch.tensor(max_val ** 2 / max(mse, 1e-10))).item()

        outputs[tau_val] = z_pred.float().clone()

        print(f"{tau_val:6.2f} | {z_tau.float().mean().item():10.4f} | {z_tau.float().std().item():9.4f} | "
              f"{z_pred.float().mean().item():11.4f} | {z_pred.float().std().item():10.4f} | "
              f"{mse:10.6f} | {psnr:6.1f}")

# Test 2: Pairwise output differences
print("\n=== Test 2: Pairwise output differences ===")
for t1, t2 in [(0.1, 0.9), (0.0, 1.0), (0.3, 0.7)]:
    diff = (outputs[t1] - outputs[t2]).abs().mean().item()
    print(f"  |out(tau={t1}) - out(tau={t2})| = {diff:.6f}")

# Test 3: Is the model just outputting the mean?
print("\n=== Test 3: Is model outputting dataset mean? ===")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # Get a few different samples
    for i in range(3):
        z_sample = dataset[i * 100]["latents"].unsqueeze(0).to(device)
        tau = torch.full((1, T), 0.5, device=device)
        z_tau, _ = schedule.add_noise(z_sample, tau)
        z_pred = model(z_tau, tau, step_size=torch.ones(1, dtype=torch.long, device=device))
        mse_vs_input = ((z_pred.float() - z_tau.float()) ** 2).mean().item()
        mse_vs_clean = ((z_pred.float() - z_sample.float()) ** 2).mean().item()
        print(f"  Sample {i}: MSE(pred, z_tau)={mse_vs_input:.4f}  MSE(pred, z_0)={mse_vs_clean:.4f}")

# Test 4: Check tau embedding weights — are they diverse?
print("\n=== Test 4: Tau embedding diversity ===")
tau_weights = model.tau_embed.weight.data
print(f"  tau_embed shape: {tau_weights.shape}")
print(f"  tau_embed norm per index (first 5, last 5):")
norms = tau_weights.norm(dim=-1)
print(f"    indices 0-4: {norms[:5].tolist()}")
print(f"    indices 59-63: {norms[59:64].tolist()}")
# Check if embeddings have differentiated
cos_sim_0_57 = torch.nn.functional.cosine_similarity(
    tau_weights[6:7], tau_weights[57:58]
).item()
cos_sim_0_1 = torch.nn.functional.cosine_similarity(
    tau_weights[6:7], tau_weights[7:8]
).item()
print(f"  cosine_sim(idx=6[tau=0.1], idx=57[tau=0.9]) = {cos_sim_0_57:.4f}")
print(f"  cosine_sim(idx=6[tau=0.1], idx=7[tau=0.11]) = {cos_sim_0_1:.4f}")
print(f"  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")

# Test 5: Gradient magnitude for tau_embed vs spatial weights
print("\n=== Test 5: Gradient check ===")
model.train()
z_0_grad = dataset[0]["latents"].unsqueeze(0).to(device)
tau = torch.full((1, T), 0.5, device=device)
z_tau, _ = schedule.add_noise(z_0_grad, tau)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    z_pred = model(z_tau, tau, step_size=torch.ones(1, dtype=torch.long, device=device))
    loss = ((z_pred.float() - z_0_grad.float()) ** 2).mean()
loss.backward()

tau_grad_norm = model.tau_embed.weight.grad.norm().item() if model.tau_embed.weight.grad is not None else 0.0
step_grad_norm = model.step_embed.weight.grad.norm().item() if model.step_embed.weight.grad is not None else 0.0
input_grad_norm = model.input_proj.weight.grad.norm().item() if model.input_proj.weight.grad is not None else 0.0
cond_grad_norm = model.cond_proj.weight.grad.norm().item() if model.cond_proj.weight.grad is not None else 0.0

print(f"  tau_embed grad norm:  {tau_grad_norm:.6f}")
print(f"  step_embed grad norm: {step_grad_norm:.6f}")
print(f"  cond_proj grad norm:  {cond_grad_norm:.6f}")
print(f"  input_proj grad norm: {input_grad_norm:.6f}")
print(f"  ratio (input/cond):   {input_grad_norm / max(cond_grad_norm, 1e-10):.1f}x")
