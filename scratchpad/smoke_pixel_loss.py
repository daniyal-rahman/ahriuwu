"""Smoke: pixel_hud_masked_loss on a real YT clip — un-fold + decode + masked MSE,
grad flows to the dynamics but NOT the frozen tokenizer, loss finite, VRAM sane."""
import sys, torch
sys.path.insert(0, "scripts")
from train_dynamics import pixel_hud_masked_loss, _dyn_to_tok_latents
from pretokenize_replay_v7 import load_v7
from ahriuwu.models.dynamics import create_dynamics
from ahriuwu.data import PackedLatentSequenceDataset
from torch.utils.data import default_collate

dev = "cuda"
tok, _, step = load_v7("rollout_stage/transformer_tokenizer_latest.pt", dev)
for p in tok.parameters():
    p.requires_grad_(False)
tok.eval()
mask = torch.load("scratchpad/hud_valid_mask_352.pt", map_location=dev, weights_only=True).float()
print(f"tok step {step} | mask {tuple(mask.shape)} HUD={ (mask==0).float().mean()*100:.0f}%")

m = create_dynamics("medium", latent_dim=32, use_actions=False, num_kv_heads=4,
                    num_register_tokens=8, soft_cap=50.0, use_qk_norm=True,
                    gradient_checkpointing=True).to(dev)
ds = PackedLatentSequenceDataset(latents_dir="/scratch/ahriuwu/dynamics_yt_latents_v7_dim32",
                                 sequence_length=8, stride=8)
z0 = default_collate([ds[0], ds[1]])["latents"].to(dev).float()   # (2,8,32,16,16)
print("z0", tuple(z0.shape), "| unfold ->", tuple(_dyn_to_tok_latents(z0).shape))
T = z0.shape[1]
tau = torch.rand(2, T, device=dev)
zt = tau.view(2, T, 1, 1, 1) * z0 + (1 - tau.view(2, T, 1, 1, 1)) * torch.randn_like(z0)
z_pred = m(zt, tau)
torch.cuda.reset_peak_memory_stats()
loss = pixel_hud_masked_loss(tok, z_pred, z0, mask, tau, 2)
print(f"pixel loss = {loss.item():.5f}  finite={torch.isfinite(loss).item()}")
loss.backward()
gdyn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
gtok = sum(1 for p in tok.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"grad -> dynamics: {gdyn:.3f} (>0 good) | tokenizer params w/ grad: {gtok} (must be 0)")
print(f"peak VRAM for pixel loss: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
