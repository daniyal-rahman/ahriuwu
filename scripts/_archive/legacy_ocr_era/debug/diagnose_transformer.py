#!/usr/bin/env python3
"""Diagnose transformer tokenizer attention and bottleneck."""

import torch
import numpy as np
from ahriuwu.models import create_transformer_tokenizer
from ahriuwu.data import SingleFrameDataset


def analyze_bottleneck(model, frame, device):
    """Check if bottleneck preserves spatial information."""
    print("\n=== Bottleneck Analysis ===")
    
    model.eval()
    with torch.no_grad():
        # Get pre-bottleneck latents
        B, T = 1, 1
        x = frame.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 256, 256)
        x_flat = x.reshape(B * T, 3, 256, 256)
        
        patches = model.patch_embed(x_flat)
        patches = patches.reshape(B, T * model.num_patches, model.embed_dim)
        
        latents_pre = model.encoder(patches, T)  # (B, T*num_latents, embed_dim)
        
        # Apply bottleneck
        latents_post = model.bottleneck(latents_pre, T)  # (B, T*num_latents, latent_dim)
        
        # Check variance across spatial positions
        latents_pre_2d = latents_pre.reshape(1, 256, -1)
        latents_post_2d = latents_post.reshape(1, 256, -1)
        
        pre_var = latents_pre_2d.var(dim=1).mean().item()
        post_var = latents_post_2d.var(dim=1).mean().item()
        
        print(f"Pre-bottleneck spatial variance: {pre_var:.6f}")
        print(f"Post-bottleneck spatial variance: {post_var:.6f}")
        print(f"Ratio: {post_var/pre_var:.4f}")
        
        # Check if different spatial positions produce different outputs
        pos_0 = latents_post_2d[0, 0, :]
        pos_128 = latents_post_2d[0, 128, :]
        pos_diff = (pos_0 - pos_128).abs().mean().item()
        print(f"Latent diff between pos 0 and 128: {pos_diff:.6f}")
        
        return latents_post_2d


def analyze_decoder_patches(model, latents, device):
    """Check if decoder produces spatially varying patches."""
    print("\n=== Decoder Output Analysis ===")
    
    model.eval()
    with torch.no_grad():
        latents_inv = model.bottleneck_inv(latents.reshape(1, 256, -1), 1)
        patches = model.decoder(latents_inv, 1)  # (1, 256, embed_dim)
        
        patches_2d = patches.reshape(1, 256, -1)
        
        # Check variance across patches
        patch_var = patches_2d.var(dim=1).mean().item()
        print(f"Output patch variance: {patch_var:.6f}")
        
        # Check if neighboring patches differ
        patch_0 = patches_2d[0, 0, :]
        patch_1 = patches_2d[0, 1, :]
        patch_128 = patches_2d[0, 128, :]
        
        diff_01 = (patch_0 - patch_1).abs().mean().item()
        diff_0_128 = (patch_0 - patch_128).abs().mean().item()
        
        print(f"Patch diff (0 vs 1, neighbors): {diff_01:.6f}")
        print(f"Patch diff (0 vs 128, far): {diff_0_128:.6f}")
        
        if diff_01 < 0.01:
            print("WARNING: Neighboring patches nearly identical!")


def analyze_reconstruction_error(model, frame, device):
    """Check where reconstruction error is highest."""
    print("\n=== Reconstruction Error Analysis ===")
    
    model.eval()
    frame_in = frame.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(frame_in, mask_ratio=0.0)
        recon = output["reconstruction"]
        
        # Per-pixel error
        error = (frame_in - recon).abs()
        
        # Spatial error map (average over channels)
        error_map = error.mean(dim=1).squeeze()  # (256, 256)
        
        # Top vs bottom half error
        top_error = error_map[:128, :].mean().item()
        bottom_error = error_map[128:, :].mean().item()
        
        # Left vs right half error
        left_error = error_map[:, :128].mean().item()
        right_error = error_map[:, 128:].mean().item()
        
        print(f"Top half error: {top_error:.6f}")
        print(f"Bottom half error: {bottom_error:.6f}")
        print(f"Left half error: {left_error:.6f}")
        print(f"Right half error: {right_error:.6f}")
        
        # Check if error is uniform or localized
        error_std = error_map.std().item()
        error_mean = error_map.mean().item()
        print(f"Error mean: {error_mean:.6f}, std: {error_std:.6f}")
        
        if error_std < error_mean * 0.2:
            print("Note: Error is relatively uniform across image")
        else:
            print("Note: Error varies significantly across image")


def main():
    device = "cuda"
    
    print("Loading model...")
    model = create_transformer_tokenizer("small").to(device)
    ckpt = torch.load("checkpoints/transformer_tokenizer_latest.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded step {ckpt.get('global_step', 'unknown')}")
    
    print("\nLoading test image...")
    dataset = SingleFrameDataset("data/processed/frames")
    frame = dataset[0]["frame"].to(device)
    
    # Run diagnostics
    latents = analyze_bottleneck(model, frame, device)
    analyze_decoder_patches(model, latents, device)
    analyze_reconstruction_error(model, frame, device)
    
    print("\n=== Summary ===")
    print("If bottleneck spatial variance is very low, bottleneck is losing position info.")
    print("If patch diff is very low, decoder isn't producing varied outputs.")
    print("If error is directional (top vs bottom different), may indicate position issues.")


if __name__ == "__main__":
    main()
