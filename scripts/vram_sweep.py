"""VRAM sweep for the dynamics trainer: measure peak VRAM of a full training step
(forward + loss + backward) for the medium model across batch / seq-len / compile
/ pixel-HUD, and record where it OOMs. Uses synthetic latents (VRAM depends on
shapes, not values) + the real frozen v7 tokenizer for the pixel-HUD decode path.
Emits a markdown table (--out) for the experiment doc.

Run: PYTHONPATH=src python scripts/vram_sweep.py --tokenizer <v7.pt> --hud-mask <mask.pt> --out VRAM_SWEEP.md
"""
import argparse, sys, time, traceback
import torch

sys.path.insert(0, "scripts")
from ahriuwu.models.dynamics import create_dynamics
from train_dynamics import pixel_hud_masked_loss
from pretokenize_replay_v7 import load_v7


def try_step(model, tok, mask, B, T, dev, pixel, K):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    z0 = torch.randn(B, T, 32, 16, 16, device=dev)
    tau = torch.rand(B, T, device=dev)
    tb = tau.view(B, T, 1, 1, 1)
    z_tau = tb * z0 + (1 - tb) * torch.randn_like(z0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z_pred = model(z_tau, tau)
        loss = pixel_hud_masked_loss(tok, z_pred, z0, mask, tau, K) if pixel \
            else ((z_pred.float() - z0.float()) ** 2).mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    model.zero_grad(set_to_none=True)
    del z0, tau, z_tau, z_pred, loss
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="rollout_stage/transformer_tokenizer_latest.pt")
    ap.add_argument("--hud-mask", default="scratchpad/hud_valid_mask_352.pt")
    ap.add_argument("--pixel-frames", type=int, default=4)
    ap.add_argument("--out", default="VRAM_SWEEP.md")
    args = ap.parse_args()
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)

    tok, tcfg, tstep = load_v7(args.tokenizer, dev)
    for p in tok.parameters():
        p.requires_grad_(False)
    tok.eval()
    mask = torch.load(args.hud_mask, map_location=dev, weights_only=True).float()

    model = create_dynamics("medium", latent_dim=32, use_actions=False, num_kv_heads=4,
                            num_register_tokens=8, soft_cap=50.0, use_qk_norm=True,
                            gradient_checkpointing=True).to(dev).train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"GPU {name} {total:.1f}GB cap{cap} | dynamics medium {nparams/1e6:.1f}M params | "
          f"tok {tstep} num_lat={tcfg['num_latents']}", flush=True)

    # (T, pixel, K, list of batch sizes to try)
    plan = [(128, True, args.pixel_frames), (256, True, args.pixel_frames),
            (128, False, 0), (256, False, 0)]
    batches = [1, 2, 3, 4, 6, 8, 12]
    rows = []
    for (T, pixel, K) in plan:
        for B in batches:
            tag = f"B={B} T={T} pixel-HUD={'on(K=%d)' % K if pixel else 'off'}"
            try:
                peak = try_step(model, tok, mask, B, T, dev, pixel, K)
                pct = 100 * peak / total
                print(f"  {tag:34s} -> {peak:5.1f} GB  ({pct:3.0f}% of {total:.0f}GB)  OK", flush=True)
                rows.append((B, T, pixel, K, f"{peak:.1f} GB", "OK"))
            except torch.cuda.OutOfMemoryError:
                print(f"  {tag:34s} -> OOM", flush=True)
                rows.append((B, T, pixel, K, ">%.0f" % total, "OOM"))
                torch.cuda.empty_cache()
                break  # bigger batches at this (T,pixel) will also OOM
            except Exception as e:
                print(f"  {tag:34s} -> ERR {str(e)[:50]}", flush=True)
                torch.cuda.empty_cache()

    # extrapolate max batch for a 24GB card from the two smallest fitting batches
    with open(args.out, "w") as f:
        f.write(f"# Dynamics trainer VRAM sweep\n\n")
        f.write(f"- **GPU tested:** {name}, {total:.1f} GB, compute cap {cap[0]}.{cap[1]}\n")
        f.write(f"- **Model:** dynamics `medium`, latent_dim 32, {nparams/1e6:.1f}M params, "
                f"num_kv_heads 4, register_tokens 8, soft_cap 50, gradient_checkpointing ON, "
                f"AdamW (bf16 autocast)\n")
        f.write(f"- **Pixel-HUD loss:** frozen v7 decoder, frame-by-frame + gradient-checkpointed, "
                f"K={args.pixel_frames} frames/clip\n")
        f.write(f"- **Metric:** peak `torch.cuda.max_memory_allocated` for one full "
                f"forward+loss+backward on synthetic latents (B,T,32,16,16)\n\n")
        f.write("| batch | seq T | pixel-HUD | peak VRAM | status |\n|---|---|---|---|---|\n")
        for (B, T, pixel, K, mem, st) in rows:
            f.write(f"| {B} | {T} | {'on (K=%d)'%K if pixel else 'off'} | {mem} | {st} |\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
