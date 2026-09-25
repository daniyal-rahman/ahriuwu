"""Crux test: is the tokenizer's 16x16 latent grid SPATIALLY aligned to the frame?
Black out the bottom/corners of a clean replay frame, encode clean vs blacked,
and map |Δlatent| onto the 16x16 grid. If the change LOCALIZES to the bottom
rows -> spatial -> a 16x16 loss mask works. If it smears everywhere -> non-spatial."""
import sys, glob
sys.path.insert(0, "scripts")
import cv2, numpy as np, torch
from pretokenize_replay_v7 import load_v7

dev = "cuda"
model, cfg, step = load_v7("rollout_stage/transformer_tokenizer_latest.pt", dev)
size = int(cfg.get("img_size", 352))
print(f"tok step {step} num_latents={cfg['num_latents']} latent_dim={cfg['latent_dim']} size={size}")


def enc(x):
    with torch.no_grad():
        return model.encode(x)["latent"][0].float()  # (num_latents, latent_dim)


def heat(dlat, label):
    # dlat: (num_latents=512, latent_dim=16). Fold EXACTLY like the pretok:
    # view(16,16,-1) => (16,16,32) spatial grid; mean over the 32 channels.
    g = dlat.reshape(16, 16, 32).abs().mean(-1).cpu().numpy()  # (16,16)
    gn = g / (g.max() + 1e-9)
    print(f"\n{label}  (16x16 dynamics grid, brighter=more change):")
    for row in gn:
        print("  " + "".join(" .:-=+*#%@"[min(9, int(v * 9.99))] for v in row))
    print(f"  bottom-3-rows Δ={g[-3:].mean():.4f}  top-3-rows Δ={g[:3].mean():.4f}  "
          f"mid Δ={g[6:10].mean():.4f}  overall={g.mean():.4f}  max/mean ratio={g.max()/(g.mean()+1e-9):.1f}")


for fi in [80, 300, 600]:
    f = sorted(glob.glob("/mnt/nfs/datasets/lol_replays_16_9_772/*/frames/*.png"))[fi]
    img = cv2.resize(cv2.imread(f), (size, size))
    clean = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().div_(255).permute(2, 0, 1)[None].to(dev)
    blk = clean.clone()
    H = size
    blk[:, :, int(0.80 * H):, :] = 0.0                              # bottom 20% band (HUD bar)
    blk[:, :, int(0.72 * H):, :int(0.28 * size)] = 0.0             # bottom-left corner
    blk[:, :, int(0.72 * H):, int(0.72 * size):] = 0.0            # bottom-right corner (minimap)
    d = enc(blk) - enc(clean)                                      # (num_latents, latent_dim)
    heat(d, f"frame#{fi} blacked bottom+corners")
