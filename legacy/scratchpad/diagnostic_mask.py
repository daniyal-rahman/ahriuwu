"""Does the HUD mask 'snag' the dynamics loss? Directly compares teacher-forced
denoising on blacked-HUD YT clips vs real-HUD replay clips, using the live 154
checkpoint. If the mask were snagging the loss we'd expect: YT MSE >> replay MSE,
or big per-clip spread on YT (some clips much worse), or wildly different latent
absmax (outliers from the sharp black boundary distorting the max_val-normalized
PSNR). If YT and replay look similar, the mask isn't the culprit."""
import argparse, math, torch
from torch.utils.data import default_collate
from ahriuwu.models.dynamics import create_dynamics
from ahriuwu.data import PackedLatentSequenceDataset


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = create_dynamics("medium", latent_dim=32, use_actions=False, num_kv_heads=4,
                        num_register_tokens=8, soft_cap=50.0, use_qk_norm=True)
    sd = ck["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"loaded 154 step={ck.get('global_step')} missing={len(miss)} unexpected={len(unexp)}\n")
    return m.to(dev).eval()


@torch.no_grad()
def denoise(m, z0, dev, taus=(0.5, 0.7, 0.9)):
    # process ONE clip at a time (B=1) to keep VRAM tiny while 154 trains
    B = z0.shape[0]
    mv = z0.abs().max().item()
    out = {}
    for tv in taus:
        per = []
        for i in range(B):
            zi = z0[i:i+1]
            Ti = zi.shape[1]
            tau = torch.full((1, Ti), tv, device=dev)
            eps = torch.randn_like(zi)
            z_tau = tau.view(1, Ti, 1, 1, 1) * zi + (1.0 - tau.view(1, Ti, 1, 1, 1)) * eps
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_pred = m(z_tau, tau, step_size=torch.ones(1, dtype=torch.long, device=dev))
            per.append(((z_pred.float() - zi.float()) ** 2).mean().item())
            del z_pred, z_tau, eps
            torch.cuda.empty_cache()
        t = torch.tensor(per)
        mse = t.mean().item()
        out[tv] = dict(psnr=10 * math.log10(mv ** 2 / max(mse, 1e-10)), mse=mse,
                       mse_lo=t.min().item(), mse_hi=t.max().item(),
                       mse_std=t.std().item(), absmax=mv)
    return out


def batch_from(d, k, seqlen, dev):
    ds = PackedLatentSequenceDataset(latents_dir=d, sequence_length=seqlen, stride=seqlen)
    idxs = [(i * max(1, len(ds) // k)) % len(ds) for i in range(k)]
    return default_collate([ds[i] for i in idxs])["latents"].to(dev).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_yt578_pretrain/dynamics_latest.pt")
    ap.add_argument("--yt", default="/scratch/ahriuwu/dynamics_yt_latents_v7_dim32")
    ap.add_argument("--replay", default="/scratch/ahriuwu/dynamics_replay_latents_v7_dim32")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seqlen", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda"
    m = load_model(args.checkpoint, dev)

    for name, d in [("YT (blacked HUD)", args.yt), ("REPLAY (real HUD)", args.replay)]:
        z0 = batch_from(d, args.k, args.seqlen, dev)
        print(f"=== {name} === {args.k} clips x T={args.seqlen} | latent mean={z0.mean():.3f} "
              f"std={z0.std():.3f} absmax={z0.abs().max():.2f}")
        r = denoise(m, z0, dev)
        for tv, s in r.items():
            print(f"  tau{tv}: PSNR {s['psnr']:5.1f} | MSE {s['mse']:.5f} "
                  f"(per-clip {s['mse_lo']:.5f}..{s['mse_hi']:.5f}, std {s['mse_std']:.5f}) "
                  f"| latent absmax {s['absmax']:.2f}")
        print()
    print("READ: similar MSE + similar per-clip spread => mask NOT snagging the loss.")
    print("      YT MSE>>replay, or wide YT per-clip spread, or very different absmax => mask IS a factor.")


if __name__ == "__main__":
    main()
