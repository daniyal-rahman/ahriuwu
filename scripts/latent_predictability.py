#!/usr/bin/env python3
"""Why does the world model plateau? Measure how predictable the tokenizer latents
are by TRIVIAL baselines (in the eval's exact PSNR units), and compare to the WM's
rollout. If the WM barely beats persistence, the plateau is NOT a data problem:
either the latents carry little learnable frame-to-frame dynamics, or the WM failed
to capture them. If the WM >> persistence, the plateau is a real capacity/data
ceiling at a genuinely-better-than-trivial level.

    /home/dani/miniconda3/envs/ml/bin/python scripts/latent_predictability.py [latents.pt]
"""
import sys

import numpy as np
import torch

LAT = sys.argv[1] if len(sys.argv) > 1 else "rollout_stage/NA1_5549995114.pt"
# WM rollout (this run's eval, step 8850/9450): latent PSNR by horizon
WM = {1: 21.6, 2: 21.4, 4: 19.7, 8: 17.1, 16: 14.4, 32: 11.6}


def main():
    z = torch.load(LAT, weights_only=True)["latents"].float()   # (N,32,16,16)
    N = z.shape[0]
    Z = z.reshape(N, -1)                                          # (N, 8192)
    max_val = z.abs().max().item()

    def psnr(a, b):
        mse = ((a - b) ** 2).mean().item()
        return 10 * np.log10(max_val ** 2 / max(mse, 1e-10))

    print(f"latents {tuple(z.shape)}  max_val={max_val:.2f}  ({LAT.split('/')[-1]})")
    # signal scale: how much does a latent change frame-to-frame vs its own spread?
    d1 = (Z[1:] - Z[:-1])
    print(f"per-frame delta RMS={d1.pow(2).mean().sqrt():.4f}  latent RMS={Z.pow(2).mean().sqrt():.4f}  "
          f"(delta/signal={d1.pow(2).mean().sqrt() / Z.pow(2).mean().sqrt():.2%})")

    print(f"\n{'horizon':>7} | {'persist':>8} {'AR(1)':>7} | {'WM':>6} | WM-persist")
    # per-dim AR(1): z_d[t+1] = a_d z_d[t] + b_d  (fast closed form per dim)
    x, y = Z[:-1], Z[1:]
    xm, ym = x.mean(0), y.mean(0)
    a = ((x - xm) * (y - ym)).sum(0) / ((x - xm) ** 2).sum(0).clamp_min(1e-9)
    b = ym - a * xm
    for h in (1, 2, 4, 8, 16, 32):
        if h >= N:
            continue
        pers = psnr(Z[:-h], Z[h:])
        ar = psnr(a * Z[:-h] + b, Z[h:]) if h == 1 else float("nan")
        wm = WM.get(h)
        gain = f"{wm - pers:+.1f} dB" if wm else ""
        ars = f"{ar:7.1f}" if h == 1 else "      -"
        wms = f"{wm:6.1f}" if wm else "     -"
        print(f"{h:>7} | {pers:8.1f} {ars} | {wms} | {gain}")

    print("\nread: WM≈persist -> not modeling dynamics (plateau is not about data);")
    print("      WM>>persist -> real dynamics captured, plateau is a true capacity/data ceiling.")


if __name__ == "__main__":
    main()
