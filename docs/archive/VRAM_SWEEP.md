# Dynamics trainer VRAM sweep

- **GPU tested:** NVIDIA GeForce RTX 5080, 16.6 GB, compute cap 12.0
- **Model:** dynamics `medium`, latent_dim 32, 114.6M params, num_kv_heads 4, register_tokens 8, soft_cap 50, gradient_checkpointing ON, AdamW (bf16 autocast)
- **Pixel-HUD loss:** frozen v7 decoder, frame-by-frame + gradient-checkpointed, K=4 frames/clip
- **Metric:** peak `torch.cuda.max_memory_allocated` for one full forward+loss+backward on synthetic latents (B,T,32,16,16)

| batch | seq T | pixel-HUD | peak VRAM | status |
|---|---|---|---|---|
| 1 | 128 | on (K=4) | 6.1 GB | OK |
| 2 | 128 | on (K=4) | 11.0 GB | OK |
| 3 | 128 | on (K=4) | 15.8 GB | OK |
| 4 | 128 | on (K=4) | >17 | OOM |
| 1 | 256 | on (K=4) | 11.0 GB | OK |
| 2 | 256 | on (K=4) | >17 | OOM |
| 1 | 128 | off | 6.1 GB | OK |
| 2 | 128 | off | 11.0 GB | OK |
| 3 | 128 | off | 15.8 GB | OK |
| 4 | 128 | off | >17 | OOM |
| 1 | 256 | off | 11.0 GB | OK |
| 2 | 256 | off | >17 | OOM |

## Analysis

**Linear in batch:** peak ≈ **1.3 GB fixed** (model + AdamW state + frozen tokenizer + grads) **+ ~4.85 GB/batch** at T=128, **~9.7 GB/batch** at T=256 (T=256 ≈ 2× the per-clip activation, as expected). Clean and predictable.

**Pixel-HUD loss is ~free on VRAM.** Every row is *identical* with the loss on vs off (6.1 / 11.0 / 15.8 GB). The frame-by-frame + gradient-checkpointed decode through the frozen v7 decoder never holds more than a single frame's activations, so it adds ~0 to the peak. We pay a throughput cost (sequential K decodes/step), **not** a memory cost — so there's no VRAM reason to lower K.

**Note:** `torch.compile` is NOT in this sweep — it's a *throughput* lever (separate test), and it also shifts VRAM (kernel fusion). The 5080 is Blackwell sm_120 where compiled dynamics kernels hit an illegal-memory-access (hence `--no-compile` on desktop); the **4090 is Ada sm_89 where compile works** and should be enabled there.

## Extrapolation → 24 GB RTX 4090 (the Vast accel target)

Applying `1.3 + N·4.85 ≤ 24` (T=128) and `1.3 + N·9.7 ≤ 24` (T=256):

| GPU | T=128 max batch | T=256 max batch |
|---|---|---|
| RTX 5080 (16.6 GB, measured) | **3** (15.8 GB) | **1** (11.0 GB) |
| RTX 4090 (24 GB, extrapolated) | **4** (~20.7 GB) | **2** (~20.7 GB) |

**B=5/T=128 → ~25.6 GB (OOM); B=3/T=256 → ~30 GB (OOM).**

## Recommendation

- **Desktop 5080 (job 168):** `--batch-size-short 3 --batch-size-long 1` — currently **2/1**, so we're leaving one batch of headroom (15.8 GB fits, ~1 GB margin). Bumping short 2→3 is a free ~1.5× per-step throughput.
- **Vast 4090 re-launch:** `--batch-size-short 4 --batch-size-long 2` (vs the 2/1 we ran) — ~2× the batch, plus enable `torch.compile` (Ada, safe). Keep pixel-HUD K=4 (it's VRAM-free).
- Adjust `--gradient-accumulation` down proportionally to hold the effective batch constant if desired, or leave it to increase the effective batch (faster convergence per wall-clock).
