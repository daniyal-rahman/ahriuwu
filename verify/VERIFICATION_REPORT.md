# Tokenizer + Dynamics Claim Verification — Adversarial Read-Only Pass

Date: 2026-07-03 · HEAD `dbfd610` (main) · Method: code + checkpoint + on-disk data are ground truth; every card/comment/arg-description treated as a hypothesis.
Scope discipline: **no training runs launched; no edits outside `verify/`** (confirmed in §"What I did NOT do").

## 0. Bottom line — ranked by how much it should change the plan

1. 🔴 **HUD-loss bug is REAL and unfixed (A).** 35% of every YT frame is blacked; **both MSE and LPIPS** train the model to reproduce that black — `MAELoss` has no exclusion. Empirically proven (nonzero HUD error + gradient). **Fix before any further YT training run** (valid-mask in the loss).
2. 🔴 **Holdout-leak risk (K).** The 6 eval-holdout videos sit *inside* the training corpus and training excludes nothing → any full-corpus YT run scored on its own training data. Past YT-holdout PSNRs may be memorization. Add hard video-ID exclusion + a clean holdout before trusting them. (Probe is verified-disjoint/clean; run2/run3 are leak-suspect and unconfirmable read-only.)
3. 🔴 **Action-domain regression CONFIRMED (B).** −14.5 dB edge collapse on **40/40** clips — not the 2-clip fluke the card hedged. The YT-pretrained tokenizer is broken in the deploy domain's periphery until A is fixed.
4. 🟠 **The headline stage-table is measured on the wrong thing (C+K+A).** PSNR is over 35%-black frames → **+1.62 dB inflation** (full 33.56 vs center-only 31.93), and possibly leaked (K). The "+6 dB"/"+0.24 dB" gains partly reward black-reconstruction, not real-content quality; the probe-vs-run3 +0.24 is within the noise this introduces.
5. 🟠 **Data-inventory corrections (I/D):** YT=**906** (not 891/700); replay labeled=**58 h / 23.8 min-per-game** (not 9 min/22 h, not 100 h); labeled:unlabeled=**1:7.9**; training **stride=8** (60% overlap, not 20); action≈**3572** clips/game (not 560). Replan the data strategy on these.
6. 🟠 **Dynamics "dream" plateau is uninterpretable (J):** no copy-last-frame baseline; context is **6 frames/0.3 s** (not ~1 s); frozen tokenizer is step **6000** (not 7k). Add persistence baseline before reading the plateau.
7. 🟡 **Cost (E):** real ≈ **$103/epoch** (replay set) or **$818** (full YT corpus), **compute-bound** (GPU ~100%, not dataloader-starved). "$650/epoch" source not locatable read-only.
8. 🟡 **Config nits (F/G/H):** arch matches the weights but via CLI overrides (not the `large` preset); mask never reached 0.9 (capped 0.45), training is **not** "~90% masked"; `--reset-schedule` also resets the mask curriculum. Otherwise optimization/precision claims verified.

## 1. Verdict table

| Claim | Claimed value | Source | Verdict | Evidence (short) |
|---|---|---|---|---|
| **A. HUD-loss bug real + unfixed** | YT HUD blacked, MAELoss doesn't exclude it → trains black; live on HEAD | model card | ✅ **VERIFIED** | 35.5% of frame blacked; MSE penalty 0.089 + LPIPS 0.21 for non-black HUD; nonzero grad in-HUD; `MAELoss.forward` has no mask arg; no fix commit on HEAD |
| **B. Action-domain −13 dB edge collapse** | before~30/edge~31 → probe~20/edge~18 | model card (2 clips) | ✅ **VERIFIED (stronger)** | 40/40 clips: edge 31.0→16.4 (**−14.5 dB**), center held (−2.4), overall −9.2 |
| **C. Eval-domain mismatch inflates holdout** | holdout HUD-blacked; drop w/o HUD | model card | ✅ **VERIFIED** | full 33.56 (=reported) → center-only **31.93**; HUD-region 42.4 (trivial black); **inflation +1.62 dB** |
| **D. Epoch definition / stride** | seq_len 20, non-overlap implied | dataset.py | ⚠️ **stride=8, NOT 20** | `train:753` hardcodes stride=8 → 60% window overlap; clips/game≈4562; YT-corpus epoch=4.13M clips |
| **E. $650/epoch ~10× inflated?** | ~$650/epoch | notes | ⚠️ **PARTIAL — compute-bound; real $103 (replay) / $818 (full YT)** | measured 2.28 clip/s @ ~100% GPU util (NOT dataloader-starved); $650 source unpinnable read-only |
| **F. Arch == 208,352,072 / D1024 / 8+8 / te4 / 512×16** | as stated | card | ✅ **VERIFIED (attribution ⚠️)** | params exact; every axis matches saved weights — BUT these are CLI **overrides**, not the `large` preset (preset = 12+12, latent 256×64, te2) |
| **G. Objective/masking** | MSE+0.2·LPIPS RMS-norm, 16-frame LPIPS, mse_on_full_frame, tube, curric 0→0.9/2000, p_zero 0.1 | card | ⚠️ **PARTIAL — 2 corrections** | loss verified; **0.9 never reached (capped 0.45 this run)**; **"~90% masked" FALSE** (mean ≈0.1–0.2; denoising-AE, not pathological train/test gap) |
| **H. Optim/precision** | eff64=1×16×4, LR1e-4 WSD, WD0.1, fp32 AdamW no-8bit, gc ON, bf16 autocast + fp32 master + scaler-off-CUDA | card | ✅ **VERIFIED** | all confirmed in code + optimizer state (moments fp32); `--reset-schedule` ALSO resets mask curriculum (footgun confirmed) |
| **I. Data inventory** | 891 YT, 147 action, ~9min/game (~22h), 20fps, action≈560 clips/game | card | ❌ **REFUTED (multiple)** | YT=**906** (not 891/700); replay=**58h/23.8min-per-game** (not 9min/22h nor 100h); action=**~3572 clips/game** (not 560); 20fps ✅; labeled:unlabeled=**1:7.9** |
| **K. Holdout leak (adversarial)** | (unstated) | dataset behavior | 🚩 **STRUCTURAL RISK CONFIRMED** | all 6 holdout video IDs exist in training corpus; tokenizer training applies NO exclusion; which runs leaked = NEEDS-RUN (no logs). Probe (games 157-196) verified disjoint = clean |
| **L. v7-cont trains on replays not YT (adversarial)** | "trained on YT" | sbatch | ⚠️ **CLARIFIED** | `slurm_tok_train_v7_cont.sbatch` FRAMES_DIR=action-labeled replays; YT training was the separate cloud/de-risk phase |
| **J. Dynamics dream graph integrity** | ceiling=true-latent decode; dream=autoregressive; pixel-PSNR; ctx~1s; frozen 7k tokenizer; | rollout_check.py | ⚠️ **PARTIAL — 3 corrections** | ceiling/AR/pixel-PSNR/frozen-offline ✅; **ctx=6 frames (0.3s) not 20/1s**; **tokenizer step 6000 not 7k**; **NO copy-last-frame baseline** → plateau uninterpretable |

## 2. Detailed evidence

### A — HUD-loss bug (✅ VERIFIED, larger than documented)
- Mask source: `scripts/download_yt_frames.py:107-110` `apply_mask()` zeros `MASK_REGIONS_360P` (`:32-38`: top full-width strip, both champion columns, bottom-left Garen HUD, bottom-center scorecard) on the 640×360 frame, then `TARGET_SIZE=(352,352)` resize (`:40`). Aspect-squished 16:9→1:1.
- Loss has no exclusion: `MAELoss.forward(pred,target,mask_indices,patch_size,skip_lpips)` — **no valid/spatial-mask param** (empirically dumped signature). With `mse_on_full_frame=True` → `mask_indices=None` → `F.mse_loss(pred_flat,target_flat)` over the whole frame (`losses.py:280-284`); LPIPS always on full frames (`losses.py:317-327`).
- **Empirical (p0a_hud_loss.py, CPU):** YT blacked fraction **0.355** (brightness 4e-4); action in same region **0.178** (content). A recon that outputs mid-gray in the HUD → **MSE 0.0886** and **LPIPS 0.21** penalty (both nonzero); gradient nonzero in-HUD, zero outside → **HUD pixels train the model toward black**.
- Still present on HEAD: `git log --all` shows no valid-mask/HUD-loss fix (only old HUD-overlay/OCR commits). The probe (`ahriuwu_run4`) was trained through this path.
- **Impact: highest.** Both MSE and LPIPS actively reward blacking 35% of every YT frame.
- **Controlled ablation (`smoke_hud_loss.py`, tiny random-init model, 60 steps, on danilogin 1060) — closes the causal gap:** two identical models trained on rectangle-blacked frames, differing ONLY in whether the loss includes the rectangle. Grad inside the blacked rect is nonzero (6.7e-7) via the real `MAELoss`. After training, model output in the rectangle *when fed content*: **CURRENT loss 0.49→0.16 (learns to paint it black); FIXED loss (rect excluded) 0.50→0.50 (preserves content).** Train-loss identical (~0.03) both runs → **this is why it hid**: the training metric never sees it (region is always black in training data); damage only shows on content-in-region, i.e. deployment. **Verdict: the loss's inclusion of the data-level HUD mask is THE cause (not domain-shift/forgetting), and excluding it is a working fix.**

### B — action-domain regression (✅ VERIFIED on 40 clips, stronger than the 2-clip card)
- `p0b_regression.py`: BEFORE=`v7cont_step1530` (replay-trained), AFTER=`tokenizer_probe.pt` (YT-trained); both run `no_grad` fp32 on the 1060 over **40 action clips** (1 per game, 40 of 147 games); "edge/HUD" = the exact `MASK_REGIONS_360P` rects = **35.1%** of frame.
- BEFORE all **29.4** / center 28.9 / edge **31.0**  →  AFTER all **20.2** / center 26.5 / edge **16.4**.
- Paired AFTER−BEFORE: all **−9.2**, center **−2.4**, edge **−14.5 dB**; edge collapses (<−3 dB) on **40/40 clips** (center regresses >3 dB on 12/40). Matches & exceeds the card's ~−13 dB; edge-localized, exactly the HUD-loss signature (A).

### C — eval-domain black-edge inflation (✅ VERIFIED, +1.62 dB)
- `p0c_holdout_inflation.py`: probe on **42** YT-holdout clips. FULL-frame PSNR = **33.56** (reproduces the reported ~33.5 headline → confirms the stage-table eval is full-frame on the HUD-blacked holdout). CENTER-only (HUD 35% excluded) = **31.93**. HUD region alone = **42.4 dB** (trivially-reconstructed black).
- **Black-edge inflation = full − center = +1.62 dB.** Every stage-table number (seed 26.37 → probe 33.54) is on this HUD-blacked holdout, so all overstate real-content quality by ~1.6 dB — on top of the leak risk (K). The reported +0.24 probe-vs-run3 delta is within the noise this introduces.

### F/G/H — config/precision (agent-verified against checkpoint + code)
- **F:** params **208,352,072** exact; D=1024, 8+8 layers, temporal_every=4 (temporal blocks at idx 3,7), latent 512×16, patch 16, img 352 (num_patches=484=22²), RoPE — all match saved tensor shapes. ⚠️ **Correction:** these come from `v7_train_args.sh` CLI overrides, NOT the `large` preset (preset defaults = 12+12 layers, num_latents 256, latent_dim 64, temporal_every 2). Reconstructing from `create_transformer_tokenizer("large")` alone will NOT load this ckpt.
- **G:** backprop loss lives in the train loop (`train:479-481`): `(RMS(mse) + 0.2·RMS(lpips))/accum`; MAELoss's own weighted sum is discarded; MSE coeff hardcoded 1.0. LPIPS 16-frame subsample on full frames. `mse_on_full_frame=True` ✅.
  - ⚠️ **Correction G4:** curriculum ramps to 0.9 over 2000 steps, but this 1000-step reset run only reached **current_max=0.45** — the 0.9 was never used.
  - ⚠️ **Correction G5:** "trained ~90% masked" is **FALSE**. `mask_ratio_min=0.0` → per-step `U(0, ≤0.45)` with 10% at exactly 0 → mean input-mask ≈ **0.1–0.2**. With `mse_on_full_frame` (denoising-AE) the target is always the clean full frame, so this is a mild input-masking augmentation gap, NOT a "train 90% masked / eval 0%" objective mismatch.
- **H:** eff batch **64** = 1 × (64//4 per-rank=16) × 4 (`train:700,717`; saved per-rank grad_accum=16 ⇒ NGPU=4). LR 1e-4, WSD (warmup 50/decay 300 this run; args default 500/1500), WD 0.1, betas (0.9,0.999). **Full fp32 AdamW, not 8-bit** — optimizer moments are float32 (bnb gated behind `use_8bit_adam=False`). grad-checkpointing ON. bf16 autocast on CUDA (`dtype=bfloat16 if cuda`), fp32 master weights, **GradScaler disabled on CUDA** (`use_fp16 = device!='cuda'`; ckpt scaler_state={}). **Footgun confirmed:** `--reset-schedule` returns global_step=0 (`training.py:457-459`), and the mask curriculum keys off global_step (`train:442`) → resume re-ramps masking from 0.

### J — dynamics dream graph (`scripts/rollout_check.py` → `dream_psnr.png`, job 124 @ step 11473)
- ✅ Ceiling = `decode(TRUE latent from disk)` (`:161,169,176`); dream = `decode(model.rollout(...))` (`:104,162,177`) and `DynamicsTransformer.rollout` (`dynamics.py:824-899`) is genuinely **autoregressive** (commits its own predictions to the KV cache, `:891-895`); the teacher-forced 1-step number is console-only, not plotted.
- ✅ PSNR is pixel-space vs the real PNG (`px_psnr`, `:166-168`); latent-PSNR is console-only.
- ✅ Dynamics trained on **offline pre-tokenized** latents (`train_dynamics.py` hard-requires `--packed`, never instantiates a tokenizer), produced by the **frozen** v7 tokenizer (`pretokenize_replay_v7.py`, `@torch.no_grad`, `.eval()`).
- ❌ **ctx = 6 frames ≈ 0.30 s** (`--ctx` default 6, `:43,98`), NOT ~20/1 s — the "20" x-axis is the future **horizon**. Latents are 1:1 with 20 fps frames.
- ❌ Frozen tokenizer is **step 6000**, not "7k" (loaded staged `..._latest.pt` → global_step=6000). Decode tokenizer is a **moving `_latest.pt` pointer**; pretokenize-vs-decode identity not byte-verifiable from this node (provenance on desktop /scratch).
- ⚠️ **No copy-last-frame (persistence) baseline** anywhere → the ~18 dB plateau cannot be attributed to "dynamics working" vs scene stationarity (0.3–1.0 s LoL frames are highly self-similar).

### D — epoch definition / stride (⚠️ stride=8, not 20)
- `FrameSequenceDataset` (`dataset.py:126-137`) emits windows `range(0, N-seq_len+1, stride)`. Class default stride=1 (`:83`) but the **training path hardcodes stride=8** (`train_transformer_tokenizer.py:753`); seq_len=20 for v7 (`v7_train_args.sh:42`). → **60% overlap** between consecutive clips.
- clips/game = `floor((N-20)/8)+1`. YT corpus (906 games) = **4,133,664 clips/epoch** (avg 4562/game).
- Non-training call sites (eval/rollout) use stride=seq_len (non-overlap) — so eval and train use *different* stride, a subtle inconsistency.

### E — cost per epoch (⚠️ real ≈ $103 replay / $818 full-YT; compute-bound)
- **Not dataloader-starved:** GPU util measured ~100% during the real 4×5090 probe AND ~100% on the 1060 in this pass. Frames are **pre-extracted png/jpg on NVMe** (FRAMES_DIR points at frame dirs), not live video decode. So no 3–5× dataloader tax.
- **Measured** throughput (probe, real hardware): 2.28 clips/s, 28.1 s/optstep at eff-batch 64 → **$0.0127/optstep** at $1.622/hr (4×5090). LPIPS runs **every step** (16-frame subsample; in the measured number). grad-checkpointing ON (a ~30% self-imposed tax that may be droppable on 32 GB cards — optimization, not bug).
- Derived: replay-set epoch (~522k clips/8153 optsteps) ≈ **$103**; full 906-game YT epoch (64.6k optsteps) ≈ **$818**.
- **On "$650/epoch ~10× inflated":** I could not find the original $650 figure in the repo (read-only), so I can't reproduce its assumptions. If it meant the *replay* epoch (current v7-cont training) it is ~6× high and the likely cause is assuming the class-default **stride=1** (8× more windows than the real stride=8) or dataloader starvation (there is none). If it meant the *full YT corpus*, $650 is if anything an underestimate ($818). Verdict PARTIAL — real cost derived; claim source unpinnable.

### I — data inventory (❌ several REFUTED)
- YT corpus = **906 games** (NFS `yt_pretrain_garen/*.tar`=906; R2=906). Prior "891" dropped 15 legit `_`-prefixed YouTube IDs; "~700" is just wrong.
- Replay labeled = **147 games, 4,176,465 frames, fps=20 (labels.json) → 58.0 h, 23.8 min/game.** Card's "~9 min/22 h" is ~2.6× low; operator's "~100 h" is ~1.7× high.
- Per-game clips: YT ≈ **4562** (✅ ~4549); action ≈ **3572** (❌ card said 560 — off ~6.4×; no subsample keyword exists in replay_dataset/pretokenize, so 560 is unexplained unless a /scratch stage-time decimation not visible here → NEEDS-RUN on desktop).
- **Labeled:unlabeled = 1 : 7.9 by frames (58 h : 460 h); labeled ≈ 11% of total.**
- "~700 of 891 never seen" — UNVERIFIABLE read-only (no seen-game tracking; `/mnt/nfs/shared` empty; sampler is plain DistributedSampler with no exclusion).

### K — HOLDOUT LEAK (🚩 highest-impact alongside A)
- All 6 `ahriuwu_yt_holdout` video IDs (`-05oyD6OXE8`,`-4VGr4S0tPU`,`-5wqYSlDTnI`,`-8Tr0uqdA5U`,`-AdXi19JXXQ`,`-MyMfAaM2gE`) **also exist as `.tar` in the training corpus** `yt_pretrain_garen`, and tokenizer training applies **no video exclusion** (`train:764-768` plain DistributedSampler over all games in frames_dir).
- ⟹ **Any run whose `frames_dir` was the full YT corpus trained on its own eval holdout.** The de-risk/YT-pretrain phase is the prime suspect. Consequence: those PSNR numbers (and any "generalization" claim) may be train-set memorization.
- **The probe (run4) is exempt** — it trained on the disjoint slice (games 157-196; holdout=games 1-6) verified this session. run2/run3 used older `head -N` staging that likely **included** games 1-6 → their holdout numbers (32.16 / 33.30) are **leak-suspect**; not confirmable read-only (Vast boxes destroyed, no logs) → NEEDS-RUN.

### L — the current v7-cont tokenizer trains on REPLAYS, not YT (⚠️ clarification, and it's consistent with the regression)
- `slurm_tok_train_v7_cont.sbatch` (and `slurm_v7_trial.sbatch`) set `FRAMES_DIR=/scratch/ahriuwu/action_labeled_352png_train_flat` — the **replay** frames (HUD disabled), not the 906 YT games. The +6 dB "YT de-risk" and the run2/3/probe cloud runs are the YT-trained lineage.
- This is *consistent* with the A/B story: before-cloud (v7-cont, replays, content at edges) reconstructs edges fine; the YT-trained cloud runs (HUD blacked) learn black edges.

## 3. Corrections to the model card
- "the *large* config yields D=1024/8+8/512×16/te4" → these are **CLI overrides**; the bare `large` preset is 12+12 / 256×64 / te2.
- "mask curriculum 0→0.9" → **0.9 never reached; this run capped at 0.45**; and training is **not** ~90% masked (mean ≈0.1–0.2, denoising-AE).
- Dynamics eval: context is **6 frames (~0.3 s)** not ~1 s; frozen tokenizer is **step 6000** not 7k; **no persistence baseline** exists.
- YT corpus = **906 games** (card "891" dropped 15 `_`-prefixed IDs; operator "~700" wrong).
- Replay labeled = **58 h / 23.8 min-per-game / 4.18M frames** (card "~9 min/game/~22 h" is 2.6× low; operator "~100 h" is 1.7× high). Action ≈ **3572 clips/game**, not 560.
- Training **stride = 8** (60% overlap), not 20/non-overlapping. Param count **208 M** confirmed (the "~600 M" cited in earlier cost chats was wrong).
- fps: **replay = 20** (labels.json). `download_yt_frames.py` default is `fps=4.0`, but YT frame counts (~30–45k/game) imply the actual corpus was extracted at ~20 fps — the 4.0 default appears stale/unused; exact YT corpus fps not directly pinned read-only.

## 4. Load-bearing but unverified (read-only limits → NEEDS-RUN / on desktop)
- **Which past runs actually trained on the 6 holdout videos (leak K).** No job logs (`/mnt/nfs/shared` empty); Vast boxes destroyed. Probe verified-disjoint; run2/run3 unconfirmable.
- **action ≈ 560 vs 3572 clips/game.** Only reconcilable via a `/scratch` stage-time frame-subsample not visible from danilogin — inspect on desktop.
- **Pretokenize-tokenizer == decode-tokenizer byte identity** (dynamics eval). provenance.json on desktop `/scratch`. Indirect evidence: healthy ~27 dB ceiling; both reference a moving `_latest.pt`.
- **"~700 of 891 seen/unseen"** — no seen-game tracking exists.
- **Original "$650/epoch" calculation** — figure not found in-repo; can't reproduce its assumptions.
- **Exact YT-corpus extraction fps** (4.0 default vs ~20 implied) — no per-corpus fps metadata found on this node.
- Anything requiring a real training step (throughput at scale beyond the already-measured probe, dynamics epoch cost).

## 5. What I did NOT do
- Launched **zero** training runs / optimizer loops / multi-GPU jobs. All model forwards were `torch.no_grad()` eval, single-process, on ≤40 tiny clips.
- Made **no** edits/moves/deletes outside `/mnt/nfs/projects/ahriuwu/verify/` (this report) and `/tmp` scratch scripts.
