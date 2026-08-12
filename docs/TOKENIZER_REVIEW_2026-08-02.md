# Forensic design review: v7 tokenizer vs. minion-HP legibility — 2026-08-02

*Adversarial review by an independent fresh-context agent with full repo access; it ran its own
measurements on the frozen v7 checkpoint (pixel footprints, per-region loss shares) and audited the
tokenizer lineage, scaling experiments, and the parallel tokenizer session's uncommitted work.*

## 1. FORENSICS — lineage and what each experiment actually showed

| Ver | Config | Data | Outcome / evidence |
|---|---|---|---|
| CNN baseline | ~6M CNN | OCR-era frames | Compared (`d79be7d`), then deleted (`737444b`) — transformer won |
| v1 | medium, T=16, lr 3e-4 | `/mnt/storage` frames | masked-patch-only MSE while dashboards showed full-frame PSNR — "eval measured the wrong regime" lesson |
| v2/v3 | small, 1 epoch | replay flat frames | infra shakeout |
| v4→v6 | "paper-faithful" (`98e43d1`): medium ~130M, mask U(0,0.9), lr 1e-4, eff batch 32 | 142-match 352png | v6 = warm-start of v5@11216 |
| v7 trial | large D=1024 8+8, **256×32** bottleneck | same | 3000-step health check |
| **v7 prod** | large D=1024, **512×16**, temporal_every=4, T=20, eff batch 64, 6000 optsteps ≈ 0.8 epoch | replay frames | m=0 train PSNR 28.2 dB ("~v6, data-starved/hazy"); 26.8 dB re-encode verify; 208.4M params |

The buried record, in causal order:
1. **Step-5 bottleneck-shape overfit test** (`slurm_tok_train_v7.sbatch:35`): *512×16 was chosen despite the test showing 256×32 was ~7 dB better at matched params* (structure-faithfulness override).
2. **Arch upsizing bought nothing:** v6 (medium) → v7 (large) landed at the same m=0 PSNR (caveat: 0.8 epoch).
3. **Probes, three generations:** within-game 0.82 was adjacent-frame leakage (temporal-block split −0.61); cross-game full-dim 0.09; final MLP protocol 0.16/0.11. MLP≈linear and weak = information absent.
4. **Human-verified recon:** champ bar = red blob with mushy fill; "+14" popup erased; minion bars vanish.
5. **Fold is non-spatial** (`mask_spatial_test.py`): 512 perceiver latents are global readers; the 16×16×32 grid is a reshape fiction — no latent-space region masking possible.
6. **Tokenizer ranked non-binding** (GROUNDING_2026-07-29): WM teacher-forces ~9 dB below the tokenizer ceiling.
7. **Parallel tokenizer session (uncommitted):** `--exclude-blacked-regions` valid_mask in MAELoss (YT black fix), stride 8→16, `slurm_tok_train_v7_cont.sbatch` (2 more epochs, same objective, NOT yet launched). Nothing in flight targets minion HP.
8. `probe_scaling.py` is a dynamics sweep, not tokenizer.

## 2. ROOT CAUSE — measured

- Minion HP bar fill at 352×352: **~9–18 px × 1–3 px**; all bar pixels ≈ **200 px/frame = 0.16% of the frame**. Champion bar ~30×5 px.
- **The tokenizer already tolerates 22× worse error exactly there:** per-pixel MSE on minion-bar pixels 0.0234 vs background 0.00107 — at 29.5 dB frame PSNR.
- **Loss share:** minion bars = **3.4% of total MSE**; perfect bars would buy **0.16 dB**; erasing them entirely costs ~0.18 dB. LPIPS is area-weighted the same way.
- **MAE masking trains the failure:** E[mask]≈0.40 with tube masking → bar fill unpredictable from context → loss-optimal output is a *generic mean bar* (the observed artifact). ~40% of a bar's gradient is inpainting-hallucination training.
- **Bottleneck:** 8,192 dims vs 371,712 pixels (45×) under an area-weighted loss → rate allocator drops small high-freq features first; Step-5 says the chosen shape additionally under-delivers ~7 dB.
- **Resolution is NOT the killer:** the signal is human-legible in GT at 352². Its only sin is a small loss share — fixable by reweighting, not 4× compute.
- **Minion HP has no labels anywhere** (memory-read labels are hero-only; `.rofl` decode ended negative, R² 0.006). Any supervision must come from pixels — and the bars are deterministic UI, CV-readable.

Dominance: (1) objective indifference, (2) bottleneck capacity/shape, (3) MAE masking, (4) not resolution, (5) not data statistics.

## 3. OPTIONS (ranked)

**A. Hybrid wave-state track around the frozen tokenizer (RECOMMENDED).** CV bar reader → per-frame
(16×16×3) state channels (enemy-min fill, ally fill, champ HP rasterized by position) appended to the
dynamics input (latent_dim 32→35, zero-init new proj rows), predicted forward like any channel. Dreams
simulate last-hits in the state channels, not pixels. Costs: reader $0 (CPU); **no re-encode**; dynamics
**warm-start** from gs8775, 2–4k steps ≈ $8–10 Vast or 2–3 days 5080. Risks: reader noise under clumping
(min-aggregate per cell); dynamics ignoring the channels (add supervised next-state head).

**B. Bar-weighted recon fine-tune of v7** (float valid_mask weights ≈30 on CV bar masks → bars ~50% of
gradient). Cheap fine-tune (~1–1.5k steps) but the REPAY is the cost: re-encode + dynamics retrain
(~$16–25) + BC restart. Only worth piggybacking on an independently-justified retrain.

**C. Aux supervised loss on the bottleneck** (HP scalars from labels + CV pseudo-labels during tokenizer
training). Guarantees latent legibility; same repay as B; combine B+C if a retrain ever happens.
Strictly dominated by A for the near term.

**D. 256×32 reshape from scratch** — only option with direct in-repo fidelity evidence (+7 dB) but
maximum cost and fixes haze, not bar-indifference.

**E. Resolution up / dual-stream UI crop** — weakest evidence-per-dollar. Rejected.

**F. v7-cont as configured (more epochs, same objective)** — ANTI-RECOMMENDED standalone: the v6→v7
curve is flat, the loss-share math is invariant to training length, and adopting it silently commits
the full repay for ~1 dB of background haze. **Must not launch without B+C folded in.**

**G. StateHead on agent tokens (in flight)** — keep as diagnostic only. Cannot exceed what frozen
latents contain (~R²0.16 enemy HP); covers zero minions; aux 0.038 on [0,1] targets is roughly
predict-the-mean for hp_frac inflated by easy targets — do not read it as "state recovered."

## 4. ADVERSARIAL VERDICT

**Fixing the tokenizer is not worth doing now, and "minion-HP-legible latents" is the wrong gate for
"dreams simulate last-hits."** (1) The root cause is the objective, fixable only at retrain time — and
the repay (re-encode + dynamics from scratch + BC restart) is 3–6× the remaining budget and kills the
in-flight act8775 lineage. (2) The repo already measured that a sharper tokenizer buys nothing today
(WM 9 dB below tokenizer ceiling; dreams gated by controllability/persistence first). (3) Minion HP has
no labels and the bars are deterministic UI: once a CV reader exists (needed for the policy side-channel
anyway), asking a 208M MAE to rediscover 4 numbers through a 0.16%-of-pixels signal is the most
expensive possible encoding of information already held exactly. (4) The cheapest path to HP-aware
dreams bypasses pixels: explicit wave-state channels rolled forward by the dynamics (a warm-start, not a
new latent space). Tokenizer surgery re-enters only when a retrain is independently justified (YT
re-admission, 256×32 reshape) — at which point B+C are free and mandatory.

## 5. RECIPE — option A, with the tokenizer branch as gated fallback

- **Stage A — CV wave-state reader** ($0, CPU): color+shape detection (w≥2.5h, h≤3, backing-bar extent
  for fill). Gates: champ-bar read vs labels hp_frac **R² > 0.9**; minion fill within ±10% on ~50
  hand-checked crops; ≥90% of laning frames detect ≥1 enemy bar.
- **Stage B — channels alongside latents** ($0): per match `(T,16,16,3)` fp16 next to existing `.pt`
  latents. No re-encode; v7 frozen; current latents byte-identical.
- **Stage C — dynamics warm-start**: latent_dim 32→35 off gs8775, zero-init new proj rows, optional
  supervised next-state head (RMS-normalized, ~0.5). 2–4k steps ≈ $8–10 Vast (validated workflow) or
  2–3 days 5080 post-BC.
- **Go/no-go gate (held-out game, h=8 dreams, K=64):** (i) dreamed HP-channel error beats persistence;
  (ii) counterfactual sustained-AA vs no-AA diverges dreamed minion HP (doubles as the Phase-3
  controllability gate, now on a legible variable); (iii) reward head on latent+state fires BEFORE
  minion death (AUC at t−5 ≥ 0.8) — anticipation, not post-hoc death-effect keying.
- **Fallback tokenizer branch** (only if gate iii fails on reader noise AND a retrain is otherwise
  committed): fine-tune v7 with float-weighted valid_mask (bar weight ~30) + aux bottleneck-state head
  (weight ~0.1), keep 512×16, lr 3e-5, 1000–1500 steps, keep the parallel session's blacked-region/
  stride/seed fixes. Gate before paying the repay: held-out minion-fill MLP **R² ≥ 0.5** and champ
  hp_frac **R² ≥ 0.6** on the new latents + human montage check. Only then re-encode → dynamics
  retrain → BC restart.

Evidence base: `GROUNDING_2026-07-29.md`, `docs/EXPERT_REVIEW_2026-08-02.md`,
`slurm/slurm_tok_train_v7.sbatch` (Step-5 note), `scratchpad/probe_mlp.log`,
`scratchpad/hp_fulldim.txt`, `scratchpad/mask_spatial_test.py`, `docs/rofl_hp_decoder_v1.md`,
uncommitted diffs in the parallel tokenizer session.

---

## CORRECTION ADDENDUM (2026-08-05) — curve + shape investigation overturns two claims above

1. **The paper's tokenizer bottleneck IS 512×16.** Paper text: *"We reshape the (N_b=512)×(D_b=16)
   bottleneck of the tokenizer to (N_z=256)×32 for the dynamics model."* Production v7 matches the
   paper exactly; the 256×32-bottleneck trial was the deviation. The "faithfulness override" framing
   in §1 is wrong.
2. **The Step-5 "+7 dB for 256×32" does not replicate in the realistic regime.** Full-data,
   matched-samples comparison from the actual slurm logs (v7trial_69/70 vs the v7 job 75→79 resume
   chain): at ~96–98k samples, 256×32 = 25.75 dB vs 512×16 = 24.8 dB — **~+1 dB early-training, not
   +7**. The +7 was an overfit-regime optimization-ease artifact. Option D (shape reshape) is
   RETRACTED; keep the paper shape.
3. **Hard plateau confirmed** (`scratchpad/tok_curves.png`): v7 job84 climbed 26.8→28.4 dB (Jun 3–9);
   the v7c continuation (job85, Jun 9–18, seeded from step 5970, fresh WSD, 15k-step schedule) ran
   **9 more days for +0.28 dB (28.78→29.06), loss flat 0.0679→0.0680**. A never-adopted ~gs8.8k
   checkpoint exists on the desktop drive; it is moot. "More steps, same everything" is a dead lever.
4. **Token-utilization probe on the stored latents** (5,481 frames, 3 games): 2/512 dead tokens, 4.2%
   tanh saturation, adjacent-pair correlation ≈ random — but **effective rank ≈ 31/8192**
   (90% of cross-frame variance in ~750 dims, 99% in ~2,400), per-frame token-matrix rank ~9/16.
   The bottleneck is far from full ⇒ the plateau is **objective/data-limited, not capacity-limited**.
5. **Revised retrain thesis:** keep 512×16 (paper). The two evidence-backed levers are the ones never
   tried: **DATA** — v7's corpus was replays-only (~50h; commit a576758), the 450h YT never entered
   tokenizer training (needs the HUD masking fix) — and **OBJECTIVE** — bar-weighted reconstruction +
   bottleneck aux supervision (the §2 loss-share math stands). Price is unchanged from the earlier
   tiering (~$140 at 12k optsteps on 4×4090; ~$0.012/optstep), but the mechanism claim is now
   data+objective, not shape.
