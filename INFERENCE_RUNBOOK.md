# Garen agent — inference runbook (demo)

The pipeline the demo runs on, and how to bring it up. Model checkpoints referenced
are on the desktop (`/mnt/storage/data/ahriuwu/checkpoints/...`).

## Pipeline
```
frame (352x352 RGB) --v7 tokenizer.encode--> latent (32,16,16)
  --> rolling window of last `context` latents
  --> dynamics(near-clean tau, agent tokens on)  [Step 1' recovered encoder]
  --> agent token (last frame)
  --> PolicyHead.sample(offset 1)  --> abilities {Q..Stride} + movement (x,y)
```
Same math offline and live — only screen capture + input injection differ.

## Components (all built + validated)
- **Tokenizer v7** (frozen): `/mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt`
- **World-model encoder**: Step 1' recovered checkpoint `dyn179_s1prime_latentonly/dynamics_latest.pt` (tau0.9 ~26; the HUD-loss plateau is fixed — see `WM_DEBUG_LOG_2026-07-14.md`). BC is trained *on this backbone*.
- **BC policy** (Phase 2): `data/phase2_bc_garen/agent_finetune_latest.pt` (training now on the recovered backbone; watchdog + 20-min checkpoints).
- **Inference core**: `scripts/agent_infer.py` (`GarenAgent`) — plumbing-verified on real replay latents.
- **Live entrypoint**: `scripts/play_live.py` (mss capture + pynput inject; Windows).

## Offline validation (Linux, no screen) — run once BC has a checkpoint
```
PYTHONPATH=src python scripts/agent_infer.py --test-latents \
  --phase2-ckpt data/phase2_bc_garen/agent_finetune_latest.pt \
  --latents /scratch/ahriuwu/dynamics_replay_latents_v7_dim32/NA1_5549995114.pt \
  --frames 300 --temperature 0
```
Expect: varied movement (unique cells >> 1) and some ability presses. Degenerate
output (0 presses, 1 movement cell) means the policy is still ~untrained.

## Live (Windows, at home)
1. `pip install mss pynput opencv-python torch` (CUDA build for the 5080).
2. Copy the 3 checkpoints (tokenizer, Step 1' dynamics, BC) to the Windows box.
3. **Dry run first** (prints actions, sends nothing):
   ```
   python scripts/play_live.py --phase2-ckpt <bc> --tokenizer-ckpt <v7> \
     --capture-region 0,0,1920,1080 --dry-run
   ```
   Confirm the action stream looks sane (movement targets track the champion; abilities fire occasionally).
4. Go live in a **practice tool / custom game** (never a real match): drop `--dry-run`.
   Keybinds assumed: Q/W/E/R, D=Flash, F=Ignite, B=Recall, item slot 3=Stride, right-click=move/AA. Remap in `DEFAULT_KEYS` if yours differ.

## Known gaps / tuning
- **Speed**: ~7 fps at context=32 uncompiled on the 5080. For 20 fps live: lower `--context` (8–16), and/or enable `torch.compile` (works on the 5080? — test; it worked on Ada 4090). The dynamics forward over the window is the cost.
- **Policy quality** is capped by how long BC trains (it just restarted on the recovered backbone). Let it bank several epochs; re-copy the checkpoint before the demo.
- The world model is an **encoder** here (no dreaming). Phase-3 imagination (a genuinely good policy) needs action-conditioning + the mixed-dataset plumbing — that's post-demo.
