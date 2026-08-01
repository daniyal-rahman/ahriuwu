# Vast.ai Playbook — consumer-GPU rentals for ahriuwu

Everything learned renting Vast boxes (mostly 5090s) to train the v7 tokenizer on YT data.
Written after the "NCCL saga" of 2026-06-27..29 that burned real money on self-inflicted
misdiagnoses. Read this **before** renting again — most of the cost was avoidable.

Ground truth for the launcher/NCCL details is the code (`scripts/run_ddp_tok.sh`,
`scripts/vast_supervised_run.sh`, `scripts/train_transformer_tokenizer.py`). Cost/throughput
numbers are from real runs but are approximate and date-stamped — re-check live.

---

## 0. TL;DR — the five things that actually matter

1. **Consumer multi-GPU NCCL init is SLOW, not hung.** 8-way init on a no-NVLink box takes
   **~5 minutes**. Never gate readiness on a `<300s` timeout. (This one mistake cost ~$8 + hours.)
2. **NCCL SHM is HOST-SPECIFIC.** Some boxes need `NCCL_SHM_DISABLE=0` (on), others crash with an
   illegal-memory-access unless `NCCL_SHM_DISABLE=1` (off). Don't hardcode it; toggle per box.
3. **Always `trap destroy EXIT`.** A box you forget to destroy bills by the minute. Every
   provision script must destroy on every exit path.
4. **The real SSH endpoint is `public_ipaddr:direct_port_start`**, not the `ssh_host:ssh_port`
   proxy (which refuses connections while the box is otherwise "running").
5. **Re-check credit before declaring a budget block.** Assume nothing about the balance.

---

## 1. Hardware & cost cheat-sheet (approx, 2026-06)

| Box | $/hr | VRAM | Notes |
|---|---|---|---|
| 1×5090 | ~$0.35 | 33.7 GB | probes/microbench; `--direct` SSH |
| 2×4090 | ~$0.75 | 24 GB ea | DDP smoke test (provision→validate→destroy ≈ 40 min ≈ $0.50) |
| 4×5090 | ~$2.7 | 32 GB ea | needed `NCCL_SHM_DISABLE=1` (2026-06-29) |
| 8×5090 on-demand | ~$5.40 | 32 GB ea | the production box (Alberta host); spot/interruptible is cheaper |

**Model fit:** v7 "large" (208M params, 512×16 latent, D=1024, 8+8 layers) fits in **24 GB** at
batch-1 + gradient-checkpointing. On a 32 GB 5090 only ~13.6 GB is used at batch-1 → **raise batch
size to cut $/epoch** (VRAM is not the constraint; we've been leaving throughput on the table).

**Throughput:** ~9 s/optimizer-step at effective batch 64 (batch1 × accum8 × 8gpu). ~4.75 h per
1500 steps. A **full epoch on 906 games ≈ 23 h ≈ $90** at batch-1 — tune batch up first.

**Disk sizing rule:** `1.3 × (frames + keep_last_N × 2.4GB + ~40GB env)`.
≈ **250 GB** if staging frames locally, ≈ **90 GB** if streaming from R2. Pick the disk at create time.

---

## 2. Provisioning workflow

CLI lives at `/home/dani/.vastcli` (v1.1.2), API key already configured. SSH uses danilogin's
`~/.ssh/id_ed25519` (public key is pasted into the Vast account).

```bash
VAST=/home/dani/.vastcli/bin/vastai
$VAST search offers 'gpu_name=RTX_5090 num_gpus=8 inet_down>500 rentable=true' -o dph   # cheapest first
$VAST create instance <offer_id> --image <img> --disk 250 --ssh --direct                # -> STOPPED
$VAST start instance <id>                                                                # -> starting
# poll until actual_status==running AND direct_port_start is a real port (not -1/None)
$VAST show instance <id> --raw | python -c "import sys,json; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d['actual_status'], d['public_ipaddr'], d['direct_port_start'])"
printf 'y\n' | $VAST destroy instance <id>                                               # destroy needs confirmation
```

**Gotchas (all cost time or money at least once):**
- `create instance` lands in **STOPPED** — you must `start instance` then poll `running`.
- `create` can return **`success: False`** if the offer was taken between search and create →
  destroy (if anything was made) and retry a different offer.
- **`--direct` endpoint = `public_ipaddr:direct_port_start`.** The `ssh_host:ssh_port` proxy can
  refuse while the box shows `running`. Poller must *test SSH* (try direct), not trust status.
- **Fresh boxes have unstable sshd** while the container image is still pulling: the box reports
  `running` before the container's sshd is really up, and early connections flap. **Require a
  stable-sshd gate** (see §3) — e.g. 3 consecutive `ssh 'echo ok'` after a ~40s settle, with a
  ~10-min window to cover slow image pulls. Filter offers on `inet_down>500` for fast pulls.
- Launching a **long job directly over SSH hangs the client** even when backgrounded. Detach with
  `setsid env ... </dev/null >log 2>&1 &` and poll the log on a fresh connection. (On Vast this
  works; the on-box `onstart.sh` / supervisor pattern in §5 is better still. Note: on the *home
  desktop*, `setsid`/`nohup` are NOT enough — logind reaps the session scope; use tmux there. See
  memory `desktop-tmux-detach-gotcha`.)
- **Re-check `$VAST show ... balance` before ever saying "out of credit."**

---

## 3. Robust provision pattern (copy this skeleton)

The pattern that finally worked (see `scratchpad/vast_gc_probe.sh` for a full instance). Key ideas:
**always destroy on exit, gate on stable sshd, retry flaky steps.**

```bash
set -uo pipefail
IID=<instance_id>; VAST=/home/dani/.vastcli/bin/vastai
destroy(){ printf 'y\n' | timeout 60 "$VAST" destroy instance "$IID" >>"$LOG" 2>&1; }
trap destroy EXIT                                   # <-- ALWAYS destroy, every exit path
SSH(){ ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 \
           -o BatchMode=yes -o ServerAliveInterval=5 -p "$PORT" root@"$HOST" "$@"; }

# 1) wait for running + a REAL direct endpoint (port != -1/None)
# 2) settle ~40s, then require sshd STABLE: 3 consecutive 'echo ok' inside a ~10-min window
ok=0; for i in $(seq 1 60); do
  if SSH 'echo ok' >/dev/null 2>&1; then ok=$((ok+1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
  sleep 10
done
[ "$ok" -ge 3 ] || { echo "sshd never stable"; exit 1; }
# 3) retry-wrap flaky steps (ship code via tar-over-ssh, run bench, etc.)
retry(){ local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 4 ] && return 1; sleep 15; done; }
```

Ship code without git: `tar czf - -C /mnt/nfs/projects/ahriuwu src | SSH 'tar xzf - -C /root'`.

---

## 4. NCCL on consumer multi-GPU — the big one

Consumer GeForce cards (5090/4090) have **no working GPU P2P and no NVLink/InfiniBand**, so NCCL
must host-stage over PCIe/host-RAM. This produces two *completely different* failure modes that
look similar and waste days if confused.

### Failure mode A — "it's hung at init" (it is NOT)
8-way NCCL init on a no-NVLink box legitimately takes **~5 minutes**: it's a slow topology
path-search (`graph/search.cc ... Could not find a path for pattern N, falling back to simple
order`, repeated), then it passes. GPU mem sits at ~610 MB the whole time — that's *working*, not
stuck. PyTorch's default `init_process_group` NCCL timeout is **10 minutes**, so the real training
waits happily; only toy health-checks with 75–100 s timeouts "fail." **Fix: patience.** Give init
≥360 s, or just launch the real training (no timeout). To sanity-check a host, run a toy allreduce
with `timeout -k 10 360` **and** watch `nvidia-smi` — if mem >2 MB it's progressing.

### Failure mode B — init-time CUDA illegal-memory-access (a real fault)
On some boxes (seen on a 4×5090, torch 2.7.1+cu128, sm_120), multi-GPU crashes **~18 s** into NCCL
init with `CUDA error: an illegal memory access was encountered` (NCCL watchdog → SIGABRT), on
**both** compile and eager. **Fix: `NCCL_SHM_DISABLE=1`** (turn SHM off). With SHM off it trains +
checkpoints cleanly (a residual IMA at NCCL *teardown* after the save is harmless).

### The env baked into `run_ddp_tok.sh` (current, verified)
```bash
NCCL_P2P_DISABLE=1            # GeForce has no real P2P; skips the slow/broken probing
NCCL_SHM_DISABLE=${...:-0}    # HOST-SPECIFIC: default ON, pass =1 per-box if you hit mode B
NCCL_IB_DISABLE=1            # no InfiniBand on these boxes
NCCL_SOCKET_IFNAME=${...:-lo} # single-node loopback bootstrap
TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # a genuine timeout aborts+crashes so the supervisor restarts
NCCL_DEBUG=${...:-WARN}
```
The train script does **eager** `init_process_group(device_id=..., timeout=20min)` so the slow init
happens deterministically at startup.

> **SHM is host-specific — do NOT hardcode it.** The 8×5090 run2 box wanted SHM **ON**
> (`P2P off + SHM on`); a 4×5090 box wanted SHM **OFF**. Different 8× boxes have even wanted P2P
> enabled. Treat every box as a fresh calibration.

### Diagnosis recipe (do this FIRST, not last)
1. **Isolate model vs NCCL:** run `NGPU=1`. If single-GPU trains fine (reaches `loss.backward()`),
   the fault is NCCL/multi-GPU — stop blaming the model/torch/compile or swapping boxes.
2. If multi-GPU IMAs at init → toggle `NCCL_SHM_DISABLE=1`. A `MAX_STEPS=2` run confirms it
   trains+saves before you commit to the full run.
3. If it "hangs" → it's probably mode A. Run the toy allreduce with `NCCL_DEBUG=INFO`, capture the
   **full** output, read the tail. Where it stops names the layer (stops after "Bootstrap: Using
   <iface>" → bootstrap/interface → `NCCL_SOCKET_IFNAME=lo`). **Don't grep-filter the trace to
   death** — a too-narrow grep once hid the bootstrap line and I flew blind.

---

## 5. Running the real training (death-safe)

Two scripts, layered:

- **`scripts/run_ddp_tok.sh`** — the launcher. Sets the NCCL env (§4), sources the single source of
  truth `scripts/v7_train_args.sh` → `V7_ARGS`, and `exec torchrun --standalone
  --nproc_per_node=$NGPU`. Env overrides: `NGPU, FRAMES_DIR, RESUME, CHECKPOINT_DIR, MAX_STEPS,
  NUM_WORKERS, NCCL_SHM_DISABLE, RESET_SCHEDULE`.

- **`scripts/vast_supervised_run.sh`** — the self-healing supervisor. Run it via `onstart.sh` or
  `setsid bash vast_supervised_run.sh` so it survives SSH disconnect. Behavior:
  - **First launch:** resume WEIGHTS from `$INIT_RESUME` with a **fresh LR schedule** (new horizon).
  - **After a local checkpoint exists:** auto-resume **CONTINUING** (step+optimizer+scheduler) — a
    crash/preemption does NOT re-warmup or rehash.
  - **Streams the latest checkpoint to R2 every cycle** (`$R2_CKPT/tokenizer_latest.pt`) so host
    death is not catastrophic. Stop cleanly with `touch $STOP_FILE`.
  - Required env: `INIT_RESUME, CHECKPOINT_DIR, FRAMES_DIR, MAX_STEPS, R2_CKPT`.

**Data:** pull from R2 (free egress within Cloudflare). `keep-last-N` (default 3) pruning of
`step_*.pt` is in the train script (a disk-full fix). Deferred upgrades that would help: WebDataset
streaming from R2 (kills the 2×-disk untar), `torch.distributed.checkpoint` async_save.

**Schedule footgun:** `--reset-schedule` is a STRICT opt-in (`RESET_SCHEDULE=1`). A plain resume
must CONTINUE the step/LR schedule; only the *continuation-to-a-new-phase* case resets. (An earlier
`[ -n "$RESET_SCHEDULE" ]` fired even on the string "0" and wrongly reset step+LR on every resume.)

---

## 6. The DDP-only code bug (already fixed — know it exists)

The MAE mask token `mask_embed` is **unused whenever `mask_ratio==0`** (`make_mask()` returns None
→ `encode()` gets `mask_embed=None`), so it receives no gradient and DDP's reducer raises *"Parameter
indices which did not receive grad: 0"* and crashes. `mask_ratio==0` is **frequent** (`p_zero_mask`
+ the 2000-step mask warmup ramping from ~0 + `--mse-on-full-frame`). **Silent on single-GPU**
(no grad sync) — which is why Slurm single-GPU training never hit it, and why a DDP smoke test is
mandatory before a big multi-GPU spend.

**Fix (committed 0a553f5):** under DDP, add `loss = loss + 0.0 * base_model.mask_embed.sum()`
before backward so every param participates every step. Keeps `find_unused_parameters=False`
(required by torch.compile's DDPOptimizer) at ~zero cost.

---

## 7. Debugging footguns (each cost time at least once)

- **`pkill -f torchrun` matches its OWN command line** (the pattern is in argv) → kills your own
  SSH/script (exit 143/144). Use the bracket trick `pkill -f "[t]orchrun"` or `pgrep -x`.
- **`timeout N torchrun` without `-k`:** a truly-stuck torchrun ignores SIGTERM and `timeout` hangs
  forever. Always `timeout -k 15 N`.
- **`grep ... || echo HUNG`** labels a *crash* as "HUNG" (grep exits 1 on no-match). Distinguish
  exit **124** (timeout/hang) from other non-zero exits.
- **Leftover hung procs between tests** hold GPUs/ports and poison the next test (false failures).
  Kill (bracket-safely) and confirm GPUs free before re-running.
- **Block-buffered logs** from `setsid bash >log` don't flush — monitor a directly-written marker
  FILE (`echo done > /root/marker` flushes) or `stdbuf -oL`.
- **`pgrep -c` right after a launch can transiently read 0** even when the process is fine. Never
  treat a single 0 as "died" — require 2+ consecutive, and prefer a size/count/marker completion
  signal over a liveness poll. (Same class of bug bit the desktop→R2 upload monitor.)
- **Watch-script backstop epochs:** compute with `date -d`, never by hand (a fat-fingered epoch was
  ~364 days in the past → a cron tried to destroy a box mid-run). Dry-run must log "still training"
  before you install the cron. And never treat a transient `vastai show` API failure as
  "instance gone → self-destruct."

---

## 8. Results established on Vast (so far)

- **YT data helps the tokenizer.** Two independent 8×5090 runs: recon PSNR **24.79→30.85 dB
  (run1, +6.06)** and **25.54→31.56 dB (run2, +6.02)** on the same 6 held-out YT clips, all clips
  up. NOT a clean ablation (undertrained seed conflates "more steps" with "YT distribution"), but a
  clear green light that the YT corpus is good, learnable data and the full stack works end-to-end.
- **The production stack is validated:** 8-way NCCL + the mask_embed fix + torch.compile +
  checkpoint-streaming-to-R2 death-safety all survived a 3.3 h unattended run with zero crashes.
- Artifacts land in `r2:ahriuwu-yt-pretrain/_run{,2,3}/`.

---

## 9. Money hygiene

- **`trap destroy EXIT`** on every script; a forgotten box bills continuously.
- The NCCL-timeout misdiagnosis alone burned Vast credit from ~$50 to ~$14.55. Almost all of it was
  avoidable by *reading the INFO trace and being patient at init*.
- A full DDP smoke (provision→validate→destroy) is only ~40 min / ~$0.50 — always smoke-test a code
  change on 2×4090 before a multi-hundred-dollar 8×5090 run.
- The Vast API key was shared in plaintext in-session at one point — **consider rotating it.**

---

*Sources: `scripts/run_ddp_tok.sh`, `scripts/vast_supervised_run.sh`,
`scripts/train_transformer_tokenizer.py`, `scratchpad/vast_gc_probe.sh`, and memories
`ahriuwu-vast-nccl-fix`, `ahriuwu-ddp-mask-embed-fix`, `ahriuwu-yt-derisk-result`.*
