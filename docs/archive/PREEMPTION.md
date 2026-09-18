# Preemption and Checkpoint System

## Two-Mode Preemption

The training scripts support two preemption modes via Unix signals:

### Immediate Mode (SIGUSR1)
- Checked **every optimizer step**
- Saves checkpoint after the current/next optimizer step and exits
- Use for manual preemption when you want a fast, clean stop

### Delayed Mode (SIGTERM)
- Checked **only at checkpoint boundaries**
- Saves checkpoint at the next scheduled boundary and exits
- Used by Slurm preemption (`--signal=B:TERM@120`) and voluntary queue yielding (`should_yield_to_queue`)

### Sending Signals

Use the helper script:

```bash
# Immediate: save after next optimizer step
scripts/preempt.sh immediate [job_id]

# Delayed: save at next checkpoint boundary (default)
scripts/preempt.sh delayed [job_id]
# or just:
scripts/preempt.sh
```

If `job_id` is omitted, the script targets the most recent running job.

You can also send signals directly via `scancel`:

```bash
scancel --signal=USR1 <job_id>   # immediate
scancel --signal=TERM <job_id>   # delayed
```

## Dynamic Checkpoint Intervals

When `--step-save-interval` is set to `0` (default for sbatch), the checkpoint
interval is computed dynamically to target `--checkpoint-minutes` (default: 60)
between saves. The interval is recalculated based on observed optimizer
steps per second:

```
steps_per_second = global_step / elapsed_time
step_save_interval = max(1, int(checkpoint_minutes * 60 * steps_per_second))
```

This automatically adapts to different batch sizes, sequence lengths, and
gradient accumulation settings. Set `--step-save-interval` to a positive
value to override with a fixed interval.

## Shared Infrastructure

Preemption handlers and state are defined in `src/ahriuwu/utils/training.py`:

- `PreemptionState` -- holds `immediate` and `at_checkpoint` threading events
- `install_preemption_handlers(state)` -- installs SIGUSR1 and SIGTERM handlers
- `compute_dynamic_save_interval(...)` -- computes interval from timing data

Both `train_transformer_tokenizer.py` and `train_dynamics.py` use these
shared utilities. Checkpoint saving logic remains in each script since the
save arguments differ per model type.

## Slurm Integration

The sbatch scripts:
1. Set `#SBATCH --signal=B:TERM@120` to get 120s warning before preemption
2. Trap both TERM and USR1 in the shell wrapper, forwarding to the Python PID
3. Use `--step-save-interval 0 --checkpoint-minutes 60` for dynamic ~1hr checkpoints

Voluntary yielding via `should_yield_to_queue()` checks for pending jobs
with `(Resources)` reason and triggers delayed-mode exit at the next checkpoint
boundary.

## Bug Fixes Applied

1. **Step-0 checkpoint spam**: `global_step > 0` guard prevents `0 % N == 0` from
   firing on every batch during the first accumulation window.

2. **Preemption gated behind checkpoint interval**: Immediate mode (SIGUSR1) is
   now checked every optimizer step, not just at checkpoint boundaries.

3. **Interval vs accumulation mismatch**: Dynamic intervals auto-compute based
   on actual optimizer step throughput, so accumulation is accounted for.
