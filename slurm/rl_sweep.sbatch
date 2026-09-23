#!/bin/bash
#SBATCH --job-name=rl-sweep
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/train/%x-%j.out
#
# STAGE A OF THE HYPERPARAMETER SWEEP: a screen, not a stab.
#
# Why a grid and not one lr change: every hyperparameter result this project
# has is n=1. "lr 1e-5 -> 3e-4 will probably work" is almost certainly true and
# is still not a measurement, and a single arm cannot separate the lr from the
# seed. So: a shared 300-update budget, every arm launched from the SAME
# entrypoint (slurm/rl_train.sbatch, invoked as a plain shell script -- the
# #SBATCH lines are comments to bash), one variable per arm against a control.
#
# ARM 0 IS THE CONTROL, and it is not the old run. The trainer changed on
# 2026-09-23 in three ways that all touch the curve -- staggered episode
# phases, `critic_lr` actually applied, `target_kl` actually enforced -- so
# RL-002's 45.979 CS is NOT a baseline for anything below. Arm 0 pins the two
# behavioural changes back to their pre-2026-09-23 values (critic at the actor
# lr, KL unbounded) so the infrastructure change itself is measured rather than
# assumed harmless. The phase stagger cannot be pinned back and is common to
# every arm; that one is a confound this sweep does not resolve.
#
# 300 updates is chosen from the metric, not from patience: 256 envs over a
# 141-update episode yields ~1.8 completed episodes per update once the
# staggered first episode is past, so updates 141-300 carry ~290 full-episode
# cs@10min samples. At the measured sd of 7.3 CS that is a standard error near
# 0.4 -- enough to rank arms. It is NOT enough to call a winner: 300 updates is
# ~2 episodes of learning per env, and run-to-run (seed) spread is untested.
# Stage B re-runs the top two arms at 600 updates on additional seeds, and only
# that is allowed to be quoted.
set -uo pipefail                # NOT -e: one arm failing must not kill the rest
# REPO may be a SNAPSHOT worktree (`git worktree add ... <sha>`) so that edits
# to the live tree during a multi-hour sweep cannot make later arms run
# different code from earlier ones -- which happened once (sweep 1271 was
# cancelled for exactly that). The venv is not in git: symlink `.venv-gpu`
# in the snapshot to the live tree's. Runs are written to OUT_ROOT so the
# snapshot can be deleted afterwards.
REPO=${REPO:-/mnt/nfs/projects/ahriuwu-lanerl-jax}
export REPO
cd "$REPO"

UPDATES=${UPDATES:-300}
export SWEEP=${SWEEP:-sweepB}
export OUT_ROOT=${OUT_ROOT:-/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/train}
GROUP=${GROUP:-$SWEEP-$(date -u +%Y%m%d)}
export WANDB_RUN_GROUP="$GROUP"     # wandb reads this; the helper has no --group
export WANDB_INIT_TIMEOUT=60        # an unreachable wandb must not hang the GPU
export WANDB_HTTP_TIMEOUT=30

# arm name | extra args
ARMS=(
  # Sweep B (2026-09-23): the corrected simulator (SPELL-001..005, ENT-01/02,
  # COLL-004, SLOT-001) and the corrected trainer (PPO-01 head masking,
  # PPO-04/05/06/07). Nothing measured in sweep A is comparable. Dropped
  # a5/a6: lr 3e-4 already blew the critic up in chunk 0 (PPO-04); 1e-3
  # is not a candidate. Added the value-clip arms (PPO-02): the 0.2
  # absolute clip on O(10-30) returns is the leading critic hypothesis now
  # that a0-vs-a1 showed critic_lr was not.
  "a0-control|--lr 1e-5 --critic-lr 1e-5 --target-kl 1e9"
  "a1-base|--lr 1e-5"
  "a2-lr1e4|--lr 1e-4"
  "a3-lr3e4|--lr 3e-4"
  "a4-lr1e4-ent01|--lr 1e-4 --entropy-coef 0.01"
  "a7-noclip|--lr 1e-5 --no-value-clip"
  "a8-lr1e4-noclip|--lr 1e-4 --no-value-clip"
)

echo "=== sweep $GROUP | ${#ARMS[@]} arms x $UPDATES updates | $(date -u) ==="
echo "git $(git rev-parse --short HEAD) | repo $REPO | out $OUT_ROOT"
for entry in "${ARMS[@]}"; do
    name=${entry%%|*}
    args=${entry#*|}
    echo
    echo "--- arm $name : $args --- $(date -u +%H:%M:%S)"
    # shellcheck disable=SC2086
    bash slurm/rl_train.sbatch --tag "$SWEEP-$name" --updates "$UPDATES" \
        --out-root "$OUT_ROOT" --wandb-tags "$SWEEP" "$name" \
        --notes "sweep $SWEEP arm $name ($args); group $GROUP" $args
    echo "--- arm $name exit $? --- $(date -u +%H:%M:%S)"
done
echo
echo "=== sweep $GROUP done $(date -u) ==="
python3 - <<'PY'
# One table over the arms, read back from the run dirs the arms wrote. Printed
# here rather than reconstructed later so the job's own log is self-contained.
import json, glob, os
rows = []
for m in glob.glob(os.path.join(os.environ.get("OUT_ROOT", "lanerl_jax/runs/train"),
                               os.environ.get("SWEEP", "sweepA") + "-*", "manifest.json")):
    d = json.load(open(m))
    r = d.get("results") or {}
    cs = r.get("cs_at_10min") or []
    rows.append((os.path.basename(os.path.dirname(m)),
                 d["config"]["ppo"].get("lr"),
                 d["config"]["ppo"].get("entropy_coef"),
                 len(cs),
                 (sum(cs[-50:]) / len(cs[-50:])) if cs else float("nan"),
                 r.get("entropy_last"), r.get("wall_s"),
                 r.get("diverged_at_update")))
rows.sort(key=lambda x: (-(x[4] if x[4] == x[4] else -1e9)))
print(f"{'arm':<26}{'lr':>9}{'ent':>8}{'n_ep':>6}{'CS(last50)':>12}"
      f"{'entropy':>9}{'wall_s':>8}  diverged")
for r in rows:
    print(f"{r[0]:<26}{r[1]:>9.0e}{r[2]:>8.4f}{r[3]:>6}{r[4]:>12.2f}"
          f"{(r[5] or float('nan')):>9.3f}{(r[6] or 0):>8.0f}  {r[7]}")
PY
