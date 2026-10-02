#!/usr/bin/env bash
# Replay oracle extraction over lol_replays_16_9_772 (CPU partition).
#
# Two stages, same file:
#   1) extract (job array, one game per task):
#        mkdir -p /mnt/nfs/shared/replay-oracle-16.9/_code
#        cp ops/replay_oracle_extract.py /mnt/nfs/shared/replay-oracle-16.9/_code/
#        sbatch -p cpu --time=00:20:00 --array=0-146%8 \
#            -o /mnt/nfs/shared/%x-%A_%a.out ops/replay_oracle_extract.sbatch
#   2) finalize (index + compact summary), after stage 1:
#        sbatch -p cpu --time=00:20:00 --dependency=afterany:<array_jobid> \
#            --export=ALL,ORACLE_STAGE=finalize \
#            -o /mnt/nfs/shared/%x-%j.out ops/replay_oracle_extract.sbatch
#      then copy $OUT/summary/replay_16_9_observations.json* into
#      lanerl_jax/data/modern/oracle/ on the login node.
#
# The repo worktree is not visible on the desktop node, so the extractor is run
# from the NFS copy in $OUT/_code (its sha256 is recorded in every output).
# raw_mem.json / labels.json / item_track.json sha256 are computed inside the job.
#SBATCH --job-name=replay-oracle-16.9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/mnt/nfs/shared
set -euo pipefail
DATASET="${ORACLE_DATASET:-/mnt/nfs/datasets/lol_replays_16_9_772}"
OUT="${ORACLE_OUT:-/mnt/nfs/shared/replay-oracle-16.9}"
SCRIPT="${ORACLE_SCRIPT:-$OUT/_code/replay_oracle_extract.py}"
STAGE="${ORACLE_STAGE:-extract}"
mkdir -p "$OUT"
echo "host=$(hostname) stage=$STAGE dataset=$DATASET out=$OUT script=$SCRIPT sha=$(sha256sum "$SCRIPT" | cut -c1-16)"

if [[ "$STAGE" == "extract" ]]; then
    # game list: sorted NA1_* entries of the dataset root (no globbing inside game dirs)
    mapfile -t GAMES < <(python3 -c "import os,sys; [print(d) for d in sorted(os.listdir(sys.argv[1])) if d.startswith('NA1_')]" "$DATASET")
    IDX="${SLURM_ARRAY_TASK_ID:?run as a job array}"
    if (( IDX >= ${#GAMES[@]} )); then echo "index $IDX >= ${#GAMES[@]} games, nothing to do"; exit 0; fi
    G="${GAMES[$IDX]}"
    echo "task $IDX -> $G"
    python3 "$SCRIPT" extract "$DATASET/$G" --out "$OUT"
elif [[ "$STAGE" == "finalize" ]]; then
    python3 "$SCRIPT" index --out "$OUT" --dataset-root "$DATASET"
    mkdir -p "$OUT/summary"
    SUM="${ORACLE_SUMMARY:-$OUT/summary/replay_16_9_observations.json}"
    python3 "$SCRIPT" summarize --out "$OUT" --dataset-root "$DATASET" --summary "$SUM"
    sz=$(stat -c %s "$SUM")
    echo "summary bytes=$sz"
    if (( sz > 5000000 )); then
        gzip -9 -c "$SUM" > "$SUM.gz" && rm -f "$SUM"
        echo "gzipped -> $(stat -c %s "$SUM.gz") bytes"
    fi
else
    echo "unknown ORACLE_STAGE=$STAGE" >&2; exit 2
fi
