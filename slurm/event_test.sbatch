#!/bin/bash
set -eu
echo "CANARY PASSED: task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
# Keep all tasks healthy through the mandatory 180-second startup watch.
sleep 300
case "$SLURM_ARRAY_TASK_ID" in
  0) echo 'expected success'; exit 0 ;;
  1) printf 'int main( {\n' | gcc -x c -fsyntax-only - ;;
  2) echo 'expected application failure'; exit 7 ;;
  3) echo 'waiting for Slurm time limit'; sleep 600 ;;
esac
