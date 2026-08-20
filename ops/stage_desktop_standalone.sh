#!/bin/bash
# Make the desktop SELF-SUFFICIENT for live inference — no NFS, no login node —
# AND make it impossible for the deployed tree to silently drift from the repo.
#
# The rig at $DEST used to be an unmanaged FORK: 162 diff lines in play_live.py,
# 71 in agent_infer.py, no joint_noop branch, no --desktop, no --gate-bias, and a
# checkpoint the repo documents as invalid. Nothing anyone fixed in git was in
# the thing that actually played. So this script now stamps the exact commit into
# a VERSION file that play_live.py PRINTS AT STARTUP and records in every
# session's meta.json. If the banner does not match the commit you think you
# deployed, you are running something else.
#
#   ssh desktop 'bash /mnt/nfs/projects/ahriuwu/ops/stage_desktop_standalone.sh'
#
# Run ON the desktop (Linux boot) while NFS is still up. Afterwards $DEST works
# with danilogin and NFS both down. The conda 'ml' env already lives on the
# desktop-local /home.
#
# Env overrides:
#   BC_SRC=<path>   Phase-2 checkpoint to deploy (default below)
#   TOK_SRC=<path>  tokenizer checkpoint
#   DEST=<path>     deploy root
set -euo pipefail

SRC=${SRC:-/mnt/nfs/projects/ahriuwu}               # NFS repo (source)
DEST=${DEST:-/mnt/storage/ahriuwu-live}             # desktop-local durable HDD
PY=${PY:-/home/dani/miniconda3/envs/ml/bin/python}  # desktop-local env
TOK_SRC=${TOK_SRC:-$SRC/rollout_stage/transformer_tokenizer_latest.pt}
# Phase-2 BC checkpoint. Chosen by measured offline liveness on 3600 frames of
# real recorded latents at temperature 1.0 (3 replays x 600 frames x 2 seeds,
# identical inputs across candidates):
#
#   ckpt                 clicks/s  uniq cells  top-cell  non-AA casts  step
#   phase2_bc_clicks       2.70       60.0      0.057       1.36%     102420  <- deployed
#   phase2_from_vast       2.45       49.3      0.060       0.67%     100329  <- fallback
#   phase2_parity          2.06       45.7      0.071       1.08%      55216
#
# All three are alive (no one-cell collapse). bc_clicks wins on every liveness
# axis at once and its targets are screen-centred (0.498, 0.492) where parity's
# skew to (0.581, 0.397). CAVEAT: bc_clicks is the FROZEN lineage, whose
# action_embed was fitted to a different movement target and cannot adapt
# (WIRING_AUDIT 1.1) -- and liveness metrics cannot tell competence from noise,
# so a corrupted embedding could score high precisely by being noisier. If it
# looks erratic on the day, fall back to the unfrozen joint_noop one:
#   BC_SRC=$SRC/data/phase2_from_vast/vast_step90000.pt bash ops/stage_desktop_standalone.sh
BC_SRC=${BC_SRC:-$SRC/data/phase2_bc_clicks/agent_finetune_latest.pt}

[ -f "$TOK_SRC" ] || { echo "FATAL: tokenizer not found: $TOK_SRC" >&2; exit 1; }
[ -f "$BC_SRC" ]  || { echo "FATAL: phase2 ckpt not found: $BC_SRC" >&2; exit 1; }

COMMIT=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo UNKNOWN)
DIRTY=$(git -C "$SRC" status --porcelain -- src scripts 2>/dev/null | head -20)
echo "[stage] dest=$DEST  commit=${COMMIT:0:12}"
[ -n "$DIRTY" ] && echo "[stage] WARNING: src/ or scripts/ is DIRTY at stage time:" && echo "$DIRTY"

mkdir -p "$DEST/checkpoints"

# PRESERVE the rig's measured mouse calibration across a re-stage. It is the one
# artefact that only exists on the live box (measured against the real screen)
# and rsync --delete would take it out.
CAL="$DEST/scripts/keysender/mouse_calibration.json"
SAVED=""
if [ -f "$CAL" ]; then
  SAVED=$(mktemp)
  cp "$CAL" "$SAVED"
  echo "[stage] preserving measured calibration: $(cat "$CAL" | tr -d '\n')"
fi

echo "[stage] rsync code (src, scripts, tests)..."
rsync -a --delete \
  --exclude='.git' --exclude='wandb' --exclude='__pycache__' --exclude='*.pyc' \
  "$SRC/src" "$SRC/scripts" "$SRC/tests" "$SRC/pyproject.toml" "$DEST/"

if [ -n "$SAVED" ]; then
  cp "$SAVED" "$CAL"
  rm -f "$SAVED"
  echo "[stage] calibration restored -> $CAL"
else
  echo "[stage] NOTE: no mouse calibration on this box. Run ONCE before playing:"
  echo "[stage]   \$PY \$AHRIUWU/scripts/keysender/calibrate_mouse.py --host \$PI"
  echo "[stage]   (until then the sender uses the built-in fallback span 649x367)"
fi

echo "[stage] copying checkpoints to local disk..."
cp -f "$TOK_SRC" "$DEST/checkpoints/tokenizer_v7.pt"
cp -f "$BC_SRC"  "$DEST/checkpoints/phase2_bc.pt"
TOK_SHA=$(sha256sum "$DEST/checkpoints/tokenizer_v7.pt" | cut -c1-16)
BC_SHA=$(sha256sum "$DEST/checkpoints/phase2_bc.pt" | cut -c1-16)
echo "[stage]   tokenizer <- $TOK_SRC  ($TOK_SHA)"
echo "[stage]   phase2 BC <- $BC_SRC   ($BC_SHA)"

# --- the anti-drift stamp ----------------------------------------------------
cat > "$DEST/VERSION" <<EOF
commit=$COMMIT
staged_at=$(date -Is)
staged_from=$SRC
staged_by=$(whoami)@$(hostname)
dirty=$([ -n "$DIRTY" ] && echo yes || echo no)
phase2_ckpt=$BC_SRC
phase2_sha256_16=$BC_SHA
tokenizer_ckpt=$TOK_SRC
tokenizer_sha256_16=$TOK_SHA
EOF
echo "[stage] VERSION:"; sed 's/^/[stage]   /' "$DEST/VERSION"

echo "[stage] writing launch wrappers..."
cat > "$DEST/env.sh" <<EOF
# source this — all paths local, nothing on NFS/login
export AHRIUWU=$DEST
export PYTHONPATH=$DEST/src:$DEST/scripts
export PY=$PY
export TOK=$DEST/checkpoints/tokenizer_v7.pt
export BC=$DEST/checkpoints/phase2_bc.pt
export PI=\${PI:-192.168.1.144}
EOF

cat > "$DEST/preflight.sh" <<EOF
#!/bin/bash
source $DEST/env.sh
\$PY \$AHRIUWU/scripts/play_live_preflight.py \\
  --phase2-ckpt \$BC --tokenizer-ckpt \$TOK --hid-host \$PI "\$@"
EOF
chmod +x "$DEST/preflight.sh"

# temperature 1.0 is NOT a preference: greedy decode is a measured dead policy
# (0.00 clicks/s, 1 movement cell, 0 casts) on every checkpoint on disk.
cat > "$DEST/play.sh" <<EOF
#!/bin/bash
source $DEST/env.sh
\$PY \$AHRIUWU/scripts/play_live.py \\
  --phase2-ckpt \$BC --tokenizer-ckpt \$TOK \\
  --inject \${INJECT:-dry} --hid-host \$PI \\
  --movement-mode \${MOVE:-mouse} --target-fps \${FPS:-20} --temperature 1.0 "\$@"
EOF
chmod +x "$DEST/play.sh"

echo "[stage] VERIFY: load both checkpoints + one inference on a synthetic frame, LOCAL paths only..."
source "$DEST/env.sh"
"$PY" - <<'PYEOF'
import os, sys, numpy as np
sys.path.insert(0, os.environ["AHRIUWU"]+"/scripts")
sys.path.insert(0, os.environ["AHRIUWU"]+"/src")
from agent_infer import GarenAgent
ag = GarenAgent(os.environ["BC"], tokenizer_ckpt=os.environ["TOK"], device="cuda")
ag.reset()
f = np.random.rand(352,352,3).astype(np.float32)
a = ag.act_from_latent(ag.encode_frame(f), temperature=1.0)
print(f"VERIFY OK: use_actions={ag.use_actions} "
      f"movement_mode={getattr(ag.policy,'movement_mode','axis')} "
      f"gated={getattr(ag.policy,'movement_gate',False)} "
      f"move={tuple(round(x,2) for x in a['movement'])} bf16={ag.amp}")
PYEOF

echo
echo "[stage] DONE. Standalone tree at $DEST, stamped ${COMMIT:0:12}."
echo "  calibrate (once): \$PY $DEST/scripts/keysender/calibrate_mouse.py --host \$PI"
echo "  preflight       : $DEST/preflight.sh --inject hid"
echo "  dry run         : INJECT=dry $DEST/play.sh"
echo "  live            : INJECT=hid $DEST/play.sh"
