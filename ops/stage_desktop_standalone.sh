#!/bin/bash
# Make the desktop SELF-SUFFICIENT for live inference — no NFS, no login node.
# Copies the repo code + the two checkpoints inference needs to a desktop-LOCAL
# drive, writes a launch wrapper that points only at local paths, and verifies a
# dry inference runs. Run ON the desktop (Linux boot) while NFS is still up.
#
#   bash /mnt/nfs/projects/ahriuwu/scripts/stage_desktop_standalone.sh
#
# After this, /mnt/storage/ahriuwu-live is fully local: it works with danilogin
# and NFS both down. The conda 'ml' env already lives on the desktop-local /home.
set -euo pipefail

SRC=/mnt/nfs/projects/ahriuwu                       # NFS repo (source)
DEST=/mnt/storage/ahriuwu-live                      # desktop-local durable HDD
PY=/home/dani/miniconda3/envs/ml/bin/python         # desktop-local env (/home is per-node)
TOK_SRC=$SRC/rollout_stage/transformer_tokenizer_latest.pt
# Phase-2 BC checkpoint: prefer the new gated action-model, fall back to act8775.
BC_SRC=$SRC/data/phase2_bc_gate1060/agent_finetune_latest.pt
[ -f "$BC_SRC" ] || BC_SRC=$SRC/data/phase2_bc_garen_act8775/agent_finetune_latest.pt

echo "[stage] dest=$DEST"
mkdir -p "$DEST/checkpoints"

echo "[stage] copying code (src, scripts) — excluding scratchpad/.git/wandb/checkpoints symlink..."
rsync -a --delete \
  --exclude='.git' --exclude='wandb' --exclude='checkpoints' \
  --exclude='__pycache__' \
  "$SRC/src" "$SRC/scripts" "$SRC/pyproject.toml" "$DEST/" 2>/dev/null || \
  rsync -a --delete --exclude='.git' --exclude='wandb' --exclude='checkpoints' \
    --exclude='__pycache__' "$SRC/src" "$SRC/scripts" "$DEST/"

echo "[stage] copying checkpoints to local disk..."
cp -f "$TOK_SRC" "$DEST/checkpoints/tokenizer_v7.pt"
cp -f "$BC_SRC"  "$DEST/checkpoints/phase2_bc.pt"
echo "[stage]   tokenizer <- $TOK_SRC"
echo "[stage]   phase2 BC <- $BC_SRC"

echo "[stage] writing launch wrappers..."
cat > "$DEST/env.sh" <<EOF
# source this — all paths local, nothing on NFS/login
export AHRIUWU=$DEST
export PYTHONPATH=$DEST/src:$DEST/scripts
export PY=$PY
export TOK=$DEST/checkpoints/tokenizer_v7.pt
export BC=$DEST/checkpoints/phase2_bc.pt
EOF

cat > "$DEST/preflight.sh" <<EOF
#!/bin/bash
source $DEST/env.sh
\$PY \$AHRIUWU/scripts/play_live_preflight.py \\
  --phase2-ckpt \$BC --tokenizer-ckpt \$TOK "\$@"
EOF
chmod +x "$DEST/preflight.sh"

cat > "$DEST/play.sh" <<EOF
#!/bin/bash
source $DEST/env.sh
\$PY \$AHRIUWU/scripts/play_live.py \\
  --phase2-ckpt \$BC --tokenizer-ckpt \$TOK \\
  --inject \${INJECT:-dry} --hid-host \${PI:-192.168.1.144} \\
  --target-fps 20 --temperature 1.0 "\$@"
EOF
chmod +x "$DEST/play.sh"

echo "[stage] VERIFY: load both checkpoints + dry inference on a synthetic frame, LOCAL paths only..."
source "$DEST/env.sh"
"$PY" - <<'PYEOF'
import os, sys, numpy as np, torch
sys.path.insert(0, os.environ["AHRIUWU"]+"/scripts")
sys.path.insert(0, os.environ["AHRIUWU"]+"/src")
from agent_infer import GarenAgent
ag = GarenAgent(os.environ["BC"], tokenizer_ckpt=os.environ["TOK"], device="cuda")
ag.reset()
f = np.random.rand(352,352,3).astype(np.float32)
a = ag.act_from_latent(ag.encode_frame(f), temperature=1.0)
gated = getattr(ag.policy,"movement_gate",False)
print(f"VERIFY OK: use_actions={ag.use_actions} gated={gated} "
      f"move={tuple(round(x,2) for x in a['movement'])} bf16={ag.amp}")
PYEOF

echo "[stage] DONE. Standalone tree at $DEST (NFS/login can now go down)."
echo "  dry test : INJECT=dry  $DEST/play.sh --capture-region 0,0,1920,1080"
echo "  live     : INJECT=hid  $DEST/play.sh --capture-region <x,y,w,h>"
