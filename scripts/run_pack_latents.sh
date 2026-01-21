#!/bin/bash
cd /home/dani/Repos/ahriuwu
source .venv/bin/activate
python -u scripts/pack_latents.py --skip-existing --workers 2 "$@"
