#!/usr/bin/env python3
"""Standalone GarenAgent throughput on an idle GPU (the live 20fps question).
The 13fps sim-eval reading was taken while BC training pegged the same GPU."""
import sys
import time

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from agent_infer import GarenAgent

ck = sys.argv[1] if len(sys.argv) > 1 else "data/phase2_bc_garen_act8775/agent_finetune_latest.pt"
ag = GarenAgent(ck, device="cuda")
lat = [torch.randn(1, 32, 16, 16, device="cuda") for _ in range(240)]
ag.reset()
for i in range(30):
    ag.act_from_latent(lat[i])
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(30, 230):
    ag.act_from_latent(lat[i])
torch.cuda.synchronize()
fps = 200 / (time.perf_counter() - t0)
print(f"standalone agent fps (idle GPU, ctx={ag.context}, bf16={ag.amp}, "
      f"use_actions={ag.use_actions}): {fps:.1f}")
