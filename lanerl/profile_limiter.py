"""Which is the bottleneck: the servers, the policy forward, or the PPO update?"""
import os, sys, time, json, socket, subprocess
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from lanerl_rl import constants as C
from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import ACTION_KEYS, DualClipPPO, PPOConfig, RecurrentRolloutBuffer

dev = "cuda" if torch.cuda.is_available() else "cpu"
mc = ModelConfig()
print(f"device={dev}  entity={mc.entity_dim} self={mc.self_dim} global={mc.global_dim}")

# ---- 1. PPO update throughput (the learner) -------------------------------
policy = LanePolicy(mc).to(dev)
learner = DualClipPPO(policy, PPOConfig())
T, B = 256, 4
def synth():
    buf = RecurrentRolloutBuffer(T, B, mc, device=dev)
    st = policy.initial_state(B, device=dev)
    for _ in range(T):
        obs = {k: torch.zeros(B, *s, device=dev) for k, s in
               (("entities",(mc.n_slots,mc.entity_dim)),("self_vec",(mc.self_dim,)),
                ("global_vec",(mc.global_dim,)),("priv_entities",(mc.n_slots,mc.entity_dim)),
                ("priv_vec",(mc.priv_dim,)))}
        obs["entity_pad_mask"]=torch.zeros(B,mc.n_slots,dtype=torch.bool,device=dev)
        obs["priv_pad_mask"]=torch.zeros(B,mc.n_slots,dtype=torch.bool,device=dev)
        masks={"button":torch.ones(B,mc.n_buttons,dtype=torch.bool,device=dev),
               "move_x":torch.ones(B,mc.n_move_bins,dtype=torch.bool,device=dev),
               "move_z":torch.ones(B,mc.n_move_bins,dtype=torch.bool,device=dev),
               "target":torch.ones(B,mc.n_slots,dtype=torch.bool,device=dev)}
        act={k:torch.zeros(B,dtype=torch.long,device=dev) for k in ACTION_KEYS}
        buf.add(obs=obs,masks=masks,action=act,log_prob=torch.zeros(B,device=dev),
                value=torch.zeros(B,device=dev),reward=torch.zeros(B,device=dev),
                done=torch.zeros(B,device=dev),reset=torch.zeros(B,device=dev),state=st)
    buf.finish(torch.zeros(B,device=dev), learner.cfg.gamma, learner.cfg.gae_lambda)
    return buf
buf=synth(); learner.update(buf)                      # warm up
n=5; t0=time.perf_counter()
for _ in range(n): learner.update(synth())
dt=(time.perf_counter()-t0)/n
print(f"\nLEARNER  one PPO update over {T}x{B}: {dt:.2f}s  -> {3600/dt:.0f} updates/h ceiling")

# ---- 2. policy forward (the actor) ---------------------------------------
st = policy.initial_state(12, device=dev)
b = {k: torch.zeros(12,1,*s, device=dev) for k,s in
     (("entities",(mc.n_slots,mc.entity_dim)),("self_vec",(mc.self_dim,)),
      ("global_vec",(mc.global_dim,)),("priv_entities",(mc.n_slots,mc.entity_dim)),
      ("priv_vec",(mc.priv_dim,)))}
b["entity_pad_mask"]=torch.zeros(12,1,mc.n_slots,dtype=torch.bool,device=dev)
b["priv_pad_mask"]=torch.zeros(12,1,mc.n_slots,dtype=torch.bool,device=dev)
b["action_masks"]={"button":torch.ones(12,1,mc.n_buttons,dtype=torch.bool,device=dev),
   "move_x":torch.ones(12,1,mc.n_move_bins,dtype=torch.bool,device=dev),
   "move_z":torch.ones(12,1,mc.n_move_bins,dtype=torch.bool,device=dev),
   "target":torch.ones(12,1,mc.n_slots,dtype=torch.bool,device=dev)}
with torch.no_grad():
    for _ in range(20): policy.act(b, st)
    t0=time.perf_counter()
    for _ in range(200): policy.act(b, st)
    fwd=(time.perf_counter()-t0)/200
print(f"ACTOR    batched forward for 12 envs: {fwd*1000:.2f} ms -> {12/fwd:,.0f} decisions/s ceiling")
print(f"         (a 256-step rollout x 12 envs needs {256*fwd:.1f}s of forward)")
