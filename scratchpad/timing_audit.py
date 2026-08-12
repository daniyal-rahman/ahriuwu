#!/usr/bin/env python3
"""Rigorous latency audit of the live inference loop.

Prior numbers were unreliable: one benchmark skipped the tokenizer encode
entirely (claimed 26 fps), and the in-session HUD numbers were taken while the
input stream was frozen and with no stated warmup/synchronisation. This measures
each stage with explicit CUDA sync, real warmup, realistic inputs, and reports
percentiles — plus a context-length sweep that directly tests whether the
16-frame re-forward (vs a KV cache) is what costs the budget.

Run with the GPU otherwise IDLE.
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from agent_infer import GarenAgent
from ahriuwu.constants import ABILITY_KEYS

CK = "/mnt/storage/ahriuwu-live/checkpoints"
N_WARM, N_ITER = 25, 60
BUDGET_MS = 50.0            # 20 fps


def stats(x):
    x = np.array(x)
    return f"med={np.median(x):6.2f} p95={np.percentile(x,95):6.2f} min={x.min():6.2f}"


def sync():
    torch.cuda.synchronize()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free, tot = torch.cuda.mem_get_info()
    print(f"VRAM free {free/1e9:.1f}/{tot/1e9:.1f} GB  (other jobs on the GPU will skew everything)")
    ag = GarenAgent(f"{CK}/phase2_bc.pt", tokenizer_ckpt=f"{CK}/tokenizer_v7.pt", device="cuda")
    dev = ag.device
    frame = np.random.rand(720, 1280, 3).astype(np.float32)   # real stream size

    # ---------- warmup ----------
    ag.reset()
    for _ in range(N_WARM):
        ag.act_from_latent(ag.encode_frame(frame), temperature=1.0)
    sync()

    # ---------- stage timing ----------
    t_resize, t_enc, t_noise, t_dyn, t_heads, t_total = ([] for _ in range(6))
    ag.reset()
    for _ in range(N_ITER):
        s0 = time.perf_counter()
        import cv2
        f352 = cv2.resize(frame, (352, 352), interpolation=cv2.INTER_AREA)
        s1 = time.perf_counter()
        x = torch.from_numpy(f352).float().permute(2, 0, 1).unsqueeze(0).to(dev)
        with ag._ac():
            lat = ag.tok.encode(x)["latent"]
        z = lat.reshape(1, 16, 16, -1).permute(0, 3, 1, 2).contiguous().float()
        sync(); s2 = time.perf_counter()

        ag.buf.append(z)
        w = list(ag.buf)
        while len(w) < ag.context:
            w.insert(0, w[0])
        z0 = torch.stack([t.squeeze(0) for t in w], 0).unsqueeze(0).to(dev)
        B, T = z0.shape[:2]
        tau = ag.tau_ctx + torch.rand(B, T, device=dev) * (1 - ag.tau_ctx)
        z_tau, _ = ag.sched.add_noise(z0, tau)
        sync(); s3 = time.perf_counter()

        d1 = torch.ones(B, dtype=torch.long, device=dev)
        with ag._ac():
            _, agent_out = ag.dyn(z_tau, tau, step_size=d1, actions=None)
        sync(); s4 = time.perf_counter()

        h = agent_out[:, -1:, :]
        with ag._ac():
            ag.policy(h)
            ag.policy.sample(h, temperature=1.0, prev_movement_idx=None)
            if getattr(ag.policy, "movement_gate", False):
                ag.policy.gate_logits(h)
            ag.reward.predict(h)
        sync(); s5 = time.perf_counter()

        t_resize.append((s1 - s0) * 1e3); t_enc.append((s2 - s1) * 1e3)
        t_noise.append((s3 - s2) * 1e3);  t_dyn.append((s4 - s3) * 1e3)
        t_heads.append((s5 - s4) * 1e3);  t_total.append((s5 - s0) * 1e3)

    print(f"\n--- per-stage, {N_ITER} iters, CUDA-synced, ctx={ag.context} (ms) ---")
    for name, v in [("cv2 resize", t_resize), ("tokenizer encode", t_enc),
                    ("add_noise", t_noise), ("DYNAMICS forward", t_dyn),
                    ("policy+reward heads", t_heads), ("TOTAL", t_total)]:
        share = np.median(v) / np.median(t_total) * 100
        print(f"  {name:22s} {stats(v)}   {share:5.1f}% of frame")
    med = np.median(t_total)
    print(f"  => {1000/med:.1f} fps   (budget {BUDGET_MS:.0f}ms @20fps: "
          f"{'OK' if med <= BUDGET_MS else f'OVER by {med-BUDGET_MS:.0f}ms'})")

    # ---------- context-length sweep: does the 16-frame re-forward dominate? ----------
    print(f"\n--- DYNAMICS forward vs context length (tests the KV-cache hypothesis) ---")
    base = None
    for T in (1, 2, 4, 8, 16):
        zz = torch.randn(1, T, 32, 16, 16, device=dev)
        tt = ag.tau_ctx + torch.rand(1, T, device=dev) * (1 - ag.tau_ctx)
        zn, _ = ag.sched.add_noise(zz, tt)
        d1 = torch.ones(1, dtype=torch.long, device=dev)
        for _ in range(8):
            with ag._ac():
                ag.dyn(zn, tt, step_size=d1, actions=None)
        sync()
        ts = []
        for _ in range(25):
            s = time.perf_counter()
            with ag._ac():
                ag.dyn(zn, tt, step_size=d1, actions=None)
            sync()
            ts.append((time.perf_counter() - s) * 1e3)
        m = np.median(ts)
        base = base or m
        print(f"  ctx={T:2d}: {stats(ts)}   {m/base:4.1f}x vs ctx=1")
    print("\nread: if ctx=16 is ~linear in T, per-frame KV caching would cut the")
    print("dominant cost to roughly the ctx=1 number; if flat, attention isn't the driver.")


if __name__ == "__main__":
    main()
