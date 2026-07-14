#!/usr/bin/env python3
"""Inference smoke tests for GarenAgent (scripts/agent_infer.py).

Uses the REAL v7 tokenizer + REAL replay latents, but init-only (untrained) heads,
so it needs no BC checkpoint. Validates the parts that break silently:
  T0  the v7 latent fold has an exact inverse (pure tensor)
  T1  encode_frame end-to-end: decode a real latent -> frame -> re-encode ~= latent
      (a wrong fold -> garbage correlation; this is what caught the 512x16 bug)
  T2  action-space contract (9 bool abilities, movement in [0,1], finite reward)
  T3  greedy (temperature=0) determinism
  T4  rolling-window buffer: left-pad when short, maxlen=context when full
  T5  a 30-frame run doesn't crash + per-frame timing

Run:  PYTHONPATH=src python scripts/test_agent_infer.py
"""
import sys
sys.path.insert(0, "scripts")
import time
import numpy as np
import torch

from agent_infer import GarenAgent, _dyn_from_tok
from ahriuwu.constants import ABILITY_KEYS

TOK = "rollout_stage/transformer_tokenizer_latest.pt"
LAT = "rollout_stage/NA1_5549995114.pt"  # repo-relative (NFS, resolves on both nodes)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ag = GarenAgent("x", tokenizer_ckpt=TOK, context=16, device=dev, init_only=True)
    print(f"device={dev}  bf16={ag.amp}  context={ag.context}")
    z = torch.load(LAT, map_location=dev, weights_only=True)["latents"].float()  # (N,32,16,16)
    z0 = z[0:1]
    n = 0

    # T0 — fold inverse is exact
    tok = z0.permute(0, 2, 3, 1).reshape(1, 512, 16)          # (1,32,16,16) -> (1,512,16)
    assert torch.allclose(z0, _dyn_from_tok(tok), atol=1e-4), "fold inverse mismatch"
    print("T0 fold inverse round-trip ...................... OK"); n += 1

    # T1 — full encode path: decode(latent) -> frame -> encode(frame) ~= latent
    with torch.no_grad():
        rec = ag.tok.decode(tok, 1)                           # (1,1,3,H,W)
    frame = rec.squeeze().permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()  # (H,W,3)
    z_re = ag.encode_frame(frame)
    assert z_re.shape == (1, 32, 16, 16), f"bad encode shape {tuple(z_re.shape)}"
    assert torch.isfinite(z_re).all(), "non-finite encode"
    corr = torch.corrcoef(torch.stack([z_re.flatten().float(), z0.flatten()]))[0, 1].item()
    print(f"T1 encode(decode(z)) ~= z ...................... OK  (corr={corr:.3f})")
    assert corr > 0.7, f"encode round-trip correlation too low ({corr:.3f}) — fold/encode broken"
    n += 1

    # T2 — action contract
    ag.reset()
    a = ag.act_from_latent(z0, temperature=0.0)
    assert set(a["abilities"]) == set(ABILITY_KEYS), "ability keys"
    assert all(isinstance(v, bool) for v in a["abilities"].values()), "ability not bool"
    mx, my = a["movement"]
    assert -1e-4 <= mx <= 1 + 1e-4 and -1e-4 <= my <= 1 + 1e-4, f"movement out of [0,1]: {a['movement']}"
    assert isinstance(a["reward_pred"], float) and np.isfinite(a["reward_pred"]), "reward"
    print(f"T2 action-space contract ...................... OK  "
          f"(move=({mx:.2f},{my:.2f}) keys={[k for k,v in a['abilities'].items() if v] or '-'})"); n += 1

    # T3 — greedy determinism
    ag.reset(); a1 = ag.act_from_latent(z0, 0.0)
    ag.reset(); a2 = ag.act_from_latent(z0, 0.0)
    assert a1["abilities"] == a2["abilities"] and a1["movement"] == a2["movement"], "greedy nondeterministic"
    print("T3 greedy determinism ......................... OK"); n += 1

    # T4 — window/buffer: pad-when-short + maxlen
    ag.reset(); assert len(ag.buf) == 0
    ag.act_from_latent(z0, 0.0); assert len(ag.buf) == 1
    for i in range(20):
        ag.act_from_latent(z[i:i+1], 0.0)
    assert len(ag.buf) == ag.context, f"buffer maxlen {len(ag.buf)} != {ag.context}"
    print("T4 rolling-window buffer (pad + maxlen) ....... OK"); n += 1

    # T5 — 30-frame run + timing
    ag.reset(); t = time.perf_counter()
    for i in range(30):
        ag.act_from_latent(z[i:i+1], 0.0)
    if dev == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 30 * 1000
    print(f"T5 30-frame run (no crash) .................... OK  ({ms:.1f} ms/frame act-only)"); n += 1

    print(f"\nALL {n}/6 INFERENCE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
