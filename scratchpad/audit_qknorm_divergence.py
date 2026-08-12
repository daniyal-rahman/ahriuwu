"""VIBE AUDIT probe: quantify the eval-time QKNorm config drift.

scripts/eval_dynamics.py:302 reads args["use_qk_norm"], but scripts/train_dynamics.py
only ever writes args["no_qk_norm"]. The key is therefore ALWAYS absent and the
.get(..., False) default silently builds a no-QKNorm model, which ALSO flips the
attention scale (dynamics.py:176: scale = 1.0 if use_qk_norm else head_dim**-0.5).

This script builds the model both ways from the SAME checkpoint and measures how
far apart their outputs are on identical input.

Read-only. CPU-only. Does not touch any GPU.
"""
import sys
import torch

sys.path.insert(0, "/srv/nfs/projects/ahriuwu/src")
from ahriuwu.models.dynamics import create_dynamics  # noqa: E402

CKPT = "/srv/nfs/projects/ahriuwu/rollout_stage/desktop_resume_8775.pt"


def build(ck, use_qk_norm, tag):
    a = ck["args"]
    mc = ck["model_config"]
    m = create_dynamics(
        a.get("model_size", "small"),
        latent_dim=a.get("latent_dim", 32),
        use_actions=mc["use_actions"],
        use_qk_norm=use_qk_norm,
        soft_cap=a.get("soft_cap", 0.0),
        num_register_tokens=a.get("num_register_tokens", 0),
        num_kv_heads=a.get("num_kv_heads", 0),
    )
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    dropped = sum(sd[k].numel() for k in unexpected)
    print(f"  [{tag}] use_qk_norm={use_qk_norm}  attn scale="
          f"{m.blocks[0].attn.scale:.6f}  missing={len(missing)} "
          f"unexpected={len(unexpected)}  trained params DROPPED={dropped:,}")
    return m.eval()


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"checkpoint: {CKPT}")
    print(f"  model_config['use_qk_norm'] = {ck['model_config']['use_qk_norm']}  "
          f"(GROUND TRUTH)")
    print(f"  'use_qk_norm' in args        = {'use_qk_norm' in ck['args']}")
    print(f"  'no_qk_norm'  in args        = {'no_qk_norm' in ck['args']}"
          f"  value={ck['args'].get('no_qk_norm')}")
    print()

    m_correct = build(ck, True, "TRUE  (model_config)")
    m_eval = build(ck, False, "EVAL  (args.get default)")

    torch.manual_seed(0)
    B, T, C, H, W = 1, 2, ck["model_config"]["latent_dim"], 16, 16
    z = torch.randn(B, T, C, H, W)
    tau = torch.full((B,), 0.5)
    acts = {"movement": torch.rand(B, T, 2)}
    from ahriuwu.constants import ABILITY_KEYS
    acts.update({k: torch.zeros(B, T, dtype=torch.long) for k in ABILITY_KEYS})

    with torch.no_grad():
        o1 = m_correct(z, tau, actions=acts)
        o2 = m_eval(z, tau, actions=acts)
    o1 = o1[0] if isinstance(o1, tuple) else o1
    o2 = o2[0] if isinstance(o2, tuple) else o2
    if isinstance(o1, dict):
        o1, o2 = o1["latent"], o2["latent"]

    d = (o1 - o2)
    rel = d.norm() / o1.norm()
    cos = torch.nn.functional.cosine_similarity(
        o1.flatten().unsqueeze(0), o2.flatten().unsqueeze(0)).item()
    print("\n--- OUTPUT DIVERGENCE on identical input ---")
    print(f"  ||correct||          = {o1.norm():.4f}")
    print(f"  ||eval-built||       = {o2.norm():.4f}")
    print(f"  relative L2 error    = {rel:.4f}  ({100*rel:.1f}%)")
    print(f"  cosine similarity    = {cos:.4f}")
    print("\n  (cosine ~1.0 would mean the drift is harmless; anything well")
    print("   below 1.0 means every metric from eval_dynamics.py is computed")
    print("   on a materially different model than the one that was trained.)")


if __name__ == "__main__":
    main()
