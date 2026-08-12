"""VIBE AUDIT probe: quantify what `strict=False` silently discards.

Read-only. CPU-only. Reproduces the exact model-construction call used by
scripts/eval_trade_prediction.py and scripts/eval_dream_quality.py and reports
how many checkpoint tensors would be silently dropped.

Usage:
  CUDA_VISIBLE_DEVICES="" python scratchpad/audit_strict_false_probe.py
"""
import sys
import torch

sys.path.insert(0, "/srv/nfs/projects/ahriuwu/src")
from ahriuwu.models.transformer_tokenizer import create_transformer_tokenizer  # noqa: E402

CKPT = "/srv/nfs/projects/ahriuwu/data/tokenizer_v7_yt/transformer_tokenizer_latest.pt"


def report(tag, model, sd):
    msd = model.state_dict()
    missing, unexpected = [], []
    shape_mismatch = []
    for k, v in msd.items():
        if k not in sd:
            missing.append(k)
        elif tuple(sd[k].shape) != tuple(v.shape):
            shape_mismatch.append((k, tuple(sd[k].shape), tuple(v.shape)))
    for k in sd:
        if k not in msd:
            unexpected.append(k)

    n_ck = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    dropped = sum(sd[k].numel() for k, _, _ in shape_mismatch)
    dropped += sum(sd[k].numel() for k in unexpected if hasattr(sd[k], "numel"))

    print(f"\n=== {tag} ===")
    print(f"  checkpoint tensors: {len(sd)}  ({n_ck/1e6:.1f}M params)")
    print(f"  model tensors:      {len(msd)}")
    print(f"  MISSING in ckpt (stay RANDOM): {len(missing)}")
    print(f"  UNEXPECTED in ckpt (DISCARDED): {len(unexpected)}")
    print(f"  SHAPE MISMATCH (DISCARDED):    {len(shape_mismatch)}")
    print(f"  >>> trained params silently discarded: {dropped/1e6:.1f}M "
          f"({100*dropped/max(n_ck,1):.1f}% of checkpoint)")
    for k, a, b in shape_mismatch[:6]:
        print(f"      shape {k}: ckpt{a} vs model{b}")
    for k in missing[:6]:
        print(f"      missing {k}")
    return len(missing) + len(unexpected) + len(shape_mismatch)


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    cfg = ck["model_config"]
    args = ck.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)

    print("checkpoint model_config:", {k: cfg[k] for k in sorted(cfg)})

    # --- How scripts/eval_trade_prediction.py:149-152 builds it ---
    model_size = args.get("model_size", "small")
    use_rope = args.get("use_rope", True)
    m_bad = create_transformer_tokenizer(model_size, use_rope=use_rope)
    report("eval_trade_prediction.py:149  create_transformer_tokenizer"
           f"('{model_size}', use_rope={use_rope})  + strict=False", m_bad, sd)

    # --- How scripts/pretokenize_replay_v7.py:38 (load_v7) builds it (correct) ---
    from ahriuwu.models.transformer_tokenizer import TransformerTokenizer
    cfg2 = {k: v for k, v in cfg.items() if k != "size_preset"}
    m_good = TransformerTokenizer(**cfg2)
    report("pretokenize_replay_v7.load_v7  TransformerTokenizer(**model_config)"
           "  + guarded strict=False", m_good, sd)


if __name__ == "__main__":
    main()
