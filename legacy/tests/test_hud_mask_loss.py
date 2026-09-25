"""Regression test for the HUD-loss bug fix (verified 2026-07-03).

Bug: intentionally-blacked regions (e.g. the HUD rectangle blacked in YT frames) were
included in MAELoss, so the model was trained to reproduce that black — a -14.5 dB edge
collapse when deployed on HUD-disabled content. Fix: MAELoss `valid_mask` excludes those
pixels from MSE *and* LPIPS. This test asserts the excluded region contributes exactly zero
loss + gradient, while the rest of the frame still trains.

Run:  PYTHONPATH=src python tests/test_hud_mask_loss.py   (or: pytest tests/test_hud_mask_loss.py)
"""
import torch
from ahriuwu.models.losses import MAELoss


def test_valid_mask_excludes_blacked_region():
    torch.manual_seed(0)
    B, T, C, H, W = 1, 2, 3, 64, 64
    target = torch.rand(B, T, C, H, W)          # content everywhere...
    ry, rx = H - 20, W - 20                      # ...except a bottom-right 20x20 "HUD" rect
    target[..., ry:, rx:] = 0.0                  # blacked in the target (data-level mask)

    pred = target + 0.01 * torch.randn_like(target)   # small recon error OUTSIDE the rect
    pred[..., ry:, rx:] = 0.5                          # model outputs CONTENT inside the blacked rect

    vm = torch.ones(B, 1, 1, H, W)               # valid mask: 1=include, 0=ignore
    vm[..., ry:, rx:] = 0.0                       # exclude the blacked rect

    lf = MAELoss(use_lpips_lib=False)

    # (1) WITHOUT valid_mask: the blacked region IS penalized (this is the bug).
    p1 = pred.clone().requires_grad_(True)
    mse_wo = lf(p1, target, mask_indices=None, skip_lpips=True)["mse"]
    mse_wo.backward()
    g_wo_in = p1.grad[..., ry:, rx:].abs().sum().item()
    assert mse_wo.item() > 1e-3, mse_wo.item()
    assert g_wo_in > 0.0, "buggy path must put gradient in the blacked region"

    # (2) WITH valid_mask: excluded region contributes ZERO loss + ZERO gradient...
    p2 = pred.clone().requires_grad_(True)
    mse_w = lf(p2, target, mask_indices=None, skip_lpips=True, valid_mask=vm)["mse"]
    mse_w.backward()
    g_w_in = p2.grad[..., ry:, rx:].abs().sum().item()
    g_w_out = p2.grad[..., :ry, :].abs().sum().item()
    assert g_w_in == 0.0, f"grad in excluded region must be exactly 0, got {g_w_in}"
    assert g_w_out > 0.0, "the rest of the frame must still train"
    assert mse_w.item() < mse_wo.item(), "excluding the black rect must lower the MSE"

    # (3) LPIPS is also neutralized in the excluded region.
    lp_wo = lf(pred, target, mask_indices=None, skip_lpips=False)["lpips"].item()
    lp_w = lf(pred, target, mask_indices=None, skip_lpips=False, valid_mask=vm)["lpips"].item()
    assert lp_w < lp_wo, f"LPIPS must drop when the rect is excluded ({lp_w} !< {lp_wo})"

    print(f"OK  mse: {mse_wo.item():.5f} (grad_in={g_wo_in:.3e})  ->  "
          f"valid_mask {mse_w.item():.5f} (grad_in={g_w_in}, grad_out={g_w_out:.3e})  |  "
          f"lpips {lp_wo:.4f} -> {lp_w:.4f}")


if __name__ == "__main__":
    test_valid_mask_excludes_blacked_region()
    print("PASS")
