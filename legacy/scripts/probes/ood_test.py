#!/usr/bin/env python3
"""Is the live frame actually OOD to the MODEL (not just visually different)?
Three signals, on training vs live-unmasked vs live-masked frames:
  1. tokenizer recon PSNR   — can the tokenizer even represent the frame?
     (policy-independent: if low on live, latents are garbage before the policy)
  2. policy movement ENTROPY + unique argmax bins (PRE-GATE) — does the policy
     collapse to a near-constant movement target on live frames? (isolates the
     policy's response to the input from the gate/decode mechanism)
Run on the desktop (GPU free)."""
import sys, glob, cv2, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from agent_infer import GarenAgent

TOK = "/mnt/storage/ahriuwu-live/checkpoints/tokenizer_v7.pt"
BC  = "/mnt/storage/ahriuwu-live/checkpoints/phase2_bc.pt"
LIVE_MP4 = "/mnt/storage/ahriuwu-live/recordings/session_20260810_023349/model_view_352.mp4"
TRAIN = "/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5549995114/frames"
N, CTX = 200, 16
dev = "cuda"


def hud_mask_352():
    """Black-out rects for the live HUD (352 space), from the live frame layout."""
    m = np.ones((352, 352, 1), np.float32)
    m[0:352, 0:30] = 0        # left ability/item column
    m[325:352, :] = 0         # bottom scoreboard/portrait/items bar
    m[240:352, 275:352] = 0   # minimap (bottom-right)
    m[0:22, 300:352] = 0      # top-right clock/score
    return m


def load_live(mp4, n):
    cap = cv2.VideoCapture(mp4)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(tot // 5, tot - 1, n).astype(int)
    out, want = [], set(idx.tolist())
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
        i += 1
    cap.release()
    return out  # list of (352,352,3) rgb01


def load_train(root, n):
    fs = sorted(glob.glob(f"{root}/*.png"))
    idx = np.linspace(2000, len(fs) - 1, n).astype(int)
    out = []
    for i in idx:
        im = cv2.imread(fs[int(i)])
        r = cv2.resize(im, (352, 352), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    return out


@torch.no_grad()
def recon_psnr(agent, frame):
    import torch
    x = torch.from_numpy(cv2.resize(frame, (352, 352))).float().permute(2, 0, 1)[None].to(dev)
    with agent._ac():
        lat = agent.tok.encode(x)["latent"]
        rec = agent.tok.decode(lat, num_frames=1)
        rec = rec[:, 0] if rec.dim() == 5 else rec
    mse = ((rec.float().clamp(0, 1) - x) ** 2).mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-10))


@torch.no_grad()
def movement_response(agent, frames):
    """Slide a ctx window; per frame get PRE-GATE movement logits -> entropy +
    argmax bin (x axis). Returns (mean_entropy, n_unique_argmax/n)."""
    from collections import deque
    buf = deque(maxlen=CTX)
    ents, args = [], []
    for fr in frames:
        buf.append(agent.encode_frame(fr))
        w = list(buf)
        while len(w) < CTX:
            w.insert(0, w[0])
        z0 = torch.stack([t.squeeze(0) for t in w], 0)[None].to(dev).float()  # (1,T,C,16,16)
        B, T = z0.shape[:2]
        tau = agent.tau_ctx + torch.rand(B, T, device=dev) * (1 - agent.tau_ctx)
        z_tau, _ = agent.sched.add_noise(z0, tau)
        d1 = torch.ones(B, dtype=torch.long, device=dev)
        with agent._ac():
            _, ao = agent.dyn(z_tau, tau, step_size=d1, actions=None)
            _, m_logits = agent.policy(ao[:, -1:, :])          # (1,1,L,2,bins)
        p = torch.softmax(m_logits[0, 0, 1, 0].float(), -1)     # x-axis, offset 1
        ents.append(float(-(p * (p + 1e-9).log()).sum()))
        args.append(int(p.argmax()))
    return float(np.mean(ents)), len(set(args)) / len(args)


def main():
    agent = GarenAgent(BC, tokenizer_ckpt=TOK, device=dev)
    agent.reset()
    train = load_train(TRAIN, N)
    live = load_live(LIVE_MP4, N)
    mask = hud_mask_352()
    live_masked = [f * mask for f in live]

    # save a visual: live | live_masked
    ex = (np.concatenate([live[50], np.ones((352, 6, 3)), live_masked[50]], 1) * 255).astype(np.uint8)
    cv2.imwrite("/mnt/nfs/projects/ahriuwu/scratchpad/live_masked_example.png",
                cv2.cvtColor(ex, cv2.COLOR_RGB2BGR))

    import random
    samp = lambda L: [L[i] for i in np.linspace(0, len(L) - 1, 60).astype(int)]
    print("== tokenizer recon PSNR (higher=in-distribution) ==")
    for name, L in [("train", train), ("live", live), ("live-masked", live_masked)]:
        ps = [recon_psnr(agent, f) for f in samp(L)]
        print(f"  {name:12s}: {np.mean(ps):5.2f} dB  (min {np.min(ps):.1f}, max {np.max(ps):.1f})")

    print("\n== policy movement response (pre-gate, x-axis) ==")
    for name, L in [("train", train), ("live", live), ("live-masked", live_masked)]:
        agent.reset()
        ent, uniq = movement_response(agent, L)
        print(f"  {name:12s}: entropy={ent:.3f}  unique-argmax-frac={uniq:.3f}")
    print("\nread: if live recon << train AND live entropy/uniq << train -> OOD real;")
    print("      if masking moves both toward train -> HUD is the cause + mask helps.")


if __name__ == "__main__":
    main()
