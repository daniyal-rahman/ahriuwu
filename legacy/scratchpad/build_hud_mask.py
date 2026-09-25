"""Build the fixed 352x352 HUD valid-mask from actual YT frames: pixels that are
black (<=0.02) across ALL sampled frames of multiple games = the burned-in HUD
drawbox. valid=1 means real content (trained), 0 means HUD (excluded from loss)."""
import glob, tarfile, numpy as np, cv2, torch

tars = sorted(glob.glob("/srv/nfs/datasets/yt_pretrain_garen/*.tar"))[:3]
frames = []
for tp in tars:
    with tarfile.open(tp) as t:
        names = [n for n in t.getnames() if n.endswith(".jpg")]
        for n in names[:: max(1, len(names) // 20)][:20]:
            img = cv2.imdecode(np.frombuffer(t.extractfile(n).read(), np.uint8), cv2.IMREAD_COLOR)
            frames.append(cv2.resize(img, (352, 352)))
arr = np.stack(frames).astype(np.float32) / 255.0          # (N,352,352,3)
blk = (arr <= 0.02).all(axis=3).all(axis=0)                # (352,352) always-black => HUD
valid = (~blk).astype(np.float32)                          # 1=content, 0=HUD
print(f"{len(frames)} frames from {len(tars)} games | HUD(black) fraction = {blk.mean():.3f}")

small = cv2.resize(blk.astype(np.uint8), (56, 56), interpolation=cv2.INTER_AREA)
print("HUD mask (#=excluded HUD, .=trained content):")
for row in small:
    print("  " + "".join("#" if v else "." for v in row))

torch.save(torch.from_numpy(valid), "scratchpad/hud_valid_mask_352.pt")
print("saved -> scratchpad/hud_valid_mask_352.pt  shape", valid.shape)
