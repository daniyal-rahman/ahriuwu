"""Post-process: merge camera + memory data, project to screen coords, 
shift actions by 1 frame, output per-replay JSON label file."""
import json, math, sys, os

# Load data
cam = json.load(open(r"C:\tmp\cam_fast.json"))
mem = json.load(open(r"C:\tmp\pass2_mem.json"))

GAME_ID = "5528069928"
CHAMPION = "Garen"
START_TIME = 810
END_TIME = 840
FPS = 20
SW, SH = 1280, 720
NUM_FRAMES = (END_TIME - START_TIME) * FPS  # 600

# Projection math
TILT = math.radians(56.0)
FOV_V = math.radians(40.0)
FOV_H = 2 * math.atan(math.tan(FOV_V/2) * SW/SH)
TAN_H = math.tan(FOV_H / 2)
TAN_V = math.tan(FOV_V / 2)
CT = math.cos(TILT); ST = math.sin(TILT)

def project(wx, wz, cx, cy, cz):
    """Project world (wx, 52, wz) to screen coords given camera position."""
    dx = wx - cx; dy = 52.0 - cy; dz = wz - cz
    vy = dy*CT + dz*ST
    vz = -dy*ST + dz*CT
    if vz <= 10: return None
    sx = 0.5 + (dx/vz) / TAN_H * 0.5
    sy = 0.5 - (vy/vz) / TAN_V * 0.5
    px, py = int(sx*SW), int(sy*SH)
    if 0 <= px < SW and 0 <= py < SH:
        return [px, py]
    return None

def nearest(samples, gt):
    """Find nearest sample by game time."""
    best = None; best_d = 999
    for s in samples:
        d = abs(s["gt"] - gt)
        if d < best_d:
            best_d = d; best = s
    return best

def classify_spell(spell_name):
    """Classify spell into action type."""
    if not spell_name: return "idle"
    sl = spell_name.lower()
    if "attack" in sl: return "attack"
    if "recall" in sl: return "recall"
    return "ability"

# Build per-frame data
print(f"Building {NUM_FRAMES} frames...", flush=True)

frame_data = []
for fi in range(NUM_FRAMES):
    gt = START_TIME + fi / FPS
    
    best_cam = nearest(cam, gt)
    best_mem = nearest(mem, gt)
    
    if not best_cam or not best_mem:
        frame_data.append(None)
        continue
    
    cx = best_cam["cx"]; cy = best_cam["cy"]; cz = best_cam["cz"]
    heroes = best_mem.get("heroes", {})
    garen = heroes.get(CHAMPION, {})
    gp = garen.get("pos", [0, 0])
    
    # Project Garen
    garen_screen = project(gp[0], gp[1], cx, cy, cz)
    
    # Project all visible heroes
    visible = []
    for name, hd in heroes.items():
        p = hd.get("pos", [0, 0])
        sp = project(p[0], p[1], cx, cy, cz)
        if sp:
            visible.append({"name": name, "screen": sp})
    
    # Action: spell + target screen position
    spell = garen.get("spell")
    cast_target = garen.get("cast_target")
    target_hero = garen.get("target_hero")
    action_type = classify_spell(spell)
    
    action_screen = None
    if cast_target:
        action_screen = project(cast_target[0], cast_target[1], cx, cy, cz)
    
    frame_data.append({
        "gt": round(gt, 3),
        "cam": [round(cx, 1), round(cy, 1), round(cz, 1)],
        "garen_screen": garen_screen,
        "garen_world": [round(gp[0], 1), round(gp[1], 1)],
        "visible_heroes": visible,
        "action": {
            "type": action_type,
            "spell": spell,
            "screen": action_screen,
            "target_hero": target_hero,
        } if spell else {"type": "idle", "spell": None, "screen": None, "target_hero": None},
    })

# Shift actions forward by 1 frame: frame N gets frame N+1's action
# This is for ML: "given this frame, predict NEXT action"
print("Shifting actions by +1 frame...", flush=True)

frames_out = []
for fi in range(NUM_FRAMES):
    fd = frame_data[fi]
    if fd is None:
        continue
    
    # Get NEXT frame's action (or idle if last frame)
    if fi + 1 < NUM_FRAMES and frame_data[fi + 1] is not None:
        next_action = frame_data[fi + 1]["action"]
    else:
        next_action = {"type": "idle", "spell": None, "screen": None, "target_hero": None}
    
    frames_out.append({
        "frame": fi,
        "gt": fd["gt"],
        "garen_screen": fd["garen_screen"],
        "visible_heroes": fd["visible_heroes"],
        "next_action": next_action,
    })

# Build output
output = {
    "game_id": GAME_ID,
    "champion": CHAMPION,
    "start_time": START_TIME,
    "end_time": END_TIME,
    "fps": FPS,
    "resolution": [SW, SH],
    "projection": {
        "fov_v_deg": 40.0,
        "fov_h_deg": round(math.degrees(FOV_H), 1),
        "tilt_deg": 56.0,
    },
    "total_frames": len(frames_out),
    "frames": frames_out,
}

out_path = r"C:\tmp\NA1-5528069928_labels.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# Stats
action_types = {}
for fr in frames_out:
    t = fr["next_action"]["type"]
    action_types[t] = action_types.get(t, 0) + 1

print(f"Saved: {out_path}", flush=True)
print(f"Frames: {len(frames_out)}", flush=True)
print(f"Action distribution: {action_types}", flush=True)
print(f"Sample frame 100:", flush=True)
print(json.dumps(frames_out[100], indent=2), flush=True)
