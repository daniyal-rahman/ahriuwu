#!/usr/bin/env python3
"""YouTube -> tokenizer training frames pipeline (v2, from scratch 2026-06-16).

Source: @domisumReplay-Garen (regular high-res Garen games, matchup in title).
Output: 352x352 JPG frames at 20fps, HUD-masked, in a SEPARATE pretrain folder
(replay corpus stays the fine-tune set). Matchup parsed from the video title.

Design (distribution-safety critical — MIRRORS the action-labeled replay corpus,
which is locked at 352x352 since its originals were deleted):
  - download 720p60 (format 298) — matches the replay corpus's downscale RATIO
    (720->352 = 2.05x, same as replay), so the area-filter softness signature
    matches. Cheaper + less rate-limit exposure than 1080p; 352 can't hold 1080
    detail anyway. Falls back to 1080p (299) if 720 unavailable.
  - ONE ffmpeg pass: HUD drawbox mask at native res -> fps=20 CFR decimation
    (source is native 60fps, clean 3:1, no interpolation) -> SQUISH to 352x352
    with scale=area (anamorphic, NO pad/crop — exactly like extract_replay_frames).
    Mask applied BEFORE the downscale so its edges anti-alias with the game.
  - JPG q=2. Matchup parsed from the video title (free lane-opponent labels).
  - rate-limit friendly: --sleep-requests, retries, one video at a time.

Usage:
  python yt_pipeline.py --video-id NITBjbgTLVA --out-dir /home/dani/yt_pretrain --keep-video
  python yt_pipeline.py --channel-end 50 --out-dir /home/dani/yt_pretrain --skip-ids-file done.txt
"""
import argparse, datetime, json, os, re, subprocess, sys, time, shutil, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CHANNEL = "https://www.youtube.com/@domisumReplay-Garen/videos"
TARGET = 352
FPS = 20
COOKIES = None  # set via --cookies: Netscape cookies.txt from a logged-in (throwaway)
                # YouTube account -> much higher per-IP download budget (no ~40-game bot-block)
LIMIT_RATE = None    # yt-dlp --limit-rate applied OUTSIDE the fast window (e.g. "350K" ~ 1x playback,
                     # human-like so the residential IP stays under YouTube's radar)
FAST_START_HOUR = 2  # full-speed (unthrottled) window [start, end) in the SERVER's local time
FAST_END_HOUR = 9
BLOCK_BACKOFF = 0    # seconds to sleep when a download fails on a bot-block, so a residential
                     # IP cools instead of getting hammered with failed requests (--block-backoff)
# HUD mask regions in BASE 640x360 (16:9) coords, as (y1,y2,x1,x2) — the
# domisumReplay spectator-HUD layout. Scaled to the source resolution at
# runtime so the same mask works for any download res (720p, 1080p, ...).
# Finalized 2026-06-17 against a real 1080p frame.
HUD_MASK_360 = [
    (0,   35,  128, 512),   # top scoreboard center (middle 60%, 20% game shows each side)
    (35,  45,  256, 384),   # top center cluster lower lip (middle 20%)
    (35,  215, 0,   35),    # left team cards (blue)
    (35,  215, 605, 640),   # right team cards (red)
    (257, 360, 0,   105),   # bottom-left Garen HUD + domisum logo
    (282, 360, 105, 550),   # bottom-center scoreboard
    (257, 360, 535, 640),   # bottom-right minimap (mirror of bottom-left)
]

def hud_drawbox_filter(src_w, src_h):
    """Build ffmpeg drawbox chain from the 360p mask, scaled to src resolution."""
    sx, sy = src_w / 640.0, src_h / 360.0
    parts = []
    for (y1, y2, x1, x2) in HUD_MASK_360:
        X, Y = round(x1 * sx), round(y1 * sy)
        Wd, Hd = round((x2 - x1) * sx), round((y2 - y1) * sy)
        parts.append(f"drawbox=x={X}:y={Y}:w={Wd}:h={Hd}:color=black:t=fill")
    return ",".join(parts)

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def parse_matchup(title: str) -> dict:
    """Pull matchup/rank/kda from titles like:
    'GAREN vs TEEMO (TOP) | Good KDA: 13/2/4 | KR Master | 26.12'"""
    out = {"raw_title": title, "champion": "Garen", "lane_opponent": None,
           "role": None, "region": None, "rank": None, "patch": None, "kda": None}
    m = re.search(r"GAREN\s+vs\s+([A-Z'\.\s]+?)\s*\(([A-Z]+)\)", title, re.I)
    if m:
        out["lane_opponent"] = m.group(1).strip().title().replace(" ", "").replace("'", "")
        out["role"] = m.group(2).upper()
    m = re.search(r"(\d+)\s*[./]\s*(\d+)\s*[./]\s*(\d+)", title)
    if m: out["kda"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    m = re.search(r"(\d+\.\d+)\s*$", title.strip())
    if m: out["patch"] = m.group(1)
    REGIONS = r"NA|EUW|EUNE|KR|BR|LAN|LAS|OCE|TR|RU|JP|PH|SG|VN|TH|TW|CN"
    TIERS = r"Iron|Bronze|Silver|Gold|Platinum|Emerald|Diamond|Master|Grandmaster|Challenger"
    # anchor the rank tier to a preceding server region ("EUW Diamond"), else a
    # word like the "gold" in "6k gold comeback" gets misread as the rank.
    m = re.search(rf"\b({REGIONS})\s+({TIERS})\b", title, re.I)
    if m:
        out["region"] = m.group(1).upper()
        out["rank"] = m.group(2).title()
    else:
        m = re.search(rf"\b({TIERS})\b", title, re.I)
        if m: out["rank"] = m.group(1).title()
    return out

def get_title(vid):
    r = run(["yt-dlp", "--skip-download", "--print", "%(title)s",
             f"https://www.youtube.com/watch?v={vid}"])
    return r.stdout.strip() if r.returncode == 0 else ""

def _find_dl(dst, vid):
    """yt-dlp may emit .mp4/.mkv/.webm depending on remux. Return the real file."""
    cands = [p for p in dst.glob(f"{vid}.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm")
             and not p.name.endswith(".part")]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None

def download(vid, dst, sleep_req=2.0):
    """Download 1080p60 VIDEO-ONLY (format 299; H.264, no audio — frames don't
    need it, and skipping audio avoids the mp4->mkv remux). Falls back to 720p60
    (298) then any <=1080p."""
    existing = _find_dl(dst, vid)
    if existing: return existing
    base = dst / vid  # yt-dlp adds the right extension
    # time-based throttle: full speed in the fast window, ~1x-playback cap otherwise
    hr = datetime.datetime.now().hour
    in_fast = FAST_START_HOUR <= hr < FAST_END_HOUR
    rate = [] if (in_fast or not LIMIT_RATE) else ["--limit-rate", LIMIT_RATE]
    cmd = ["yt-dlp"] + (["--cookies", COOKIES] if COOKIES else []) + rate + [
           "-f", "298/136/bestvideo[height<=720]/299",
           "--sleep-requests", str(sleep_req), "--retries", "5",
           "--fragment-retries", "10", "-o", f"{base}.%(ext)s",
           f"https://www.youtube.com/watch?v={vid}"]
    r = run(cmd)
    found = _find_dl(dst, vid)
    if not found and BLOCK_BACKOFF and r.stderr and ("not a bot" in r.stderr or "Sign in to confirm" in r.stderr):
        print(f"  BOT-BLOCKED — backing off {BLOCK_BACKOFF//60}min to let the IP cool", flush=True)
        time.sleep(BLOCK_BACKOFF)
    return found

def probe_fps(path):
    r = run(["ffprobe", "-v", "0", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate,width,height,nb_frames",
             "-of", "json", str(path)])
    try: return json.loads(r.stdout)["streams"][0]
    except Exception: return {}

def extract_frames(video, out_frames_dir, src_w, src_h):
    """ONE ffmpeg pass: HUD mask (drawbox at native res) -> fps=20 CFR ->
    SQUISH to 352x352 (scale=area). Matches the action-labeled replay corpus:
    anamorphic squish (no pad/crop), area downscale. Mask applied BEFORE the
    downscale so its edges anti-alias with the game (best distribution fidelity).
    """
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    mask = hud_drawbox_filter(src_w, src_h)
    vf = f"{mask},fps={FPS}:round=down,scale={TARGET}:{TARGET}:flags=area"
    t0 = time.time()
    r = run(["ffmpeg", "-y", "-i", str(video), "-vf", vf, "-vsync", "cfr",
             "-q:v", "2", "-start_number", "0",
             str(out_frames_dir / "%06d.jpg")])
    dt = time.time() - t0
    n = len(list(out_frames_dir.glob("*.jpg")))
    return n, dt, r.returncode

def process_one(vid, out_dir, keep_video=False, title=None, gcs_bucket=None,
                cleanup=False, work_dir=None):
    """Download -> HUD-mask+squish to 352 -> pack the whole game into ONE tar
    (frames + labels.json). One object per game instead of ~31k tiny files:
    fast transport, cheap storage ops, trivial to stage (untar -> the identical
    loose-frame layout the replay corpus uses). Frames extract to a LOCAL
    work_dir (fast disk); the single tar then lands in out_dir or streams to GCS."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir) if work_dir else out_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    tar_local = out_dir / f"{vid}.tar"
    sidecar = out_dir / f"{vid}.labels.json"
    # resume guard for the local/NFS path; GCS mode relies on the runner's done-file
    if tar_local.exists():
        print(f"[{vid}] already packaged, skip", flush=True); return "skip"
    vwork = work_dir / "_videos"; vwork.mkdir(exist_ok=True)
    frames_dir = work_dir / vid
    if title is None: title = get_title(vid)
    meta = parse_matchup(title)
    print(f"[{vid}] {title}", flush=True)
    print(f"  matchup: Garen vs {meta['lane_opponent']} ({meta['role']}) {meta['rank']} patch {meta['patch']}", flush=True)
    vpath = download(vid, vwork)
    if not vpath:
        print(f"  DOWNLOAD FAILED", flush=True); return None
    sz = vpath.stat().st_size / 1e9
    fps_info = probe_fps(vpath)
    sw, sh = int(fps_info.get("width", 1280)), int(fps_info.get("height", 720))
    print(f"  downloaded {sz:.2f}GB  src={sw}x{sh} r_fps={fps_info.get('r_frame_rate')}", flush=True)
    n, dt, rc = extract_frames(vpath, frames_dir, sw, sh)
    meta.update({"video_id": vid, "n_frames": n, "fps": FPS, "target": TARGET,
                 "src_w": sw, "src_h": sh, "src_size_gb": round(sz, 3),
                 "extract_sec": round(dt, 1), "downscale": "INTER_AREA(ffmpeg scale=area), squish"})
    (frames_dir / "labels.json").write_text(json.dumps(meta, indent=2))
    vpath.unlink(missing_ok=True)  # raw video no longer needed
    print(f"  -> {n} frames in {dt:.0f}s ({n/dt:.0f} fps), HUD-masked+squished 352", flush=True)
    if n == 0:
        print(f"  NO FRAMES — abort", flush=True); shutil.rmtree(frames_dir, ignore_errors=True); return None
    # pack frames+labels into one tar (members are bare 000000.jpg ... + labels.json)
    if gcs_bucket:
        dest = f"{gcs_bucket}/{vid}.tar"
        tar = subprocess.Popen(["tar", "-cf", "-", "-C", str(frames_dir), "."], stdout=subprocess.PIPE)
        up = subprocess.run(["gsutil", "-q", "cp", "-", dest], stdin=tar.stdout)
        tar.stdout.close(); tar.wait()
        if up.returncode != 0 or tar.returncode != 0:
            print(f"  GCS TAR FAILED (tar={tar.returncode} up={up.returncode})", flush=True)
            shutil.rmtree(frames_dir, ignore_errors=True); return None
        run(["gsutil", "-q", "cp", str(frames_dir / "labels.json"), f"{gcs_bucket}/{vid}.labels.json"])
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"  -> {dest} ({n} frames, {sz:.2f}GB raw) local freed", flush=True)
    else:
        tmp = out_dir / f"{vid}.tar.tmp"
        r = run(["tar", "-cf", str(tmp), "-C", str(frames_dir), "."])
        if r.returncode != 0:
            print(f"  TAR FAILED: {r.stderr[-200:]}", flush=True)
            tmp.unlink(missing_ok=True); shutil.rmtree(frames_dir, ignore_errors=True); return None
        os.replace(tmp, tar_local)  # atomic publish on NFS
        shutil.copy(frames_dir / "labels.json", sidecar)
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"  -> {tar_local} ({n} frames)", flush=True)
    return meta


def run_batch(items, out_dir, workers, gcs_bucket, cleanup, done_file, work_dir=None, max_games=0):
    """Parallel driver over (video_id, title) pairs. Resumable via done_file
    (one completed id per line). Downloads/extracts are subprocesses so the GIL
    is released — N threads = N videos in flight. Titles come from the bulk
    metadata pull, so there are ZERO per-video metadata requests (only the
    actual media download), which keeps the per-IP request rate low."""
    done = set()
    if done_file and Path(done_file).exists():
        done = {x.strip() for x in Path(done_file).read_text().split("\n") if x.strip()}
    todo = [(v, t) for (v, t) in items if v not in done]
    if max_games and max_games > 0:
        todo = todo[:max_games]   # "u40": stop under the per-IP block cap, then rotate IP
    lock = threading.Lock()
    print(f"batch: {len(todo)} to process ({len(done)} already done), {workers} workers -> {out_dir}", flush=True)

    def work(item):
        vid, title = item
        try:
            r = process_one(vid, out_dir, keep_video=False, title=title,
                            gcs_bucket=gcs_bucket, cleanup=cleanup, work_dir=work_dir)
            ok = r is not None
            if ok and done_file:
                with lock, open(done_file, "a") as f:
                    f.write(vid + "\n")
            return vid, ("ok" if ok else "fail")
        except Exception as e:
            return vid, f"err:{type(e).__name__}:{e}"

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for fut in as_completed(futs):
            vid, status = fut.result()
            if status == "ok": ok += 1
            else: fail += 1
            print(f"  [{ok+fail}/{len(todo)}] {vid}: {status}  (ok={ok} fail={fail})", flush=True)
    print(f"BATCH DONE ok={ok} fail={fail}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id"); ap.add_argument("--channel-end", type=int)
    ap.add_argument("--ids-file", help="newline list of 'VIDEOID|Title' (title optional)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", help="local fast disk for extraction scratch (default: out-dir)")
    ap.add_argument("--workers", type=int, default=1, help="parallel videos in flight")
    ap.add_argument("--gcs-bucket", help="gs://... stream per-game tar here then free local disk")
    ap.add_argument("--done-file", help="resumable progress log (one done id per line)")
    ap.add_argument("--cookies", help="Netscape cookies.txt (logged-in throwaway acct) for higher rate limit")
    ap.add_argument("--limit-rate", help="yt-dlp rate cap OUTSIDE the fast window, e.g. 350K (~1x playback)")
    ap.add_argument("--fast-hours", help="full-speed window as START-END in 24h server-local time, e.g. 2-9")
    ap.add_argument("--block-backoff", type=int, default=0, help="sleep N sec on a bot-block before next try (lets a residential IP cool)")
    ap.add_argument("--max-games", type=int, default=0, help="stop after N games (stay under per-IP block cap, then rotate IP)")
    ap.add_argument("--keep-video", action="store_true")
    ap.add_argument("--dump-clean", action="store_true")
    ap.add_argument("--skip-ids-file")
    a = ap.parse_args()
    if a.cookies:
        global COOKIES
        COOKIES = a.cookies
    if a.limit_rate:
        global LIMIT_RATE
        LIMIT_RATE = a.limit_rate
    if a.fast_hours:
        global FAST_START_HOUR, FAST_END_HOUR
        _s, _e = a.fast_hours.split("-")
        FAST_START_HOUR, FAST_END_HOUR = int(_s), int(_e)
    if a.block_backoff:
        global BLOCK_BACKOFF
        BLOCK_BACKOFF = a.block_backoff
    if a.video_id:
        process_one(a.video_id, a.out_dir, a.keep_video)
    elif a.ids_file:
        items = []
        for line in Path(a.ids_file).read_text().split("\n"):
            line = line.strip()
            if not line: continue
            vid, title = (line.split("|", 1) + [None])[:2] if "|" in line else (line, None)
            items.append((vid.strip(), title))
        run_batch(items, a.out_dir, a.workers, a.gcs_bucket,
                  cleanup=bool(a.gcs_bucket), done_file=a.done_file, work_dir=a.work_dir,
                  max_games=a.max_games)
    elif a.channel_end:
        r = run(["yt-dlp", "--flat-playlist", "--playlist-end", str(a.channel_end),
                 "--print", "%(id)s", CHANNEL])
        ids = [x.strip() for x in r.stdout.strip().split("\n") if x.strip()]
        skip = set()
        if a.skip_ids_file and Path(a.skip_ids_file).exists():
            skip = set(Path(a.skip_ids_file).read_text().split())
        ids = [i for i in ids if i not in skip]
        print(f"processing {len(ids)} videos", flush=True)
        for i, vid in enumerate(ids):
            print(f"\n=== [{i+1}/{len(ids)}] ===", flush=True)
            process_one(vid, a.out_dir, a.keep_video)
            time.sleep(3)  # be polite between videos
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
