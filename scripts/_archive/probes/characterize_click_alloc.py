"""Characterize the click-dest alloc's lifetime and identity.

Three phases run back-to-back so we can answer concrete questions:

Phase A (STABILITY):
  Continuous playback, no seeks. Scan every ~2s for 60s. At each scan:
    - find all triple-mirror candidates (the click-dest shape family)
    - record {addr -> Vec3} and which addresses persist across scans
  Answers: do addresses stay the same while playing? do Vec3s drift (unit moving to destination) or jump (new click)?

Phase B (SEEK STRESS):
  Pause, seek -10s, resume, scan. Diff against last scan of Phase A.
  Answers: does seeking invalidate / reallocate candidates?

Phase C (IDENTITY):
  At a single moment, pull all live player positions from /liveclientdata.
  For each candidate, find the champion whose live position (+ nearby world
  positions representing recent clicks) best matches the candidate's Vec3.
  Answers: 1 candidate per unit? Can we ID Bel'Veth by position match alone?

Writes C:\\tmp\\click_alloc_characterization.json
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time
from collections import defaultdict
import numpy as np
import urllib.request, ssl

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True); _orig_print(*a, **k)
builtins.print = print

_k = ctypes.windll.kernel32
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
MEM_COMMIT = 0x1000; MEM_PRIVATE = 0x20000
PAGE_RW = 0x04 | 0x08 | 0x40

REPLAY_HOST = "https://127.0.0.1:2999"
SSL_CTX = ssl._create_unverified_context()

def api_get(path):
    try:
        with urllib.request.urlopen(f"{REPLAY_HOST}{path}", context=SSL_CTX, timeout=3) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_err": str(e)}

def api_post(path, body):
    try:
        req = urllib.request.Request(f"{REPLAY_HOST}{path}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=3) as r:
            return r.read()
    except Exception as e:
        return str(e).encode()

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def readable_heap_regions(h):
    """All RW committed private-heap regions, no address-range filter."""
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        b = mbi.BaseAddress or 0; s = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE
                and (mbi.Protect & PAGE_RW)):
            yield b, s
        addr = b + s
        if addr <= b: break

def read_region(h, base, size):
    if size > 128 * 1024 * 1024: return None
    out = bytearray(size); v = memoryview(out); o = 0; CH = 4 * 1024 * 1024
    while o < size:
        n = min(CH, size - o)
        buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h, ctypes.c_void_p(base + o), buf, n, ctypes.byref(r)) or r.value == 0:
            return None if o == 0 else bytes(v[:o])
        v[o:o+r.value] = buf[:r.value]; o += r.value
    return bytes(out)

def read_bytes(h, addr, n):
    buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

# World-space sanity: League's map is ~15000x15000
def valid_world(x, y, z):
    return (-500 < x < 16000 and -200 < y < 1000 and -500 < z < 16000
            and (abs(x) > 1 or abs(z) > 1))

def scan_triple_mirrors(h, verbose=False):
    """Find triple-mirror click-dest candidates. Fully vectorized.
    Shape: same (x,y,z) float triple at +0x000, +0x308, +0x374.
    Click-dest allocs live in small-object heap — filter regions to <=32MB.
    """
    candidates = []
    SB = 0x308 // 4
    SC = 0x374 // 4
    n_scanned = n_skipped_big = 0
    t_read = t_scan = 0.0
    for base, size in readable_heap_regions(h):
        if size > 32 * 1024 * 1024:
            n_skipped_big += 1; continue
        t0 = time.time()
        data = read_region(h, base, size)
        t_read += time.time() - t0
        if not data or len(data) < 0x400: continue
        n_scanned += 1
        t0 = time.time()
        n_f = len(data) // 4
        if n_f <= SC + 3: continue
        arr = np.frombuffer(data, dtype=np.float32, count=n_f)
        L = n_f - SC - 3
        if L <= 0: continue
        x0 = arr[0:L]; y0 = arr[1:L+1]; z0 = arr[2:L+2]
        xb = arr[SB:SB+L]; yb = arr[SB+1:SB+1+L]; zb = arr[SB+2:SB+2+L]
        xc = arr[SC:SC+L]; yc = arr[SC+1:SC+1+L]; zc = arr[SC+2:SC+2+L]
        eps = 1e-2
        # Click-dest Vec3s: y very close to floor (52), x+z clearly in playable map,
        # and NOT (0,0,0) or trivial. Suppress numpy warnings for NaN/Inf compares.
        with np.errstate(invalid='ignore', over='ignore'):
            mask = ((x0 > 100) & (x0 < 15000) & (z0 > 100) & (z0 < 15000)
                    & (y0 > 45) & (y0 < 65)
                    & (np.abs(x0) + np.abs(z0) > 500)
                    & (np.abs(x0 - xb) < eps) & (np.abs(y0 - yb) < eps) & (np.abs(z0 - zb) < eps)
                    & (np.abs(x0 - xc) < eps) & (np.abs(y0 - yc) < eps) & (np.abs(z0 - zc) < eps))
        idxs = np.nonzero(mask)[0]
        for i in idxs:
            addr = base + int(i) * 4
            candidates.append((addr, (float(x0[i]), float(y0[i]), float(z0[i]))))
        t_scan += time.time() - t0
    if verbose:
        print(f"    [regions_scanned={n_scanned} skipped_big={n_skipped_big} "
              f"read_t={t_read:.1f}s scan_t={t_scan:.1f}s cands={len(candidates)}]")
    return candidates

def main():
    pid = find_pid()
    if not pid: print("ERR no league"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}")

    result = {"pid": pid, "phases": {}}

    # ---------- Phase A: stability sweep ----------
    print("\n========== Phase A: STABILITY (30 scans, ~2s each, no seeks) ==========")
    api_post("/replay/playback", {"speed": 1.0, "paused": False})
    pre = api_get("/replay/playback")
    print(f"  starting playback at t={pre.get('time')}, paused={pre.get('paused')}")

    scans = []
    t0 = time.time()
    N_SCANS = 15
    for i in range(N_SCANS):
        ts = time.time() - t0
        tgame = api_get("/replay/playback").get("time", -1)
        scan_t = time.time()
        cands = scan_triple_mirrors(h, verbose=(i == 0))
        scan_dur = time.time() - scan_t
        print(f"  scan[{i:2d}] wall={ts:5.1f}s game_t={tgame:6.1f}  candidates={len(cands)}  scan_took={scan_dur:.2f}s", flush=True)
        scans.append({"i": i, "wall": ts, "game_t": tgame, "scan_dur": scan_dur,
                      "cands": [(hex(a), [round(v, 2) for v in xyz]) for a, xyz in cands]})
    result["phases"]["A_stability"] = scans

    # Analyze stability: which addresses persisted across all 30 scans?
    addr_sets = [set(int(a, 16) for a, _ in s["cands"]) for s in scans]
    if addr_sets:
        persistent = set.intersection(*addr_sets)
        union = set.union(*addr_sets)
        never_moved = []
        for a in persistent:
            vecs = []
            for s in scans:
                for addr, xyz in s["cands"]:
                    if int(addr, 16) == a: vecs.append(xyz)
            # is Vec3 stable or jumping?
            if len(vecs) > 1:
                xs = [v[0] for v in vecs]
                zs = [v[2] for v in vecs]
                dx = max(xs) - min(xs)
                dz = max(zs) - min(zs)
                never_moved.append((a, len(vecs), dx, dz, vecs[0], vecs[-1]))
        never_moved.sort(key=lambda r: -r[1])
        print(f"\n  Of {len(union)} unique addresses seen:")
        print(f"    persistent across ALL {N_SCANS} scans: {len(persistent)}")
        print(f"    ephemeral (came & went): {len(union) - len(persistent)}")
        print(f"\n  Top 15 persistent addresses (addr, seen_cnt, Δx, Δz, first Vec3, last Vec3):")
        for a, cnt, dx, dz, first, last in never_moved[:15]:
            motion = "STATIC" if (dx < 1 and dz < 1) else f"MOVED Δ=({dx:.0f},{dz:.0f})"
            print(f"    0x{a:X}  n={cnt}  {motion}")
            print(f"      first=({first[0]:.0f},{first[1]:.0f},{first[2]:.0f}) last=({last[0]:.0f},{last[1]:.0f},{last[2]:.0f})")
        result["phases"]["A_stability_summary"] = {
            "n_scans": N_SCANS,
            "persistent_count": len(persistent),
            "union_count": len(union),
            "persistent_addrs": [hex(a) for a in list(persistent)[:100]],
        }

    # ---------- Phase B: seek stress ----------
    print("\n========== Phase B: SEEK STRESS (seek -10s, rescan) ==========")
    last_scan = scans[-1] if scans else None
    pre_addrs = set(int(a, 16) for a, _ in last_scan["cands"]) if last_scan else set()
    pre_game_t = last_scan["game_t"] if last_scan else 0
    target_t = max(0, pre_game_t - 10)
    print(f"  pre-seek addrs: {len(pre_addrs)}   seeking to game_t={target_t:.1f}")
    api_post("/replay/playback", {"time": target_t, "paused": False})
    time.sleep(2.0)  # let it stabilize
    post = scan_triple_mirrors(h)
    post_addrs = set(a for a, _ in post)
    kept = pre_addrs & post_addrs
    lost = pre_addrs - post_addrs
    new = post_addrs - pre_addrs
    print(f"  post-seek addrs: {len(post_addrs)}")
    print(f"    kept:  {len(kept)}")
    print(f"    lost:  {len(lost)}  (pre-seek addrs no longer showing triple-mirror)")
    print(f"    new:   {len(new)}")
    if kept:
        sample = list(kept)[:5]
        print(f"    samples kept: {[hex(a) for a in sample]}")
    result["phases"]["B_seek_stress"] = {
        "pre_count": len(pre_addrs), "post_count": len(post_addrs),
        "kept": len(kept), "lost": len(lost), "new": len(new),
        "kept_addrs_sample": [hex(a) for a in list(kept)[:30]],
    }

    # ---------- Phase C: identity mapping ----------
    print("\n========== Phase C: IDENTITY mapping (candidates → champions) ==========")
    api_post("/replay/playback", {"paused": True})
    time.sleep(0.5)
    playerlist = api_get("/liveclientdata/playerlist")
    # Live client doesn't have position, only summoner info. Use /replay/render
    # selection iteration OR /liveclientdata/allgamedata. Try allgamedata.
    allg = api_get("/liveclientdata/allgamedata")
    active = api_get("/liveclientdata/activeplayername")
    print(f"  active player: {active}")
    # Build champion -> summoner map from playerlist
    players = []
    if isinstance(playerlist, list):
        for p in playerlist:
            players.append({
                "championName": p.get("championName"),
                "summonerName": p.get("summonerName"),
                "team": p.get("team"),
            })
    print(f"  {len(players)} players: {[p['championName'] for p in players]}")

    # Now re-scan (unpaused briefly) for current candidate Vec3s
    api_post("/replay/playback", {"paused": False})
    time.sleep(1.0)
    final = scan_triple_mirrors(h)
    api_post("/replay/playback", {"paused": True})
    # Bucket candidates by spatial proximity (within 500 units = same unit)
    final.sort(key=lambda r: (r[1][0], r[1][2]))
    clusters = []
    for addr, xyz in final:
        placed = False
        for cl in clusters:
            cx, cy, cz = cl["centroid"]
            if abs(cx - xyz[0]) < 500 and abs(cz - xyz[2]) < 500:
                cl["members"].append((addr, xyz))
                xs = [m[1][0] for m in cl["members"]]
                zs = [m[1][2] for m in cl["members"]]
                cl["centroid"] = (sum(xs)/len(xs), xyz[1], sum(zs)/len(zs))
                placed = True; break
        if not placed:
            clusters.append({"centroid": xyz, "members": [(addr, xyz)]})
    print(f"\n  Spatial clustering: {len(final)} candidates → {len(clusters)} clusters")
    print(f"  (each cluster = likely one unit's click-dest family)")
    for i, cl in enumerate(sorted(clusters, key=lambda c: -len(c["members"]))[:15]):
        cx, cy, cz = cl["centroid"]
        print(f"    cluster[{i:2d}]  centroid=({cx:.0f},{cy:.0f})  size={len(cl['members'])}  "
              f"sample_addr={hex(cl['members'][0][0])}")
    result["phases"]["C_identity"] = {
        "n_players": len(players),
        "players": players,
        "n_candidates": len(final),
        "n_clusters": len(clusters),
        "clusters": [{
            "centroid": [round(c, 1) for c in cl["centroid"]],
            "size": len(cl["members"]),
            "members": [(hex(a), [round(v, 1) for v in xyz]) for a, xyz in cl["members"]],
        } for cl in clusters],
    }

    api_post("/replay/playback", {"paused": False})
    with open(r"C:\tmp\click_alloc_characterization.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nwrote C:\\tmp\\click_alloc_characterization.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
