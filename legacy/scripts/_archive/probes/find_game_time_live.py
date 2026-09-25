"""Find game_time RVA via multi-time verification.

The scanner's Phase 1b picks the first f32 matching api_gt within ±1s, with no
advance-over-time check. On 16.9.772 it locked onto 0x1D8C278 which is a stale
snapshot — gt sits at 102.358 forever during pass 1.

Usage: run while a replay is PLAYING (not paused) at 1x or 2x speed.
"""
from __future__ import annotations
import json, struct, time, sys, ctypes, os, datetime
from ctypes import wintypes as wt
from pathlib import Path
import urllib.request, ssl

EXPECTED_MOD_SIZE = 0x20E0000

# --- Win32 RPM helpers (mirror scan_offsets) ---
_k = ctypes.WinDLL("kernel32", use_last_error=True)
_k.OpenProcess.restype = wt.HANDLE
_k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
_k.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_k.CloseHandle.argtypes = [wt.HANDLE]
_k.CreateToolhelp32Snapshot.restype = wt.HANDLE
_k.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
_k.Module32First.argtypes = [wt.HANDLE, ctypes.c_void_p]
_k.Module32Next.argtypes = [wt.HANDLE, ctypes.c_void_p]


class Mem:
    def __init__(self, pid):
        self.h = _k.OpenProcess(0x10, False, pid)
        if not self.h: raise OSError(ctypes.get_last_error())
    def close(self): _k.CloseHandle(self.h); self.h = None
    def read(self, a, n):
        b = (ctypes.c_ubyte * n)(); read = ctypes.c_size_t(0)
        if _k.ReadProcessMemory(self.h, a, b, n, ctypes.byref(read)) and read.value == n:
            return bytes(b)
        return None
    def f32(self, a): d = self.read(a, 4); return struct.unpack('<f', d)[0] if d else None


def find_league_pid():
    import subprocess
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower():
            return int(l.strip('"').split('","')[1])
    return None


def find_module_base(pid):
    """Toolhelp32 module enum — same approach as scan_offsets.find_base."""
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("a",ctypes.c_ulong),("b",ctypes.c_ulong),
                  ("c",ctypes.c_ulong),("d",ctypes.c_ulong),
                  ("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
                  ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),
                  ("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME(); me.dwSize = ctypes.sizeof(ME)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                _k.CloseHandle(snap)
                return ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value, me.modBaseSize
            if not _k.Module32Next(snap, ctypes.byref(me)):
                break
    return None, None


def get_api_gt():
    ctx = ssl._create_unverified_context()
    try:
        r = urllib.request.urlopen("https://127.0.0.1:2999/liveclientdata/gamestats", context=ctx, timeout=2)
        return json.loads(r.read()).get("gameTime", 0)
    except Exception as e:
        return None


def replay_post(ep, body):
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(f"https://127.0.0.1:2999{ep}",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    try:
        urllib.request.urlopen(req, context=ctx, timeout=2)
    except Exception as e:
        pass


def lcu_post(ep, body):
    """Read riotclient lockfile, POST to LCU."""
    import base64
    lockfile = Path(os.environ.get("LOCALAPPDATA", "C:/Users/daniz/AppData/Local")) / "Riot Games" / "Riot Client" / "Config" / "lockfile"
    # LCU lockfile is at C:\Riot Games\League of Legends\lockfile
    league_lock = Path("C:/Riot Games/League of Legends/lockfile")
    if league_lock.exists():
        lockfile = league_lock
    if not lockfile.exists():
        print(f"ERROR: LCU lockfile not found at {lockfile}")
        return None
    parts = lockfile.read_text().split(":")
    port, password = parts[2], parts[3]
    auth = base64.b64encode(f"riot:{password}".encode()).decode()
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(f"https://127.0.0.1:{port}{ep}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Basic {auth}", "Content-Type":"application/json"})
    try:
        return urllib.request.urlopen(req, context=ctx, timeout=5).read()
    except Exception as e:
        print(f"LCU error: {e}")
        return None


def launch_replay_and_wait(game_id):
    print(f"Launching replay {game_id}...")
    lcu_post(f"/lol-replays/v1/rofls/{game_id}/watch", {"componentType":"replay"})
    for i in range(120):
        gt = get_api_gt()
        if gt is not None:
            print(f"Game loaded after {i*2}s, gt={gt}")
            return True
        time.sleep(2)
    print("TIMEOUT: replay didn't load")
    return False


def scan_f32_in_range(m, base, mod_size, lo, hi):
    """Scan module memory for all f32 in [lo, hi]. Returns list of RVAs."""
    hits = []
    SCAN_START = 0x1000000
    SCAN_END = mod_size - 4
    CHUNK = 0x100000
    for chunk_off in range(SCAN_START, SCAN_END, CHUNK):
        chunk_size = min(CHUNK, SCAN_END - chunk_off)
        buf = m.read(base + chunk_off, chunk_size)
        if not buf: continue
        for i in range(0, len(buf) - 3, 4):
            try:
                v = struct.unpack("<f", buf[i:i+4])[0]
            except struct.error:
                continue
            if lo <= v <= hi:
                hits.append(chunk_off + i)
    return hits


def main():
    game_id = sys.argv[1] if len(sys.argv) > 1 else None
    if game_id:
        if not launch_replay_and_wait(game_id):
            return 1
        # Let replay reach mid-game so gt is well past 0
        print("Waiting 30s for replay to advance past load screen...")
        time.sleep(30)
        # Bring to 1x speed unpause to verify gt advances
        replay_post("/replay/playback", {"speed":1.0, "paused":False})
        time.sleep(2)

    pid = find_league_pid()
    if not pid:
        print("ERROR: League PID not found"); return 1
    base, mod_size = find_module_base(pid)
    if not base:
        print("ERROR: module base not found"); return 1
    print(f"PID={pid} base=0x{base:X} mod_size=0x{mod_size:X}")

    m = Mem(pid)

    api_t0 = get_api_gt()
    if api_t0 is None or api_t0 < 30:
        print(f"ERROR: api_gt={api_t0} too small / replay not playing"); return 1
    print(f"T0: api_gt={api_t0:.2f}")

    print(f"Scanning {(mod_size - 0x1000000)//1024//1024}MB for f32 in [{api_t0-1:.1f}, {api_t0+1:.1f}]...")
    t = time.time()
    cands = scan_f32_in_range(m, base, mod_size, api_t0 - 1.0, api_t0 + 1.0)
    print(f"  found {len(cands)} candidates in {time.time()-t:.1f}s")

    SLEEP = 5.0
    print(f"Sleeping {SLEEP}s for replay to advance...")
    time.sleep(SLEEP)

    api_t1 = get_api_gt()
    delta = (api_t1 or 0) - api_t0
    print(f"T1: api_gt={api_t1}  delta={delta:.2f}")
    if delta < 1.0:
        print("ERROR: api_gt did not advance — replay paused?"); return 1

    survivors = []
    for rva in cands:
        v = m.f32(base + rva)
        if v is None: continue
        if abs(v - api_t1) < 1.0:
            survivors.append((rva, v))
    print(f"Survivors (advanced ~{delta:.1f}s): {len(survivors)}")
    for rva, v in survivors[:30]:
        print(f"  RVA 0x{rva:X} now reads {v:.2f}")
    if not survivors:
        print("✗ No live game_time found"); return 1

    survivors.sort(key=lambda x: abs(x[1] - api_t1))
    best_rva, best_v = survivors[0]
    print(f"\n✓ LIVE game_time at RVA 0x{best_rva:X} (read={best_v:.2f}, api={api_t1:.2f})")

    # Update offsets JSON in place
    repo_root = Path(__file__).parent
    json_path = repo_root / "offsets_16_9_772.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}
    old = data.get("game_time")
    data["game_time"] = best_rva
    versions = data.setdefault("_offset_versions", {})
    iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    versions["game_time"] = iso
    data["_scanned_at"] = iso
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {json_path}")
    print(f"  game_time: 0x{old:X} → 0x{best_rva:X}" if old else f"  game_time: → 0x{best_rva:X}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
