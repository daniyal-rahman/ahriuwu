"""Probe the 3 addresses identified by the process-wide scan as click-destination
holders. For each window, pause the replay and read:
  - The Vec3 at each of the 3 addresses
  - Surrounding 64 bytes to see struct layout
  - The containing allocation's base (one common parent?)

Also try to find a pointer-chain from the module or from Bel'Veth's hero struct
that leads to this allocation.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time
import ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

ADDRS = [0x2075BF70114, 0x2075BF7041C, 0x2075BF70488]
FOCUSED_HERO_PTR_RVA = 0x1E13490
WINDOWS = [("win1", 40.0, 3124.0, 8122.0),
           ("win2", 44.5, 3736.0, 8358.0),
           ("win3", 49.0, 4398.0, 8444.0)]

_k = ctypes.windll.kernel32
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t

class ME32(ctypes.Structure):
    _fields_ = [("dwSize",wt.DWORD),("a",wt.DWORD),("pid",wt.DWORD),
                ("b",wt.DWORD),("c",wt.DWORD),
                ("modBase",ctypes.POINTER(ctypes.c_byte)),("modSize",wt.DWORD),
                ("hMod",wt.HMODULE),("szMod",ctypes.c_char*256),
                ("szPath",ctypes.c_char*260)]
def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
def find_base(pid):
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME32(); me.dwSize = ctypes.sizeof(ME32)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szMod.lower():
                b = ctypes.cast(me.modBase, ctypes.c_void_p).value
                _k.CloseHandle(snap); return b, me.modSize
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap); return None, None

def read(h, a, sz):
    buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
    return buf.raw[:n.value] if ok else b""

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def find_enclosing_alloc(h, addr):
    mbi = MBI()
    _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
    return mbi.AllocationBase or 0, mbi.RegionSize, mbi.BaseAddress or 0

def fmt_vec3(b, off):
    if off + 12 > len(b): return "-"
    x, y, z = struct.unpack_from("<fff", b, off)
    return f"({x:.1f}, {y:.2f}, {z:.1f})"

def main():
    pid = find_pid(); base, size = find_base(pid)
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"PID={pid} module base=0x{base:X} size=0x{size:X}")

    # First seek to 36 paused
    _post({"time": 36.0, "speed": 0.25, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break

    # Enumerate the allocation containing our addresses
    alloc_base, region_size, region_base = find_enclosing_alloc(h, ADDRS[0])
    print(f"\nContaining allocation: base=0x{alloc_base:X} region_base=0x{region_base:X} size=0x{region_size:X}")
    min_addr = min(ADDRS); max_addr = max(ADDRS) + 12
    print(f"Addresses span: 0x{min_addr:X} .. 0x{max_addr:X} (range 0x{max_addr-min_addr:X} bytes)")

    # For each window, unpause briefly to let replay reach that time, then pause & read
    for label, gt_target, ex, ez in WINDOWS:
        _post({"time": gt_target - 1.5, "speed": 0.5, "paused": True})
        for _ in range(20):
            time.sleep(0.3)
            st = _get()
            if not st["seeking"] and st["paused"]: break
        _post({"speed": 0.5, "paused": False})
        # Wait till at gt_target
        for _ in range(60):
            time.sleep(0.1)
            st = _get()
            if st["time"] >= gt_target: break
        _post({"speed": 0.5, "paused": True})
        time.sleep(0.3)
        st = _get()
        print(f"\n=== {label}: paused at gt={st['time']:.2f}, searching for ({ex},?,{ez}) ===")

        # Read the containing allocation
        # Cap to ~200KB around our addresses for readability
        span_base = min_addr - 0x200
        span_size = max_addr - span_base + 0x200
        raw = read(h, span_base, span_size)
        print(f"Read {len(raw)} bytes from 0x{span_base:X}")
        # Print Vec3 at each of our 3 addresses
        for a in ADDRS:
            off = a - span_base
            if 0 <= off < len(raw):
                v = struct.unpack_from("<fff", raw, off) if off + 12 <= len(raw) else None
                tag = " MATCH" if v and abs(v[0]-ex) < 5 and abs(v[2]-ez) < 5 else ""
                print(f"  0x{a:X}  = {fmt_vec3(raw, off)}{tag}")
        # Print 64 bytes surrounding each address
        print("\nHex dumps around each match:")
        for a in ADDRS:
            off = a - span_base
            dump_start = max(0, off - 32)
            dump = raw[dump_start:off + 64]
            print(f"  0x{span_base + dump_start:X} (base+{dump_start-off}):")
            for i in range(0, len(dump), 16):
                hex_s = " ".join(f"{b:02x}" for b in dump[i:i+16])
                asc_s = "".join(chr(b) if 32 <= b < 127 else '.' for b in dump[i:i+16])
                addr_here = span_base + dump_start + i
                marker = " <<<" if addr_here <= a <= addr_here + 11 else ""
                print(f"    0x{addr_here:X}  {hex_s}  |{asc_s}|{marker}")

    # Now find who points to this allocation. Scan the module image.
    print(f"\n=== finding pointers to the click-buffer allocation (scanning module image) ===")
    mod_raw = read(h, base, size)
    print(f"Read {len(mod_raw)} bytes of module")
    target_lo = alloc_base or min_addr - 0x1000
    target_hi = (alloc_base or min_addr) + max(region_size, 0x10000)
    # Scan u64 pointers in module that land inside the target allocation
    refs = []
    import numpy as np
    arr = np.frombuffer(mod_raw, dtype=np.uint64)
    mask = (arr >= target_lo) & (arr < target_hi)
    idx = np.nonzero(mask)[0]
    for i in idx[:50]:
        ptr_rva = int(i) * 8
        ptr_val = int(arr[i])
        refs.append((ptr_rva, ptr_val))
    print(f"{len(idx)} module pointers land inside 0x{target_lo:X}..0x{target_hi:X}")
    for rva, val in refs[:20]:
        print(f"  module+0x{rva:X} -> 0x{val:X}")

    # Also scan hero struct for pointers into this allocation
    hero = struct.unpack("<Q", read(h, base + FOCUSED_HERO_PTR_RVA, 8))[0]
    hero_raw = read(h, hero, 128 * 1024)
    print(f"\nHero struct at 0x{hero:X}:")
    hero_arr = np.frombuffer(hero_raw, dtype=np.uint64)
    h_mask = (hero_arr >= target_lo) & (hero_arr < target_hi)
    h_idx = np.nonzero(h_mask)[0]
    print(f"{len(h_idx)} hero-struct pointers land inside target allocation:")
    for i in h_idx[:20]:
        off = int(i) * 8
        val = int(hero_arr[i])
        print(f"  hero+0x{off:04X} -> 0x{val:X}")

if __name__ == "__main__":
    main()
