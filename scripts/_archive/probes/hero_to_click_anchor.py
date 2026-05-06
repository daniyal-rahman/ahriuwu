"""Check if Bel'Veth's hero struct contains any pointer to her click-dest
allocation, OR to anywhere near it.

Path:
  1. Read focused_hero_ptr at module+0x1E13490 → hero_base
  2. Read the first 0x20000 bytes of the hero struct
  3. Scan every u64 slot for pointers that land in [alloc, alloc+0x400)
     OR in a wider range around the alloc (its heap neighborhood)
  4. Also: scan the hero struct for any u64 pointing into the SAME heap
     region as the alloc — those could be sibling struct pointers

Usage: python hero_to_click_anchor.py --alloc 0x1F37A7793A4
"""
import ctypes, ctypes.wintypes as wt
import argparse, struct, subprocess, sys, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

_k = ctypes.windll.kernel32
FOCUSED_HERO_RVA = 0x1E13490
HERO_STRUCT_SIZE = 0x20000

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def module_range(pid):
    psapi = ctypes.WinDLL("psapi.dll")
    h = _k.OpenProcess(0x0410, False, pid)
    HMODULE = wt.HMODULE
    psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, HMODULE, wt.LPWSTR, wt.DWORD]
    class MINFO(ctypes.Structure):
        _fields_ = [("lpBaseOfDll", ctypes.c_void_p), ("SizeOfImage", wt.DWORD), ("EntryPoint", ctypes.c_void_p)]
    psapi.GetModuleInformation.argtypes = [wt.HANDLE, HMODULE, ctypes.POINTER(MINFO), wt.DWORD]
    mods = (HMODULE * 1024)(); needed = wt.DWORD(0)
    psapi.EnumProcessModulesEx(h, mods, ctypes.sizeof(mods), ctypes.byref(needed), 3)
    n = needed.value // ctypes.sizeof(HMODULE)
    for i in range(n):
        name = ctypes.create_unicode_buffer(260)
        psapi.GetModuleFileNameExW(h, mods[i], name, 260)
        if name.value.lower().endswith("league of legends.exe"):
            mi = MINFO()
            psapi.GetModuleInformation(h, mods[i], ctypes.byref(mi), ctypes.sizeof(mi))
            return mi.lpBaseOfDll, mi.SizeOfImage
    return None, None

def read_u64(h, addr):
    buf = (ctypes.c_char * 8)(); r = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, 8, ctypes.byref(r))
    return struct.unpack("<Q", bytes(buf))[0] if ok and r.value == 8 else None

def read_bytes(h, addr, n):
    buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--neighborhood", default="0x100000",
                    help="heap neighborhood around alloc to consider (bytes)")
    args = ap.parse_args()
    alloc = int(args.alloc, 16)
    neighborhood = int(args.neighborhood, 16)

    pid = find_pid()
    if not pid:
        print("ERR: no league"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    mod_base, _ = module_range(pid)
    print(f"pid={pid} module=0x{mod_base:X}")

    # Step 1: resolve focused hero
    fhp_addr = mod_base + FOCUSED_HERO_RVA
    hero_base = read_u64(h, fhp_addr)
    if not hero_base:
        print(f"ERR: couldn't read focused_hero_ptr at 0x{fhp_addr:X}"); return 1
    print(f"focused_hero_ptr=0x{fhp_addr:X} -> hero_base=0x{hero_base:X}")

    # Quick sanity: read champion name at hero+0x4360
    name = read_bytes(h, hero_base + 0x4360, 32)
    name_s = (name.split(b"\x00")[0] if name else b"?").decode("ascii", errors="replace")
    print(f"  hero champion_name = '{name_s}'")
    pos_bytes = read_bytes(h, hero_base + 0x200, 12)
    if pos_bytes:
        pos = struct.unpack("<fff", pos_bytes)
        print(f"  hero position = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    # Step 2: read the full hero struct
    data = read_bytes(h, hero_base, HERO_STRUCT_SIZE)
    if not data:
        print("ERR: couldn't read hero struct"); return 1
    print(f"read {len(data)} bytes of hero struct")

    # Step 3: find pointers into [alloc, alloc+0x400) from hero struct
    targets_exact = set(range(alloc, alloc + 0x400, 1))
    arr = np.frombuffer(data[:(len(data) & ~7)], dtype=np.uint64)
    print(f"\n== Scanning {len(arr)} u64 slots in hero struct for ptrs into alloc ==")
    hits_exact = []
    for i in range(len(arr)):
        v = int(arr[i])
        if alloc <= v < alloc + 0x400:
            hits_exact.append((hero_base + i * 8, v))
    print(f"  exact alloc match: {len(hits_exact)}")
    for src, tgt in hits_exact:
        off = src - hero_base
        print(f"    hero+0x{off:X} -> alloc+0x{tgt-alloc:X}")

    # Step 4: pointers into wider neighborhood (alloc ± neighborhood)
    lo, hi = alloc - neighborhood, alloc + neighborhood
    hits_nb = []
    for i in range(len(arr)):
        v = int(arr[i])
        if lo <= v < hi and not (alloc <= v < alloc + 0x400):
            hits_nb.append((hero_base + i * 8, v))
    print(f"\n  ptrs to heap neighborhood [-0x{neighborhood:X},+0x{neighborhood:X}]: "
          f"{len(hits_nb)}")
    for src, tgt in hits_nb[:20]:
        off = src - hero_base
        print(f"    hero+0x{off:X} -> 0x{tgt:X}  (alloc{tgt-alloc:+d})")

    # Also check 4-byte aligned (u64 read at offset 4)
    arr4 = np.frombuffer(data[4:4+((len(data)-4) & ~7)], dtype=np.uint64)
    hits4 = []
    for i in range(len(arr4)):
        v = int(arr4[i])
        if alloc <= v < alloc + 0x400:
            hits4.append((hero_base + 4 + i * 8, v))
    print(f"\n  4-byte aligned alloc hits: {len(hits4)}")
    for src, tgt in hits4:
        off = src - hero_base
        print(f"    hero+0x{off:X} -> alloc+0x{tgt-alloc:X}")

    out = {
        "pid": pid, "module_base": hex(mod_base),
        "hero_base": hex(hero_base), "champion": name_s,
        "alloc": hex(alloc),
        "hits_exact_8aligned": [{"hero_off": hex(s-hero_base), "tgt": hex(t)} for s,t in hits_exact],
        "hits_exact_4aligned": [{"hero_off": hex(s-hero_base), "tgt": hex(t)} for s,t in hits4],
        "hits_neighborhood": [{"hero_off": hex(s-hero_base), "tgt": hex(t)} for s,t in hits_nb[:100]],
    }
    with open(r"C:\tmp\hero_to_click_anchor.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote C:\\tmp\\hero_to_click_anchor.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
