"""Verify the vtable hypothesis. Two checks:

  1. At known A1 = 0x18447A4F834, read qword at A1-0x14 and confirm it equals
     module_base + 0x192BF90.
  2. Scan all heap RW pages for qwords equal to module_base + 0x192BF90, then
     for each hit confirm there's a valid Vec3 at hit+0x14, and a triple-mirror
     at hit+0x14+0x308 / hit+0x14+0x374.
     Compare result count against |T|=27 / |N|=18 from earlier.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig=builtins.print
def print(*a,**k): k.setdefault("flush",True); _orig(*a,**k)
builtins.print=print

VTABLE_RVA = 0x192BF90
VEC3_OFFSET_FROM_VPTR = 0x14
A1_KNOWN = 0x18447A4F834

_k = ctypes.windll.kernel32
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
MEM_COMMIT=0x1000; MEM_PRIVATE=0x20000; PAGE_RW=0x04|0x08|0x40

def find_pid():
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                     capture_output=True,text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

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
            _k.CloseHandle(h)
            return mi.lpBaseOfDll, mi.SizeOfImage
    _k.CloseHandle(h)
    return None, None

def read_bytes(h, addr, n):
    buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
    if not _k.ReadProcessMemory(h,ctypes.c_void_p(addr),buf,n,ctypes.byref(r)): return None
    return bytes(buf[:r.value]) if r.value else None

def read_u64(h, addr):
    b=read_bytes(h, addr, 8)
    return struct.unpack("<Q", b)[0] if b and len(b)==8 else None

def regions(h, max_size=64*1024*1024):
    addr=0; mbi=MBI()
    while addr<0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h,ctypes.c_void_p(addr),ctypes.byref(mbi),ctypes.sizeof(mbi)): break
        b=mbi.BaseAddress or 0; s=mbi.RegionSize
        if (mbi.State==MEM_COMMIT and mbi.Type==MEM_PRIVATE
                and (mbi.Protect&PAGE_RW) and s<=max_size):
            yield b,s
        addr=b+s
        if addr<=b: break

def read_region(h, base, size):
    out=bytearray(size); v=memoryview(out); o=0; CH=4*1024*1024
    while o<size:
        n=min(CH,size-o)
        buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h,ctypes.c_void_p(base+o),buf,n,ctypes.byref(r)) or r.value==0:
            return None if o==0 else bytes(v[:o])
        v[o:o+r.value]=buf[:r.value]; o+=r.value
    return bytes(out)

def main():
    pid = find_pid()
    if not pid: print("ERR: League not running"); return 1
    mod_base, mod_size = module_range(pid)
    if not mod_base: print("ERR module"); return 1
    print(f"pid={pid}  module_base=0x{mod_base:X}")

    target_vptr = mod_base + VTABLE_RVA
    print(f"expected vptr value = 0x{target_vptr:X}  (module+0x{VTABLE_RVA:X})")

    h = _k.OpenProcess(0x0410, False, pid)

    # --- Test 1: read A1-0x14 from the known A1 address ---
    val = read_u64(h, A1_KNOWN - 0x14)
    if val is None:
        print(f"\nTEST 1: A1 0x{A1_KNOWN:X} no longer readable (alloc died?)")
    else:
        print(f"\nTEST 1: at known A1-0x14 = 0x{A1_KNOWN-0x14:X}, qword = 0x{val:X}")
        if val == target_vptr:
            print(f"        MATCH ✓ — vtable hypothesis confirmed")
        else:
            diff = val - target_vptr
            print(f"        MISMATCH — diff = 0x{diff:X}")

    # --- Test 2: scan all heap for qwords equal to target_vptr ---
    print(f"\nTEST 2: scanning heap for qwords == 0x{target_vptr:X} ...")
    t0 = time.time()
    target_bytes = struct.pack("<Q", target_vptr)
    candidates = []
    rcount = 0
    for base, size in regions(h):
        rcount += 1
        data = read_region(h, base, size)
        if not data: continue
        # find all 8-byte aligned positions with target qword
        n = len(data) // 8 * 8
        if n < 8: continue
        arr = np.frombuffer(data[:n], dtype=np.uint64)
        idxs = np.nonzero(arr == np.uint64(target_vptr))[0]
        for i in idxs:
            candidates.append(base + int(i)*8)
    elapsed = time.time() - t0
    print(f"  scanned {rcount} regions in {elapsed:.1f}s — {len(candidates)} qwords match")

    # For each, validate Vec3 + triple-mirror at +0x14+...
    SB = 0x308; SC = 0x374
    valid = []
    for vptr_addr in candidates:
        vec3_addr = vptr_addr + VEC3_OFFSET_FROM_VPTR
        bv = read_bytes(h, vec3_addr, 12)
        if not bv or len(bv) < 12: continue
        x, y, z = struct.unpack("<fff", bv)
        if not (100 < x < 15000 and 100 < z < 15000 and 45 < y < 65): continue
        # Mirror B/C
        bb = read_bytes(h, vec3_addr + SB, 12)
        bc = read_bytes(h, vec3_addr + SC, 12)
        if not (bb and bc and len(bb)==12 and len(bc)==12): continue
        xb, yb, zb = struct.unpack("<fff", bb)
        xc, yc, zc = struct.unpack("<fff", bc)
        if abs(x-xb)<0.01 and abs(y-yb)<0.01 and abs(z-zb)<0.01 and abs(x-xc)<0.01 and abs(z-zc)<0.01:
            valid.append((vptr_addr, vec3_addr, (x,y,z)))

    print(f"  vptr matches with valid Vec3 + triple-mirror: {len(valid)}")
    for vptr, vec3_addr, vec in valid:
        is_a1 = " <- A1" if vec3_addr == A1_KNOWN else ""
        print(f"    vptr@0x{vptr:X}  vec3@0x{vec3_addr:X}  ({vec[0]:.1f}, {vec[1]:.2f}, {vec[2]:.1f}){is_a1}")

    print()
    print("="*60)
    if len(valid) == 0:
        print("VERDICT: no vtable hits — RVA wrong or process state changed")
    elif len(valid) <= 5:
        print(f"VERDICT: STRONG signature — {len(valid)} click-dest objects identified")
    elif len(valid) <= 30:
        print(f"VERDICT: vtable + triple-mirror gives {len(valid)} candidates")
        print(f"         (compare against earlier: |T|=27, |N|<nil>=18)")
    else:
        print(f"VERDICT: vtable scan gives unexpectedly many ({len(valid)})")
    return 0

if __name__=="__main__":
    sys.exit(main())
