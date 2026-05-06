"""Scan League process memory for Bel'Veth's champion-name string and derive
a verified hero struct pointer. Outputs the struct address + position.

Also finds the hero_array by searching the module for a pointer-to-that-struct.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

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

class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok else b""
    def u64(self, a):
        d = self.read(a, 8); return struct.unpack("<Q", d)[0] if len(d)==8 else None
    def vec3(self, a):
        d = self.read(a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)

def scan_regions(m, pattern, min_addr=0, max_addr=0x7FFFFFFFFFFF, readable_only=True):
    """Yield (addr, data) tuples for all committed RW regions containing `pattern`."""
    addr = min_addr
    mbi = MBI()
    while addr < max_addr:
        ok = _k.VirtualQueryEx(m.h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
        if not ok: break
        base = mbi.BaseAddress or 0
        size = mbi.RegionSize
        state = mbi.State
        protect = mbi.Protect
        if state == 0x1000 and protect & 0x44:  # MEM_COMMIT and (RW-ish)
            # only scan heap-like (private+commit) to skip modules/mapped files
            data = m.read(base, min(size, 200*1024*1024))
            if pattern in data:
                off = 0
                while True:
                    i = data.find(pattern, off)
                    if i < 0: break
                    yield base + i
                    off = i + 1
        addr = base + size
        if addr <= base: break

def main():
    pid = find_pid()
    base, mod_size = find_base(pid)
    print(f"PID={pid} module base=0x{base:X} size=0x{mod_size:X}")
    m = Mem(pid)

    # Search for "Belveth\0" bytes — League internal champion names usually lowercase after first
    for pat in (b"Belveth\x00", b"belveth\x00", b"BelVeth\x00", b"BELVETH\x00"):
        hits = list(scan_regions(m, pat))
        if hits:
            print(f"Pattern {pat!r}: {len(hits)} hits")
            for h in hits[:20]:
                # Read surrounding bytes to see what struct this is in
                tail = m.read(h, 32)
                print(f"  0x{h:X}  {tail!r}")
            # Best guess: the champion_name offset in hero struct is 0x4360.
            # So hero struct base candidate = hit - 0x4360.
            cands = []
            for h in hits:
                cand = h - 0x4360
                x, y, z = m.vec3(cand + 0x200)
                if -200 < y < 200 and 0 < x < 16000 and 0 < z < 16000:
                    cands.append((cand, x, y, z))
            print(f"\n{len(cands)} candidates pass the 'pos looks like map coords' filter (x,z in [0,16000], y in [-200,200])")
            for c, x, y, z in cands[:10]:
                print(f"  hero_struct=0x{c:X}  pos=({x:.0f},{y:.0f},{z:.0f})")

            if cands:
                # Now find pointer-to-this-struct inside the module (exe image)
                # to locate hero_array
                target_ptrs = {c for c, *_ in cands}
                print(f"\nSearching module image (0x{base:X}..0x{base+mod_size:X}) for pointers to {len(target_ptrs)} candidate(s)...")
                chunk = m.read(base, mod_size)
                refs = []
                for i in range(0, len(chunk)-8, 8):
                    v = struct.unpack_from("<Q", chunk, i)[0]
                    if v in target_ptrs:
                        refs.append((base + i, v))
                print(f"{len(refs)} pointers found in module image")
                for a, v in refs[:30]:
                    rva = a - base
                    print(f"  module+0x{rva:X}  -> 0x{v:X}")
                # Also search heap regions (non-module) for pointers
                return
            break
    print("No Bel'Veth name bytes found.")

if __name__ == "__main__":
    main()
