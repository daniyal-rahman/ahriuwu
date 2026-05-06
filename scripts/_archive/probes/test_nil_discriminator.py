"""Test whether the bytes "<nil" at offset +0x0C of a triple-mirror Vec3 alloc
uniquely identifies click-destination allocs (vs turrets / camps / etc).

Two scans:
  T  = triple-mirror filter alone (mirrors at +0/+0x308/+0x374, on-map Vec3)
  N  = T + (alloc[+0x0C..+0x0F] == "<nil")

Reports:
  |T|, |N|, |T \\ N|  — i.e. triple-mirror hits that lack <nil
  Sample of N hits and T-only hits with their Vec3s

If |N| << |T| we have a cheap discriminator. If |N| ≈ |T| <nil isn't
distinctive. If |N| == 0 the hypothesis is wrong.

Usage (paused replay):  python scripts\\test_nil_discriminator.py
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig=builtins.print
def print(*a,**k): k.setdefault("flush",True); _orig(*a,**k)
builtins.print=print

NIL_BYTES = b"<nil"  # 0x6C696E3C little-endian when read as u32
NIL_OFFSET = 0x0C
SB = 0x308
SC = 0x374

_k = ctypes.windll.kernel32
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
MEM_COMMIT=0x1000; MEM_PRIVATE=0x20000
PAGE_RW=0x04|0x08|0x40

def find_pid():
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                     capture_output=True,text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def regions(h, max_size=32*1024*1024):
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

def scan(h):
    """Return (triple_mirror_set, nil_filtered_set, candidates_with_metadata)."""
    SB_F = SB // 4; SC_F = SC // 4
    NIL_F = NIL_OFFSET // 4
    triple = []     # list of (addr, vec3)
    with_nil = []
    rcount=0
    t0=time.time()
    for base,size in regions(h):
        rcount+=1
        data=read_region(h,base,size)
        if not data or len(data)<0x400: continue
        n_f=len(data)//4
        if n_f<=SC_F+3: continue
        arr=np.frombuffer(data, dtype=np.float32, count=n_f)
        L=n_f-SC_F-3
        if L<=0: continue
        x0=arr[:L]; y0=arr[1:L+1]; z0=arr[2:L+2]
        xb=arr[SB_F:SB_F+L]; yb=arr[SB_F+1:SB_F+1+L]; zb=arr[SB_F+2:SB_F+2+L]
        xc=arr[SC_F:SC_F+L]; yc=arr[SC_F+1:SC_F+1+L]; zc=arr[SC_F+2:SC_F+2+L]
        with np.errstate(invalid='ignore', over='ignore'):
            mask = ((x0>100)&(x0<15000)&(z0>100)&(z0<15000)
                   &(y0>45)&(y0<65)
                   &(np.abs(x0-xb)<1e-2)&(np.abs(y0-yb)<1e-2)&(np.abs(z0-zb)<1e-2)
                   &(np.abs(x0-xc)<1e-2)&(np.abs(z0-zc)<1e-2))
        idxs = np.nonzero(mask)[0]
        # also pull the 4 bytes at +0x0C for each hit, check for "<nil"
        u32arr = np.frombuffer(data, dtype=np.uint32, count=n_f)
        for i in idxs:
            addr = base + int(i)*4
            vec = (float(x0[i]), float(y0[i]), float(z0[i]))
            triple.append((addr, vec))
            # Check tag at i + NIL_F (4-byte units)
            tag_idx = int(i) + NIL_F
            if tag_idx < n_f:
                tag = int(u32arr[tag_idx])
                # "<nil" little-endian as u32 = 0x6C696E3C
                if tag == 0x6C696E3C:
                    with_nil.append((addr, vec, hex(tag)))
    elapsed = time.time()-t0
    print(f"  scanned {rcount} regions in {elapsed:.1f}s")
    return triple, with_nil

def main():
    pid=find_pid()
    if not pid: print("ERR no League"); return 1
    h=_k.OpenProcess(0x0410, False, pid)
    if not h: print("ERR OpenProcess"); return 1
    print(f"pid={pid}")
    print("scanning...")
    triple, with_nil = scan(h)
    print(f"\n|T| triple-mirror candidates:                {len(triple)}")
    print(f"|N| triple-mirror + <nil at +0x0C:           {len(with_nil)}")
    print(f"|T\\N| triple without <nil:                   {len(triple)-len(with_nil)}")
    nil_addrs = set(a for a,_,_ in with_nil)
    t_only = [(a,v) for a,v in triple if a not in nil_addrs]

    print(f"\n-- Sample of N hits (with <nil) --")
    for a,v,t in with_nil[:30]:
        print(f"  0x{a:X}  vec=({v[0]:7.1f},{v[1]:6.2f},{v[2]:7.1f})  tag={t}")
    if len(with_nil) > 30:
        print(f"  ... and {len(with_nil)-30} more")

    print(f"\n-- Sample of T-only hits (no <nil) --")
    for a,v in t_only[:15]:
        print(f"  0x{a:X}  vec=({v[0]:7.1f},{v[1]:6.2f},{v[2]:7.1f})")
    if len(t_only) > 15:
        print(f"  ... and {len(t_only)-15} more")

    # Verdict
    print("\n" + "="*60)
    if len(triple) == 0:
        print("VERDICT: no triple-mirror hits — replay state suspect")
    elif len(with_nil) == 0:
        print("VERDICT: <nil> hypothesis WRONG (no triple-mirror has <nil)")
    elif len(with_nil) <= 5 and len(triple) > 20:
        print(f"VERDICT: <nil> is a STRONG discriminator ({len(with_nil)} of {len(triple)})")
    elif len(with_nil) < len(triple) * 0.3:
        print(f"VERDICT: <nil> is a partial discriminator ({len(with_nil)} of {len(triple)})")
    else:
        print(f"VERDICT: <nil> doesn't discriminate well ({len(with_nil)} of {len(triple)})")
    return 0

if __name__=="__main__":
    sys.exit(main())
