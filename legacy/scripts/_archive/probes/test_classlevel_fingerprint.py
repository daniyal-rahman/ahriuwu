"""For each of the 27 triple-mirror candidates, check the 28 offsets we found
identical between A1 and A2. Count matches, cluster.

If the click-dest CLASS is distinct from turret/camp triple-mirror structs,
the candidates will partition: a small group (containing A1, A2) matching
on most/all 28 class-level offsets, plus the rest with low match counts.

If all 27 match all 28 fields, those offsets are pan-triple-mirror noise,
not class-specific — and we have no signature.

Usage (paused replay): python scripts\\test_classlevel_fingerprint.py
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig=builtins.print
def print(*a,**k): k.setdefault("flush",True); _orig(*a,**k)
builtins.print=print

# 28 offsets that were identical between A1 (0x18447A4F834) and A2 (0x18444C9DA84)
# in the prior deep-classify run. Negative offsets = the heap header / preceding
# qwords (still readable since the alloc is on the same page).
CLASS_OFFSETS = [
    -0x30, -0x18,
    0x18, 0x20,
    0x48, 0x58,
    0x88, 0x98,
    0xC8, 0xD8,
    0x108, 0x118,
    0x1C8, 0x1D8,
    0x208, 0x218,
    0x248, 0x258,
    0x288, 0x298,
    0x2C8, 0x2F0,
    0x318,
    0x390, 0x3C0, 0x3D0, 0x3E0, 0x3E8,
]

# Known references from the prior run
A1_REF = 0x18447A4F834
A2_REF = 0x18444C9DA84

SB = 0x308; SC = 0x374

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

def read_bytes(h, addr, n):
    buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
    if not _k.ReadProcessMemory(h,ctypes.c_void_p(addr),buf,n,ctypes.byref(r)): return None
    return bytes(buf[:r.value]) if r.value else None

def fingerprint(h, alloc):
    """Return list of 28 u64 values at the class offsets, or None if read fails."""
    fp=[]
    for off in CLASS_OFFSETS:
        b=read_bytes(h, alloc+off, 8)
        if not b or len(b)<8: return None
        fp.append(struct.unpack("<Q", b)[0])
    return fp

def scan_triple_mirrors(h):
    """Re-scan to enumerate current triple-mirror candidates."""
    SB_F=SB//4; SC_F=SC//4
    cands=[]
    for base,size in regions(h):
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
            mask=((x0>100)&(x0<15000)&(z0>100)&(z0<15000)
                 &(y0>45)&(y0<65)
                 &(np.abs(x0-xb)<1e-2)&(np.abs(y0-yb)<1e-2)&(np.abs(z0-zb)<1e-2)
                 &(np.abs(x0-xc)<1e-2)&(np.abs(z0-zc)<1e-2))
        for i in np.nonzero(mask)[0]:
            cands.append((base+int(i)*4,(float(x0[i]),float(y0[i]),float(z0[i]))))
    return cands

def main():
    pid=find_pid()
    if not pid: print("ERR no League"); return 1
    h=_k.OpenProcess(0x0410, False, pid)
    if not h: print("ERR OpenProcess"); return 1
    print(f"pid={pid}")

    # 1. Reference fingerprint (A1)
    a1_fp = fingerprint(h, A1_REF)
    a2_fp = fingerprint(h, A2_REF)
    if not a1_fp:
        print(f"ERR: A1 0x{A1_REF:X} not readable — alloc died?"); return 1
    print(f"\nA1 0x{A1_REF:X} fingerprint loaded ({len(a1_fp)} fields)")
    if a2_fp:
        same_a1_a2 = sum(1 for x,y in zip(a1_fp, a2_fp) if x==y)
        print(f"  A1/A2 still agree on {same_a1_a2}/{len(a1_fp)} fields")

    # 2. Re-scan for triple-mirror candidates
    print("\nRe-scanning triple-mirror...")
    t0=time.time()
    cands = scan_triple_mirrors(h)
    print(f"  {len(cands)} candidates in {time.time()-t0:.1f}s")

    # 3. Score each by # of matches to A1's fingerprint
    print("\n  addr             vec                      matches/28   non-matching offs")
    rows=[]
    for addr,vec in sorted(cands):
        fp=fingerprint(h, addr)
        if not fp:
            rows.append((addr, vec, -1, "READ_FAIL")); continue
        matches = sum(1 for x,y in zip(fp, a1_fp) if x==y)
        diffs = [hex(off) for off,x,y in zip(CLASS_OFFSETS, fp, a1_fp) if x!=y]
        rows.append((addr, vec, matches, ",".join(diffs[:6])))
        is_a1 = " <- A1" if addr==A1_REF else ""
        is_a2 = " <- A2" if addr==A2_REF else ""
        print(f"  0x{addr:X}  ({vec[0]:7.1f},{vec[1]:6.2f},{vec[2]:7.1f})   {matches:>2}/{len(a1_fp):<2}      {','.join(diffs[:5]):<40}{is_a1}{is_a2}")

    # 4. Partition
    bucket = {}
    for addr,vec,matches,_ in rows:
        bucket.setdefault(matches, []).append((addr,vec))
    print(f"\n== Match-count distribution ==")
    for k in sorted(bucket.keys(), reverse=True):
        print(f"  {k:>2}/{len(a1_fp)} matches: {len(bucket[k])} candidates")

    # 5. Verdict
    print("\n" + "="*60)
    full_match = bucket.get(len(a1_fp), [])
    near_match = [r for r in rows if r[2] >= len(a1_fp)-2 and r[2] != len(a1_fp)]
    if len(full_match) == len(rows):
        print("VERDICT: all candidates match all 28 — fingerprint is GENERIC, not class-specific")
    elif len(full_match) <= 5:
        print(f"VERDICT: STRONG fingerprint — {len(full_match)} candidates match all 28")
        for a,v in full_match:
            tag = " <- A1" if a==A1_REF else (" <- A2" if a==A2_REF else "")
            print(f"  0x{a:X}  ({v[0]:.1f}, {v[1]:.2f}, {v[2]:.1f}){tag}")
    elif len(full_match) + len(near_match) <= 8:
        print(f"VERDICT: NEAR-fingerprint — {len(full_match)} exact + {len(near_match)} near-match (≥26/28)")
    else:
        print(f"VERDICT: weak fingerprint — {len(full_match)} match all 28")

    return 0

if __name__=="__main__":
    sys.exit(main())
