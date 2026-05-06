"""Deep-classify the click-destination alloc: forward pointer walk + A1/A2 diff.

Successor to analyze_belveth_alloc.py. That script dumped a single alloc and
ran a reverse-pointer scan (nothing pointed back into it). This one goes the
other direction: for every qword field in the alloc, follow it one or two
hops, and classify where it terminates. The goal is to find a stable anchor
back into module-static territory so we don't need a 90s process scan per
replay.

Usage (run on Windows, League replay actively playing, tightpoll has already
identified a valid alloc address):

    python scripts\\alloc_deep_classify.py \\
        --alloc 0x1f280743324 \\
        --alloc2 0x1f2820df044 \\
        --hero  0x1f2801234567 \\
        --size  0x400

--alloc2 optional: if given, produces an aligned A1/A2 diff so we can see
which fields are class-level (identical in both) vs per-instance.
--hero optional: if given, flags alloc pointers that land inside the hero
struct (hero .. hero+0x20000).

Outputs a human-readable table to stdout and JSON to
C:\\tmp\\alloc_deep_classify.json.
"""
import ctypes, ctypes.wintypes as wt
import argparse, struct, subprocess, sys, json, time
import numpy as np

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
MEM_COMMIT = 0x1000; MEM_PRIVATE = 0x20000; MEM_IMAGE = 0x1000000
PAGE_READABLE = 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80
PAGE_EXECUTABLE = 0x10 | 0x20 | 0x40 | 0x80

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
            _k.CloseHandle(h)
            return mi.lpBaseOfDll, mi.SizeOfImage
    _k.CloseHandle(h)
    return None, None

def read_bytes(h, addr, n):
    buf = (ctypes.c_char * n)()
    r = ctypes.c_size_t(0)
    if not _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r)):
        return None
    return bytes(buf[:r.value]) if r.value else None

def build_region_map(h):
    """Build a sorted list of (base, end, protect, type) covering all committed
    readable pages. Used to classify any address in O(log n)."""
    regs = []
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        b = mbi.BaseAddress or 0; s = mbi.RegionSize
        if mbi.State == MEM_COMMIT and (mbi.Protect & PAGE_READABLE):
            regs.append((b, b + s, int(mbi.Protect), int(mbi.Type)))
        addr = b + s
        if addr <= b: break
    regs.sort()
    return regs

def region_of(regs, addr):
    """Return (base, end, protect, type) for the region containing addr, or None."""
    # linear ok for now; regs typically ~1-2k entries
    for b, e, p, t in regs:
        if b <= addr < e:
            return (b, e, p, t)
        if b > addr:
            return None
    return None

def classify_target(addr, mod_base, mod_size, alloc_base, alloc_end, hero_base, hero_size, regs):
    """Return a short tag describing what `addr` points into."""
    if addr == 0:
        return "NULL"
    if addr < 0x10000:  # small int, not a pointer
        return ""
    if mod_base is not None and mod_base <= addr < mod_base + mod_size:
        rva = addr - mod_base
        reg = region_of(regs, addr)
        subsection = ""
        if reg and (reg[2] & PAGE_EXECUTABLE):
            subsection = ".text"
        elif reg:
            subsection = ".rdata/.data"
        return f"MODULE+0x{rva:X} ({subsection})"
    if alloc_base <= addr < alloc_end:
        return f"SELF+0x{addr - alloc_base:X}"
    if hero_base is not None and hero_base <= addr < hero_base + hero_size:
        return f"HERO+0x{addr - hero_base:X}"
    reg = region_of(regs, addr)
    if reg:
        b, e, p, t = reg
        kind = "HEAP" if t == MEM_PRIVATE else ("IMAGE" if t == MEM_IMAGE else "MAPPED")
        return f"{kind}@0x{addr:X}"
    return f"UNMAPPED 0x{addr:X}"

def two_hop_module(h, addr, mod_base, mod_size):
    """Read 0x80 at addr, return True if it contains a module pointer."""
    data = read_bytes(h, addr, 0x80)
    if not data: return False, None
    for i in range(0, len(data) - 7, 8):
        u = struct.unpack_from("<Q", data, i)[0]
        if mod_base <= u < mod_base + mod_size:
            return True, (addr + i, u - mod_base)
    return False, None

def classify_alloc(h, alloc_base, size, mod_base, mod_size, hero_base, hero_size, regs, prefix_scan=0x40):
    """Walk every qword in [alloc-prefix, alloc+size] and classify."""
    data = read_bytes(h, alloc_base - prefix_scan, prefix_scan + size)
    if not data:
        return None
    fields = []
    for i in range(0, len(data) - 7, 8):
        off = i - prefix_scan  # offset relative to alloc_base
        u64 = struct.unpack_from("<Q", data, i)[0]
        fa = struct.unpack_from("<f", data, i)[0]
        fb = struct.unpack_from("<f", data, i + 4)[0]
        alloc_end = alloc_base + size
        tgt_tag = classify_target(u64, mod_base, mod_size, alloc_base, alloc_end,
                                   hero_base, hero_size, regs)
        # 2-hop module reachability for HEAP targets
        two_hop = None
        if tgt_tag.startswith("HEAP@") and u64 > 0x10000:
            ok, info = two_hop_module(h, u64, mod_base, mod_size)
            if ok:
                two_hop = info
        # ASCII sniff
        raw = data[i:i+8]
        ascii_repr = ""
        if all(32 <= b <= 126 or b in (0, 9, 10) for b in raw):
            s = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw).rstrip("\x00 ").rstrip(".")
            if s and len(s) >= 2:
                ascii_repr = s
        fields.append({
            "off": off,
            "u64": u64,
            "f32a": fa,
            "f32b": fb,
            "target": tgt_tag,
            "two_hop_module": two_hop,
            "ascii": ascii_repr,
        })
    return fields

def print_alloc_table(label, fields, mod_base):
    print(f"\n== {label} ==")
    print(f"  off    qword              f32a       f32b        target                           ascii       2hop-module")
    interesting = []
    for f in fields:
        off = f["off"]
        # Highlight known mirror positions
        mark = ""
        if off == 0x000: mark = " [MirA]"
        elif off == 0x308: mark = " [MirB]"
        elif off == 0x374: mark = " [MirC]"
        fmt = lambda v: f"{v:>10.2f}" if 0.01 < abs(v) < 1e7 else (f"{v:>10.2e}" if v != 0 else "      0.00")
        two_hop = ""
        if f["two_hop_module"]:
            src, rva = f["two_hop_module"]
            two_hop = f"via +0x{src & 0x7F:X} -> MOD+0x{rva:X}"
            interesting.append((off, f["target"], rva))
        line = f"  +{off:+05X} {f['u64']:016X}  {fmt(f['f32a'])} {fmt(f['f32b'])}  {f['target']:<30}  {f['ascii']:<10} {two_hop}{mark}"
        print(line)
        # Also flag direct module pointers
        if f["target"].startswith("MODULE+"):
            interesting.append((off, f["target"], None))
    if interesting:
        print(f"\n  Anchors-of-interest from this alloc:")
        for off, tgt, via in interesting[:20]:
            v = f" (2hop->MOD+0x{via:X})" if via is not None else ""
            print(f"    +0x{off:X} -> {tgt}{v}")
    return interesting

def diff_allocs(fa, fb):
    """Print field-by-field diff of two classifications."""
    print("\n== A1 vs A2 field diff ==")
    print("  off    a1_qword          a2_qword          same?   a1_target                a2_target")
    same_anchors = []
    for x, y in zip(fa, fb):
        if x["off"] != y["off"]:
            continue
        same = "YES" if x["u64"] == y["u64"] else "no "
        off = x["off"]
        if x["u64"] == y["u64"] and x["target"] and not x["target"].startswith(("NULL", "SELF")):
            same_anchors.append((off, x["u64"], x["target"]))
        # trim log: show only interesting rows
        interesting = (x["u64"] != y["u64"]
                      or x["target"].startswith("MODULE+")
                      or x["target"].startswith("HERO+"))
        if interesting or off in (0, 0x308, 0x374):
            print(f"  +{off:+05X} {x['u64']:016X}  {y['u64']:016X}  {same}   {x['target']:<25} {y['target']:<25}")
    if same_anchors:
        print(f"\n  Fields IDENTICAL in both allocs (candidate class-level anchors):")
        for off, val, tgt in same_anchors[:40]:
            print(f"    +0x{off:X} = 0x{val:X}  ({tgt})")
    return same_anchors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True, help="primary alloc addr, e.g. 0x1f280743324")
    ap.add_argument("--alloc2", default=None, help="optional second alloc for A1/A2 diff")
    ap.add_argument("--hero", default=None, help="hero struct base, optional")
    ap.add_argument("--size", default="0x400", help="bytes after alloc to analyze")
    ap.add_argument("--hero-size", default="0x20000")
    ap.add_argument("--out", default=r"C:\tmp\alloc_deep_classify.json")
    args = ap.parse_args()

    a1 = int(args.alloc, 16)
    a2 = int(args.alloc2, 16) if args.alloc2 else None
    hero_base = int(args.hero, 16) if args.hero else None
    hero_size = int(args.hero_size, 16)
    size = int(args.size, 16)

    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    if not h:
        print("ERR: OpenProcess failed (Vanguard? running as admin?)"); return 1
    mod_base, mod_size = module_range(pid)
    if not mod_base:
        print("ERR: module_range failed"); return 1
    print(f"pid={pid}  module=0x{mod_base:X}+0x{mod_size:X}")
    print(f"building region map...")
    regs = build_region_map(h)
    print(f"  {len(regs)} readable regions")

    # Sanity-check: confirm A1 still has a live Vec3
    v1 = read_bytes(h, a1, 12)
    if not v1:
        print(f"ERR: A1 0x{a1:X} not readable"); return 1
    print(f"A1 @ 0x{a1:X} Vec3 = {struct.unpack('<fff', v1)}")

    fa = classify_alloc(h, a1, size, mod_base, mod_size, hero_base, hero_size, regs)
    if fa is None:
        print(f"ERR: could not read A1"); return 1
    anchors_a1 = print_alloc_table(f"A1 = 0x{a1:X}", fa, mod_base)

    fb = None; anchors_a2 = []; same_anchors = []
    if a2:
        v2 = read_bytes(h, a2, 12)
        if v2:
            print(f"\nA2 @ 0x{a2:X} Vec3 = {struct.unpack('<fff', v2)}")
            fb = classify_alloc(h, a2, size, mod_base, mod_size, hero_base, hero_size, regs)
            if fb:
                anchors_a2 = print_alloc_table(f"A2 = 0x{a2:X}", fb, mod_base)
                same_anchors = diff_allocs(fa, fb)
        else:
            print(f"\nA2 0x{a2:X} not readable — skipping diff")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mod_ptrs_a1 = [f for f in fa if f["target"].startswith("MODULE+")]
    two_hop_a1 = [f for f in fa if f["two_hop_module"] is not None]
    hero_ptrs_a1 = [f for f in fa if f["target"].startswith("HERO+")]
    print(f"A1 direct module pointers:  {len(mod_ptrs_a1)}")
    for f in mod_ptrs_a1:
        print(f"  +0x{f['off']:X} -> {f['target']}")
    print(f"A1 two-hop module reach:    {len(two_hop_a1)}")
    for f in two_hop_a1:
        src, rva = f["two_hop_module"]
        print(f"  +0x{f['off']:X} -> HEAP -> MOD+0x{rva:X}")
    print(f"A1 -> hero-struct pointers: {len(hero_ptrs_a1)}")
    for f in hero_ptrs_a1:
        print(f"  +0x{f['off']:X} -> {f['target']}")
    if same_anchors:
        print(f"\nA1/A2 identical anchor fields: {len(same_anchors)}")
        print("  (these are class-level — pick one with a MODULE target for cross-replay signature)")

    with open(args.out, "w") as fout:
        json.dump({
            "pid": pid,
            "module_base": hex(mod_base), "module_size": mod_size,
            "a1": hex(a1),
            "a1_fields": [{**f, "u64": hex(f["u64"]),
                            "two_hop_module": [hex(f["two_hop_module"][0]), hex(f["two_hop_module"][1])]
                                               if f["two_hop_module"] else None}
                          for f in fa],
            "a2": hex(a2) if a2 else None,
            "a2_fields": [{**f, "u64": hex(f["u64"]),
                            "two_hop_module": [hex(f["two_hop_module"][0]), hex(f["two_hop_module"][1])]
                                               if f["two_hop_module"] else None}
                          for f in fb] if fb else None,
            "same_anchors": [[hex(off), hex(val), tgt] for off, val, tgt in same_anchors],
        }, fout, indent=2)
    print(f"\nwrote {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
