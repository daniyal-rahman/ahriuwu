"""Inspect the first 0x100 bytes of the click-destination allocation.

Goal: find a byte signature (ideally a vtable pointer into League.exe's image,
or a stable constant/struct-type-id) that lets us re-locate the alloc on any
session by scanning heap for that signature — instead of chasing a fragile
multi-hop ephemeral pointer chain.

Reuses find_click_alloc() from trace_click_anchor.py.
"""
import sys, json, struct
sys.path.insert(0, r"C:\Users\daniz")
from trace_click_anchor import (
    find_pid, open_proc, module_range, read_region, enumerate_regions,
    find_click_alloc, api_post, api_get,
)
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

def main():
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = open_proc(pid)
    mod_base, mod_size = module_range(pid)
    print(f"pid={pid}  module=[0x{mod_base:X},0x{mod_base+mod_size:X})")

    try:
        api_post("/replay/render", {"interfaceAll": True, "selectionName": "Belveth"})
    except Exception: pass

    alloc_base, mirrors = find_click_alloc(h)
    if alloc_base is None:
        print("ERR: alloc not found"); return 1

    # Read 0x200 bytes starting at alloc_base (and 0x40 before, in case the
    # "base" we picked is actually mirror-A and not the struct start).
    read_from = alloc_base - 0x40
    data = read_region(h, read_from, 0x400)
    if data is None:
        print("ERR: couldn't read"); return 1

    print(f"\nRaw bytes around alloc_base=0x{alloc_base:X} (shown as offset from alloc_base):")
    for i in range(0, len(data), 8):
        off = read_from + i - alloc_base
        if i + 8 > len(data): break
        u64 = struct.unpack_from("<Q", data, i)[0]
        f_a = struct.unpack_from("<f", data, i)[0]
        f_b = struct.unpack_from("<f", data, i + 4)[0] if i + 8 <= len(data) else 0.0
        # Classify u64
        cls = ""
        if mod_base <= u64 < mod_base + mod_size:
            cls = f"MODULE+0x{u64-mod_base:X}"
        elif 0x10000000000 < u64 < 0x80000000000:
            cls = "HEAP"
        marker = ""
        base_off = read_from + i
        if base_off == alloc_base: marker = "  <-- alloc_base"
        elif base_off == alloc_base + 0x308: marker = "  <-- mirror B"
        elif base_off == alloc_base + 0x374: marker = "  <-- mirror C"
        print(f"  [+{off:+4d} 0x{off & 0xFFFFFFFF:03X}]  0x{u64:016X}  "
              f"f=({f_a:.2f},{f_b:.2f})  {cls}{marker}")

    # Additionally: scan for a reasonable vtable-candidate in the first 0x30 bytes
    print("\nModule-pointer candidates in first 0x40 bytes:")
    for i in range(0, 0x40, 8):
        u64 = struct.unpack_from("<Q", data, 0x40 + i)[0]  # 0x40 is the shift to alloc_base
        if mod_base <= u64 < mod_base + mod_size:
            print(f"  alloc+0x{i:X}  -> module+0x{u64-mod_base:X}  (RVA 0x{u64-mod_base:X})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
