"""Compare two click-dest candidate addresses' parent struct layouts.

For each candidate (vec3 addr), parent_addr = vec3_addr - 0x14 (parent+0x00 = vptr,
parent+0x14 = vec3). Dumps a 256-byte slice of the parent struct, then:

  - Side-by-side hex (8-byte rows)
  - Diff: which u64 slots differ
  - For each differing slot, mark if it looks like a pointer (0x7FFxxxxxxxxx range
    for module pointers, or heap range)
  - Cross-check pointer targets: dereference each pointer and read another 64
    bytes to see if it has any recognizable strings (champion names, "Garen",
    "client", etc.)

Goal: find a field in the parent struct that distinguishes the REAL Garen
click-dest from sibling instances — likely "owner unit pointer" or
"target unit ID" that points back to the controlling champion.

Usage:
  python click_addr_diff.py 0x1c3baf25eb4 0x1c3baf23034 [more_addrs...]
"""
import sys, ctypes, ctypes.wintypes as wt, struct, subprocess, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

VEC3_OFFSET = 0x14   # parent+0x14 = vec3
DUMP_BYTES = 256
HERO_NAME_OFF = 0x4360   # if a ptr leads to a hero struct, name is here

_k = ctypes.windll.kernel32

def find_pid():
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                     capture_output=True,text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def read_bytes(h, addr, n):
    buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

def is_likely_ptr(v):
    return 0x10000 <= v < 0x7FFFFFFFFFFF

def read_cstring(h, addr, n=64):
    b = read_bytes(h, addr, n)
    if not b: return None
    nul = b.find(b'\x00')
    s = b[:nul if nul >= 0 else n]
    try: return s.decode('ascii')
    except: return None

def dump_struct(h, vec3_addr):
    parent = vec3_addr - VEC3_OFFSET
    raw = read_bytes(h, parent, DUMP_BYTES)
    return parent, raw

def hex_rows(raw):
    """Return list of (offset, u64_hex, ascii) for each 8-byte slot."""
    rows = []
    for i in range(0, len(raw), 8):
        chunk = raw[i:i+8]
        if len(chunk) < 8: break
        v = struct.unpack("<Q", chunk)[0]
        ascii_repr = "".join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        rows.append((i, v, ascii_repr))
    return rows

def main():
    if len(sys.argv) < 3:
        print("usage: python click_addr_diff.py <addr1> <addr2> [...]"); return 1
    addrs = [int(x, 16) for x in sys.argv[1:]]
    pid = find_pid()
    if not pid: print("no League"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"PID={pid}  comparing {len(addrs)} candidates\n")

    dumps = []
    for a in addrs:
        parent, raw = dump_struct(h, a)
        if not raw or len(raw) < DUMP_BYTES:
            print(f"  {hex(a)}: read failed"); continue
        dumps.append((a, parent, raw, hex_rows(raw)))

    # Side-by-side hex of all candidates
    print("=== HEX (parent+0x00..0xFF, 8-byte rows) ===")
    print(f"{'off':>5}", *(f" {'addr=' + hex(a)[-9:]:>20}" for a,_,_,_ in dumps))
    n_rows = min(len(rows) for _,_,_,rows in dumps)
    for i in range(n_rows):
        offset = dumps[0][3][i][0]
        row = f"{offset:>5}"
        for _,_,_,rows in dumps:
            _, v, ascii_r = rows[i]
            tag = "P" if is_likely_ptr(v) else " "
            row += f"  {v:016X}{tag} {ascii_r}"
        # mark if any difference
        vals = [rows[i][1] for _,_,_,rows in dumps]
        diff = " DIFF" if len(set(vals)) > 1 else ""
        print(row + diff)

    # For each diff'ing slot, follow pointers and look for hero struct
    print("\n=== POINTER CHASE on diff'ing slots ===")
    for i in range(n_rows):
        vals = [rows[i][1] for _,_,_,rows in dumps]
        if len(set(vals)) <= 1: continue
        offset = dumps[0][3][i][0]
        if not all(is_likely_ptr(v) for v in vals): continue
        print(f"\n--- slot +0x{offset:X} (pointers) ---")
        for (a,_,_,_), v in zip(dumps, vals):
            print(f"  addr={hex(a)}  -> ptr=0x{v:X}")
            # Check if v + 0x4360 holds a champion-name-like string
            nm = read_cstring(h, v + HERO_NAME_OFF, 32)
            if nm and nm.isalpha() and len(nm) >= 3:
                print(f"     *(ptr+0x4360) = {nm!r}  ← CHAMPION NAME via hero offset")
            # Also try reading first 32 bytes as ASCII (in case it's a string)
            head = read_bytes(h, v, 32)
            if head:
                ascii_head = "".join(chr(b) if 32<=b<127 else '.' for b in head)
                print(f"     ptr+0x00 = {head[:16].hex()}  {ascii_head}")
            # Check for vtable: if ptr+0 is a code address it's a vptr
            vp = read_bytes(h, v, 8)
            if vp:
                vptr = struct.unpack("<Q", vp)[0]
                if 0x7FF000000000 <= vptr <= 0x7FFFFFFFFFFF:
                    print(f"     ptr+0x00 = vptr 0x{vptr:X}  (object with vtable)")

    print("\n=== END ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
