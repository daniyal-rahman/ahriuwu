"""Inspect a candidate Lua GC object: dump its header + first 0x200 bytes,
and scan for pointers that land in heap (especially our click-dest alloc
regions 0x1F37Axxxxx). If this is a Lua Table, we'll see its array/node
pointers and can walk fields by name.

Lua 5.3 Table struct:
  CommonHeader: next (8), tt_ (1), marked (1)        = 16 w/ padding
  lu_byte flags (1) + lu_byte lsizenode (1) + padding
  sizearray (int = 4)
  TValue *array  (8)  -> array part
  Node *node     (8)  -> hash part
  Node *lastfree (8)
  Table *metatable (8)
  GCObject *gclist (8)

Usage: python lua_inspect_obj.py 0x1F37AB303D0 0x1F2C38DD120 ...
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
_k = ctypes.windll.kernel32

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def read(h, addr, n):
    buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
    if not _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r)): return None
    return bytes(buf[:r.value])

def classify(v):
    if 0x10000000000 < v < 0x80000000000: return "HEAP"
    if 0x7FF600000000 < v < 0x7FFFFFFFFFFF: return "MODULE?"
    return ""

def main():
    pid = find_pid()
    if not pid: print("no league"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    addrs = [int(a, 16) for a in sys.argv[1:]]
    if not addrs:
        print("Usage: lua_inspect_obj.py <addr1> <addr2> ..."); return 1

    for a in addrs:
        print(f"\n========== 0x{a:X} ==========")
        data = read(h, a, 0x200)
        if not data:
            print("  read failed"); continue
        # Lua 5.3 CommonHeader: 8-byte next ptr, 1-byte tt, 1-byte marked
        next_ptr = struct.unpack_from("<Q", data, 0)[0]
        tt = data[8]
        marked = data[9]
        # Table-specific
        flags = data[10] if len(data) > 10 else 0
        lsizenode = data[11] if len(data) > 11 else 0
        sizearray = struct.unpack_from("<I", data, 12)[0]
        arr_ptr = struct.unpack_from("<Q", data, 16)[0]
        node_ptr = struct.unpack_from("<Q", data, 24)[0]
        lastfree_ptr = struct.unpack_from("<Q", data, 32)[0]
        meta_ptr = struct.unpack_from("<Q", data, 40)[0]

        print(f"  CommonHeader: next=0x{next_ptr:X} tt=0x{tt:02X} marked=0x{marked:02X}")
        print(f"  flags=0x{flags:02X} lsizenode={lsizenode} sizearray={sizearray}")
        print(f"  array=0x{arr_ptr:X}  {classify(arr_ptr)}")
        print(f"  node=0x{node_ptr:X}  {classify(node_ptr)}")
        print(f"  lastfree=0x{lastfree_ptr:X}  {classify(lastfree_ptr)}")
        print(f"  metatable=0x{meta_ptr:X}  {classify(meta_ptr)}")

        # Dump first 0x200 bytes annotated
        print(f"  bytes (u64 view):")
        for off in range(0, min(0x100, len(data)), 8):
            u64 = struct.unpack_from("<Q", data, off)[0]
            f32a = struct.unpack_from("<f", data, off)[0]
            f32b = struct.unpack_from("<f", data, off+4)[0] if off+8 <= len(data) else 0.0
            cls = classify(u64)
            # ASCII?
            raw = data[off:off+8]
            ascii_r = ""
            if all(32 <= b <= 126 or b == 0 for b in raw) and any(b >= 32 for b in raw):
                ascii_r = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw)
                ascii_r = f"'{ascii_r}'"
            print(f"    +{off:03X}  0x{u64:016X}  f=({f32a:>9.2f},{f32b:>9.2f})  {cls:<8} {ascii_r}")

        # Follow the node pointer (if it looks valid) and dump 2 nodes (80 bytes)
        if classify(node_ptr) == "HEAP":
            print(f"\n  Following node array at 0x{node_ptr:X}:")
            nodes = read(h, node_ptr, 40 * (1 << lsizenode) if lsizenode < 6 else 40 * 8)
            if nodes:
                n_nodes = min(len(nodes) // 40, 8)
                for i in range(n_nodes):
                    off = i * 40
                    # Node: i_val (TValue: 16 bytes), i_key TKey (24 bytes)
                    ival_val = struct.unpack_from("<Q", nodes, off)[0]
                    ival_tt = nodes[off+8]
                    ikey_val = struct.unpack_from("<Q", nodes, off+16)[0]
                    ikey_tt = nodes[off+24]
                    next_ofs = struct.unpack_from("<i", nodes, off+28)[0] if off+32 <= len(nodes) else 0
                    # If key is string (tt=4 or 0x14 with collectable bit), try to read it
                    key_str = ""
                    if (ikey_tt & 0x0F) == 4 and classify(ikey_val) == "HEAP":
                        # TString header is ~24 bytes, then chars
                        chars = read(h, ikey_val + 24, 64)
                        if chars:
                            key_str = chars.split(b"\x00")[0].decode("ascii", errors="replace")
                    print(f"    node[{i}]  val=0x{ival_val:016X} tt=0x{ival_tt:02X}  "
                          f"key=0x{ikey_val:016X} ktt=0x{ikey_tt:02X}  "
                          f"next={next_ofs}  key_str='{key_str}'")

    return 0

if __name__ == "__main__":
    sys.exit(main())
