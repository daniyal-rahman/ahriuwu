"""Walk Lua-object references to 'Belveth' and decode the TValue values
that sit at the key position (Lua Table.Node layout).

Lua 5.1 Node layout (40 bytes):
  offset  0..16 : TValue i_val   (value + tt)  [the mapped value]
  offset 16..32 : TValue i_key   (value + tt)  [key = "Belveth" string]
  offset 32..36 : int next       (linked hash chain offset)
  offset 36..40 : padding

Lua 5.3 Node layout (32 bytes):
  offset  0..16 : TValue i_val
  offset 16..24 : Value i_key.nk.value
  offset 24..28 : int tt
  offset 28..32 : int next

For each pointer to a Belveth-TString, we read the 16 bytes before it and
interpret as a TValue. If tt is 5 (table), 7 (userdata), or 6 (function),
the value is a GCObject* pointing to the associated object.

We then dedupe those GCObject* values and rank by number of refs. Top
candidates are the primary Belveth objects (champion table, spell scripts,
etc.). One of them will eventually lead to the click-dest struct via a
field-name lookup.
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json
from collections import Counter

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

LUA_TNIL      = 0
LUA_TBOOL     = 1
LUA_TLIGHT    = 2
LUA_TNUMBER   = 3
LUA_TSTRING   = 4
LUA_TTABLE    = 5
LUA_TFUNCTION = 6
LUA_TUSERDATA = 7
LUA_TTHREAD   = 8
TYPE_NAMES = {0:"nil", 1:"bool", 2:"lightuserdata", 3:"number", 4:"string",
              5:"table", 6:"function", 7:"userdata", 8:"thread"}

def decode_tvalue(b16):
    """Attempt both Lua 5.1 and 5.3 TValue layouts."""
    if len(b16) < 16: return None
    # Lua 5.1: value(8), tt(4), pad(4)
    v51_val = struct.unpack_from("<Q", b16, 0)[0]
    v51_tt  = struct.unpack_from("<I", b16, 8)[0]
    # Lua 5.3/5.4: value(8) then tt_ byte + padding
    v53_val = struct.unpack_from("<Q", b16, 0)[0]
    v53_tt  = b16[8]
    return v51_val, v51_tt, v53_val, v53_tt

def main():
    pid = find_pid()
    if not pid: print("no league"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}")

    lua = json.load(open(r"C:\tmp\belveth_lua_refs.json"))
    refs = lua["belveth_string_refs"]
    print(f"Loading {len(refs)} refs...")

    # For each ref, read 16 bytes BEFORE the ref (Node i_val slot).
    tvals_51 = []  # (gc_ptr, tt) interpreted as Lua 5.1
    tvals_53 = []
    for r in refs:
        src = int(r["src"], 16)
        b = read(h, src - 16, 16)
        if not b or len(b) < 16: continue
        dec = decode_tvalue(b)
        if not dec: continue
        v51_val, v51_tt, v53_val, v53_tt = dec
        tvals_51.append((v51_val, v51_tt))
        tvals_53.append((v53_val, v53_tt))

    # Lua 5.1 type-tag distribution
    def tag_name(tt): return TYPE_NAMES.get(tt, f"?0x{tt:X}")
    cnt51 = Counter(t for _, t in tvals_51)
    cnt53 = Counter(t for _, t in tvals_53)
    print(f"\n== Lua 5.1 interpretation of preceding TValue ==")
    for tt, n in cnt51.most_common(15):
        print(f"  tt={tt} ({tag_name(tt)}): {n}")
    print(f"\n== Lua 5.3 interpretation (tt byte at +8) ==")
    for tt, n in cnt53.most_common(15):
        print(f"  tt=0x{tt:02X} ({tag_name(tt & 0x0F)}): {n}")

    # Under 5.1 interpretation: filter to "looks like a GC pointer and type in 5..8"
    # (table/function/userdata/thread)
    gc_candidates = []
    for v, t in tvals_51:
        if t in (LUA_TTABLE, LUA_TFUNCTION, LUA_TUSERDATA, LUA_TTHREAD):
            if 0x10000000000 < v < 0x80000000000:
                gc_candidates.append((v, t))
    print(f"\n== Lua 5.1 Node i_val candidates (tt in [5..8] + heap ptr) ==")
    print(f"   {len(gc_candidates)} candidates")
    gc_cnt = Counter(gc_candidates)
    for (v, t), n in gc_cnt.most_common(25):
        print(f"  0x{v:X}  tt={t} ({tag_name(t)})  refs={n}")

    # Lua 5.3: the Node is 32 bytes: i_val(16), i_key(16). i_key is
    # (Value 8)(tt_ 1)(pad 7). So the TString* for Belveth is at offset 16
    # (not 24 or whatever). The ref points AT the TString*, so bytes
    # before the ref at -16..-0 are the Value, and at -24..-16 are i_val.
    # We tried -16 above. Now try -32..-16 for Lua 5.3 i_val.
    print(f"\n== Lua 5.3 Node interpretation (i_val at -32..-16 from ref) ==")
    tvals_53b = []
    for r in refs:
        src = int(r["src"], 16)
        b = read(h, src - 32, 16)
        if not b or len(b) < 16: continue
        v_val = struct.unpack_from("<Q", b, 0)[0]
        v_tt = b[8]
        tvals_53b.append((v_val, v_tt))
    c = Counter(t for _, t in tvals_53b)
    for tt, n in c.most_common(10):
        print(f"  tt=0x{tt:02X} ({tag_name(tt & 0x0F)}): {n}")
    gc53b = [(v, t) for v, t in tvals_53b
             if (t & 0x0F) in (LUA_TTABLE, LUA_TFUNCTION, LUA_TUSERDATA)
             and 0x10000000000 < v < 0x80000000000]
    print(f"  Lua-5.3-GC candidates: {len(gc53b)}")
    gc53_cnt = Counter(gc53b)
    for (v, t), n in gc53_cnt.most_common(15):
        print(f"  0x{v:X}  tt=0x{t:02X}  refs={n}")

    out = {
        "pid": pid,
        "top_51_gc": [{"addr": hex(v), "tt": t, "name": tag_name(t), "refs": n}
                       for (v, t), n in gc_cnt.most_common(30)],
        "top_53_gc": [{"addr": hex(v), "tt": t, "name": tag_name(t & 0x0F), "refs": n}
                       for (v, t), n in gc53_cnt.most_common(30)],
    }
    with open(r"C:\tmp\lua_belveth_objects.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote C:\\tmp\\lua_belveth_objects.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
