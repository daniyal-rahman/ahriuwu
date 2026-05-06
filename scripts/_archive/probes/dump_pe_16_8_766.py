"""Dump .text, .rdata, .data of the running League of Legends.exe (16.8.766)
to a local pe_dump dir for static reverse-engineering. RPM-only — no debug
APIs, no writes, no hooks. Works under Vanguard the same way our other
memory-scrape scripts do.

Output layout (Windows-side):
    C:\\tmp\\pe_dump_16.8.766\\text.bin
    C:\\tmp\\pe_dump_16.8.766\\rdata.bin
    C:\\tmp\\pe_dump_16.8.766\\data.bin
    C:\\tmp\\pe_dump_16.8.766\\sections.json   (RVA + size + module base)

Run:
    ssh windows python -u C:\\Users\\daniz\\Repos\\ahriuwu\\scripts\\dump_pe_16_8_766.py
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig=builtins.print
def print(*a,**k): k.setdefault("flush",True); _orig(*a,**k)
builtins.print=print

_PATCH = sys.argv[1] if len(sys.argv) > 1 else "16.8.766"
OUT_DIR = rf"C:\tmp\pe_dump_{_PATCH}"

_k = ctypes.windll.kernel32

def find_pid():
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                     capture_output=True,text=True)
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

def read_bytes(h, addr, n, chunk=4*1024*1024):
    """Read n bytes starting at addr. Returns whatever it can, or None if 0 bytes."""
    out = bytearray(n)
    v = memoryview(out)
    o = 0
    while o < n:
        c = min(chunk, n-o)
        buf = (ctypes.c_char*c)()
        r = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr+o), buf, c, ctypes.byref(r))
        if not ok or r.value == 0:
            return None if o == 0 else bytes(v[:o])
        v[o:o+r.value] = buf[:r.value]
        o += r.value
        if r.value < c:
            return bytes(v[:o])
    return bytes(out)

def parse_pe_sections(h, base):
    """Parse the PE in-memory headers and return list of (name, virt_addr, virt_size)."""
    # DOS header
    dos = read_bytes(h, base, 0x40)
    if not dos or dos[:2] != b'MZ': raise RuntimeError("no MZ")
    e_lfanew = struct.unpack_from("<I", dos, 0x3C)[0]
    # NT headers
    nth = read_bytes(h, base+e_lfanew, 0x108)
    if not nth or nth[:4] != b'PE\x00\x00': raise RuntimeError("no PE")
    file_header_off = 4
    n_sections = struct.unpack_from("<H", nth, file_header_off+2)[0]
    size_of_optional = struct.unpack_from("<H", nth, file_header_off+16)[0]
    sections_off = e_lfanew + 4 + 20 + size_of_optional
    sec_table = read_bytes(h, base+sections_off, 40*n_sections)
    sections = []
    for i in range(n_sections):
        rec = sec_table[i*40:(i+1)*40]
        name = rec[0:8].rstrip(b'\x00').decode('latin1')
        virt_size = struct.unpack_from("<I", rec, 8)[0]
        virt_addr = struct.unpack_from("<I", rec, 12)[0]
        raw_size = struct.unpack_from("<I", rec, 16)[0]
        sections.append({"name": name, "rva": virt_addr, "vsize": virt_size, "rsize": raw_size})
    return sections

def main():
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    base, mod_size = module_range(pid)
    if not base:
        print("ERR: module base unknown"); return 1
    print(f"pid={pid}  base=0x{base:X}  size=0x{mod_size:X}")

    h = _k.OpenProcess(0x0410, False, pid)
    if not h:
        print("ERR: OpenProcess"); return 1

    sections = parse_pe_sections(h, base)
    print("Sections:")
    for s in sections:
        print(f"  {s['name']:<10} rva=0x{s['rva']:08X}  vsize=0x{s['vsize']:08X}")

    os.makedirs(OUT_DIR, exist_ok=True)
    targets = {".text": "text.bin", ".rdata": "rdata.bin", ".data": "data.bin"}
    saved = []
    for s in sections:
        if s["name"] not in targets: continue
        addr = base + s["rva"]
        size = s["vsize"]
        print(f"  reading {s['name']} ({size} bytes) ...", end="")
        data = read_bytes(h, addr, size)
        if not data:
            print(f" FAILED"); continue
        out_path = os.path.join(OUT_DIR, targets[s["name"]])
        with open(out_path, "wb") as f:
            f.write(data)
        print(f" wrote {len(data)} bytes -> {out_path}")
        saved.append({"name": s["name"], "rva": s["rva"], "vsize": s["vsize"],
                      "actual_bytes": len(data), "file": targets[s["name"]]})

    meta = {
        "pid": pid,
        "module_base": hex(base),
        "module_size": mod_size,
        "patch": _PATCH,
        "sections": saved,
    }
    with open(os.path.join(OUT_DIR, "sections.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {OUT_DIR}\\sections.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
