"""Probe hero_array + champion_name offset by broader search.

Approach:
1. Scan module RVA (broad range) for any pointer with consecutive valid pointers
2. For each candidate hero array, search the 10 hero structs for champion names
   anywhere in +0x4000..+0x5000 (512-byte stride, then refine)
3. Report hero_array RVA and the exact champion_name offset.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, time, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
_k = ctypes.windll.kernel32

CHAMP_NAMES = {"Jayce","Diana","Garen","Anivia","Amumu","Darius","LeeSin",
    "Lee Sin","Ahri","Mel","Bard","Mordekaiser","Graves","Viktor","Kaisa","Rell",
    "Naafiri","Brand","Ezreal","Alistar","Karthus","Hwei","Smolder","Nami",
    "Malphite","Azir","Ashe","Seraphine","Aatrox","Akali","Aphelios","Blitzcrank",
    "Braum","Caitlyn","Camille","Cassiopeia","Corki","Draven","Ekko","Elise",
    "Evelynn","Fiora","Fizz","Galio","Gangplank","Gragas","Hecarim","Heimerdinger",
    "Illaoi","Irelia","Ivern","Janna","JarvanIV","Jax","Jhin","Jinx","Kalista",
    "Karma","Kassadin","Katarina","Kayn","Kennen","Khazix","Kindred","Kled",
    "KogMaw","Leblanc","Lillia","Lissandra","Lucian","Lulu","Lux","Malzahar",
    "Maokai","MasterYi","MissFortune","Morgana","Nasus","Nautilus","Neeko",
    "Nidalee","Nocturne","Nunu","Olaf","Orianna","Ornn","Pantheon","Poppy","Pyke",
    "Qiyana","Quinn","Rakan","Rammus","RekSai","Renata","Renekton","Rengar",
    "Riven","Rumble","Ryze","Samira","Sejuani","Senna","Sett","Shaco","Shen",
    "Shyvana","Singed","Sion","Sivir","Skarner","Sona","Soraka","Swain","Sylas",
    "Syndra","TahmKench","Taliyah","Talon","Taric","Teemo","Thresh","Tristana",
    "Trundle","Tryndamere","TwistedFate","Twitch","Udyr","Urgot","Varus","Vayne",
    "Veigar","Velkoz","Vex","Vi","Viego","Vladimir","Volibear","Warwick","Wukong",
    "Xayah","Xerath","XinZhao","Yasuo","Yone","Yorick","Yuumi","Zac","Zed","Zeri",
    "Ziggs","Zilean","Zoe","Zyra","DrMundo"}
LO, HI = 0x10000000, 0x7FFFFFFFFFFF

class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self.read(a,8); return struct.unpack('<Q',d)[0] if d else None

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe',
                        '/FO','CSV','/NH'], capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def find_base(pid):
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("a",ctypes.c_ulong),("b",ctypes.c_ulong),
            ("c",ctypes.c_ulong),("d",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),
            ("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap=_k.CreateToolhelp32Snapshot(0x18,pid);me=ME();me.dwSize=ctypes.sizeof(ME)
    if _k.Module32First(snap,ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                _k.CloseHandle(snap)
                return ctypes.cast(me.modBaseAddr,ctypes.c_void_p).value, me.modBaseSize
            if not _k.Module32Next(snap,ctypes.byref(me)): break
    return None, None

def find_c_string(data, pos, max_len=40):
    """Read NUL-terminated ASCII starting at pos. Return None if not printable."""
    end = data.find(b'\x00', pos, pos + max_len)
    if end < 0: return None
    s = data[pos:end]
    if len(s) < 3 or len(s) > 20: return None
    try:
        t = s.decode('ascii')
        if all(c.isalpha() or c == ' ' for c in t): return t
    except: pass
    return None

def main():
    pid = find_pid()
    base, mod_size = find_base(pid)
    m = Mem(pid)
    print(f"PID={pid} base=0x{base:X} mod_size=0x{mod_size:X}")

    # Pass 1: build list of candidate hero_array RVAs
    # Strategy: scan RVA range, find any RVA where 10 consecutive u64s look like pointers
    # to allocations in the high user range (typical game heap).
    print("\n=== Pass 1: scanning for pointer arrays ===")
    CANDIDATE_RANGE = (0x1C00000, 0x1FFFFFF)  # reasonable .data section
    candidates = []
    step = 8
    total = (CANDIDATE_RANGE[1] - CANDIDATE_RANGE[0]) // step
    print(f"  scanning {total} RVAs...")
    for rva in range(CANDIDATE_RANGE[0], CANDIDATE_RANGE[1], step):
        ptr = m.u64(base + rva)
        if not ptr or not (LO < ptr < HI): continue
        # Check if this pointer leads to an array of 10 hero pointers
        good_ptrs = 0
        first_hp = None
        for i in range(10):
            hp = m.u64(ptr + i*8)
            if hp and LO < hp < HI:
                good_ptrs += 1
                if not first_hp: first_hp = hp
        if good_ptrs >= 8:
            candidates.append((rva, ptr, good_ptrs, first_hp))
    print(f"  {len(candidates)} candidate arrays with >=8 valid pointers")

    # Pass 2: for each candidate, read ALL 10 hero structs and search for champion names
    # at any offset in 0x3000..0x6000 (widened)
    print(f"\n=== Pass 2: scanning champion name strings in first hero ptr of each candidate ===")
    SCAN_START, SCAN_LEN = 0x3000, 0x3000
    best_overall = None
    for idx, (rva, arr_ptr, n_good, first_hp) in enumerate(candidates):
        # Quick filter: read buffer at first hero and search for any champion name
        buf = m.read(first_hp + SCAN_START, SCAN_LEN)
        if not buf: continue
        first_hits = []  # (pos, name)
        for pos in range(len(buf) - 4):
            if buf[pos] == 0: continue
            s = find_c_string(buf, pos, 20)
            if s and s in CHAMP_NAMES:
                first_hits.append((pos, s))
        if not first_hits: continue
        # Found at least one champion name at first hero — now check ALL 10 heroes
        # at each candidate offset to confirm
        for pos, sname in first_hits:
            off = SCAN_START + pos
            names = {}
            for slot in range(10):
                hp = m.u64(arr_ptr + slot*8)
                if not hp or not (LO < hp < HI): continue
                s = find_c_string(m.read(hp + off, 24) or b'\x00', 0, 20)
                if s and s in CHAMP_NAMES:
                    names[slot] = s
            # Must be 10 DISTINCT champions (real game)
            distinct = set(names.values())
            if len(names) >= 8 and len(distinct) == len(names):
                print(f"\n  hero_array RVA 0x{rva:X} → ptr 0x{arr_ptr:X}")
                print(f"    champion_name at hero+0x{off:X} ({len(names)}/10 slots, {len(distinct)} distinct):")
                for slot in sorted(names.keys()):
                    print(f"      slot {slot}: {names[slot]}")
                if best_overall is None or len(names) > len(best_overall[2]):
                    best_overall = (rva, off, names, arr_ptr)
                if len(names) >= 9:
                    break
        if best_overall and len(best_overall[2]) >= 9:
            break

    if best_overall:
        rva, off, names, arr_ptr = best_overall
        print(f"\n>>> BEST MATCH")
        print(f"    hero_array RVA = 0x{rva:X}  (anchor 0x1DBEEE8, shift {hex(rva-0x1DBEEE8)})")
        print(f"    champion_name  = 0x{off:X}  (anchor 0x4328, shift {hex(off-0x4328)})")
        print(f"    {len(names)}/10 slots matched")

if __name__ == '__main__':
    main()
