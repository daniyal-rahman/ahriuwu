"""
League of Legends Memory Reader — FINAL
Extracts hero positions, movement targets, and auto-attack events from replay memory.

Confirmed working offsets (patch 26.7, module size 0x207C000):
  HeroArrayPtr:     base + 0x1DBEEE8 -> ptr to array of 10 hero pointers
  Position:         hero + 0x25C      (Vec3: x, y, z)
  CharacterName:    hero + 0x4328     (ASCII string)
  NetId:            hero + 0xCC       (u32)
  Team:             inferred from hero array index (0-4=blue, 5-9=red)
  ActiveSpellCast:  hero + 0x3120     -> active spell ptr
    SpellInfo:        spell + 0x008   -> SpellInfo ptr
      SpellNamePtr:     info + 0x28   -> ptr to name string (double deref)
    CastStartPos:     spell + 0x0D0   (Vec3)
    CastTargetPos:    spell + 0x0DC   (Vec3)

AiManager is XOR-encrypted at hero+0x41F0 — cannot dereference.
Movement targets derived from position delta sampling instead.
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
from ctypes import wintypes

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_MODULE_NAME32 = 255
MAX_PATH = 260

# Confirmed offsets
HERO_ARRAY_RVA = 0x1DBEEE8
OFF_POSITION = 0x25C
OFF_NAME = 0x4328
OFF_NETID = 0xCC
OFF_ACTIVE_SPELL = 0x3120  # hero + 0x3120 -> ActiveSpellCast ptr
OFF_SPELLINFO = 0x008      # spell + 0x008 -> SpellInfo ptr
OFF_SPELL_NAME_PTR = 0x28  # info + 0x28 -> ptr to name string
OFF_CAST_START = 0x0D0     # spell + 0x0D0 -> Vec3 cast start
OFF_CAST_TARGET = 0x0DC    # spell + 0x0DC -> Vec3 cast target


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_char * MAX_PATH),
    ]

class MODULEENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
        ("modBaseSize", wintypes.DWORD), ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_char * (MAX_MODULE_NAME32 + 1)),
        ("szExePath", ctypes.c_char * MAX_PATH),
    ]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def find_league():
    """Find League of Legends process ID."""
    s = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pe = PROCESSENTRY32()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
    kernel32.Process32First(s, ctypes.byref(pe))
    while True:
        if b"League of Legends" in pe.szExeFile:
            pid = pe.th32ProcessID
            kernel32.CloseHandle(s)
            return pid
        if not kernel32.Process32Next(s, ctypes.byref(pe)):
            break
    kernel32.CloseHandle(s)
    return None


def find_base(pid):
    """Find League module base address and size."""
    s = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    me = MODULEENTRY32()
    me.dwSize = ctypes.sizeof(MODULEENTRY32)
    kernel32.Module32First(s, ctypes.byref(me))
    while True:
        if b"league of legends" in me.szModule.lower():
            base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
            size = me.modBaseSize
            kernel32.CloseHandle(s)
            return base, size
        if not kernel32.Module32Next(s, ctypes.byref(me)):
            break
    kernel32.CloseHandle(s)
    return None, None


class Mem:
    """Process memory reader via ReadProcessMemory."""

    def __init__(self, pid):
        self.h = kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid
        )
        if not self.h:
            raise OSError(f"OpenProcess failed: {ctypes.get_last_error()}")

    def read(self, addr, sz):
        buf = ctypes.create_string_buffer(sz)
        n = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(
            self.h, ctypes.c_void_p(addr), buf, sz, ctypes.byref(n)
        )
        return buf.raw[: n.value] if ok and n.value == sz else None

    def u32(self, a):
        d = self.read(a, 4)
        return struct.unpack("<I", d)[0] if d else None

    def u64(self, a):
        d = self.read(a, 8)
        return struct.unpack("<Q", d)[0] if d else None

    def f32(self, a):
        d = self.read(a, 4)
        return struct.unpack("<f", d)[0] if d else None

    def vec3(self, a):
        d = self.read(a, 12)
        return struct.unpack("<fff", d) if d else None

    def string(self, a, n=64):
        d = self.read(a, n)
        if not d:
            return None
        return d.split(b"\x00")[0].decode("ascii", errors="replace") or None

    def close(self):
        kernel32.CloseHandle(self.h)


def is_heap(v):
    return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF


def is_map_pos(v):
    if v is None:
        return False
    x, y, z = v
    return 100 < x < 15000 and -500 < y < 1000 and 100 < z < 15000


# ======================================================================
# Core API functions
# ======================================================================

def read_heroes(m, base):
    """Read all 10 hero positions, names, teams, and netIDs.

    Returns list of dicts with keys:
        ptr, name, team, net_id, position (x, y, z), index
    """
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    if not is_heap(arr_ptr):
        return []

    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp):
            continue

        name = m.string(hp + OFF_NAME, 32) or f"unknown_{i}"
        pos = m.vec3(hp + OFF_POSITION)
        team = 100 if i < 5 else 200  # blue=0-4, red=5-9
        net_id = m.u32(hp + OFF_NETID)

        heroes.append({
            "index": i,
            "ptr": hp,
            "name": name,
            "team": team,
            "net_id": net_id,
            "position": pos,  # (x, y, z) tuple
        })

    return heroes


def read_movement_targets(m, base, heroes, dt=0.1):
    """Estimate movement direction/speed from position delta sampling.

    Since AiManager is XOR-encrypted (hero+0x41F0), we derive movement
    from two position samples taken dt seconds apart.

    Returns dict mapping hero_name -> {
        speed, direction (dx, dz), is_moving,
        position (current), projected_target (2s ahead estimate)
    }
    """
    # Sample 1: current positions
    snap1 = {}
    for h in heroes:
        snap1[h["name"]] = m.vec3(h["ptr"] + OFF_POSITION)

    time.sleep(dt)

    # Sample 2: positions after dt
    snap2 = {}
    for h in heroes:
        snap2[h["name"]] = m.vec3(h["ptr"] + OFF_POSITION)

    results = {}
    for h in heroes:
        p1 = snap1[h["name"]]
        p2 = snap2[h["name"]]
        if not p1 or not p2:
            results[h["name"]] = {
                "speed": 0, "direction": (0, 0), "is_moving": False,
                "position": p2 or p1, "projected_target": None,
            }
            continue

        dx = p2[0] - p1[0]
        dz = p2[2] - p1[2]
        dist = (dx * dx + dz * dz) ** 0.5
        speed = dist / dt

        is_moving = speed > 30  # threshold for noise

        if is_moving:
            # Normalize direction
            norm_dx = dx / dist if dist > 0 else 0
            norm_dz = dz / dist if dist > 0 else 0
            # Project 2 seconds ahead as rough target estimate
            proj_x = p2[0] + norm_dx * speed * 2.0
            proj_z = p2[2] + norm_dz * speed * 2.0
            projected = (proj_x, p2[1], proj_z)
        else:
            norm_dx = norm_dz = 0
            projected = None

        results[h["name"]] = {
            "speed": speed,
            "direction": (norm_dx, norm_dz),
            "is_moving": is_moving,
            "position": p2,
            "projected_target": projected,
        }

    return results


def read_auto_attacks(m, base, heroes):
    """Read current auto-attack / spell cast events for all heroes.

    Uses ActiveSpellCast at hero+0x3120:
      spell+0x008 -> SpellInfo -> +0x28 -> name_ptr -> "ChampBasicAttack"
      spell+0x0D0 -> cast start position (Vec3)
      spell+0x0DC -> cast target position (Vec3)

    For each hero with an active spell:
      - Parse the champion name from the spell name prefix
      - The hero at hero+0x3120 is the CASTER
      - Cross-reference cast_target position with all hero positions to ID target

    Returns list of dicts:
        {caster, target, spell_name, caster_pos, target_pos, cast_start_pos}
    """
    # Build hero position lookup
    hero_positions = {}
    for h in heroes:
        if h["position"]:
            hero_positions[h["name"]] = h["position"]

    events = []
    for h in heroes:
        spell_ptr = m.u64(h["ptr"] + OFF_ACTIVE_SPELL)
        if not is_heap(spell_ptr):
            continue

        # Read spell name via double-deref chain
        spell_info = m.u64(spell_ptr + OFF_SPELLINFO)
        if not is_heap(spell_info):
            continue

        name_ptr = m.u64(spell_info + OFF_SPELL_NAME_PTR)
        if not is_heap(name_ptr):
            continue

        spell_name = m.string(name_ptr, 64)
        if not spell_name or len(spell_name) < 3:
            continue

        cast_start = m.vec3(spell_ptr + OFF_CAST_START)
        cast_target = m.vec3(spell_ptr + OFF_CAST_TARGET)

        # Identify target by matching cast_target position to hero positions
        target_name = None
        target_pos = None
        target_dist = None
        if cast_target and is_map_pos(cast_target):
            best_dist = 300  # max match distance for hero targets
            for h2_name, h2_pos in hero_positions.items():
                if h2_name == h["name"]:
                    continue  # skip self
                dx = cast_target[0] - h2_pos[0]
                dz = cast_target[2] - h2_pos[2]
                dist = (dx * dx + dz * dz) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    target_name = h2_name
                    target_pos = h2_pos
                    target_dist = dist

        events.append({
            "caster": h["name"],
            "caster_team": h["team"],
            "caster_pos": cast_start,
            "target": target_name,  # None if targeting non-hero (minion/monster)
            "target_pos": target_pos or cast_target,
            "target_dist": target_dist,
            "spell_name": spell_name,
            "cast_target_raw": cast_target,
        })

    return events


# ======================================================================
# Display helpers
# ======================================================================

def vec_str(v, fmt="({:.0f},{:.0f})"):
    if v is None:
        return "N/A"
    return fmt.format(v[0], v[2])


def print_hero_table(heroes):
    print(f"{'#':>2} {'Champion':12s} {'Team':>6} {'NetID':>10} {'Position':>18}")
    print("-" * 55)
    for h in heroes:
        pos = vec_str(h["position"])
        nid = f"0x{h['net_id']:08X}" if h["net_id"] else "?"
        team_str = "BLUE" if h["team"] == 100 else "RED"
        print(f"{h['index']:2d} {h['name']:12s} {team_str:>6} {nid:>10} {pos:>18}")


def print_movement_table(movement):
    print(f"{'Champion':12s} {'Speed':>6} {'Dir':>12} {'Position':>14} {'Projected':>14}")
    print("-" * 65)
    for name, m in movement.items():
        if m["is_moving"]:
            spd = f"{m['speed']:.0f}"
            d = m["direction"]
            dir_str = f"({d[0]:.2f},{d[1]:.2f})"
            proj = vec_str(m["projected_target"])
        else:
            spd = "0"
            dir_str = "---"
            proj = "---"
        pos = vec_str(m["position"])
        print(f"{name:12s} {spd:>6} {dir_str:>12} {pos:>14} {proj:>14}")


def print_attack_table(attacks, heroes):
    if not attacks:
        print("  (no active spells)")
        return

    hero_teams = {h["name"]: h["team"] for h in heroes}

    for a in attacks:
        caster = a["caster"]
        spell = a["spell_name"]
        c_pos = vec_str(a["caster_pos"])
        t_pos = vec_str(a["target_pos"])

        # Determine spell type
        is_aa = "BasicAttack" in spell or "CritAttack" in spell
        spell_type = "AA" if is_aa else "SPELL"

        # Show team context
        c_team = "B" if a.get("caster_team") == 100 else "R"

        if a["target"]:
            target = a["target"]
            t_team = "B" if hero_teams.get(target) == 100 else "R"
            target_str = f"{target}({t_team})"
        else:
            target_str = "minion/other"

        print(
            f"  [{spell_type:5s}] {caster}({c_team}) -> {target_str}: "
            f"'{spell}' from={c_pos} to={t_pos}"
        )


# ======================================================================
# Main — real-time monitoring
# ======================================================================

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("League Memory Reader — FINAL")
    print("=" * 70)

    pid = find_league()
    if not pid:
        print("League of Legends not found!")
        return
    base, mod_size = find_base(pid)
    if not base:
        print("Module not found!")
        return
    print(f"PID={pid} Base=0x{base:X} ModSize=0x{mod_size:X}")

    m = Mem(pid)
    assert m.read(base, 2) == b"MZ", "ReadProcessMemory failed!"
    print("ReadProcessMemory: OK\n")

    # --- Read heroes ---
    heroes = read_heroes(m, base)
    if not heroes:
        print("No heroes found!")
        m.close()
        return

    print(f"--- Heroes ({len(heroes)}) ---")
    print_hero_table(heroes)

    # --- Real-time monitoring loop ---
    print(f"\n{'='*70}")
    print("Real-time monitoring (10 seconds, ~1 sample/sec)")
    print("=" * 70)

    for t in range(10):
        # Re-read hero positions
        heroes = read_heroes(m, base)

        # Read movement (takes ~0.1s for the delta sample)
        movement = read_movement_targets(m, base, heroes, dt=0.1)

        # Read attacks
        attacks = read_auto_attacks(m, base, heroes)

        print(f"\n--- Sample {t} ---")
        print(f"\n  Movement:")
        print(f"  {'Champion':12s} {'Speed':>6} {'Position':>14} {'Projected':>14}")
        for name, mv in movement.items():
            if mv["is_moving"]:
                proj = vec_str(mv["projected_target"])
                pos = vec_str(mv["position"])
                print(f"  {name:12s} {mv['speed']:6.0f} {pos:>14} {proj:>14}")

        print(f"\n  Active spells:")
        print_attack_table(attacks, heroes)

        # Sleep remaining time for ~1s total per iteration
        time.sleep(0.8)

    m.close()
    print(f"\n{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()
