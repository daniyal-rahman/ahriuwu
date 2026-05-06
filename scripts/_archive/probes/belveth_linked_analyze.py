"""Offline-on-Windows analyzer for C:\\tmp\\bv_linked.

Finds Vec3 offsets inside each linked allocation that match the expected
click destination (x, z) across all 3 windows:
  W1 gt∈[38.72, 43.41], dest (3124, 8122)
  W2 gt∈[43.54, 47.77], dest (3736, 8358)
  W3 gt∈[47.90, 50.86], dest (4398, 8444)

Criterion: ≥50% of snapshots in EACH window must read Vec3 (x,z) within
40u of expected. Prints top candidates and the evolution of their values.
"""
import json, struct, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

ROOT = r"C:\tmp\bv_linked"
with open(os.path.join(ROOT, "index.json")) as f:
    index = json.load(f)

WINDOWS = [(38.72, 43.41, 3124, 8122),
           (43.54, 47.77, 3736, 8358),
           (47.90, 50.86, 4398, 8444)]
TOL = 40.0

# Classify each snapshot by window
win_snaps = [[], [], []]
for i, rec in enumerate(index):
    gt = rec.get("gt") or 0
    for wi, (lo, hi, _, _) in enumerate(WINDOWS):
        if lo < gt < hi:
            win_snaps[wi].append(i)
for wi, snaps in enumerate(win_snaps):
    print(f"W{wi+1} [{WINDOWS[wi][0]}-{WINDOWS[wi][1]}] dest={WINDOWS[wi][2:]}: {len(snaps)} snaps")

# Allocations stable across ALL snapshots
ptrsets = []
for rec in index:
    ptrsets.append(set(rec["linked"].keys()))
common = set.intersection(*ptrsets) if ptrsets else set()
print(f"\n{len(common)} allocations present in all {len(index)} snapshots")

def read_xyz(buf, off):
    if off + 12 > len(buf): return None
    return struct.unpack_from("<fff", buf, off)
def ok_xz(v, ex, ez):
    if v is None: return False
    x, y, z = v
    if x != x or z != z or abs(x) > 1e7 or abs(z) > 1e7: return False
    return abs(x - ex) < TOL and abs(z - ez) < TOL

# For memory, process allocations streaming — read each snap's file for this alloc, scan offsets
def scan_alloc(ptr_hex):
    # Load buffer for this allocation at every snapshot
    bufs = []
    for rec in index:
        path = os.path.join(ROOT, f"s{rec['idx']:03d}", f"p_{int(ptr_hex,16):X}.bin")
        try:
            with open(path, "rb") as f: bufs.append(f.read())
        except FileNotFoundError:
            bufs.append(b"")
    L = min(len(b) for b in bufs) if bufs else 0
    if L < 12: return []
    hits = []
    for off in range(0, L - 12, 4):
        ok_all = True
        counts = []
        for wi, (lo, hi, ex, ez) in enumerate(WINDOWS):
            if not win_snaps[wi]: continue
            hc = 0; tc = len(win_snaps[wi])
            for si in win_snaps[wi]:
                if ok_xz(read_xyz(bufs[si], off), ex, ez):
                    hc += 1
            counts.append((hc, tc))
            if hc < max(1, tc // 2):
                ok_all = False; break
        if ok_all:
            hits.append((off, counts, [read_xyz(b, off) for b in bufs]))
    return hits

# Scan all common allocations
print("\nScanning linked allocations...")
total_hits = []
done = 0
for ptr_hex in common:
    hits = scan_alloc(ptr_hex)
    for off, counts, vals in hits:
        total_hits.append((ptr_hex, off, counts, vals))
    done += 1
    if done % 100 == 0:
        print(f"  scanned {done}/{len(common)} allocations; {len(total_hits)} hits so far")

print(f"\n=== {len(total_hits)} total hits ===")
total_hits.sort(key=lambda x: -sum(c[0]/c[1] for c in x[2]))
for ptr_hex, off, counts, vals in total_hits[:20]:
    hit_str = "  ".join(f"{c[0]}/{c[1]}" for c in counts)
    # sample mid-window values
    sample_txt = []
    for wi in range(3):
        if win_snaps[wi]:
            si = win_snaps[wi][len(win_snaps[wi])//2]
            v = vals[si]
            sample_txt.append(f"W{wi+1}:({v[0]:.0f},{v[2]:.0f})" if v else "-")
    print(f"  {ptr_hex}+0x{off:04X}  {hit_str}  {'  '.join(sample_txt)}")
