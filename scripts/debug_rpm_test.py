"""Test if ReadProcessMemory works on League process from SSH."""
import ctypes, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

_k = ctypes.windll.kernel32

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe',
                        '/FO','CSV','/NH'], capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

pid = find_pid()
h = _k.OpenProcess(0x0410, False, pid)
print(f"pid={pid} handle={h}")

# Try reading at a known-valid address first: stack region we query
class MBI(ctypes.Structure):
    import ctypes.wintypes as wt
    _fields_ = [("BaseAddress",ctypes.c_void_p),("AllocationBase",ctypes.c_void_p),
                ("AllocationProtect",ctypes.c_ulong),("__a1",ctypes.c_ulong),
                ("RegionSize",ctypes.c_size_t),("State",ctypes.c_ulong),
                ("Protect",ctypes.c_ulong),("Type",ctypes.c_ulong),
                ("__a2",ctypes.c_ulong)]

mbi = MBI()
_k.VirtualQueryEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t

# Walk regions; find any readable committed region; attempt RPM
addr = 0
n_queried = 0; n_committed = 0; n_readable = 0; n_image = 0
first_image = None
while addr < 0x00007FFFFFFFFFFF:
    rc = _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
    if not rc:
        break
    n_queried += 1
    a = mbi.BaseAddress or 0; sz = mbi.RegionSize or 0
    if sz == 0: break
    if mbi.State == 0x1000:  # MEM_COMMIT
        n_committed += 1
        if mbi.Protect & 0xEE:  # readable
            n_readable += 1
            if mbi.Type == 0x1000000:  # MEM_IMAGE
                n_image += 1
                if first_image is None:
                    first_image = (a, sz, mbi.Protect)
                    # Try to read first 2 bytes
                    buf = ctypes.create_string_buffer(2)
                    nr = ctypes.c_size_t(0)
                    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(a), buf, 2, ctypes.byref(nr))
                    print(f"  RPM@0x{a:X}: ok={ok}, n={nr.value}, data={buf.raw[:nr.value]}, lasterr={_k.GetLastError()}")
    addr = a + sz
print(f"\nqueried={n_queried} committed={n_committed} readable={n_readable} image={n_image}")
if first_image:
    print(f"first image: base=0x{first_image[0]:X} size=0x{first_image[1]:X} protect=0x{first_image[2]:X}")
