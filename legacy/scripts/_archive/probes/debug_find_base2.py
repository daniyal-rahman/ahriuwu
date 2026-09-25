"""Debug: enable SeDebugPrivilege, try VirtualQueryEx-based module scan."""
import ctypes, ctypes.wintypes as wt, subprocess, struct, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

_k = ctypes.windll.kernel32
advapi = ctypes.windll.advapi32

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe',
                        '/FO','CSV','/NH'], capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def enable_debug_priv():
    class LUID(ctypes.Structure):
        _fields_=[("LowPart",ctypes.c_ulong),("HighPart",ctypes.c_long)]
    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_=[("Luid",LUID),("Attributes",ctypes.c_ulong)]
    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_=[("PrivilegeCount",ctypes.c_ulong),
                  ("Privileges",LUID_AND_ATTRIBUTES*1)]
    hToken = wt.HANDLE()
    if not advapi.OpenProcessToken(_k.GetCurrentProcess(), 0x20|0x8, ctypes.byref(hToken)):
        print(f"  OpenProcessToken err={_k.GetLastError()}"); return False
    luid = LUID()
    if not advapi.LookupPrivilegeValueW(None, "SeDebugPrivilege", ctypes.byref(luid)):
        print(f"  LookupPrivilegeValueW err={_k.GetLastError()}"); return False
    tp = TOKEN_PRIVILEGES()
    tp.PrivilegeCount = 1
    tp.Privileges[0].Luid = luid
    tp.Privileges[0].Attributes = 0x2  # SE_PRIVILEGE_ENABLED
    if not advapi.AdjustTokenPrivileges(hToken, False, ctypes.byref(tp), 0, None, None):
        print(f"  AdjustTokenPrivileges err={_k.GetLastError()}"); return False
    lastErr = _k.GetLastError()
    if lastErr == 1300:  # ERROR_NOT_ALL_ASSIGNED
        print("  ERROR_NOT_ALL_ASSIGNED — debug priv not granted"); return False
    print("  SeDebugPrivilege enabled"); return True

print("Enabling SeDebugPrivilege...")
enable_debug_priv()

pid = find_pid()
print(f"PID = {pid}")

# Open with extended flags incl PROCESS_QUERY_INFORMATION (0x0400) and VM_READ (0x10)
h = _k.OpenProcess(0x0410, False, pid)
print(f"OpenProcess(0x0410) = {h}")
# Also try the more limited query
h2 = _k.OpenProcess(0x1000 | 0x10, False, pid)  # QUERY_LIMITED | VM_READ
print(f"OpenProcess(0x1010) = {h2}")

# Method: VirtualQueryEx walk, find MZ headers
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress",ctypes.c_void_p),("AllocationBase",ctypes.c_void_p),
                ("AllocationProtect",ctypes.c_ulong),("__a1",ctypes.c_ulong),
                ("RegionSize",ctypes.c_size_t),("State",ctypes.c_ulong),
                ("Protect",ctypes.c_ulong),("Type",ctypes.c_ulong),
                ("__a2",ctypes.c_ulong)]

_k.VirtualQueryEx.argtypes=[wt.HANDLE,ctypes.c_void_p,ctypes.POINTER(MBI),ctypes.c_size_t]
_k.VirtualQueryEx.restype=ctypes.c_size_t

def rpm(handle, a, sz):
    buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(handle, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
    return buf.raw[:n.value] if ok and n.value else None

print("\nScanning for MZ headers...")
for handle in [h, h2]:
    if not handle: continue
    addr = 0; mbi = MBI(); found = 0; mz_hits = []
    while addr < 0x7FFFFFFFFFFF:
        rc = _k.VirtualQueryEx(handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
        if not rc: break
        a = mbi.BaseAddress or 0; sz = mbi.RegionSize or 0
        if sz == 0: break
        # MEM_IMAGE = 0x1000000
        if mbi.State == 0x1000 and mbi.Type == 0x1000000 and sz > 0x1000000:
            d = rpm(handle, a, 2)
            if d == b'MZ':
                # read PE size — SizeOfImage at PE+0x50
                pe_off_buf = rpm(handle, a + 0x3C, 4)
                if pe_off_buf:
                    pe_off = struct.unpack('<I', pe_off_buf)[0]
                    sz_buf = rpm(handle, a + pe_off + 0x50, 4)
                    if sz_buf:
                        img_size = struct.unpack('<I', sz_buf)[0]
                        # Get image name via readback: try name at section (harder). Just record.
                        mz_hits.append((a, sz, img_size))
        addr = a + sz
    print(f"Handle {handle}: {len(mz_hits)} large MZ modules:")
    for a, regsz, imgsz in mz_hits[:10]:
        print(f"  base=0x{a:X}  region_sz=0x{regsz:X}  image_sz=0x{imgsz:X}")
    break  # just need first working handle
