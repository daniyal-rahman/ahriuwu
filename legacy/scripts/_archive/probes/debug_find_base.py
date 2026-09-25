"""Debug find_module_base on League process."""
import ctypes, ctypes.wintypes as wt, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

_k = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe',
                        '/FO','CSV','/NH'], capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

pid = find_pid()
print(f"PID = {pid}")
if not pid: sys.exit(1)

h = _k.OpenProcess(0x0410, False, pid)
print(f"OpenProcess handle = {h}, lasterr = {_k.GetLastError()}")

# Method 1: Toolhelp32Snapshot
class ME(ctypes.Structure):
    _fields_=[("dwSize",ctypes.c_ulong),("a",ctypes.c_ulong),("b",ctypes.c_ulong),
        ("c",ctypes.c_ulong),("d",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
        ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),
        ("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]

snap = _k.CreateToolhelp32Snapshot(0x18, pid)
print(f"CreateToolhelp32Snapshot = 0x{snap & 0xFFFFFFFFFFFFFFFF:X}, lasterr = {_k.GetLastError()}")
if snap != -1 and snap != 0xFFFFFFFFFFFFFFFF:
    me = ME(); me.dwSize = ctypes.sizeof(ME)
    ok = _k.Module32First(snap, ctypes.byref(me))
    print(f"Module32First ok={ok}, lasterr={_k.GetLastError()}")
    count = 0
    while ok and count < 20:
        name = me.szModule.decode(errors='replace')
        base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
        print(f"  mod[{count}] base=0x{base or 0:X} size=0x{me.modBaseSize:X} name={name}")
        count += 1
        ok = _k.Module32Next(snap, ctypes.byref(me))
    _k.CloseHandle(snap)

# Method 2: EnumProcessModulesEx via psapi
print("\n-- psapi --")
hMods = (ctypes.c_void_p * 1024)()
needed = ctypes.c_ulong(0)
ok = psapi.EnumProcessModulesEx(h, ctypes.byref(hMods), ctypes.sizeof(hMods),
                                 ctypes.byref(needed), 0x03)
print(f"EnumProcessModulesEx ok={ok}, needed={needed.value}, lasterr={_k.GetLastError()}")
if ok:
    n = needed.value // ctypes.sizeof(ctypes.c_void_p)
    class MODINFO(ctypes.Structure):
        _fields_ = [("lpBaseOfDll", ctypes.c_void_p),
                    ("SizeOfImage", ctypes.c_ulong),
                    ("EntryPoint", ctypes.c_void_p)]
    for i in range(min(n, 3)):
        name = ctypes.create_string_buffer(260)
        psapi.GetModuleBaseNameA(h, hMods[i], name, 260)
        mi = MODINFO()
        psapi.GetModuleInformation(h, hMods[i], ctypes.byref(mi), ctypes.sizeof(mi))
        print(f"  mod[{i}] base=0x{mi.lpBaseOfDll or 0:X} size=0x{mi.SizeOfImage:X} name={name.value.decode()}")
