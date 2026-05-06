"""Session-1 helper to lock camera via hardware scan-code SendInput.

Pynput / PostMessage / keybd_event all emit virtual-key events that many
games (League included) ignore because their input layer polls hardware
state directly. SendInput with `KEYEVENTF_SCANCODE` generates a hardware-
flavored event that DirectInput/Raw-Input pick up.

Usage (via schtasks /IT):
    pythonw.exe lock_cam_once.py <key> [mode]
"""
import ctypes, ctypes.wintypes as wt, sys, time

LOG = r"C:\tmp\lock_cam.log"
def log(s):
    try:
        ts = time.strftime('%H:%M:%S')
        with open(LOG, "a") as f: f.write(f"[{ts}] {s}\n")
    except Exception as e:
        # Also try writing the exception to a fallback path
        try:
            with open(r"C:\tmp\lock_cam.err", "a") as ff: ff.write(f"log err: {e}\n")
        except Exception: pass

# ── SendInput with scan-codes ────────────────────────────────────────
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class _II(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _II)]
INPUT_KEYBOARD    = 1
KEYEVENTF_KEYUP   = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# Virtual-key -> hardware scan code (US QWERTY)
VK_TO_SCAN = {
    '1':0x02, '2':0x03, '3':0x04, '4':0x05, '5':0x06,
    '6':0x07, '7':0x08, '8':0x09, '9':0x0A, '0':0x0B,
    'q':0x10, 'w':0x11, 'e':0x12, 'r':0x13, 't':0x14,
    'y':0x15, 'u':0x16, 'i':0x17, 'o':0x18, 'p':0x19,
    'space':0x39,
}

def send_scan(scan, up=False):
    extra = ctypes.c_ulong(0)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    ki = KeyBdInput(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=ctypes.pointer(extra))
    inp = Input(type=INPUT_KEYBOARD, ii=_II(ki=ki))
    n = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    return n == 1

def tap(scan):
    send_scan(scan, up=False)
    time.sleep(0.06)
    send_scan(scan, up=True)

def find_game_hwnd():
    u32 = ctypes.windll.user32
    hwnds = []
    def cb(h, _):
        if not u32.IsWindowVisible(h): return True
        n = u32.GetWindowTextLengthW(h)
        if n <= 0: return True
        buf = ctypes.create_unicode_buffer(n + 1)
        u32.GetWindowTextW(h, buf, n + 1)
        cls = ctypes.create_unicode_buffer(256)
        u32.GetClassNameW(h, cls, 256)
        if cls.value == "RiotWindowClass":
            hwnds.append((h, cls.value, buf.value))
        return True
    u32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    return hwnds[0] if hwnds else (None, None, None)

def focus_game():
    u32 = ctypes.windll.user32; k = ctypes.windll.kernel32
    h, cls, title = find_game_hwnd()
    if not h: log("no hwnd"); return None
    log(f"hwnd=0x{h:X} class={cls!r} title={title!r}")
    u32.SystemParametersInfoW(0x2001, 0, None, 0)
    fg = u32.GetForegroundWindow()
    ft = u32.GetWindowThreadProcessId(fg, None)
    ct = k.GetCurrentThreadId()
    u32.AttachThreadInput(ct, ft, True)
    u32.keybd_event(0x12, 0, 0, 0); u32.keybd_event(0x12, 0, 2, 0)
    u32.ShowWindow(h, 9); u32.BringWindowToTop(h); u32.SetForegroundWindow(h)
    u32.AttachThreadInput(ct, ft, False)
    time.sleep(0.3)
    log(f"post-focus fg=0x{u32.GetForegroundWindow():X} target=0x{h:X}")
    return h

def main():
    open(LOG, "w").close()
    if len(sys.argv) < 2: return
    key = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "double"
    scan = VK_TO_SCAN.get(key.lower())
    if scan is None:
        log(f"unknown key {key}"); return
    log(f"key={key!r} scan=0x{scan:X} mode={mode}")
    h = focus_game()
    if not h: return
    if mode == "single":
        tap(scan)
    elif mode == "triple":
        for _ in range(3): tap(scan); time.sleep(0.25)
    else:
        tap(scan); time.sleep(0.28); tap(scan)
    time.sleep(0.3)
    log("done")

if __name__ == "__main__":
    main()
