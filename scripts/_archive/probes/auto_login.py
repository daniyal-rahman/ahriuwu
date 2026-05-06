"""Launch Riot Client, auto-type creds from .env, wait for LeagueClient LCU.

Reads C:\\Users\\daniz\\.env (lines like RIOT_USERNAME=x / RIOT_PASSWORD=y).
Uses AttachThreadInput + pynput to focus & type into the Riot Client window.

Usage:  python auto_login.py
Exit 0 if LCU session is SUCCEEDED within timeout, else non-zero.
"""
import base64, ctypes, ctypes.wintypes as wt, json, os, ssl, subprocess, sys, time
import urllib.request
from pynput.keyboard import Controller as KbCtl, Key

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

ENV_PATH    = r"C:\Users\daniz\.env"
LOCKFILE    = r"C:\Riot Games\League of Legends\lockfile"
RIOT_CLIENT = r"C:\Riot Games\Riot Client\RiotClientServices.exe"

_k   = ctypes.windll.kernel32
_u32 = ctypes.windll.user32
_kb  = KbCtl()
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE

def load_env():
    env = {}
    for line in open(ENV_PATH, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def find_window(title_fragment):
    """Find top-level window whose title contains fragment (case-insensitive)."""
    hwnds = []
    frag = title_fragment.lower()
    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if not _u32.IsWindowVisible(hwnd): return True
        n = _u32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            _u32.GetWindowTextW(hwnd, buf, n + 1)
            if frag in buf.value.lower():
                hwnds.append((hwnd, buf.value))
        return True
    _u32.EnumWindows(cb, 0)
    return hwnds

def focus(hwnd):
    """Force-focus a window using the AttachThreadInput trick."""
    _u32.SystemParametersInfoW(0x2001, 0, None, 0)
    fg = _u32.GetForegroundWindow()
    ft = _u32.GetWindowThreadProcessId(fg, None)
    ct = _k.GetCurrentThreadId()
    _u32.AttachThreadInput(ct, ft, True)
    _u32.keybd_event(0x12, 0, 0, 0); _u32.keybd_event(0x12, 0, 2, 0)  # ALT kick
    _u32.ShowWindow(hwnd, 9)
    _u32.BringWindowToTop(hwnd)
    _u32.SetForegroundWindow(hwnd)
    _u32.AttachThreadInput(ct, ft, False)
    time.sleep(0.3)

def type_str(s, delay=0.03):
    for ch in s:
        _kb.press(ch); time.sleep(delay); _kb.release(ch)

def lcu_session_state():
    """Return LCU login session state, or None if API not ready."""
    try:
        p = open(LOCKFILE).read().strip().split(':')
        port, token = p[2], p[3]
        auth = base64.b64encode(f'riot:{token}'.encode()).decode()
        req = urllib.request.Request(
            f"https://127.0.0.1:{port}/lol-login/v1/session",
            headers={'Authorization': 'Basic ' + auth})
        with urllib.request.urlopen(req, context=_ctx, timeout=4) as r:
            return json.loads(r.read())
    except Exception:
        return None

def main():
    env = load_env()
    u = env.get('RIOT_USERNAME'); p = env.get('RIOT_PASSWORD')
    if not u or not p:
        print(f"ERROR: .env missing RIOT_USERNAME / RIOT_PASSWORD"); sys.exit(2)
    print(f"loaded creds for user={u[:3]}***")

    # Kill any stale clients
    subprocess.run(['taskkill','/F','/IM','LeagueClient.exe','/T'], capture_output=True)
    subprocess.run(['taskkill','/F','/IM','RiotClientServices.exe','/T'], capture_output=True)
    subprocess.run(['taskkill','/F','/IM','RiotClientUx.exe','/T'], capture_output=True)
    subprocess.run(['taskkill','/F','/IM','League of Legends.exe','/T'], capture_output=True)
    time.sleep(2)

    print("Launching Riot Client...")
    subprocess.Popen([RIOT_CLIENT, '--launch-product=league_of_legends',
                      '--launch-patchline=live'])

    # Wait for login window
    print("Waiting for login window...")
    t0 = time.time(); login_hwnd = None
    while time.time() - t0 < 60:
        # Riot Client login screen window title is "Riot Client"
        wins = find_window('riot client')
        if wins:
            login_hwnd = wins[0][0]
            print(f"  found: {wins[0][1]!r} (hwnd={login_hwnd})")
            break
        time.sleep(1)
    if not login_hwnd:
        print("TIMEOUT waiting for Riot Client login window"); sys.exit(3)

    # Already logged in check — if lockfile appears quickly we skip typing
    time.sleep(3)
    for _ in range(5):
        s = lcu_session_state()
        if s and s.get('state') == 'SUCCEEDED':
            print("Already logged in via persistent session.")
            sys.exit(0)
        time.sleep(1)

    # Type creds
    print("Typing credentials...")
    focus(login_hwnd)
    time.sleep(0.8)

    # Clear username field (Ctrl+A, Del), type
    _kb.press(Key.ctrl); _kb.press('a'); _kb.release('a'); _kb.release(Key.ctrl)
    time.sleep(0.1)
    _kb.press(Key.delete); _kb.release(Key.delete)
    time.sleep(0.1)
    type_str(u)
    time.sleep(0.3)
    _kb.press(Key.tab); _kb.release(Key.tab)
    time.sleep(0.3)
    type_str(p)
    time.sleep(0.3)
    _kb.press(Key.enter); _kb.release(Key.enter)
    print("  submitted")

    # Wait for LCU session
    print("Waiting for LCU session SUCCEEDED...")
    t0 = time.time()
    while time.time() - t0 < 180:
        s = lcu_session_state()
        if s:
            state = s.get('state'); connected = s.get('connected')
            sys.stdout.write(f"  [{int(time.time()-t0):3d}s] state={state} connected={connected}\n")
            sys.stdout.flush()
            if state == 'SUCCEEDED':
                print("LOGIN_OK"); sys.exit(0)
        time.sleep(3)
    print("TIMEOUT waiting for login"); sys.exit(4)

if __name__ == '__main__':
    main()
