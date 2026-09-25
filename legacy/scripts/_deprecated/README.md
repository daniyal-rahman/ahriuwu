# Deprecated — kept for reference, NOT wired to anything

- `hid_server.py`, `setup_hid_combo.sh` — a JSON/absolute-coordinate HID gadget.
  The Pi actually in use speaks plain-text `press <key>` / `release <key>`
  (relative), served by `scripts/keysender/hybrid_sender.py`. These were built
  against the wrong protocol and never ran. Do not resurrect without checking
  what the Pi is running first.
