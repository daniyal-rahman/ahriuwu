# ops/ — production infrastructure (NOT disposable)

These run unattended and other things depend on them. `scratchpad/` is for
throwaway experiments; anything a live run needs lives here.

| file | what it does |
|---|---|
| `bc_night.sh supervise` | owns the nightly BC window (06:00-18:00 UTC = 11pm-11am PT). Launch detached on the desktop; does NOT survive a reboot, re-arm after one. |
| `bc5080_gate_watchdog.sh` | keeps the BC trainer alive inside the window; resumes from the last checkpoint on crash. Started by bc_night.sh. |
| `tok_eval_watcher.py` | scores tokenizer checkpoints on fixed held-out sets as they appear. |
| `stage_desktop_standalone.sh` | builds the login/NFS-independent inference bundle at /mnt/storage/ahriuwu-live. |

Status: `bash ops/bc_night.sh status`
