#!/usr/bin/env python3
"""
Windows-side watcher that uploads completed pipeline output to NFS via SSH.

Polls --src for match dirs whose labels.json exists (= post-process is done),
streams the full match dir over a single ssh-tar pipe to
danilogin:/mnt/nfs/datasets/<dataset>/<match_id>/, verifies the upload via
remote `find -type f | wc -l`, and optionally deletes the local copy on
success.

Tar-streaming avoids the per-file overhead of scp/rsync — with ~33k tiny
PNGs per game, scp spent ~5min on filesystem syscalls per game; tar pipes
the same bytes in ~1-2min over a single TCP connection.

Sentinel lives at <src>/.synced/<match_id>.json (sibling to match dirs) so
it survives --delete-local. pipeline.py also checks this path as an extra
skip signal during resume — this prevents the "delete-local breaks resume"
incident from 2026-05-07.

Usage:
    python scripts/aggregation/sync_to_nfs.py \\
        --src C:\\tmp\\replay_data \\
        --dataset lol_replays_16_9_772 \\
        --remote danilogin:/mnt/nfs/datasets \\
        --poll 30 \\
        [--delete-local] [--once]

Prerequisite: `ssh windows ssh danilogin` must work without prompts (key auth
+ host key already accepted).
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

LABELS_FILE = "labels.json"
SENTINEL_DIR = ".synced"           # subdir of --src holding per-match sentinels
LEGACY_SENTINEL = ".synced"        # in-dir file from pre-2026-05-08 layout (auto-migrated)
DEAD_LETTER_FILE = ".sync_failures.json"  # under <src>/<SENTINEL_DIR>/
MAX_RETRIES = 3                    # consecutive fails per match before dead-lettering
BACKLOG_WARN_GAMES = 5             # warn when this many matches are queued
BACKLOG_WARN_GB = 30.0             # or this much local data is queued

EXCLUDE_DIRS = {"logs", SENTINEL_DIR}


def _run(cmd, **kw):
    """Run a shell command. Returns (rc, stdout, stderr)."""
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, p.stdout, p.stderr


def _parse_remote(remote):
    """'host:path' → (host, path). Caller pre-validated the colon presence."""
    host, root = remote.split(":", 1)
    return host, root


def _ssh(host, cmd):
    """Run a single shell command on `host` via ssh. Returns (rc, stdout, stderr)."""
    return _run(["ssh", host, cmd])


def _remote_count(remote, sub_path):
    """Count files in a remote path. Returns int or None on error."""
    host, root = _parse_remote(remote)
    rc, out, _ = _ssh(host, f"find {shlex.quote(root + '/' + sub_path)} -type f 2>/dev/null | wc -l")
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def _local_count_and_bytes(path):
    """Single walk: returns (n_files, n_bytes). Used for both verification and backlog metrics."""
    n_files = 0
    n_bytes = 0
    for root, _, files in os.walk(path):
        for name in files:
            n_files += 1
            try:
                n_bytes += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass  # file vanished mid-walk (race with pipeline cleanup) — ignore
    return n_files, n_bytes


def _sentinel_path(src, match_id):
    return os.path.join(src, SENTINEL_DIR, f"{match_id}.json")


def _has_sentinel(src, match_id):
    """True if either new-layout sentinel OR legacy in-dir sentinel exists."""
    if os.path.exists(_sentinel_path(src, match_id)):
        return True
    legacy = os.path.join(src, match_id, LEGACY_SENTINEL)
    return os.path.exists(legacy)


def _migrate_legacy_sentinels(src):
    """One-shot: any pre-existing <src>/<mid>/.synced files get promoted to <src>/.synced/<mid>.json
    so resume semantics are preserved across the layout change. Idempotent."""
    sentinel_dir = os.path.join(src, SENTINEL_DIR)
    os.makedirs(sentinel_dir, exist_ok=True)
    migrated = 0
    for name in os.listdir(src):
        match_dir = os.path.join(src, name)
        if not os.path.isdir(match_dir) or name in EXCLUDE_DIRS:
            continue
        legacy = os.path.join(match_dir, LEGACY_SENTINEL)
        if not os.path.exists(legacy):
            continue
        new_path = _sentinel_path(src, name)
        if not os.path.exists(new_path):
            try:
                shutil.move(legacy, new_path)
                migrated += 1
            except OSError as e:
                print(f"  [migrate] failed for {name}: {e}", flush=True)
    if migrated:
        print(f"[sync_to_nfs] migrated {migrated} legacy sentinels to {sentinel_dir}", flush=True)


def _tar_stream_dir(local_dir, remote, dataset, match_id):
    """Stream local_dir to remote/<dataset>/<match_id>/ over a single ssh-tar pipe.

    Single TCP stream, no per-file fsync. Reproduces the original directory tree
    on the remote (PNGs aren't recompressed since they're already deflate)."""
    host, root = _parse_remote(remote)
    remote_parent = f"{root}/{dataset}"

    rc, _, err = _ssh(host, f"mkdir -p {shlex.quote(remote_parent)}")
    if rc != 0:
        return False, f"remote mkdir failed: {err.strip()}"

    # Pre-clean any partial leftover from a previous failed transfer so we
    # don't end up with a hybrid file count that never converges.
    rc, _, err = _ssh(host, f"rm -rf {shlex.quote(remote_parent + '/' + match_id)}")
    if rc != 0:
        return False, f"remote pre-clean failed: {err.strip()}"

    name = os.path.basename(local_dir.rstrip(os.sep))
    ssh_cmd = ["ssh", host, f"tar -xf - -C {shlex.quote(remote_parent)}"]
    proc = subprocess.Popen(ssh_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            tar.add(local_dir, arcname=name)
    except Exception as e:
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait()
        return False, f"tar stream error: {e!r}"
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read().decode("utf-8", "replace")
        return False, f"remote tar failed (rc={rc}): {err.strip()[:300]}"
    return True, None


def list_completed_matches(src):
    """Yield match_id for each <src>/<match_id>/labels.json that exists and isn't yet synced."""
    if not os.path.isdir(src):
        return
    for name in sorted(os.listdir(src)):
        if name in EXCLUDE_DIRS:
            continue
        match_dir = os.path.join(src, name)
        if not os.path.isdir(match_dir):
            continue
        if not os.path.exists(os.path.join(match_dir, LABELS_FILE)):
            continue
        if _has_sentinel(src, name):
            continue
        yield name


def _backlog_metrics(src, pending):
    """Sum file count and bytes across pending matches. Returns (n_files, n_bytes)."""
    total_files = 0
    total_bytes = 0
    for mid in pending:
        n_f, n_b = _local_count_and_bytes(os.path.join(src, mid))
        total_files += n_f
        total_bytes += n_b
    return total_files, total_bytes


def sync_one(src, match_id, remote, dataset, delete_local=False):
    """Tar-stream a single match dir, verify by file-count, write sentinel, optionally
    delete local. Returns (ok, msg, stats). stats is a dict of {n_files,n_bytes,wall,mb_s,files_s}."""
    local_dir = os.path.join(src, match_id)
    n_local, n_bytes = _local_count_and_bytes(local_dir)
    size_mb = n_bytes / (1024 * 1024)
    print(f"  [{match_id}] streaming ({n_local} files / {size_mb:.0f} MB via tar)...", flush=True)
    t0 = time.time()
    ok, err = _tar_stream_dir(local_dir, remote, dataset, match_id)
    if not ok:
        return False, err, None
    elapsed = time.time() - t0
    n_remote = _remote_count(remote, f"{dataset}/{match_id}")
    if n_remote is None:
        return False, "remote file-count failed", None
    if n_remote != n_local:
        return False, f"file-count mismatch: local={n_local} remote={n_remote}", None

    files_s = n_local / elapsed if elapsed > 0 else 0
    mb_s = size_mb / elapsed if elapsed > 0 else 0
    print(f"  [{match_id}] verified {n_remote} files in {elapsed:.0f}s "
          f"({files_s:.0f} files/s, {mb_s:.1f} MB/s)", flush=True)

    sentinel = _sentinel_path(src, match_id)
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    with open(sentinel, "w") as f:
        json.dump({
            "match_id": match_id,
            "uploaded_at": time.time(),
            "n_files": n_local,
            "n_bytes": n_bytes,
            "remote": f"{remote}/{dataset}/{match_id}",
            "wall_seconds": round(elapsed, 1),
            "mb_per_sec": round(mb_s, 1),
        }, f)

    if delete_local:
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"  [{match_id}] deleted local", flush=True)

    return True, None, {"n_files": n_local, "n_bytes": n_bytes, "wall": elapsed,
                        "files_s": files_s, "mb_s": mb_s}


def _load_dead_letter(src):
    """Read prior dead-letter list (matches abandoned after MAX_RETRIES failures)."""
    p = os.path.join(src, SENTINEL_DIR, DEAD_LETTER_FILE)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_dead_letter(src, dead):
    p = os.path.join(src, SENTINEL_DIR, DEAD_LETTER_FILE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(dead, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Watch pipeline output dir and ship completed matches to NFS")
    ap.add_argument("--src", required=True, help="local pipeline output dir (e.g. C:\\tmp\\replay_data)")
    ap.add_argument("--dataset", required=True, help="subdir name under remote root (e.g. lol_replays_16_9_772)")
    ap.add_argument("--remote", required=True, help="ssh-host:remote-root (e.g. danilogin:/mnt/nfs/datasets)")
    ap.add_argument("--poll", type=int, default=30, help="seconds between scans (default 30)")
    ap.add_argument("--delete-local", action="store_true",
                    help="delete the local match dir after verified upload")
    ap.add_argument("--once", action="store_true", help="scan once and exit (default: forever)")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"FATAL: src dir does not exist: {args.src}", file=sys.stderr)
        sys.exit(1)
    if ":" not in args.remote:
        print(f"FATAL: --remote must be host:path", file=sys.stderr)
        sys.exit(1)

    _migrate_legacy_sentinels(args.src)

    # Per-match failure counters live across poll cycles within a single process.
    # Restart resets them. Dead-letter persists across processes.
    fail_counts = {}
    dead_letter = _load_dead_letter(args.src)
    if dead_letter:
        print(f"[sync_to_nfs] dead-letter has {len(dead_letter)} match(es) — they'll be skipped",
              flush=True)

    print(f"[sync_to_nfs] watching {args.src} → {args.remote}/{args.dataset} every {args.poll}s "
          f"(delete_local={args.delete_local}, max_retries={MAX_RETRIES})", flush=True)

    while True:
        pending = [m for m in list_completed_matches(args.src) if m not in dead_letter]
        if pending:
            n_files, n_bytes = _backlog_metrics(args.src, pending)
            backlog_gb = n_bytes / (1024 ** 3)
            warn = " WARN" if (len(pending) >= BACKLOG_WARN_GAMES or backlog_gb >= BACKLOG_WARN_GB) else ""
            shown = pending[:5]
            print(f"[sync_to_nfs]{warn} BACKLOG: {len(pending)} match(es) / {backlog_gb:.1f} GB "
                  f"to upload: {shown}{'...' if len(pending) > 5 else ''}", flush=True)

        for mid in pending:
            ok, err, _stats = sync_one(args.src, mid, args.remote, args.dataset,
                                       delete_local=args.delete_local)
            if ok:
                fail_counts.pop(mid, None)
                continue

            fail_counts[mid] = fail_counts.get(mid, 0) + 1
            print(f"  [{mid}] FAIL ({fail_counts[mid]}/{MAX_RETRIES}): {err}", flush=True)
            if fail_counts[mid] >= MAX_RETRIES:
                print(f"  [{mid}] DEAD-LETTERED after {MAX_RETRIES} failures — won't retry "
                      f"until you delete <src>/.synced/{DEAD_LETTER_FILE}", flush=True)
                dead_letter[mid] = {
                    "first_failed_at": time.time(),
                    "n_attempts": fail_counts[mid],
                    "last_error": err,
                }
                _save_dead_letter(args.src, dead_letter)
                fail_counts.pop(mid, None)

        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
