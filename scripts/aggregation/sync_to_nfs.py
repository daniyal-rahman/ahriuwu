#!/usr/bin/env python3
"""
Windows-side watcher that uploads completed pipeline output to NFS via SSH.

Polls --src for match dirs whose labels.json exists (= post-process is done),
scp's the full match dir to danilogin:/mnt/nfs/datasets/<dataset>/<match_id>/,
verifies the upload via remote `find -type f | wc -l`, and optionally deletes
the local copy on success.

Designed to run alongside pipeline.py so transfers don't block the next game's
pass1.

Usage:
    python scripts/aggregation/sync_to_nfs.py \
        --src C:\\tmp\\replay_data \
        --dataset lol_replays_16_9_772 \
        --remote danilogin:/mnt/nfs/datasets \
        --poll 30 \
        [--delete-local] [--once]

Prerequisite: `ssh windows ssh danilogin` must work without prompts (key auth
+ host key already accepted). See scripts/README.md "NFS sink" for setup.
"""
import argparse, json, os, shlex, subprocess, sys, time

LABELS_FILE = "labels.json"
SENTINEL = ".synced"  # written into local match dir after successful upload


def _run(cmd, **kw):
    """Run a shell command. Returns (rc, stdout, stderr)."""
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, p.stdout, p.stderr


def _remote_count(remote, sub_path):
    """Count files in a remote path. Returns int or None on error."""
    host, root = remote.split(":", 1)
    rc, out, _ = _run(["ssh", host, f"find {shlex.quote(root + '/' + sub_path)} -type f 2>/dev/null | wc -l"])
    if rc != 0: return None
    try: return int(out.strip())
    except ValueError: return None


def _local_count(path):
    n = 0
    for _, _, files in os.walk(path):
        n += len(files)
    return n


def _scp_dir(local_dir, remote, dataset, match_id):
    """scp -r local_dir to remote/<dataset>/<match_id>/. Creates parent on remote first."""
    host, root = remote.split(":", 1)
    remote_parent = f"{root}/{dataset}"
    # mkdir -p on remote (idempotent)
    rc, _, err = _run(["ssh", host, f"mkdir -p {shlex.quote(remote_parent)}"])
    if rc != 0:
        return False, f"remote mkdir failed: {err.strip()}"
    rc, _, err = _run(["scp", "-rq", local_dir, f"{host}:{remote_parent}/"])
    if rc != 0:
        return False, f"scp failed: {err.strip()}"
    return True, None


def list_completed_matches(src):
    """Yield match_id for each <src>/<match_id>/labels.json that exists and isn't yet marked synced."""
    if not os.path.isdir(src): return
    for name in sorted(os.listdir(src)):
        match_dir = os.path.join(src, name)
        if not os.path.isdir(match_dir): continue
        if name == "logs": continue  # pipeline writes per-game logs here, skip
        labels = os.path.join(match_dir, LABELS_FILE)
        if not os.path.exists(labels): continue
        if os.path.exists(os.path.join(match_dir, SENTINEL)): continue
        yield name


def sync_one(src, match_id, remote, dataset, delete_local=False):
    """scp a single match dir, verify by file-count match, mark sentinel locally,
    and optionally delete the local copy. Returns (ok, msg)."""
    local_dir = os.path.join(src, match_id)
    n_local = _local_count(local_dir)
    print(f"  [{match_id}] uploading ({n_local} files)...", flush=True)
    t0 = time.time()
    ok, err = _scp_dir(local_dir, remote, dataset, match_id)
    if not ok:
        return False, err
    elapsed = time.time() - t0
    n_remote = _remote_count(remote, f"{dataset}/{match_id}")
    if n_remote is None:
        return False, "remote file-count failed"
    if n_remote != n_local:
        return False, f"file-count mismatch: local={n_local} remote={n_remote}"
    print(f"  [{match_id}] verified {n_remote} files in {elapsed:.0f}s "
          f"({n_local/elapsed:.0f} files/s)", flush=True)
    # mark sentinel so we don't re-upload on next poll
    with open(os.path.join(local_dir, SENTINEL), "w") as f:
        f.write(json.dumps({"uploaded_at": time.time(), "n_files": n_local,
                            "remote": f"{remote}/{dataset}/{match_id}",
                            "wall_seconds": round(elapsed, 1)}))
    if delete_local:
        import shutil
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"  [{match_id}] deleted local", flush=True)
    return True, None


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

    print(f"[sync_to_nfs] watching {args.src} → {args.remote}/{args.dataset} every {args.poll}s "
          f"(delete_local={args.delete_local})", flush=True)
    while True:
        pending = list(list_completed_matches(args.src))
        if pending:
            print(f"[sync_to_nfs] {len(pending)} match(es) to upload: {pending[:5]}{'...' if len(pending)>5 else ''}",
                  flush=True)
        for mid in pending:
            ok, err = sync_one(args.src, mid, args.remote, args.dataset, delete_local=args.delete_local)
            if not ok:
                print(f"  [{mid}] FAIL: {err}", flush=True)
        if args.once: break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
