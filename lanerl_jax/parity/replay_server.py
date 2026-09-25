"""Replay a recorded action stream on the source server without a learner.

Used for server regressions, including the historical Q-retarget freeze.
Original entity IDs are mapped by creation rank by ReplayWireDriver; this
diagnostic API is not the actor's screen-click interface.
"""
import argparse
import json
import time
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

from .policy_driver import PolicyActionLog, scan_cast_freeze
from .policy_divergence import ReplayWireDriver
from .record import record_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("actions", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port-base", type=int, default=49100)
    p.add_argument("--server-dir", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--decisions", type=int, help="Replay only this many recorded decisions")
    p.add_argument("--decision-trace", action="store_true")
    args = p.parse_args()
    run_replay(args)


def run_replay(args):
    from lanerl_train import paths
    from lanerl_jax.train.run_manifest import file_sha256, git_provenance, git_environment
    log = PolicyActionLog.load(args.actions)
    decisions = len(log) if args.decisions is None else args.decisions
    if not 1 <= decisions <= len(log):
        raise ValueError("decisions must be within the recorded stream")
    step_ticks = int(log.meta.get("step_ticks", 2))
    if step_ticks < 1:
        raise ValueError("recorded step_ticks must be positive")
    server = (args.server_dir or paths.server_dir()).resolve()
    config = (args.config or paths.default_game_config()).resolve()
    provenance = dict(actions=str(args.actions.resolve()), actions_sha256=file_sha256(args.actions),
        decisions=decisions, step_ticks=step_ticks, decision_trace=args.decision_trace,
        server_dir=str(server), server_binary_sha256=file_sha256(server / "GameServerLib.dll"),
        config=str(config), config_sha256=file_sha256(config), source=git_provenance())
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                   cwd=root, env=git_environment(root)).decode().split("\0")
    with tarfile.open(args.out / "source.tar.gz", "w:gz") as archive:
        for name in sorted(set(names)):
            path = root / name
            if name and path.is_file() and path.suffix in (".py", ".sh", ".html", ".toml", ".patch", ".md", ".json"):
                archive.add(path, arcname=name)
    provenance["snapshot_sha256"] = file_sha256(args.out / "source.tar.gz")
    vendor = paths.server_dir().parents[3]
    (args.out / "vendor.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=vendor, env=git_environment(vendor)))
    provenance["vendor_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=vendor, env=git_environment(vendor)).decode().strip()
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.out / "command.txt").write_text(shlex.join(
        [sys.executable, "-m", "lanerl_jax.parity.replay_server", *sys.argv[1:]]) + "\n")
    driver = ReplayWireDriver(log)
    started = time.monotonic()
    record_trace(args.out, decisions=decisions, port_base=args.port_base, step_ticks=step_ticks,
                 driver=driver, tag="replay", server_dir=server,
                 config_path=config, extra_env={"LANERL_AUTOBUY": "0",
                     "LANERL_DECISION_TRACE": "1" if args.decision_trace else "0"})
    with (args.out / "replay_obs.jsonl").open() as f:
        frames = [json.loads(line) for line in f]
    detector = scan_cast_freeze(frames)
    result = {"seconds": time.monotonic() - started,
              "unresolved_targets": driver.unresolved,
              "freeze": detector.report()}
    (args.out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if detector.invalid or driver.unresolved:
        raise SystemExit(1)


def compare_replay_prefix(original, off, on, decisions):
    """Compare parsed wire and tick hashes, and count unaliased cast events.

    The original is a policy recording; off/on are replay_server outputs.
    Canonical hashes include all ticks through the trace-off recording's end.
    """
    import hashlib
    import re
    from lanerl_jax.train.run_manifest import file_sha256
    original, off, on = map(Path, (original, off, on))
    started = time.monotonic()
    paths={'original':original/'policy_obs.jsonl', 'off':off/'replay_obs.jsonl', 'on':on/'replay_obs.jsonl'}
    wire={}
    frames={}
    for name,path in paths.items():
        values=[]
        with path.open() as f:
            for i,line in enumerate(f):
                if i==decisions: break
                values.append(json.loads(line))
        frames[name]=values
        wire[name]={'frames':len(values), 'last_t_ms':values[-1]['t'], 'blue_final_cs':next(u['cs'] for u in values[-1]['u'] if u.get('k')=='Champion' and u['tm']==100)}
    comparisons={}
    for a,b in [('off','on'),('original','off'),('original','on')]:
        mismatches=[i for i,(x,y) in enumerate(zip(frames[a],frames[b])) if x!=y]
        comparisons[a+'_'+b]={'equal':len(frames[a])==len(frames[b]) and not mismatches,'mismatched_frames':len(mismatches),'first_mismatch':mismatches[0] if mismatches else None}
    blue=next(u['id'] for u in frames['on'][0]['u'] if u.get('k')=='Champion' and u['tm']==100)
    # Include the initial tick and every canonical state hash through the replay's last dumped tick.
    hash_re=re.compile(r'LANERL_STATEHASH t=(-?\d+) n=(\d+) h=([0-9a-f]{16})')
    logs={'off':off/'replay/instance000.log','on':on/'replay/instance000.log','original':original/'policy/instance000.log'}
    hashes={};events={}; cutoff=None
    for name,path in logs.items():
        hs=[];es=[]
        with path.open(errors='replace') as f:
            for line in f:
                m=hash_re.search(line)
                if m:
                    t=int(m[1])
                    if cutoff is not None and t>cutoff:break
                    hs.append((t,int(m[2]),m[3]))
                if 'LANERL_DECISION' in line and 'k=FinishCasting ' in line and f'id={blue} ' in line:
                    es.append(line.strip())
        hashes[name]=hs
        events[name]={'blue_finish_casting':len(es),'blue_auto_finish_casting':sum('auto=True' in e for e in es),'records':es}
        if name=='off':cutoff=hs[-1][0]
    result={'wire':wire,'wire_comparisons':comparisons,'canonical':{name:{'ticks':len(hs),'sha256':hashlib.sha256(json.dumps(hs).encode()).hexdigest()} for name,hs in hashes.items()},'canonical_off_on_equal':hashes['off']==hashes['on'],'canonical_original_off_equal':hashes['original']==hashes['off'],'events':events,'blue_net_id':blue,'cutoff_ms':cutoff,'inputs':{str(p):file_sha256(p) for p in [*paths.values(),*logs.values()]},'wall_s':time.monotonic()-started}
    return result


if __name__ == "__main__":
    main()
