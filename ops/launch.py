#!/usr/bin/env python3
"""THE launcher for training runs and frozen evaluations. No shell path games.

    ops/launch.py E07 [--seed N] [--init-from CKPT | --resume CKPT] [--dry-run] [--no-canary]
    ops/launch.py eval --ckpt CKPT [--opponent frozen --opponent-ckpt CKPT] [--envs 4]

An experiment is a JSON spec in experiments/<ID>.json (args + slurm resources +
port base). The launcher:
  1. resolves every path in Python and REFUSES if a checkpoint, manifest,
     server build or opponent checkpoint is missing (both mount spellings);
  2. refuses port bases >= 32768 (OPS-003) and duplicate live job names;
  3. runs a CANARY first: the exact argument list with 1 server, 4-step
     rollouts and 2 updates on the desktop (about 3 minutes) and only then
     submits the real job -- a launch bug fails in minutes, not hours;
  4. writes the exact command and the resolved spec into the run directory
     and prints the job id.
Why: on 2026-09-26 four runs were lost to a dropped trailing slash from
`ls -d` in zsh, an inherited update counter/lr schedule, an environment
variable collision that starved evaluations of --opponent-ckpt, and memory
allocations that did not add up. All four are refused or caught here.
"""
import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path

REPO_SRV = Path("/srv/nfs/projects/ahriuwu-lanerl-jax")
REPO_MNT = "/mnt/nfs/projects/ahriuwu-lanerl-jax"
SERVER_DIR = "/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/ClickV3/net6.0"
ENV = ("env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 "
       "LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor ./.venv-gpu/bin/python -m lanerl_jax.train.server_train")


def srv(path: str) -> Path:
    """Both mount spellings resolve to the login node's view for existence checks."""
    return Path(str(path).replace("/mnt/nfs/", "/srv/nfs/", 1))


def mnt(path) -> str:
    return str(path).replace("/srv/nfs/", "/mnt/nfs/", 1)


def must_exist(label, path):
    if not srv(path).exists():
        sys.exit(f"REFUSED: {label} does not exist: {path}")
    return mnt(path)


def live_jobs():
    out = subprocess.run(["squeue", "-h", "-o", "%j %T %C %m"], capture_output=True, text=True).stdout
    return [l.split() for l in out.splitlines() if l.strip()]


def build_args(spec, a):
    args = dict(spec["args"])
    if a.seed is not None:
        args["seed"] = a.seed
    seed = int(args.get("seed", 0))
    args["port-base"] = int(spec["port_base"]) + 40 * seed
    if args["port-base"] >= 32768:
        sys.exit("REFUSED: port base inside the ephemeral range (OPS-003)")
    args["out"] = f"lanerl_jax/runs/{spec['id']}/seed{seed}"
    args["server-dir"] = must_exist("server build", spec.get("server_dir", SERVER_DIR))
    if args.get("opponent") == "frozen":
        args["opponent-ckpt"] = must_exist("opponent checkpoint", a.opponent_ckpt or spec.get("opponent_ckpt", ""))
        must_exist("opponent manifest", Path(srv(args["opponent-ckpt"])).parent / "manifest.json")
    init = a.init_from or spec.get("init_from")
    if a.resume and init:
        sys.exit("REFUSED: --resume and --init-from are exclusive")
    if a.resume:
        args["resume"] = must_exist("resume checkpoint", a.resume)
        must_exist("resume manifest", Path(srv(a.resume)).parent / "manifest.json")
    elif init:
        args["init-from"] = must_exist("init checkpoint", init)
        must_exist("init manifest", Path(srv(init)).parent / "manifest.json")
    return args


def argv_of(args):
    out = []
    for k, v in args.items():
        if v is True: out.append(f"--{k}")
        elif v is False or v is None: continue
        elif isinstance(v, list): out += [f"--{k}", *map(str, v)]
        else: out += [f"--{k}", str(v)]
    return out


def canary(args, name):
    """The exact config, shrunk: 1 server, 4-step rollouts, 2 updates."""
    c = dict(args); c.update({"envs": 1, "rollout": 4, "updates": 2, "minibatches": 1,
                              "port-base": args["port-base"] + 30, "out": args["out"] + "-canary",
                              "save-updates": []})
    c.pop("resume", None)           # a resume's counter would exceed 2 updates; canary the code path with init-from
    if "init-from" not in c and "resume" in args:
        c["init-from"] = args["resume"]
    cmd = ["srun", "-p", "cpu", "-w", "desktop", "--cpus-per-task=2", "--mem=4G", "--time=15",
           f"--chdir={REPO_MNT}", f"--job-name=canary-{name}", "bash", "-c",
           ENV + " " + shlex.join(argv_of(c))]
    print("canary:", shlex.join(argv_of(c))[:300], flush=True)
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "absl" not in l and "cudart" not in l)[-1500:]
    if r.returncode != 0 or '"update": 2' not in r.stdout:
        sys.exit(f"CANARY FAILED after {time.time()-t:.0f}s (rc {r.returncode}):\n{tail}")
    print(f"canary passed in {time.time()-t:.0f}s", flush=True)


def submit(spec, args, name, dry):
    res = spec["slurm"]
    argv = argv_of(args)
    if res.get("partition") == "gpup":
        cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--cpus-per-task={res['cpus']}", f"--mem={res['mem']}",
               "slurm/server_train.sbatch", *argv]
    else:
        cmd = ["srun", "-p", "cpu", "-w", "desktop", f"--cpus-per-task={res['cpus']}", f"--mem={res['mem']}",
               f"--time={res.get('time', '24:00:00')}", f"--chdir={REPO_MNT}", f"--job-name={name}",
               "bash", "-c", ENV + " " + shlex.join(argv)]
    print("command:", shlex.join(cmd)[:400], flush=True)
    if dry:
        return None
    out_dir = srv(REPO_SRV / args["out"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "launch.json").write_text(json.dumps({"spec": spec, "args": args, "command": cmd,
                                                      "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    if cmd[0] == "sbatch":
        jid = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_SRV).stdout.strip()
        print("job", jid); return jid
    log = out_dir / f"{name}.out"
    subprocess.Popen(cmd, stdout=log.open("ab"), stderr=subprocess.STDOUT, cwd=REPO_SRV, start_new_session=True)
    print("srun started; log", log); return "srun"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", help="experiment ID (experiments/<ID>.json) or 'eval'")
    p.add_argument("--seed", type=int); p.add_argument("--init-from"); p.add_argument("--resume")
    p.add_argument("--ckpt", help="eval: checkpoint to evaluate"); p.add_argument("--opponent", default=None)
    p.add_argument("--opponent-ckpt"); p.add_argument("--envs", type=int, default=4); p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--no-canary", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="eval: argmax actions (diagnostic)")
    a = p.parse_args()
    if a.experiment == "eval":
        ck = must_exist("checkpoint", a.ckpt); must_exist("manifest", Path(srv(a.ckpt)).parent / "manifest.json")
        spec = {"id": "EVAL", "port_base": 24300, "slurm": {"partition": "cpu", "cpus": 2, "mem": "4G", "time": "4:00:00"},
                "args": {"envs": a.envs, "opponent": a.opponent or "mirror", "start-near-wave": True, "step-ticks": 6,
                         "episode-s": 600, "eval-episodes": a.episodes, "seed": 0}}
        args = build_args(spec, argparse.Namespace(seed=None, init_from=None, resume=ck, opponent_ckpt=a.opponent_ckpt))
        args["out"] = "lanerl_jax/runs/EVAL"
        if a.deterministic: args["deterministic"] = True
        submit(spec, args, "EVAL-det" if a.deterministic else "EVAL", a.dry_run); return
    spec = json.loads((REPO_SRV / "experiments" / f"{a.experiment}.json").read_text())
    name = f"{spec['id']}-s{a.seed if a.seed is not None else spec['args'].get('seed', 0)}"
    if any(j[0] == name for j in live_jobs()):
        sys.exit(f"REFUSED: a job named {name} is already in the queue")
    args = build_args(spec, a)
    if not a.dry_run and not a.no_canary:
        canary(args, name)
    submit(spec, args, name, a.dry_run)


if __name__ == "__main__":
    main()
