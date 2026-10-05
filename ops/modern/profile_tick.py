"""Attribute modern world tick device time to source files and lines.

TOOL (MODERN-021). Builds the same world and scripted scan as
``ops.modern.bench`` (same variant flags), warms it, traces a few steady
calls with ``jax.profiler.trace`` and joins every executed XLA op (``hlo_op``
stat in the trace) to the innermost user source location recorded in the
compiled HLO (``FileLocations``/``StackFrames`` tables). Fusions take the most
common location among their fused instructions. Container ops (while, call,
conditional) are skipped so time is not counted twice.

    python -m ops.modern.profile_tick --envs 512 --ticks 100 [--top 40] [--trace-dir DIR]

Prints JSON lines: totals, then per-file and per-(file, line) shares of op time.
On CPU, parallel thunks overlap, so shares are of summed op time, not wall time.
On GPU, CUDA command buffers are disabled while profiling (they replace kernel names
with ``command_buffer_N``), so absolute times run a little slower than the bench.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import tempfile
import time

import jax

from ops.modern.bench import add_world_args, build_world, init_batch, warm_up, world_label

CONTAINERS = {"while", "call", "conditional", "async-start", "async-done", "async-update"}
_COMP = re.compile(r"^%?([\w.\-]+) .*\{$")


def source_map(hlo: str) -> tuple[dict, dict]:
    """``(location_of, opcode_of)`` for every instruction name in the HLO text."""
    files, locs, frames = {}, {}, {}
    section = None
    for line in hlo.splitlines():
        head = line.strip()
        if head in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            section = head
            continue
        if section and re.match(r"^\d+ ", head):
            k, rest = head.split(" ", 1)
            if section == "FileNames":
                files[int(k)] = rest.strip('"')
            elif section == "FileLocations":
                m = re.search(r"file_name_id=(\d+).*?line=(\d+)", rest)
                locs[int(k)] = (int(m.group(1)), int(m.group(2)))
            elif section == "StackFrames":
                frames[int(k)] = int(re.search(r"file_location_id=(\d+)", rest).group(1))
            continue
        section = None
    where = {f: (files.get(locs[l][0], "?"), locs[l][1]) for f, l in frames.items() if l in locs}

    loc_of, op_of, calls_of, members = {}, {}, {}, collections.defaultdict(list)
    comp = None
    for line in hlo.splitlines():
        if line.endswith("{") and not line.startswith(" "):
            m = _COMP.match(line)
            comp = m.group(1) if m else None
            continue
        if " = " not in line:
            continue
        name = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+) = ", line)
        if not name:
            continue
        name = name.group(1)
        op = re.search(r"\} ([\w\-]+)\(|\] ([\w\-]+)\(|\) ([\w\-]+)\(| ([\w\-]+)\(", line)
        op_of[name] = next((g for g in op.groups() if g), "?") if op else "?"
        sf = re.search(r"stack_frame_id=(\d+)", line)
        if sf and int(sf.group(1)) in where:
            loc_of[name] = where[int(sf.group(1))]
            if comp:
                members[comp].append(loc_of[name])
        cl = re.search(r"calls=%?([\w.\-]+)", line)
        if cl:
            calls_of[name] = cl.group(1)
    for name, comp_name in calls_of.items():
        if name not in loc_of and members.get(comp_name):
            loc_of[name] = collections.Counter(members[comp_name]).most_common(1)[0][0]
    return loc_of, op_of


def short(path: str) -> str:
    i = path.find("lanerl_jax/")
    return path[i:] if i >= 0 else path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--warm-ticks", type=int, default=1200)
    ap.add_argument("--calls", type=int, default=2, help="traced steady calls")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--keep-command-buffers", action="store_true",
                    help="GPU: keep CUDA command buffers (they hide per-kernel names from the trace)")
    add_world_args(ap)
    args = ap.parse_args()
    if not args.keep_command_buffers:      # read at backend start, so before any jax computation
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer=").strip()
    from lanerl_jax.jax_cache import enable_compile_cache
    enable_compile_cache()
    from jax.profiler import ProfileData

    cfg, run = build_world(args)
    timed = jax.jit(jax.vmap(lambda s: run(s, args.ticks)))
    compiled = timed.lower(init_batch(cfg, args.envs)).compile()
    out, _ = warm_up(compiled, init_batch(cfg, args.envs), args.warm_ticks, args.ticks)
    loc_of, op_of = source_map(compiled.as_text())

    trace_dir = args.trace_dir or tempfile.mkdtemp(prefix="modern_profile_")
    t0 = time.time()
    with jax.profiler.trace(trace_dir):
        for _ in range(args.calls):
            out, _ = compiled(out)
        jax.block_until_ready(out.t)
    wall = time.time() - t0
    by_file, by_line = collections.Counter(), collections.Counter()
    total = unmatched = 0.0
    for path in glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True):
        for plane in ProfileData.from_file(path).planes:
            for line in plane.lines:
                for ev in line.events:
                    op = dict(ev.stats).get("hlo_op")
                    if not op or op_of.get(op, "?") in CONTAINERS or op.split(".")[0] in CONTAINERS:
                        continue
                    d = ev.duration_ns
                    total += d
                    if op in loc_of:
                        f, ln = loc_of[op]
                        by_file[short(f)] += d
                        by_line[(short(f), ln)] += d
                    else:
                        unmatched += d
    steps = args.calls * args.ticks
    print(json.dumps({"backend": jax.default_backend(), **world_label(args), "envs": args.envs,
                      "traced_ticks": steps, "wall_s": round(wall, 3), "wall_ms_per_tick": 1e3 * wall / steps,
                      "op_ms_per_tick": total / 1e6 / steps, "unattributed_share": unmatched / max(total, 1.0),
                      "trace_dir": trace_dir}), flush=True)
    for f, d in by_file.most_common():
        print(json.dumps({"file": f, "share": round(d / total, 4), "ms_per_tick": round(d / 1e6 / steps, 4)}))
    for (f, ln), d in by_line.most_common(args.top):
        print(json.dumps({"line": f"{f}:{ln}", "share": round(d / total, 4),
                          "ms_per_tick": round(d / 1e6 / steps, 4)}))


if __name__ == "__main__":
    main()
