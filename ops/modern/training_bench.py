"""Pinned modern vs 4.20 lane PPO timings (MODERN-043).

Prepare on the login node, then submit slurm/modern_training_bench.sbatch::

  python -m ops.modern.training_bench prepare --out /mnt/nfs/shared/champ-expansion/NAME \
    --modern-repo . --modern-ref 78abf40f --legacy-repo /mnt/nfs/projects/ahriuwu-lanerl-jax \
    --legacy-ref bb2e5b46
  BENCH=/mnt/nfs/shared/champ-expansion/NAME sbatch slurm/modern_training_bench.sbatch

Only committed trees are exported. Legacy untracked route data and required vendor JSON/ngrid
are copied and hashed; neither checkout is changed. Results/logs are artifacts, findings belong in
JAX_FIDELITY_LEDGER.md. Each cell/phase uses a fresh process and no persistent compile cache.
Rollout includes observations, policy sampling, actions, simulation, rewards, resets and trajectory
materialization. PPO additionally includes bootstrap, GAE, epochs/minibatches and trainer metrics.
These are different games, never a fidelity-equivalent speedup or pure-sim training extrapolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

MODERN_REF = "78abf40ff0142764a8a5ebcd2ed50ec126f5b4c7"
LEGACY_REF = "bb2e5b466be278e7510868d1e081fb77ae33a200"


def rates(envs, decisions, seconds, tick_hz, step_ticks, acting_agents, learning_agents):
    """One env decision advances one game; agent count must never multiply game time."""
    values = (envs, decisions, seconds, tick_hz, step_ticks, acting_agents, learning_agents)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("counts, rates and duration must be positive and finite")
    env_rate = envs * decisions / seconds
    return dict(env_decisions_per_s=env_rate, agent_decisions_per_s=env_rate * acting_agents,
                learner_agent_decisions_per_s=env_rate * learning_agents,
                simulated_game_seconds_per_wall_s=env_rate * step_ticks / tick_hz)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def export_tree(repo, ref, dest):
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"{ref}^{{commit}}"], text=True).strip()
    dest.mkdir()
    # Stream archives; never include unrelated dirty work or run git across other directories.
    proc = subprocess.Popen(["git", "-C", str(repo), "archive", pin], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
        archive.extractall(dest, filter="data")
    if proc.wait():
        raise RuntimeError("git archive failed")
    return pin


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    pins = {"modern": export_tree(a.modern_repo, a.modern_ref, a.out / "modern"),
            "legacy": export_tree(a.legacy_repo, a.legacy_ref, a.out / "legacy")}
    routes = a.legacy_repo / "data/jax_routes/map1_garen_r35_o50_v2"
    shutil.copytree(routes, a.out / "legacy/data/jax_routes/map1_garen_r35_o50_v2")
    # Copy the small content inputs, not the server binaries or full content tree.
    content = a.vendor / "LoLServer/Content/LeagueSandbox-Default"
    target = a.out / "vendor/LoLServer/Content/LeagueSandbox-Default"
    for pattern in ("Stats/*/*.json", "Spells/*/*.json", "Maps/Map1/*.json", "AIMesh/Map1/*.aimesh_ngrid"):
        for src in sorted(content.glob(pattern)):
            dst = target / src.relative_to(content)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    shutil.copy2(__file__, a.out / "training_bench.py")
    manifest = dict(pins=pins, files=file_manifest(a.out), modern_pin_reason="parent before benchmark/champion work",
                    legacy_pin_reason="requested bb2e5b46; excludes later E165 docs and dirty BC work",
                    assets={"routes": str(routes), "vendor": str(content)})
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(out=str(a.out), pins=pins)), flush=True)


def setup(a):
    import jax
    import jax.numpy as jnp
    if a.engine == "modern":
        from lanerl_jax.modern import train as T
        from lanerl_jax.modern import obs as O
        cfg = T.ModernVecConfig(n_envs=a.envs, rollout_steps=a.rollout)
        if a.preset == "matched":
            cfg = cfg._replace(policy=cfg.policy._replace(core="gru", core_norm=True,
                                core_residual=True, detach_critic=False), lr_anneal=False)
        world = T.MS.build_config(T.default_loadouts(), fog="rays",
                                 lanes=(2,) if a.layout == "top" else (0, 1, 2),
                                 jungle=a.layout == "full", objectives=a.layout == "full")
        env = T.make_env(world)
        bank = T.prepare_modern_bank(cfg, env, a.seed)
        built = T.make_modern_vec_train(cfg, env, bank)
        update = lambda r: built["run_chunk"](r, 1)
        desc = dict(ruleset="26.19", champions=["Garen", "Jax"], fog="rays", layout=a.layout,
                    units=world.layout.n_units if hasattr(world.layout, "n_units") else len(world.unit_kind),
                    packet_capacity=world.layout.packet_capacity, loadouts=[repr(x) for x in world.loadouts],
                    reset="16 scripted 60s hold-point starts; 600s episode", tick_hz=30,
                    differences="modern items/runes/shop/wards/role quest; full adds all lanes/jungle/objectives",
                    observation=dict(profile=O.PROFILE, entities=[O.N_SLOTS, O.MODERN_ENTITY_DIM],
                                     self_width=O.MODERN_WORLD_SELF_DIM, global_width=O.GLOBAL_DIM,
                                     projectile_rows=False),
                    action="19 buttons, 7 masked choice/level buttons")
        learn_agents = cfg.learn_agents
    else:
        from lanerl_jax.train import vec_train as T
        from lanerl_jax.sim.config import SimConfig
        from lanerl_jax.train.wave_scenario import prepare_scenario_bank, park_afk_opponent
        cfg = T.VecConfig(n_envs=a.envs, rollout_steps=a.rollout)
        if a.preset == "matched":
            # Equal trunk/core widths, GRU, critic gradients, self play, PPO scalars and decision Hz.
            cfg = cfg._replace(opponent="self", lr_anneal=False,
                policy=cfg.policy._replace(value_grad_trunk=True),
                ppo=cfg.ppo._replace(lr=2.5e-4, discount=.99, entropy_coef=.01 / 3))
        sim = SimConfig.training(route_artifact=a.root / "legacy/data/jax_routes/map1_garen_r35_o50_v2")
        sim = sim.replace(step_ticks=cfg.step_ticks)
        bank = prepare_scenario_bank(sim, a.work / "bank", [-300., -100., 100., 300.], a.seed)
        if cfg.opponent == "afk":
            bank = park_afk_opponent(bank)
        built = T.make_vec_train(cfg, sim, bank)
        update = built["update"]
        desc = dict(ruleset="4.20 server parity, not modern", champions=["Garen", "Garen"],
                    fog="map-grid-supercover-v1", layout="historical top lane", units=bank.x.shape[-1],
                    reset="level3 120s tower starts, 12 preplaced minions, asymmetric .7/1 HP; 240s end",
                    tick_hz=60, simulator=sim.describe(),
                    differences="historical runes/masteries, no items/jungle/objectives/modern systems; projectile encoder",
                    observation="56x16 (32 unit + 24 projectile), self16, global6", action="8 buttons; natural AFK samples both rows, then forces red noop")
        learn_agents = 2 if cfg.opponent == "self" else 1
    runner = built["initial_runner"](jax.random.key(a.seed))
    jax.block_until_ready(runner)
    desc.update(policy=cfg.policy._asdict(), ppo=cfg.ppo._asdict(),
                vec={k: v for k, v in cfg._asdict().items() if k not in ("policy", "ppo")},
                parameter_count=sum(x.size for x in jax.tree.leaves(runner.params)),
                acting_agents=2, learning_agents=learn_agents, decision_hz=cfg.decision_hz,
                step_ticks=cfg.step_ticks, seed=a.seed,
                matmul_precision=str(jax.config.jax_default_matmul_precision),
                matched_limit="input/action widths and projectile encoder differ; rewards/reset distributions differ")
    if a.phase == "rollout":
        # Return the real rollout so XLA cannot prune trajectory construction.
        fn = built["collect"]
    else:
        fn = update
    return runner, fn, desc


def worker(a):
    sys.path.insert(0, str(a.root / a.engine))
    os.environ["LANERL_VENDOR_ROOT"] = str(a.root / "vendor")
    import jax
    import flax
    import optax
    import numpy as np
    jax.config.update("jax_enable_compilation_cache", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.default_backend() != a.backend:
        raise RuntimeError(f"expected {a.backend}, got {jax.default_backend()}")
    start = time.perf_counter()
    runner, fn, desc = setup(a)
    init_s = time.perf_counter() - start
    emit = lambda row: print(json.dumps(row, allow_nan=False), flush=True)
    metadata = dict(engine=a.engine, preset=a.preset, envs=a.envs, phase=a.phase,
                    pin=json.loads((a.root / "manifest.json").read_text())["pins"][a.engine],
                    scenario=desc, backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
                    versions=dict(jax=jax.__version__, flax=flax.__version__, optax=optax.__version__, numpy=np.__version__),
                    init_and_bank_compile_s=init_s, compile_cache=False)
    emit(dict(event="initialized", **metadata))
    if a.validate_only:
        shapes = jax.eval_shape(fn, runner)
        emit(dict(event="validated", **metadata, output_leaves=len(jax.tree.leaves(shapes))))
        return
    t = time.perf_counter()
    executable = jax.jit(fn).lower(runner).compile()
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    out = jax.block_until_ready(executable(runner))
    first_s = time.perf_counter() - t
    runner = out[0]
    del out
    samples = []
    for i in range(a.samples):
        t = time.perf_counter()
        out = jax.block_until_ready(executable(runner))
        dt = time.perf_counter() - t
        runner = out[0]
        if a.phase == "ppo":
            metrics = out[1]
            for key in ("policy_loss", "value_loss", "loss_nonfinite", "sim_overflow_max"):
                if key in metrics:
                    value = np.asarray(metrics[key])
                    if not np.isfinite(value).all() or (key in ("loss_nonfinite", "sim_overflow_max") and (value > 0).any()):
                        raise RuntimeError(f"invalid {key}: {value}")
        elif hasattr(out[1], "overflow") and np.asarray(out[1].overflow).max() > 0:
            raise RuntimeError("simulation overflow")
        del out
        samples.append(dt)
        emit(dict(event="sample", index=i, seconds=dt,
                  **rates(a.envs, a.rollout, dt, desc["tick_hz"], desc["step_ticks"], 2, desc["learning_agents"])))
    mem = jax.local_devices()[0].memory_stats() or {}
    median = statistics.median(samples)
    emit(dict(event="result", **metadata, rollout_steps=a.rollout, samples_s=samples,
              median_s=median, min_s=min(samples), max_s=max(samples),
              update_s=median if a.phase == "ppo" else None,
              rollout_s=median if a.phase == "rollout" else None,
              compile_s=compile_s, first_execution_s=first_s,
              peak_device_bytes=mem.get("peak_bytes_in_use"), memory_stats=mem,
              **rates(a.envs, a.rollout, median, desc["tick_hz"], desc["step_ticks"], 2, desc["learning_agents"])))


def matrix(a):
    manifest = json.loads((a.root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(a.root / name) != digest:
            raise RuntimeError(f"snapshot changed: {name}")
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.root / "manifest.json", a.output / "manifest.json")
    failures = []
    for preset in a.presets:
        for n in a.env_counts:
            for layout in a.layouts:
                engines = ("legacy", "modern") if layout == "top" else ("modern",)
                for engine in engines:
                    for phase in ("rollout", "ppo"):
                        name = f"{preset}-{n}-{layout}-{engine}-{phase}"
                        work = a.output / name
                        work.mkdir()
                        cmd = [sys.executable, str(a.root / "training_bench.py"), "worker", "--root", str(a.root),
                               "--engine", engine, "--preset", preset, "--envs", str(n), "--layout", layout,
                               "--phase", phase, "--work", str(work), "--samples", str(a.samples),
                               "--rollout", str(a.rollout), "--seed", str(a.seed), "--backend", a.backend]
                        print(json.dumps(dict(event="start", case=name, command=cmd)), flush=True)
                        with (work / "stdout.jsonl").open("w") as out, (work / "stderr.log").open("w") as err:
                            proc = subprocess.run(cmd, cwd=a.root / engine, stdout=out, stderr=err)
                        rows = [json.loads(line) for line in (work / "stdout.jsonl").read_text().splitlines()
                                if line.startswith('{')]
                        result = next((row for row in rows if row.get("event") == "result"), None)
                        error = (work / "stderr.log").read_text()
                        if proc.returncode or result is None:
                            oom = "RESOURCE_EXHAUSTED" in error or "out of memory" in error.lower()
                            result = dict(event="failure", case=name, returncode=proc.returncode,
                                          status="oom" if oom else "error")
                            failures.append(result)
                        with (a.output / "results.jsonl").open("a") as out:
                            out.write(json.dumps({**result, "case": name}) + "\n")
                        print(json.dumps({**result, "case": name}), flush=True)
    if any(x["status"] != "oom" for x in failures):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--modern-repo", type=Path, required=True)
    prep.add_argument("--modern-ref", default=MODERN_REF)
    prep.add_argument("--legacy-repo", type=Path, required=True)
    prep.add_argument("--legacy-ref", default=LEGACY_REF)
    prep.add_argument("--vendor", type=Path, default=Path("/mnt/nfs/projects/lanerl-vendor"))
    prep.add_argument("--out", type=Path, required=True)
    for name in ("matrix", "worker"):
        s = sub.add_parser(name)
        s.add_argument("--root", type=Path, required=True)
        s.add_argument("--samples", type=int, default=5)
        s.add_argument("--rollout", type=int, default=128)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
        if name == "matrix":
            s.add_argument("--output", type=Path, required=True)
            s.add_argument("--env-counts", nargs="+", type=int, default=[256, 1024, 4096])
            s.add_argument("--presets", nargs="+", choices=("matched", "natural"), default=["matched", "natural"])
            s.add_argument("--layouts", nargs="+", choices=("top", "full"), default=["top", "full"])
        else:
            s.add_argument("--validate-only", action="store_true")
            s.add_argument("--engine", choices=("modern", "legacy"), required=True)
            s.add_argument("--preset", choices=("matched", "natural"), required=True)
            s.add_argument("--envs", type=int, required=True)
            s.add_argument("--layout", choices=("top", "full"), required=True)
            s.add_argument("--phase", choices=("rollout", "ppo"), required=True)
            s.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    if a.command != "prepare" and (a.samples < 1 or a.rollout < 1):
        p.error("samples and rollout must be positive")
    {"prepare": prepare, "matrix": matrix, "worker": worker}[a.command](a)


if __name__ == "__main__":
    main()
