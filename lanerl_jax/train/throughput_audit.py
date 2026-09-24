"""Throughput audit: where the training loop's time goes (docs/THROUGHPUT_AUDIT.md).

Not a gate and not a training entrypoint. It times the REAL trainer programs
(`make_train`'s `rollout` and `run_chunk`) on a realistic, heterogeneous state
pool, separately, compile excluded, `block_until_ready` around every timing
(`docs/EXPERIMENT_METHOD.md` section 6).

Units, fixed here so no number is quoted in two of them:
  env-step        one env advanced one decision (both champions act; 2 sim ticks)
  champion-dec    2 per env-step
  game-min/hour   env-steps/s * (1/30 s) * 3600 / 60  (lane-game minutes per wall hour)

Modes (one process per mode/config so `peak_bytes_in_use` is per config):
  pool    warm a 256-env pool with the trainer's own rollout under a trained
          checkpoint, save it, and log per-rollout time against game clock
  split   rollout vs full update (learner = difference) for one config, VRAM peak,
          optional XLA profile of each program
  phases  per-phase sim-step cost (obs / policy / decode / apply_orders routed vs
          raw / env_advance and its controls) on the pool or the gate warm state
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HZ = 30.0


def units(env_steps_per_s: float) -> dict:
    return {"env_steps_per_s": round(env_steps_per_s, 1),
            "champion_decisions_per_s": round(2 * env_steps_per_s, 1),
            "game_min_per_hour": round(env_steps_per_s / HZ * 60, 0)}


def mem_gib() -> dict:
    s = jax.devices()[0].memory_stats() or {}
    return {k: round(s.get(k, 0) / 2**30, 3)
            for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")}


def timed(fn, reps: int):
    """fn() -> output; returns list of seconds, block_until_ready per rep."""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t)
    return ts


def tile(tree, n):
    def f(a):
        a = jnp.asarray(a)
        m = a.shape[0]
        k = -(-n // m)
        return jnp.tile(a, (k,) + (1,) * (a.ndim - 1))[:n]
    return jax.tree.map(f, tree)


def load_ckpt(runner, path):
    from flax.serialization import from_bytes
    if not path:
        return runner, "random-init"
    try:
        p = from_bytes({"params": runner.params, "opt_state": runner.opt_state,
                        "step": runner.step}, Path(path).read_bytes())
        return runner._replace(params=p["params"], opt_state=p["opt_state"]), str(path)
    except Exception as exc:  # structure moved: say so, do not guess
        print(f"ckpt load failed ({type(exc).__name__}: {str(exc)[:200]}); random init")
        return runner, f"random-init (ckpt failed: {type(exc).__name__})"


_SIM = {}


def build(n_envs, rollout, minibatches, epochs, route=True):
    from ..sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
    from .trainer import TrainConfig, make_train
    from .ppo import PPOConfig
    if route not in _SIM:      # load the 231 MiB artifact once per process
        _SIM[route] = SimConfig.training(
            route_artifact=DEFAULT_ROUTE_ARTIFACT if route else None)
    sim = _SIM[route]
    cfg = TrainConfig(n_envs=n_envs, rollout_steps=rollout, n_updates=1,
                      n_minibatches=minibatches,
                      ppo=PPOConfig()._replace(epochs=epochs))
    return cfg, sim, make_train(cfg, sim_config=sim)


def _is_key(x):
    return jnp.issubdtype(jnp.asarray(x).dtype, jax.dtypes.prng_key)


def save_pool(path, tree):
    leaves = jax.tree.leaves(tree)
    arrs = {f"a{i}": np.asarray(jax.random.key_data(x) if _is_key(x) else x)
            for i, x in enumerate(leaves)}
    with open(path, "wb") as f:
        np.savez(f, **arrs)


def load_pool(path, template):
    leaves, treedef = jax.tree.flatten(template)
    z = np.load(path)
    out = []
    for i, t in enumerate(leaves):
        a = jnp.asarray(z[f"a{i}"])
        out.append(jax.random.wrap_key_data(a, impl=jax.random.key_impl(t))
                   if _is_key(t) else a)
    return jax.tree.unflatten(treedef, out)


def pool_template(train):
    r = train.initial_runner(jax.random.key(0))
    return {"env_state": r.env_state, "reward_state": r.reward_state,
            "deadline_ms": r.deadline_ms}


# --------------------------------------------------------------------------
def cmd_pool(a):
    from flax.serialization import to_bytes
    cfg, sim, train = build(a.envs, a.rollout, 4, 4)
    runner = train.initial_runner(jax.random.key(a.seed))
    runner, src = load_ckpt(runner, a.ckpt)
    roll = jax.jit(train.rollout)
    t = time.perf_counter()
    runner, tr = roll(runner)
    jax.block_until_ready(tr)
    comp = time.perf_counter() - t
    rows = []
    for i in range(a.warm_rollouts):
        t = time.perf_counter()
        runner, tr = roll(runner)
        jax.block_until_ready(tr)
        dt = time.perf_counter() - t
        tms = np.asarray(runner.env_state.t_ms)
        alive = np.asarray(runner.env_state.alive).sum(-1)
        rows.append({"i": i, "rollout_s": round(dt, 4),
                     "t_ms_mean": float(tms.mean()), "alive_mean": float(alive.mean())})
    del tr
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"env_state": runner.env_state, "reward_state": runner.reward_state,
               "deadline_ms": runner.deadline_ms}
    save_pool(out, payload)
    tms = np.asarray(runner.env_state.t_ms) / 1000
    alive = np.asarray(runner.env_state.alive).sum(-1)
    res = {"mode": "pool", "policy": src, "compile_plus_first_s": round(comp, 1),
           "rollouts": rows,
           "pool_t_s_quantiles": np.quantile(tms, [0, .1, .25, .5, .75, .9, 1]).round(1).tolist(),
           "pool_alive_quantiles": np.quantile(alive, [0, .1, .5, .9, 1]).tolist(),
           "mem": mem_gib()}
    print(json.dumps(res))
    return res


# --------------------------------------------------------------------------
def cmd_split(a):
    from flax.serialization import from_bytes
    if a.precision:
        jax.config.update("jax_default_matmul_precision", a.precision)
    cfg, sim, train = build(a.envs, a.rollout, a.minibatches, a.epochs)
    runner = train.initial_runner(jax.random.key(a.seed))
    runner, src = load_ckpt(runner, a.ckpt)
    if a.pool:
        _, _, tp = build(a.pool_envs, 128, 4, 4)
        pool = load_pool(a.pool, pool_template(tp))
        pool = tile(pool, a.envs)
        runner = runner._replace(env_state=pool["env_state"],
                                 reward_state=pool["reward_state"],
                                 deadline_ms=pool["deadline_ms"])
    jax.block_until_ready(runner)
    n_env_steps = a.envs * a.rollout
    res = {"mode": "split", "envs": a.envs, "rollout": a.rollout,
           "minibatches": a.minibatches, "epochs": a.epochs,
           "samples_per_update": 2 * n_env_steps,
           "minibatch_size": 2 * n_env_steps // a.minibatches,
           "precision": a.precision or "default", "policy": src,
           "pool": a.pool, "prealloc": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
           "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")}

    roll = jax.jit(train.rollout)
    state = {"r": runner}

    def do_roll():
        r, tr = roll(state["r"])
        state["r"] = r
        return tr
    t = time.perf_counter()
    jax.block_until_ready(do_roll())
    res["rollout_compile_plus_first_s"] = round(time.perf_counter() - t, 2)
    rt = timed(do_roll, a.reps)
    res["rollout_s"] = [round(x, 4) for x in rt]
    res["mem_after_rollout"] = mem_gib()

    step = jax.jit(train.run_chunk, static_argnums=1)

    def do_upd(n=1):
        r, m = step(state["r"], n)
        state["r"] = r
        return m
    t = time.perf_counter()
    try:
        jax.block_until_ready(do_upd())
    except Exception as exc:
        res["update_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        res["mem_after_update"] = mem_gib()
        print(json.dumps(res))
        return res
    res["update_compile_plus_first_s"] = round(time.perf_counter() - t, 2)
    ut = timed(do_upd, a.reps)
    res["update_s"] = [round(x, 4) for x in ut]
    res["mem_after_update"] = mem_gib()
    r_med, u_med = statistics.median(rt), statistics.median(ut)
    res.update(rollout_med_s=round(r_med, 4), update_med_s=round(u_med, 4),
               learner_med_s=round(u_med - r_med, 4),
               learner_frac=round((u_med - r_med) / u_med, 3),
               rollout_ms_per_step=round(1000 * r_med / a.rollout, 3),
               acting_only=units(n_env_steps / r_med),
               trained=units(n_env_steps / u_med))
    if a.chunk_check:
        t = time.perf_counter()
        jax.block_until_ready(do_upd(a.chunk_check))
        res["chunk_compile_plus_first_s"] = round(time.perf_counter() - t, 2)
        ct = timed(lambda: do_upd(a.chunk_check), 2)
        res["chunk_per_update_s"] = [round(x / a.chunk_check, 4) for x in ct]
    if a.profile:
        pdir = Path(a.profile)
        for name, fn in (("rollout", do_roll), ("update", do_upd)):
            d = pdir / name
            jax.profiler.start_trace(str(d))
            t = time.perf_counter()
            jax.block_until_ready(fn())
            wall = time.perf_counter() - t
            jax.profiler.stop_trace()
            res[f"profile_{name}"] = summarise_profile(d, wall)
    print(json.dumps(res))
    return res


def summarise_profile(d: Path, wall_s: float, top: int = 25) -> dict:
    """Top GPU ops by total time, kernel count, and GPU busy fraction over wall."""
    from jax.profiler import ProfileData
    files = sorted(Path(d).rglob("*.xplane.pb"))
    if not files:
        return {"error": "no xplane"}
    pd = ProfileData.from_file(str(files[-1]))
    out = {"file": str(files[-1]), "wall_s": round(wall_s, 4)}
    for plane in pd.planes:
        if "GPU" not in plane.name:
            continue
        lines = {ln.name: ln for ln in plane.lines}
        out["lines"] = list(lines)
        # kernel-level busy: stream lines hold the kernels
        ivs, by_kernel, by_op, n_k = [], {}, {}, 0
        for ln in plane.lines:
            nm = ln.name
            if nm.startswith("Stream") or "stream" in nm.lower():
                for ev in ln.events:
                    ivs.append((ev.start_ns, ev.start_ns + ev.duration_ns))
                    by_kernel[ev.name] = by_kernel.get(ev.name, 0) + ev.duration_ns
                    n_k += 1
            if nm == "XLA Ops":
                for ev in ln.events:
                    key = ev.name
                    by_op[key] = by_op.get(key, 0) + ev.duration_ns
        ivs.sort()
        busy, cur_s, cur_e = 0, None, None
        for s, e in ivs:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
        span = (ivs[-1][1] - ivs[0][0]) if ivs else 0
        tot_k = sum(by_kernel.values()) or 1
        out.update(
            n_kernels=n_k, kernel_busy_s=round(busy / 1e9, 4),
            kernel_span_s=round(span / 1e9, 4),
            busy_frac_of_span=round(busy / span, 3) if span else None,
            busy_frac_of_wall=round(busy / 1e9 / wall_s, 3),
            mean_kernel_us=round(tot_k / 1e3 / max(n_k, 1), 2),
            top_kernels=[(k[:90], round(v / 1e6, 2), round(v / tot_k, 3))
                         for k, v in sorted(by_kernel.items(), key=lambda x: -x[1])[:top]],
            top_xla_ops=[(k[:90], round(v / 1e6, 2))
                         for k, v in sorted(by_op.items(), key=lambda x: -x[1])[:top]])
        break
    return out


# --------------------------------------------------------------------------
def cmd_phases(a):
    """Separate jits per phase, fixed inputs, median of reps x calls."""
    from flax.serialization import from_bytes
    from ..obs.builder import build_observation
    from ..obs.frame import make_lane_frame
    from ..sim.init import TOP_OUTER_TURRET
    from ..sim.state import Team
    from ..sim.step import env_advance, env_apply
    from .policy import LanePolicy, apply_flattened_batch
    from .trainer import BLUE_NEXUS, RED_NEXUS, _sample, orders_from

    cfg, sim, train = build(a.pool_envs, 128, 4, 4)
    runner = train.initial_runner(jax.random.key(a.seed))
    runner, src = load_ckpt(runner, a.ckpt)
    params = runner.params
    if a.gate_state:
        from .benchmark import warm_state
        base = warm_state(150.0)
        states = jax.tree.map(lambda x: jnp.broadcast_to(x, (a.envs,) + x.shape), base)
        state_src = "gate warm_state(150 s) broadcast"
    else:
        pool = load_pool(a.pool, pool_template(train))
        states = tile(pool["env_state"], a.envs)
        state_src = f"pool {a.pool} tiled"
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE], TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    red = make_lane_frame(TOP_OUTER_TURRET[Team.RED], TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS)
    policy = LanePolicy(cfg.policy)
    P = sim.params

    def _obs(s):
        b = build_observation(s, 0, frame, params=P, horizon_s=cfg.episode_s)
        r = build_observation(s, 1, red, params=P, horizon_s=cfg.episode_s)
        return jax.tree.map(lambda x, y: jnp.stack([x, y]), b, r)

    obs_fn = jax.jit(jax.vmap(_obs))
    pol_fn = jax.jit(lambda p, o: apply_flattened_batch(
        policy, p, o.entities, o.entity_pad_mask, o.self_vec, o.global_vec))

    def _dec(lg, s, su, k):
        act, _, _ = _sample(lg, k, su >= 0)
        return orders_from(act, s, su, frame)
    dec_fn = jax.jit(jax.vmap(_dec))
    raw = sim.replace(route_table=None, terrain=None)
    cfgs = {
        "apply_routed": sim,
        "apply_raw_control": raw,
    }
    adv = {
        "advance_training": sim,
        "advance_no_terrain_repair": sim.replace(defer_collision_terrain=False),
        "advance_no_collision": sim.replace(enable_collision=False,
                                            defer_collision_terrain=False),
    }
    keys = jax.random.split(jax.random.key(1), a.envs)
    obs = jax.block_until_ready(obs_fn(states))
    lg = jax.block_until_ready(pol_fn(params, obs))
    orders = jax.block_until_ready(dec_fn(lg, states, obs.slot_unit, keys))
    res = {"mode": "phases", "envs": a.envs, "state": state_src, "policy": src,
           "live_entities_mean": float(np.asarray(states.alive).sum(-1).mean()),
           "t_s_median": float(np.median(np.asarray(states.t_ms)) / 1000),
           "ms": {}}

    def bench(name, f):
        jax.block_until_ready(f())
        per = []
        for _ in range(a.reps):
            t = time.perf_counter()
            for _ in range(a.calls):
                o = f()
            jax.block_until_ready(o)
            per.append((time.perf_counter() - t) / a.calls * 1000)
        res["ms"][name] = round(statistics.median(per), 3)
        print(name, res["ms"][name], flush=True)

    bench("observation", lambda: obs_fn(states))
    bench("policy_forward", lambda: pol_fn(params, obs))
    bench("sample_decode", lambda: dec_fn(lg, states, obs.slot_unit, keys))
    ordered = None
    for nm, c in cfgs.items():
        f = jax.jit(jax.vmap(lambda s, o, c=c: env_apply(s, o, c)))
        bench(nm, lambda f=f: f(states, orders))
        if nm == "apply_routed":
            ordered = jax.block_until_ready(f(states, orders))
    for nm, c in adv.items():
        f = jax.jit(jax.vmap(lambda s, c=c: env_advance(s, c)))
        bench(nm, lambda f=f: f(ordered))
    ms = res["ms"]
    ms["routing_delta"] = round(ms["apply_routed"] - ms["apply_raw_control"], 3)
    ms["terrain_repair_delta"] = round(ms["advance_training"] - ms["advance_no_terrain_repair"], 3)
    ms["sum_parts"] = round(ms["observation"] + ms["policy_forward"] + ms["sample_decode"]
                            + ms["apply_routed"] + ms["advance_training"], 3)
    print(json.dumps(res))
    return res


# --------------------------------------------------------------------------
def _digest(tree) -> str:
    import hashlib
    h = hashlib.sha256()
    for x in jax.tree.leaves(tree):
        x = jax.random.key_data(x) if _is_key(x) else x
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()[:16]


def cmd_determ(a):
    """Two from-scratch runs of `--updates` updates, same seed, same process,
    exactly as run_train starts (initial_runner(seed), no ckpt, no pool).
    Hashes of metrics and params per run; a second PROCESS gives the
    cross-process comparison (autotuning can pick different kernels)."""
    cfg, sim, train = build(a.envs, a.rollout, a.minibatches, a.epochs)
    step = jax.jit(train.run_chunk, static_argnums=1)
    res = {"mode": "determ", "xla_flags": os.environ.get("XLA_FLAGS", ""),
           "envs": a.envs, "updates": a.updates, "seed": a.seed, "runs": []}
    ref = None
    for i in range(2):
        runner = train.initial_runner(jax.random.key(a.seed))
        jax.block_until_ready(runner)
        t = time.perf_counter()
        runner, m = step(runner, a.updates)
        jax.block_until_ready(m)
        wall = time.perf_counter() - t
        mh = {k: np.asarray(v) for k, v in m.items()}
        row = {"wall_s": round(wall, 2), "metrics_sha": _digest(mh),
               "params_sha": _digest(runner.params),
               "reward_last": float(mh["reward"][-1]),
               "entropy_last": float(mh["entropy"][-1]),
               "lane_dist": [round(float(x), 3) for x in mh["lane_dist"]]}
        if ref is not None:
            diff = [k for k in mh if not np.array_equal(mh[k], ref[k], equal_nan=True)]
            first = None
            for u in range(a.updates):
                if not np.array_equal(mh["reward"][u], ref["reward"][u]):
                    first = u
                    break
            row.update(metrics_differ=sorted(diff), first_reward_diff_update=first,
                       max_abs_reward_diff=float(np.nanmax(np.abs(mh["reward"] - ref["reward"]))))
        ref = mh
        res["runs"].append(row)
        print(json.dumps(row), flush=True)
    # run 0 includes compile; run 1 is steady
    res["steady_s_per_update"] = round(res["runs"][1]["wall_s"] / a.updates, 4)
    res["trained"] = units(a.envs * a.rollout * a.updates / res["runs"][1]["wall_s"])
    print(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "split", "phases", "determ"])
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--pool-envs", type=int, default=256,
                    help="env count the pool file was written with")
    ap.add_argument("--out", default="lanerl_jax/runs/perf/pool256.msgpack")
    ap.add_argument("--warm-rollouts", type=int, default=70)
    ap.add_argument("--precision", default=None,
                    help="jax_default_matmul_precision (report-only knob)")
    ap.add_argument("--chunk-check", type=int, default=0)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--gate-state", action="store_true")
    ap.add_argument("--jsonl", default=None, help="append the result row here")
    a = ap.parse_args()
    print(f"device {jax.devices()[0]} jax {jax.__version__}", flush=True)
    res = {"pool": cmd_pool, "split": cmd_split, "phases": cmd_phases,
           "determ": cmd_determ}[a.mode](a)
    if a.jsonl:
        res["argv"] = vars(a)
        with open(a.jsonl, "a") as f:
            f.write(json.dumps(res, default=str) + "\n")


if __name__ == "__main__":
    main()
