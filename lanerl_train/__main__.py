"""Real ``lane-rl`` PPO training entrypoint.

Nobody had ever assembled ``lanerl_train.run.TrainingLoop`` against the real
server before today: ``TrainingLoop`` was unit-tested only against
``FakeInstance``. This module is the CLI that actually does it, using
``lanerl_train.lane_wiring`` for the (observation, action, reward, policy)
plumbing.

Usage::

    python -m lanerl_train --run-name my-run --num-actors 2 --envs-per-actor 2

Resuming an existing run::

    python -m lanerl_train --run-name my-run --resume

SIGTERM
-------
``gpup`` Slurm jobs on this cluster get SIGTERM 60s before pre-emption
(danilogin's ``~/CLAUDE.md``). On SIGTERM this process stops the training
loop, forces an out-of-band checkpoint (not waiting for
``checkpoint_every``), writes ``state.json``, and exits -- well under the
budget. Actor threads are daemons and the process exit takes them with it;
their in-flight rollouts are simply lost, which is cheaper than the
alternative of blocking exit on a live network read from a game server that
may itself be mid-preemption.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

import torch

from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import DualClipPPO, PPOConfig

from . import paths
from .anchor_eval import AnchorEvalConfig, AnchorEvaluator, make_anchor_driver_factory
from .eval import DEFAULT_RUN_ANCHORS, Evaluator, anchors_for_run
from .lane_wiring import LanePolicyActor, collect_rollout, make_collect_fn, make_lane_adapters
from .ports import PortAllocator
from .run import GpuProbe, RunConfig, StalledRun, TrainingLoop
from .vec import EpisodeSpec, SideAssignment, VecDriver, VecLaneEnv

log = logging.getLogger("lanerl_train.__main__")

SELF = "self"

# Generous headroom per actor's port block: PortAllocator already verifies
# freeness and raises rather than colliding, but starting each actor far
# apart makes a leaked/lingering instance from a previous run obvious in the
# logs instead of eating into the next actor's range.
PORTS_PER_ACTOR_STRIDE = 64


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real lane-rl 1v1 Garen PPO training run")
    p.add_argument("--run-name", required=True)
    p.add_argument("--num-actors", type=int, default=2)
    p.add_argument("--envs-per-actor", type=int, default=2, help="real server instances per actor thread")
    p.add_argument(
        "--rollout-steps", type=int, default=129,
        help="decisions collected per actor rollout; buffer rows = this - 1 (see collect_rollout)",
    )
    p.add_argument("--max-staleness", type=int, default=1)
    p.add_argument("--queue-capacity", type=int, default=2)
    p.add_argument("--total-updates", type=int, default=1_000_000)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--keep-last-checkpoints", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--horizon-s", type=float, default=30.0, help="PPOConfig's discount horizon in seconds")
    p.add_argument("--anneal-clock", choices=("env_steps", "updates"), default="env_steps")
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--port-base", type=int, default=21000,
        help="first actor's LANERL_PORT_BASE; later actors offset by "
        f"{PORTS_PER_ACTOR_STRIDE} x envs_per_actor from this",
    )
    # -- evaluation against a frozen opponent ----------------------------
    p.add_argument(
        "--anchors", default=",".join(DEFAULT_RUN_ANCHORS),
        help="comma-separated frozen opponents to evaluate against on --eval-every. "
        "A named anchor whose resource is missing FAILS AT STARTUP; pass an empty "
        "string only together with --eval-every 0.",
    )
    p.add_argument(
        "--bc-checkpoint", type=Path, default=None,
        help="checkpoint for the bc_policy anchor; required if it is in --anchors",
    )
    p.add_argument(
        "--anchor-envs", type=int, default=1,
        help="server instances per anchor. Each anchor keeps its own alive for the "
        "life of the run (a restart is ~12s against 0.23ms for an episode reset).",
    )
    p.add_argument(
        "--anchor-episodes", type=int, default=1,
        help="games per anchor per eval cycle; an anchor game is a real 10-minute game",
    )
    p.add_argument(
        "--anchor-all-per-eval", action="store_true",
        help="evaluate EVERY anchor each cycle instead of one round-robin. Three "
        "10-minute games per cycle is roughly 40%% of throughput at --eval-every 400.",
    )
    p.add_argument(
        "--anchor-port-base", type=int, default=31000,
        help="port base for the anchors' own servers; must not overlap --port-base",
    )
    return p


def _build_driver_for_actor(
    actor_idx: int,
    envs_per_actor: int,
    port_base: int,
    model_cfg: ModelConfig,
    run_dir: Path,
    device: str,
    train_step_source,
) -> Tuple[VecDriver, LanePolicyActor, dict]:
    actor_base = port_base + actor_idx * envs_per_actor * PORTS_PER_ACTOR_STRIDE
    allocator = PortAllocator(base=actor_base)
    ports = allocator.allocate(envs_per_actor)
    adapters = make_lane_adapters(train_step_source=train_step_source)
    policy = LanePolicy(model_cfg).to(device)
    actor = LanePolicyActor(policy, device=device)
    env = VecLaneEnv(n=envs_per_actor, ports=ports, log_dir=run_dir / f"actor{actor_idx}_logs")
    driver = VecDriver(
        env=env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=SELF) for _ in range(envs_per_actor)],
        episode=EpisodeSpec(),
    )
    log.info("actor %d: starting %d real server instance(s) on ports base=%d", actor_idx, envs_per_actor, actor_base)
    driver.start()
    return driver, actor, adapters.reward_contexts


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_argparser().parse_args(argv)

    torch.manual_seed(args.seed)

    run_dir = paths.runs_root() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig()
    ppo_cfg = PPOConfig(lr=args.lr, horizon_s=args.horizon_s)
    policy = LanePolicy(model_cfg).to(args.device)
    learner = DualClipPPO(policy, ppo_cfg)

    run_cfg = RunConfig(
        run_dir=run_dir,
        num_actors=args.num_actors,
        envs_per_actor=args.envs_per_actor,
        rollout_steps=args.rollout_steps,
        max_staleness=args.max_staleness,
        queue_capacity=args.queue_capacity,
        total_updates=args.total_updates,
        checkpoint_every=args.checkpoint_every,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        keep_last_checkpoints=args.keep_last_checkpoints,
        seed=args.seed,
        anneal_clock=args.anneal_clock,
    )

    # Resolved config + seed, written for reproducibility BEFORE anything can
    # fail -- a run that dies on its first update should still leave behind
    # exactly what was asked for.
    resolved = {
        "args": vars(args),
        "run_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(run_cfg).items()},
        "model_config": dataclasses.asdict(model_cfg),
        "ppo_config": dataclasses.asdict(ppo_cfg),
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True, default=str))
    log.info("resolved config written to %s", run_dir / "resolved_config.json")

    # -- the frozen-opponent ladder --------------------------------------
    # anchors_for_run RAISES on a missing resource. That is the whole point:
    # the previous behaviour was a warning ("anchor bc_policy has no resource
    # configured; it will be skipped") followed by 37 eval reports that said
    # (None, 0) against every anchor while the policy learned nothing.
    anchor_names = [n.strip() for n in args.anchors.split(",") if n.strip()]
    anchors = anchors_for_run(anchor_names, args.bc_checkpoint) if anchor_names else []
    anchor_evaluator = None
    if anchors:
        # Two servers on one control port means the second dies during
        # start-up, which from the ladder's point of view reads as "the anchor
        # is unbeatable" rather than as a port collision.
        actor_hi = args.port_base + args.num_actors * args.envs_per_actor * PORTS_PER_ACTOR_STRIDE
        anchor_hi = args.anchor_port_base + len(anchors) * args.anchor_envs * PORTS_PER_ACTOR_STRIDE
        if args.port_base < anchor_hi and args.anchor_port_base < actor_hi:
            raise SystemExit(
                f"--port-base range [{args.port_base}, {actor_hi}) overlaps the anchor "
                f"range [{args.anchor_port_base}, {anchor_hi}). Move one of them."
            )
        anchor_evaluator = AnchorEvaluator(
            anchors=anchors,
            driver_factory=make_anchor_driver_factory(
                build_policy_actor=lambda: LanePolicyActor(
                    LanePolicy(model_cfg).to(args.device), device=args.device
                ),
                policy_key=SELF,
                port_base=args.anchor_port_base,
                log_dir=run_dir,
                adapter_factory_for=lambda: make_lane_adapters(train_step_source=lambda: 0),
                envs=args.anchor_envs,
                seed=args.seed,
            ),
            agent_id_fn=lambda: loop.agent_id(),
            config=AnchorEvalConfig(
                episodes_per_anchor=args.anchor_episodes,
                rotate=not args.anchor_all_per_eval,
            ),
        )

    loop = TrainingLoop(
        run_cfg,
        learner,
        evaluator=Evaluator(anchors=anchors),
        anchor_eval=anchor_evaluator,
        gpu=GpuProbe(args.device),
    )
    # Refuse to start a run whose evaluation would measure nothing.
    loop.require_anchor_eval()

    if args.resume or (loop.state_path.exists()):
        loop.resume()

    # VecLaneEnv actually owns the subprocess/socket lifecycle. TrainingLoop
    # has no idea drivers exist -- it only knows about actor threads -- so
    # this process keeps its own handle on every driver it builds, purely to
    # close them explicitly on the way out. Without this, a clean exit still
    # happens today (LanerlControl.OnTick sees EOF on the closed socket and
    # exits the server itself), but that is the server's own safety net, not
    # something this script should depend on as ITS shutdown path.
    built_drivers: list = []

    def build_driver_for_actor(actor_idx: int):
        driver, actor, reward_contexts = _build_driver_for_actor(
            actor_idx, args.envs_per_actor, args.port_base, model_cfg, run_dir, args.device, loop.train_steps
        )
        built_drivers.append(driver)
        return driver, actor, reward_contexts

    loop.collect = make_collect_fn(build_driver_for_actor, SELF, args.rollout_steps, ppo_cfg.gamma, ppo_cfg.gae_lambda)

    # -- SIGTERM: gpup gives 60s notice before pre-emption. Force a
    # checkpoint and exit well inside that budget rather than trusting the
    # in-flight run() loop to reach its own checkpoint_every boundary. -----
    stop_requested = threading.Event()

    def _on_sigterm(signum, frame):
        log.error("SIGTERM received; forcing a checkpoint and exiting")
        stop_requested.set()
        loop.stop_event.set()

    signal.signal(signal.SIGTERM, _on_sigterm)

    loop.start_actors()
    # step_once(timeout=...) both waits for a rollout AND treats exceeding
    # that wait as StalledRun -- using cfg.stall_timeout_s (900s default)
    # directly here would make the loop deaf to SIGTERM for that whole
    # window. Poll on a short timeout instead, and track real staleness
    # ourselves: a short poll finding nothing is normal while actors are
    # still booting real servers, not a stall.
    last_progress = time.monotonic()
    try:
        while not stop_requested.is_set() and loop.state.update < run_cfg.total_updates:
            try:
                if loop.step_once(timeout=5.0):
                    last_progress = time.monotonic()
            except StalledRun:
                if stop_requested.is_set():
                    break
                if time.monotonic() - last_progress > run_cfg.stall_timeout_s:
                    raise
    finally:
        payload = learner.state_payload()
        ckpt = loop.checkpoints.save(loop.state.update, payload)
        loop.save_state(ckpt)
        log.info("final checkpoint: %s (update=%d)", ckpt, loop.state.update)
        loop.shutdown()
        if anchor_evaluator is not None:
            anchor_evaluator.close()
        for driver in built_drivers:
            try:
                driver.env.close()
            except Exception:
                log.error("error closing a driver's server instances", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
