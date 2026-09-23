"""Run a training job and print the curve. `python -m lanerl_jax.train.run_train`."""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from .ppo import MAX_FACTORED_ENTROPY
from .run_manifest import RunDir, file_sha256
from .trainer import TrainConfig, make_train

# The shared wandb helpers live outside the package tree (`src/ahriuwu/...`) and
# are used by every dreamer script, so reuse them rather than starting a second
# logging convention.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# `DEFAULT_ROUTE_ARTIFACT` lives in `sim/config.py` with `SimConfig`; the
# name is re-exported here because parity tools import it from this module.
__all__ = ["main", "build_parser", "cli_record", "manifest_config",
           "DEFAULT_ROUTE_ARTIFACT"]


def cli_record(args: argparse.Namespace) -> dict:
    """The parsed command line as recorded in the manifest and replayed by
    the README's "Reproducing" block (`run_manifest.reproduce_command`).

    Paths are recorded as strings rather than dropped: the old record left
    out ``--route-artifact``, ``--out-root`` and ``--resume`` altogether.
    ``None`` means "not given, default applies" and is omitted; wandb options
    are the logging helper's, not the run's.
    """
    out = {}
    for k, v in vars(args).items():
        if (k.startswith(("wandb", "_")) or k == "run_name"
                or v is None):
            continue
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def manifest_config(cfg: TrainConfig, args: argparse.Namespace, sim_config,
                    n_dec: int) -> dict:
    """Everything that defines the run, as recorded in `manifest.json`.

    `PPO-12` added the policy config (it used to be a hard-coded
    `PolicyConfig()` nothing recorded) and the route artifact by CONTENT;
    library versions and `XLA_FLAGS` are in the manifest's `software`
    block (`run_manifest.software_provenance`).
    """
    if sim_config.route_artifact is not None:
        art = Path(sim_config.route_artifact)
        route_prov = {
            "path": str(art), "resolved": str(art.resolve()),
            # The loader re-hashes the arrays against these and raises on a
            # mismatch, so this digest is of the CONTENT that ran.
            "content_sha256": sim_config.route_digest,
            "manifest_sha256": file_sha256(art / "manifest.json"),
        }
    else:
        route_prov = None
    return {
        "train": cfg, "ppo": cfg.ppo, "policy": cfg.policy,
        "cli": cli_record(args),
        "reward_weights": cfg.reward.weights,
        "env_decisions": n_dec,
        "route_artifact": route_prov,
        # `STRUCT-003`: the ONE step configuration this run's env step used,
        # and a digest of its arrays.
        "sim_config": {**sim_config.describe(),
                       "fingerprint": sim_config.fingerprint()},
    }


def build_parser() -> argparse.ArgumentParser:
    """The real parser, also used by `test_run_manifest` to parse the
    README's reproduce command back."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-s", type=float, default=600.0,
                    help="episode length in seconds. NOT the discount horizon "
                         "(PPOConfig.horizon_s); see TrainConfig.")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--route-artifact", type=Path, default=DEFAULT_ROUTE_ARTIFACT,
                    help="Map1 local-route artifact (default: %(default)s)")
    ap.add_argument(
        "--no-route-table", action="store_true",
        help="explicitly run the PATH-001 two-point approximation; intended "
             "only for comparison/debugging")
    ap.add_argument(
        "--tag", default="rl",
        help="run label. The run directory is <tag>-<timestamp>-<sha>, so two "
             "runs of the same tag never collide and the sha is visible without "
             "opening anything.")
    ap.add_argument("--notes", default="",
                    help="free text recorded in the manifest and the README")
    ap.add_argument("--out-root", type=Path,
                    default=Path("lanerl_jax/runs/train"))
    ap.add_argument(
        "--chunk", type=int, default=20,
        help="updates per jitted call. The loop used to be ONE jit call, so no "
             "metric and no checkpoint was observable until it finished -- a "
             "node failure at update ~450 of 600 lost the whole run. Chunking "
             "returns to Python to log and checkpoint; shapes are identical "
             "each call so there is still exactly one compile.")
    ap.add_argument("--ckpt-every", type=int, default=5,
                    help="checkpoint every N chunks (0 = only at the end)")
    ap.add_argument(
        "--resume", type=Path, default=None,
        help="resume params AND optimiser state from a ckpt_*.msgpack. "
             "Checkpoints were being written and nothing read one, so a 12-hour "
             "run that died in hour 11 restarted from zero -- which has already "
             "happened once, at update ~450 of 600. The chunked loop is half "
             "that fix; this is the other half. NOTE: the environment and RNG "
             "state are NOT restored (they are not in the checkpoint), so a "
             "resumed run continues the POLICY, not the episode -- it is a "
             "crash recovery, not a bit-exact continuation, and the manifest "
             "records which checkpoint it came from.")
    ap.add_argument("--lr", type=float, default=None,
                    help="override PPO learning rate. The inherited 1e-5 comes "
                         "from a BC FINE-TUNE config, chosen to avoid destroying "
                         "a behaviour-cloned prior -- and `trainer.py` says there "
                         "is no BC prior here yet, so from-scratch runs are "
                         "training ~30x slower than a normal PPO rate for a "
                         "reason that does not apply.")
    ap.add_argument("--entropy-coef", type=float, default=None)
    ap.add_argument("--no-value-clip", action="store_true",
                    help="unclipped value loss. The clipped form bounds each "
                         "state's value move to value_clip_eps=0.2 PER UPDATE in "
                         "raw return units, and returns here are O(10-30) and "
                         "unnormalised, so a +-15 kill swing takes ~75 updates "
                         "to fit (PPO-02). Andrychowicz 2021 / Engstrom 2020 "
                         "found the clip neutral-to-harmful.")
    ap.add_argument("--critic-lr", type=float, default=None,
                    help="value-head learning rate. Declared in PPOConfig at "
                         "3e-4 but read by nothing until 2026-09-23, so every "
                         "earlier run trained the critic at --lr.")
    ap.add_argument("--target-kl", type=float, default=None,
                    help="stop the remaining minibatches of an update once the "
                         "k3 KL estimate exceeds this. Also declared and "
                         "unenforced before 2026-09-23.")
    ap.add_argument("--value-coef", type=float, default=None)
    ap.add_argument(
        "--time-steady", action="store_true",
        help="run the whole job TWICE to separate compile from steady state. "
             "Exact, and it doubles the cost -- `n_updates` is the scan length "
             "and therefore baked into the graph, so there is no cheaper honest "
             "way. Use it for a benchmark, not for a real training run.")
    try:
        from ahriuwu.utils.logging import add_wandb_args
        add_wandb_args(ap)
        ap.set_defaults(_have_wandb_args=True)
    except Exception as exc:                        # pragma: no cover
        print(f"wandb helpers unavailable ({exc}); --wandb disabled")
        ap.set_defaults(_have_wandb_args=False)
    return ap


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()
    _have_wandb_args = a._have_wandb_args

    ppo = TrainConfig().ppo
    if a.lr is not None:
        ppo = ppo._replace(lr=a.lr)
    if a.entropy_coef is not None:
        ppo = ppo._replace(entropy_coef=a.entropy_coef)
    if a.critic_lr is not None:
        ppo = ppo._replace(critic_lr=a.critic_lr)
    if a.target_kl is not None:
        ppo = ppo._replace(target_kl=a.target_kl)
    if a.no_value_clip:
        ppo = ppo._replace(clip_value_loss=False)
    if a.value_coef is not None:
        ppo = ppo._replace(value_coef=a.value_coef)
    cfg = TrainConfig(n_envs=a.envs, rollout_steps=a.rollout,
                      n_updates=a.updates, n_minibatches=a.minibatches,
                      episode_s=a.episode_s, ppo=ppo)
    n_dec = cfg.n_envs * cfg.rollout_steps * cfg.n_updates
    print(f"device {jax.devices()[0]}")
    print(f"{cfg.n_envs} envs x {cfg.rollout_steps} steps x {cfg.n_updates} "
          f"updates = {n_dec:,} env-decisions "
          f"({n_dec * 2:,} champion-decisions)")
    print(f"episode {cfg.episode_s:.0f}s ({cfg.episode_steps:,} decisions) | "
          f"discount horizon {cfg.ppo.horizon_s:.0f}s "
          f"(gamma {cfg.ppo.gamma:.6f}) | "
          f"{cfg.n_updates * cfg.rollout_steps / cfg.episode_steps:.1f} "
          f"episodes per env")

    if not a.no_route_table:
        if not a.route_artifact.exists():
            ap.error(
                f"route artifact not found: {a.route_artifact}. Build it with: "
                "python -m lanerl_jax.data.local_route_artifact "
                f"--out {a.route_artifact} --radius 35 --offset-radius 50")
        sim_config = SimConfig.training(route_artifact=a.route_artifact)
        nh = sim_config.route_table.next_hop
        print(f"local routes {a.route_artifact.name} "
              f"({nh.nbytes / 2**20:.1f} MiB next-hop table)")
    else:
        sim_config = SimConfig.training(route_artifact=None)
        print("WARNING: local routing disabled; Move uses the PATH-001 raw segment")

    built = make_train(cfg, sim_config=sim_config)
    config = manifest_config(cfg, a, sim_config, n_dec)
    sim_desc = config["sim_config"]
    print(f"sim config {sim_desc['name']} fingerprint {sim_desc['fingerprint']}")

    # ---- run directory, manifest, wandb ---------------------------------
    run = RunDir(a.out_root, a.tag, config, notes=a.notes)
    print(f"run dir {run.path}")
    # The wandb run NAME is the run-dir id, so a chart and a checkpoint can be
    # matched without opening either. init_wandb() reads args.run_name.
    a.run_name = run.run_id
    wb = None
    if _have_wandb_args:
        try:
            from ahriuwu.utils.logging import finish_wandb, init_wandb, log_step
            wb = init_wandb(a, job_type="lanerl_rl",
                            extra_config=run.manifest["config"])
            if wb is not None:
                run.manifest["wandb"] = {
                    "run_name": getattr(wb, "name", None),
                    "run_path": getattr(wb, "path", None),
                    "url": getattr(wb, "url", None)}
                run.write()
                print(f"wandb {run.manifest['wandb'].get('url')}")
        except Exception as exc:
            print(f"wandb init failed ({exc}); continuing without it")

    # ---- chunked loop ----------------------------------------------------
    chunk = max(1, min(a.chunk, cfg.n_updates))
    n_chunks, rem = divmod(cfg.n_updates, chunk)
    step_fn = jax.jit(built.run_chunk, static_argnums=1)
    runner = built.initial_runner(jax.random.key(a.seed))
    if a.resume is not None:
        from flax.serialization import from_bytes
        if not a.resume.exists():
            ap.error(f"--resume checkpoint not found: {a.resume}")
        # Deserialised INTO the freshly built runner's own pytrees, so a
        # checkpoint whose structure no longer matches the model fails here
        # rather than loading something plausible. Params and optimiser state
        # both: resuming the weights without Adam's moments throws away the
        # second-moment estimate and the first few updates after the resume are
        # then effectively at a different learning rate.
        raw = a.resume.read_bytes()
        try:
            payload = from_bytes(
                {"params": runner.params, "opt_state": runner.opt_state,
                 "step": runner.step}, raw)
        except (ValueError, KeyError):
            # checkpoints written before `step` was saved (PPO-06): the
            # zero-sum alpha anneal restarted from 0.5 on every resume.
            payload = from_bytes(
                {"params": runner.params, "opt_state": runner.opt_state}, raw)
            payload["step"] = runner.step
            print("WARNING: checkpoint has no `step`; the step counter "
                  "restarts from 0", flush=True)
        # A resumed run must not replay the seed's RNG stream from the
        # start: fold the restored step in.
        rng = jax.random.fold_in(runner.rng, int(np.asarray(payload["step"])))
        runner = runner._replace(params=payload["params"],
                                 opt_state=payload["opt_state"],
                                 step=jnp.asarray(payload["step"], jnp.int32),
                                 rng=rng)
        run.manifest["resumed_from"] = {
            "checkpoint": str(a.resume),
            "note": "params + opt_state only; env state and RNG are fresh, so "
                    "this continues the policy and not the episode",
        }
        run.write()
        print(f"resumed params + opt_state from {a.resume}")
    jax.block_until_ready(runner)

    parts, t0, diverged = [], time.perf_counter(), False
    for ci in range(n_chunks + (1 if rem else 0)):
        n = chunk if ci < n_chunks else rem
        runner, mc = step_fn(runner, n)
        jax.block_until_ready(mc)
        parts.append(mc)
        upd = (ci + 1) * chunk if ci < n_chunks else cfg.n_updates
        # MEAN over the chunk, not the last update of it. A single update's
        # `reward` and `entropy` are noisy enough that a chunk-end sample and a
        # chunk mean can point different directions, and the chunk boundary is
        # an artefact of --chunk.
        with warnings.catch_warnings():
            # cs_at_10min is all-NaN on any chunk where no episode ended, and
            # `nanmean` of that is NaN with a RuntimeWarning. NaN is the right
            # answer here -- "no sample" -- so the warning is noise.
            warnings.simplefilter("ignore", RuntimeWarning)
            row = {k: float(np.nanmean(np.asarray(v))) for k, v in mc.items()}
        # cs_at_10min is NaN on updates where no episode ended, so its mean is
        # over the episodes that DID end; the count is the denominator.
        row["cs_episodes"] = float(np.asarray(mc["cs_episodes"]).sum())
        row.update(update=upd, chunk=ci,
                   step=int(np.asarray(runner.step)),
                   wall_s=round(time.perf_counter() - t0, 1))
        run.log(row)
        if wb is not None:
            log_step({f"train/{k}": v for k, v in row.items() if k != "update"},
                     step=upd)
        cs = np.asarray(mc["cs_at_10min"])
        done = cs[~np.isnan(cs)]
        print(f"  chunk {ci:>3} upd {upd:>5} reward {row['reward']:+.5f} "
              f"entropy {row['entropy']:.3f} kl {row['approx_kl']:.4f} "
              f"vloss {row['value_loss']:.3f} "
              + (f"cs {np.mean(done):.2f} (n={row['cs_episodes']:.0f} champion-episodes)"
                 if done.size else "cs -"), flush=True)
        # DIVERGENCE GUARD. A run whose loss has gone non-finite is producing
        # nothing but wall-clock, and the RL-002 log shows `value_loss` going
        # 0.0024 -> 2.0032 -> 542.7 without anything stopping it. Checkpoint,
        # record why, and get off the GPU so the next arm can have it.
        # Checked PER UPDATE, not on the chunk mean: a NaN that starts
        # mid-chunk hides behind the finite updates before it under
        # `nanmean` (`PPO-07`), and `reward` stays finite regardless
        # because the sim is driven by sampled actions.
        # The loss metrics are means over APPLIED minibatches (`PPO-11`), and
        # a NaN-KL minibatch is never applied, so `loss_nonfinite` (over ALL
        # minibatches) is what sees a NaN that the stop withheld.
        bad = [k for k in ("policy_loss", "value_loss", "entropy", "approx_kl")
               if not np.all(np.isfinite(np.asarray(mc[k])))]
        if np.any(np.asarray(mc["loss_nonfinite"]) > 0):
            bad.append("loss_nonfinite")
        params_finite = all(bool(np.all(np.isfinite(np.asarray(p))))
                            for p in jax.tree.leaves(runner.params))
        if a.ckpt_every and (ci + 1) % a.ckpt_every == 0 and not bad \
                and params_finite:
            run.save(int(np.asarray(runner.step)), upd,
                     {"params": runner.params, "opt_state": runner.opt_state,
                      "step": runner.step})
        if bad or not params_finite:
            # Saved for the post-mortem, but NOT as `ckpt_latest`: that is
            # the documented `--resume` target and a NaN payload there
            # poisons the next run.
            run.save(int(np.asarray(runner.step)), upd,
                     {"params": runner.params, "opt_state": runner.opt_state,
                      "step": runner.step}, latest=False)
            if not params_finite:
                bad = bad + ["params"]
            run.set_results(diverged_at_update=upd, diverged_metrics=bad)
            print(f"\nDIVERGED at update {upd}: {bad} non-finite. "
                  f"Stopping; see {run.path}", flush=True)
            diverged = True
            break

    m = jax.tree.map(lambda *xs: np.concatenate([np.asarray(x) for x in xs]),
                     *parts) if len(parts) > 1 else \
        jax.tree.map(np.asarray, parts[0])
    out = (runner, m)
    # `step` too, like the periodic saves: without it a resume from the final
    # checkpoint took the "old checkpoint" path and restarted the counter.
    # NOT after a divergence: this used to run after the guard's `break` and
    # overwrite `ckpt_latest` with the non-finite params the guard had just
    # kept out of it (`PPO-07`).
    if not diverged:
        run.save(int(np.asarray(runner.step)), cfg.n_updates,
                 {"params": runner.params, "opt_state": runner.opt_state,
                  "step": runner.step})
    jax.block_until_ready(out)
    first = time.perf_counter() - t0

    if a.time_steady:
        # `n` is a STATIC argument: the chunks above compiled `n=chunk`, so
        # a call with `n=cfg.n_updates` is a new XLA program and its time
        # includes a second compile (`PPO-08`). Time one chunk and scale.
        t0 = time.perf_counter()
        jax.block_until_ready(step_fn(
            built.initial_runner(jax.random.key(a.seed + 1)), chunk))
        steady = (time.perf_counter() - t0) * cfg.n_updates / chunk
        print(f"first call {first:.1f}s (compile ~{first - steady:.1f}s) | "
              f"steady {steady:.1f}s -> {n_dec / steady:,.0f} env-decisions/s "
              f"END TO END, gradient step included")
    else:
        print(f"wall {first:.1f}s -> {n_dec / first:,.0f} env-decisions/s "
              f"including compile (pass --time-steady to separate them)")

    print()
    cols = ("reward", "entropy", "approx_kl", "clip_frac", "value_loss",
            "value_explained_var", "grad_norm", "lane_dist", "cs_at_10min")
    print(f"{'upd':>5}" + "".join(f"{c:>20}" if len(c) > 11 else f"{c:>12}"
                                    for c in cols))
    # len(m), not n_updates: a diverged run stops early and indexing past
    # its last update raised before the results were written.
    for i in range(0, len(np.asarray(m["reward"])), a.every):
        row = "".join(
            f"{float(np.asarray(m[c])[i]):>20.4f}" if len(c) > 11
            else f"{float(np.asarray(m[c])[i]):>12.4f}" for c in cols)
        print(f"{i:>5}{row}")
    print()
    e0 = float(np.asarray(m["entropy"])[0])
    e1 = float(np.asarray(m["entropy"])[-1])
    # The MASKED ceiling (`PPO-01`); 14.099 counted heads no button uses.
    print(f"entropy {e0:.3f} -> {e1:.3f} of a {MAX_FACTORED_ENTROPY:.3f} "
          f"masked maximum ({100 * e1 / MAX_FACTORED_ENTROPY:.0f}%)")
    print(f"mean reward first/last update: "
          f"{float(np.asarray(m['reward'])[0]):+.5f} -> "
          f"{float(np.asarray(m['reward'])[-1]):+.5f}")

    # --- the two numbers the sampled table above cannot show ---------------
    #
    # cs_at_10min is NaN on updates where no episode ended. Before the phase
    # stagger (`RunnerState.deadline_ms`) that was all but 1 update in 141 --
    # every env resetting on the same step -- so printing every `--every`th row
    # reported NaN throughout an 800-update run while the number existed. With
    # the stagger the samples arrive continuously, but the row-level NaNs remain
    # real, so the per-episode list stays.
    cs = np.asarray(m["cs_at_10min"])
    done_at = np.flatnonzero(~np.isnan(cs))
    if len(done_at):
        print(f"\nCS@10min, per episode that ENDED ({len(done_at)} of them):")
        for i in done_at:
            print(f"  update {int(i):>5}   {float(cs[i]):.3f} CS per champion")
    else:
        print(f"\nNo episode finished: {cfg.n_updates} updates x "
              f"{cfg.rollout_steps} steps = "
              f"{cfg.n_updates * cfg.rollout_steps:,} decisions against an "
              f"{cfg.episode_steps:,}-decision episode. Run at least "
              f"{cfg.episode_steps // cfg.rollout_steps + 1} updates.")

    # lane_dist is dominated by WHERE in the episode each rollout falls: a
    # rollout just after a reset has both champions back at the fountain at
    # ~8,000. The minimum over the run is what says whether they ever arrive.
    _cs = np.asarray(m["cs_at_10min"])
    _done = _cs[~np.isnan(_cs)]
    run.set_results(
        cs_at_10min=[round(float(x), 3) for x in _done],
        cs_at_10min_last=(round(float(_done[-1]), 3) if _done.size else None),
        episodes_ended=int(_done.size),
        reward_first=round(float(np.asarray(m["reward"])[0]), 6),
        reward_last=round(float(np.asarray(m["reward"])[-1]), 6),
        entropy_first=round(float(np.asarray(m["entropy"])[0]), 4),
        entropy_last=round(float(np.asarray(m["entropy"])[-1]), 4),
        wall_s=round(first, 1),
        env_decisions_per_s=round(n_dec / first, 1),
        # Which reward term drove the total, over the whole run. A rebalance is
        # the change this answers directly rather than by inference.
        reward_terms={k[len("reward_"):]: round(float(np.asarray(v).mean()), 8)
                      for k, v in m.items() if k.startswith("reward_")},
        value_explained_var_last=round(
            float(np.asarray(m["value_explained_var"])[-1]), 4),
        grad_clipped_frac=round(float(np.asarray(m["grad_clipped"]).mean()), 4),
    )
    if wb is not None:
        try:
            log_step({"final/cs_at_10min": run.manifest["results"]["cs_at_10min_last"]
                      or float("nan")}, step=cfg.n_updates)
            finish_wandb()
        except Exception:
            pass
    run.close()
    print(f"\nrun dir {run.path}\n  manifest.json / README.md / metrics.jsonl / "
          f"ckpt_latest.msgpack")

    ld = np.asarray(m["lane_dist"])
    print(f"\nlane distance: start {ld[0]:,.0f} -> min {ld.min():,.0f} "
          f"(0 = inside the lane corridor, ~8,000 = the fountain)")


if __name__ == "__main__":
    main()
