"""Real ``lane-rl`` PPO training entrypoint.

Nobody had ever assembled ``lanerl_train.run.TrainingLoop`` against the real
server before today: ``TrainingLoop`` was unit-tested only against
``FakeInstance``. This module is the CLI that actually does it, using
``lanerl_train.lane_wiring`` for the (observation, action, reward, policy)
plumbing.

Usage::

    python -m lanerl_train --run-name my-run --num-actors 2 --envs-per-actor 2

Phase 1 -- against the frozen scripted bot rather than against itself::

    python -m lanerl_train --run-name phase1 --opponent scripted:bronze

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
from typing import Dict, List, Tuple

import torch

from lanerl_rl.constants import DEFAULT_HORIZON_S as C_DEFAULT_HORIZON_S
from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import DualClipPPO, PPOConfig
from lanerl_rl.reward import LaneRewardConfig

from . import paths
from .anchor_eval import AnchorEvalConfig, AnchorEvaluator, make_anchor_driver_factory
from .eval import DEFAULT_RUN_ANCHORS, Evaluator, anchors_for_run
from .lane_wiring import LanePolicyActor, collect_rollout, make_collect_fn, make_lane_adapters
from .ports import PortAllocator
from .run import GpuProbe, RunConfig, StalledRun, TrainingLoop
from .vec import EpisodeSpec, ServerLaunchSpec, SideAssignment, VecDriver, VecLaneEnv

log = logging.getLogger("lanerl_train.__main__")

SELF = "self"
#: Policy key for the red side when it is a sampled LEAGUE opponent.
OPPONENT = "opponent"
#: ``--opponent league`` -- red is a checkpoint drawn from the PFSP pool.
LEAGUE = "league"
#: ``--opponent none`` -- red is left IDLE in its fountain. A no-enemy
#: curriculum phase: the agent learns to farm without also being harassed.
#:
#: Worth having because the policy has never seen a game without an enemy, so
#: the solo setting is out of distribution -- observed live, a top-lane policy
#: with no opponent wandered to MID and died to a turret. That also means the
#: uncontested eval number (37.5 CS against 45.4 contested) says as much about
#: distribution shift as it does about farming skill.
NO_ENEMY = "none"

# Generous headroom per actor's port block: PortAllocator already verifies
# freeness and raises rather than colliding, but starting each actor far
# apart makes a leaked/lingering instance from a previous run obvious in the
# logs instead of eating into the next actor's range.
PORTS_PER_ACTOR_STRIDE = 64

#: The scripted difficulties ``--opponent scripted:<name>`` knows by name.
#: The same three JSONs the anchor ladder is built from (``eval.default_anchors``),
#: so "the opponent I train against" and "the opponent I am measured against"
#: can be made the same thing deliberately rather than by coincidence.
SCRIPTED_CONFIGS: Dict[str, str] = {
    "bronze": "anchor_bronze.json",
    "gold": "anchor_gold.json",
    "diamond": "anchor_diamond.json",
}

#: ``LanerlConfig.Seed`` -- the server's default, and therefore the seed every
#: training instance has silently used so far, all of them the same one.
SERVER_DEFAULT_BOT_SEED = 1234

#: Room for 10,000 envs under one ``--seed`` before two runs' bot streams can
#: overlap.  ``--seed 0`` env 0 lands back on exactly 1234, so a single-env
#: run still reproduces the historical bot.
BOT_SEED_RUN_STRIDE = 10_000


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real lane-rl 1v1 Garen PPO training run")
    p.add_argument("--run-name", required=True)
    p.add_argument("--num-actors", type=int, default=2)
    p.add_argument("--envs-per-actor", type=int, default=2, help="real server instances per actor thread")
    p.add_argument(
        "--rollout-steps", type=int, default=129,
        help="decisions collected per actor rollout; buffer rows = this - 1 (see collect_rollout)",
    )
    p.add_argument(
        "--max-staleness", type=int, default=None,
        help="versions of lag a rollout may carry and still be trained on. "
             "Default: derived as queue_capacity + num_actors - 1, the worst "
             "case the pipeline can produce. Setting it lower discards work "
             "that has ALREADY been collected -- the old default of 1 binned "
             "45-80%% of every rollout, all of them stale by exactly 2.",
    )
    p.add_argument("--queue-capacity", type=int, default=2)
    p.add_argument(
        "--lane-presence", type=float, default=None,
        help="per-decision reward for being inside the top-lane corridor. "
             "Default: RewardWeights.lane_presence (0.0001, ~1.8 an episode "
             "against ~40 for a 40-CS game). Set 0 to disable -- it is a "
             "DENSE reward for occupying a state, which is the shape of "
             "shaping term most likely to be gamed, so it needs to be "
             "switchable without a code edit to be testable at all.",
    )
    p.add_argument(
        "--lane-approach", type=float, default=None,
        help="reward per 1000 game units of distance CLOSED toward the lane "
             "corridor, paid as a potential. Default: "
             "RewardWeights.lane_approach (0.07, so 0.478 for the whole 6,835 "
             "unit walk from the fountain). Set 0 to ablate. Unlike "
             "--lane-presence this is a Ng-et-al. potential and so cannot "
             "change which policy is optimal, only which ones are found -- "
             "but it is switchable because it did NOT turn out to be the fix "
             "for rl-screen-0914 (that agent reached lane in 29%% of its "
             "steps and died there; see RewardWeights.lane_approach), so its "
             "value is still unmeasured.",
    )
    p.add_argument(
        "--agent-type", choices=("main", "main_exploiter", "league_exploiter"),
        default="main",
        help="this lineage's role in the league (AlphaStar, Nature 2019). "
             "'main' trains on the ordinary mixture and never resets. "
             "'main_exploiter' targets the CURRENT main agent to find its "
             "weaknesses. 'league_exploiter' PFSPs the whole league. The "
             "exploiters are what make a league more than self-play -- a main "
             "agent playing only its own history can cycle, and faces no "
             "pressure to be robust to a strategy nobody in that history "
             "happened to try. Needs --league-dir to see the other lineages.",
    )
    p.add_argument(
        "--league-dir", default=None,
        help="directory shared by every lineage. Each publishes its snapshots "
             "there and reads the others back. WITHOUT IT concurrently "
             "training agents cannot see each other and the 'league' is N "
             "independent self-play runs.",
    )
    p.add_argument(
        "--actor-mode", choices=("thread", "process"), default="process",
        help="'process' (default) gives each actor its own interpreter and its "
             "own CUDA context. 'thread' runs them in this process, sharing one "
             "GIL that the observation build holds for ~55%% of every decision, "
             "which caps the whole run at ~1.8 of 16 cores however many actors "
             "you ask for. Measured at 12 instances / 4 actors: thread 1.85 "
             "cores and 1,191 decisions/s, process 4.85 cores and 1,949.",
    )
    p.add_argument("--total-updates", type=int, default=1_000_000)
    p.add_argument(
        "--chunk-len", type=int, default=None,
        help="decisions per BPTT chunk -- the GRADIENT-CARRYING context, at "
             "30 Hz. Default: PPOConfig.chunk_len (16, i.e. 0.53 s). Note "
             "this is NOT the observation horizon: the core is a GRU, so the "
             "hidden state carries arbitrarily far at inference; this bounds "
             "only how far back a gradient can assign credit.",
    )
    p.add_argument(
        "--burn-in", type=int, default=None,
        help="decisions replayed before each chunk to warm the recurrent "
             "state, WITHOUT gradient (R2D2). Default: PPOConfig.burn_in (8). "
             "Guards against a chunk starting from a stale or zero hidden "
             "state, which is why it grows with --chunk-len.",
    )
    p.add_argument(
        "--minibatch-chunks", type=int, default=None,
        help="chunks per minibatch. Default: PPOConfig.minibatch_chunks (32). "
             "Activation memory scales as this * (burn_in + chunk_len), so "
             "raising --chunk-len without lowering this is how a long-context "
             "run OOMs.",
    )
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--keep-last-checkpoints", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--horizon-s", type=float, default=C_DEFAULT_HORIZON_S,
        help="PPOConfig discount horizon in seconds (default %(default)s). A "
             "shove-and-bounce cycle is 1-3 minutes, so at 30 a payoff two "
             "minutes out is worth 1.8%% of face value and wave management is "
             "invisible; at 120 it is 36.8%%.")
    p.add_argument("--anneal-clock", choices=("env_steps", "updates"), default="env_steps")
    p.add_argument("--device", default="cpu")
    # -- the reward's zero-sum coefficient -------------------------------
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="zero-sum coefficient in r = r_self - alpha * r_opponent. HELD "
        "CONSTANT unless --alpha-anneal-steps says otherwise. There was no flag "
        "at all, so every run inherited LaneRewardConfig's 0.5 -> 1.0 anneal "
        "over 2,000,000 rows of the anneal clock: at ~255 rows/update that is "
        "~7,800 updates, so a six-hour run spent its ENTIRE life on a moving "
        "reward scale. Measured in runs/rl-bc4-0912, ep_return by 500-update "
        "bucket went 30.0 -> 26.6 -> 24.4 -> 34.9 -> 18.7 -> 6.5 while CS@10 "
        "stayed flat at ~36 -- the curve was the reward definition changing, "
        "not the policy. The default is the value every measured run actually "
        "spent its first bucket at, so the only thing that changes is that it "
        "no longer MOVES. 1.0 makes the lane exactly zero-sum; against a frozen "
        "scripted opponent (--opponent scripted:...) a lower alpha is arguable, "
        "since the subtracted stream is then one the agent cannot influence.",
    )
    p.add_argument(
        "--alpha-anneal-steps", type=int, default=0,
        help="anneal alpha from LaneRewardConfig's 0.5 up to --alpha over this "
        "many steps of --anneal-clock. 0 (the default) holds it constant, which "
        "is the only setting under which a return curve measures the policy.",
    )
    # -- who the agent plays ---------------------------------------------
    p.add_argument(
        "--opponent", default=SELF,
        help="'self' (both champions driven by the live policy), 'league' "
        "(red is a checkpoint drawn from the PFSP pool -- the sampler exists "
        "and until 2026-09-13 had ZERO callers, so every game was a live "
        "mirror and no past checkpoint was ever played), or "
        "'scripted:<name-or-path>[,<name-or-path>...]'. Phase 1 is scripted: in a "
        "mirror the opponent's farming stream is uncontrollable noise in every "
        "advantage, and the training opponent is not comparable with the "
        "evaluation one, so CS arrives as ~1 sample per 400 updates from the "
        "anchor ladder instead of ~35 per 200 from training itself. Names are "
        f"{sorted(SCRIPTED_CONFIGS)} (or the 'scripted_' anchor ids); anything "
        "else is a path to a bot config JSON. More than one is OPT-IN "
        "population play -- see build_training_specs for what varies and what "
        "cannot.",
    )
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
        "--init-from", type=Path, default=None,
        help="behaviour-cloning checkpoint to INITIALISE the policy from, and to "
        "freeze as the KL reference. Without this the policy starts random, and "
        "a random policy provably cannot reach lane: 345 u/s / 30 Hz = 11.5 "
        "units per decision, so a random walk needs ~1.06M steps (9.9 h of game) "
        "to cross 11,866 units against a 10-minute episode.",
    )
    p.add_argument(
        "--critic-lr", type=float, default=None,
        help="Adam lr for the VALUE head only (default: PPOConfig's 3e-4). The "
        "critic starts from random weights -- behaviour cloning has no returns "
        "to fit it against -- so it must not inherit the actor's fine-tuning "
        "rate. One shared 1e-5 is why value_loss climbed 0.047 -> 0.55 over a "
        "2,690-update run while the critic never caught its moving target.",
    )
    p.add_argument(
        "--critic-warmup-updates", type=int, default=None,
        help="train ONLY the value head for this many updates first (default 0). "
        "The actor is frozen, so a random critic cannot feed garbage advantages "
        "into a BC prior that is already good.",
    )
    p.add_argument(
        "--entropy-coef", type=float, default=None,
        help="override PPOConfig.entropy_coef (default 0.01). When fine-tuning "
        "from a BC prior this must come DOWN: the prior is confident (entropy "
        "1.64 of a 9.94 maximum), so an entropy bonus sized for a random init "
        "pays the policy to throw the prior away -- measured, entropy ROSE "
        "1.64 -> 3.96 over 140 updates while CS went nowhere.",
    )
    p.add_argument(
        "--end-on-death", dest="end_on_death", action="store_true", default=None,
        help="terminate an episode at the first champion death (the old default)",
    )
    p.add_argument(
        "--no-end-on-death", dest="end_on_death", action="store_false",
        help="play the full 10 minutes through deaths, as a real lane does. "
        "Required for cs_at_10 to exist at all: it is an ABSOLUTE 10-minute "
        "metric, so an episode that ends at the first death (mean 3,094 steps = "
        "103 s in the first BC-init run) can never contribute one, and all 43 "
        "episodes reported cs_at_10 = None. Death stays punished by the reward's "
        "death term, which fires on the transition either way.",
    )
    p.add_argument(
        "--kl-ref-anneal-steps", type=int, default=0,
        help="decay --kl-ref-coef linearly to ZERO over this many steps of the "
        "--anneal-clock. 0 keeps it flat forever, which tethers the agent to a "
        "prior it is supposed to beat: measured at 69%% of the policy-gradient "
        "magnitude, with kl_ref still climbing at the end of a 2,700-update run.",
    )
    p.add_argument(
        "--kl-ref-coef", type=float, default=0.0,
        help="weight on KL(reference || policy) toward the --init-from prior. "
        "Only has any effect together with --init-from; 0 disables it.",
    )
    p.add_argument(
        "--anchor-envs", type=int, default=4,
        help="server instances per anchor. Each anchor keeps its own alive for the "
        "life of the run (a restart is ~12s against 0.23ms for an episode reset). "
        "This is a WALL-CLOCK knob: anchor games run in waves of this many, so at "
        "the default 35 episodes 1 env is ~35 sequential 10-minute games and 4 is "
        "~9 waves.",
    )
    p.add_argument(
        "--anchor-episodes", type=int, default=None,
        help="games per anchor per eval cycle (default: AnchorEvalConfig's, which is "
        "sized for statistical power). One game CANNOT resolve anything: self-play "
        "CS@10 has sd 7.3, so n=1 has a minimum detectable difference of ~29 CS -- "
        "it could not tell 36 CS from 7. n=35 resolves ~5 CS.",
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


def reward_config(gamma: float, alpha: float, alpha_anneal_steps: int) -> LaneRewardConfig:
    """The ONE place a training ``LaneRewardConfig`` is built.

    Both fields have to be set together or the flag does nothing:
    ``LaneRewardConfig.alpha`` returns ``zero_sum_alpha_end`` when
    ``zero_sum_anneal_steps <= 0`` and interpolates from ``_start`` otherwise,
    so "constant at x" is ``end=x, steps=0`` and "anneal to x" is
    ``start=0.5, end=x, steps=N``. Setting ``end`` alone would have looked
    exactly like a working ``--alpha`` while the 2,000,000-step anneal from 0.5
    carried on underneath it.
    """
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(
            f"--alpha {alpha} is outside [0, 1]. Above 1 the opponent's reward "
            f"outweighs the agent's own; below 0 the agent is paid for the "
            f"opponent's farm."
        )
    if alpha_anneal_steps > 0:
        return LaneRewardConfig(
            gamma=gamma,
            zero_sum_alpha_start=LaneRewardConfig().zero_sum_alpha_start,
            zero_sum_alpha_end=alpha,
            zero_sum_anneal_steps=int(alpha_anneal_steps),
        )
    return LaneRewardConfig(
        gamma=gamma,
        zero_sum_alpha_start=alpha,
        zero_sum_alpha_end=alpha,
        zero_sum_anneal_steps=0,
    )


def resolve_bot_configs(text: str) -> List[Tuple[str, Path]]:
    """``--opponent scripted:<text>`` -> ``[(label, bot config path), ...]``.

    A name that does not resolve is fatal here rather than at the first server
    launch: ``LANERL_BOT_CONFIG`` pointing at a file that does not exist leaves
    ``LanerlConfig`` on its defaults, which is a DIFFERENT and much weaker bot,
    and nothing downstream can tell that apart from a difficulty that was
    chosen.
    """
    out: List[Tuple[str, Path]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name = item[len("scripted_"):] if item.startswith("scripted_") else item
        if name in SCRIPTED_CONFIGS:
            path, label = paths.bot_config_dir() / SCRIPTED_CONFIGS[name], name
        else:
            path = Path(item).expanduser()
            label = path.stem
        if not path.exists():
            raise SystemExit(
                f"--opponent names {item!r}, which is neither one of "
                f"{sorted(SCRIPTED_CONFIGS)} nor an existing file ({path})."
            )
        out.append((label, path))
    if not out:
        raise SystemExit("--opponent scripted: needs at least one config name or path")
    return out


def build_training_specs(
    opponent: str,
    seed: int,
    actor_idx: int,
    envs_per_actor: int,
) -> Tuple[List[ServerLaunchSpec], Dict[int, str], bool]:
    """``(launch spec per instance, instance -> opponent label, red is ours)``.

    Self-play leaves every instance on the default spec (``bot_teams="none"``)
    and hands RED to the same policy.

    Scripted play sets ``bot_teams="purple"`` and points ``LANERL_BOT_CONFIG``
    at the difficulty JSON, and RED is then assigned to NO policy -- the action
    line omits the key, ``LanerlControl.ApplyActions`` finds no object for it,
    and the bot's own orders stand (see ``anchor_eval.anchor_launch_spec``,
    which does the same thing for the ladder).

    **The bot seed varies per instance, and that is the most it can vary.**
    ``LANERL_BOT_SEED`` is read once by ``LanerlConfig.FromEnv`` at process
    start; ``LanerlBot._rng`` is built from it once in the constructor, and
    ``LanerlBot.OnEpisodeReset`` (which the in-process reset calls) clears
    every other piece of bot state but deliberately does NOT re-seed that RNG.
    So within one process the stream carries on across episodes and successive
    episodes DO see different jitters, coin flips and ability rolls -- but two
    processes launched on the same seed replay each other exactly, which until
    now every training instance did, from the server's own default of 1234.
    Per-episode *reseeding* would need a server change (a seed on the reset
    line, or a reseed in ``OnEpisodeReset``); it is not reachable from here.

    **Difficulty is drawn per instance, not per episode, for the same reason:**
    ``LANERL_BOT_CONFIG`` is process-scoped and nothing reloads it on reset.
    Opt-in, by naming more than one config: a deterministic round-robin over
    the population rather than a random draw, so the mixture is exactly
    balanced across a handful of envs (a draw leaves difficulties unsampled at
    N=4) and a resume faces the same population it left.
    """
    if opponent in (SELF, LEAGUE, NO_ENEMY):
        # LEAGUE and NO_ENEMY launch exactly like self-play as far as the
        # SERVER is concerned: LANERL_BOT stays "none" either way, so no
        # scripted bot is attached. What differs is purely who drives red --
        # the live policy (self), a pool checkpoint (league), or nobody at all
        # (none, which leaves the champion standing in its fountain). Handled here rather than mapped at each
        # call site because main() pre-flights this function with the raw
        # --opponent string, and mapping at one site only left that check
        # rejecting a mode the rest of the stack supports.
        return [ServerLaunchSpec() for _ in range(envs_per_actor)], {}, True
    if not opponent.startswith("scripted:"):
        raise SystemExit(
            f"--opponent {opponent!r} is not 'self', 'league', 'none', or "
            f"'scripted:<name-or-path>'"
        )
    configs = resolve_bot_configs(opponent[len("scripted:"):])
    specs: List[ServerLaunchSpec] = []
    labels: Dict[int, str] = {}
    for env_idx in range(envs_per_actor):
        global_idx = actor_idx * envs_per_actor + env_idx
        label, path = configs[global_idx % len(configs)]
        specs.append(
            ServerLaunchSpec(
                bot_teams="purple",
                bot_config=path,
                # Distinct per env AND distinct from the anchor evaluator's
                # (which passes --seed straight through, deliberately frozen),
                # so an eval game is never a replay of a game just trained on.
                bot_seed=SERVER_DEFAULT_BOT_SEED + seed * BOT_SEED_RUN_STRIDE + global_idx,
            )
        )
        # Namespaced so training games never pool into the anchor ladder's
        # win rate; see lane_wiring._opponent_of.
        labels[env_idx] = f"train:scripted_{label}"
    return specs, labels, False


def _build_driver_for_actor(
    actor_idx: int,
    envs_per_actor: int,
    port_base: int,
    model_cfg: ModelConfig,
    run_dir: Path,
    device: str,
    train_step_source,
    reward_cfg: LaneRewardConfig,
    end_on_death: bool,
    opponent: str,
    seed: int,
    league: bool = False,
) -> Tuple[VecDriver, LanePolicyActor, dict, Dict[int, str]]:
    actor_base = port_base + actor_idx * envs_per_actor * PORTS_PER_ACTOR_STRIDE
    allocator = PortAllocator(base=actor_base)
    ports = allocator.allocate(envs_per_actor)
    # ONE discount, and ONE alpha schedule. LaneRewardConfig.gamma defaults to
    # gamma_for_horizon(30), independently of PPOConfig, so --horizon-s moved
    # the trainer's gamma and left the shaping gamma behind -- and
    # potential-based shaping is only policy-invariant when the two agree (its
    # own docstring says so). The config is built once, in main(), and passed
    # in, so the anchor evaluator and the actors cannot drift apart either.
    adapters = make_lane_adapters(train_step_source=train_step_source, reward_cfg=reward_cfg)
    policy = LanePolicy(model_cfg).to(device)
    actor = LanePolicyActor(policy, device=device)
    # A SECOND policy for the red side, so the opponent can be a past
    # checkpoint rather than a live mirror. The weights are swapped in on every
    # parameter push; when the league samples "latest" the live payload goes
    # into both and the behaviour is byte-identical to the old mirror.
    #
    # Two policies, not two assignments: the driver's SideAssignment is fixed
    # when the driver is built, but the OPPONENT has to change per sample --
    # so the identity that varies is the weights, not the wiring.
    opp_actor = None
    if league:
        opp_actor = LanePolicyActor(LanePolicy(model_cfg).to(device), device=device)
    specs, labels, red_is_ours = build_training_specs(
        # A league game is a self-play game as far as the SERVER is concerned:
        # both champions are driven by our control channel, so the launch spec
        # is identical. Only which weights drive red differs.
        "self" if league else opponent, seed, actor_idx, envs_per_actor
    )
    env = VecLaneEnv(
        n=envs_per_actor, specs=specs, ports=ports,
        log_dir=run_dir / f"actor{actor_idx}_logs",
    )
    if opponent == NO_ENEMY:
        red_key = None          # nobody drives red; it never leaves the fountain
    elif league and red_is_ours:
        red_key = OPPONENT
    elif red_is_ours:
        red_key = SELF
    else:
        red_key = None          # the in-server scripted bot has it
    policies = {SELF: actor}
    if opp_actor is not None:
        policies[OPPONENT] = opp_actor
    driver = VecDriver(
        env=env,
        policies=policies,
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[
            SideAssignment(blue=SELF, red=red_key)
            for _ in range(envs_per_actor)
        ],
        episode=EpisodeSpec(end_on_death=end_on_death),
    )
    log.info(
        "actor %d: starting %d real server instance(s) on ports base=%d against %s",
        actor_idx, envs_per_actor, actor_base,
        "itself" if red_is_ours else f"{sorted(set(labels.values()))} "
        f"seeds={[s.bot_seed for s in specs]}",
    )
    driver.start()
    return driver, actor, adapters.reward_contexts, labels, opp_actor


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_argparser().parse_args(argv)

    # Resolve --opponent HERE, not on the actor thread that first needs it.
    # build_driver_for_actor is called lazily inside ActorLoop, so a typo in a
    # difficulty name would otherwise surface as an actor dying several minutes
    # into a run that has already booted its servers.
    build_training_specs(args.opponent, args.seed, 0, max(1, args.envs_per_actor))

    torch.manual_seed(args.seed)

    run_dir = paths.runs_root() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig()
    ppo_kw = dict(lr=args.lr, horizon_s=args.horizon_s,
                  kl_ref_coef=args.kl_ref_coef,
                  kl_ref_anneal_steps=args.kl_ref_anneal_steps)
    if args.entropy_coef is not None:
        ppo_kw["entropy_coef"] = args.entropy_coef
    if args.critic_lr is not None:
        ppo_kw["critic_lr"] = args.critic_lr
    if args.critic_warmup_updates is not None:
        ppo_kw["critic_warmup_updates"] = args.critic_warmup_updates
    ppo_cfg = PPOConfig(**ppo_kw)
    # Built ONCE and shared by the actors and the anchor evaluator, so the two
    # cannot end up measuring under different reward definitions.
    reward_cfg = reward_config(ppo_cfg.gamma, args.alpha, args.alpha_anneal_steps)
    for _name in ("chunk_len", "burn_in", "minibatch_chunks"):
        _v = getattr(args, _name)
        if _v is not None:
            setattr(ppo_cfg, _name, int(_v))
    if (ppo_cfg.burn_in + ppo_cfg.chunk_len) > (args.rollout_steps - 1):
        raise SystemExit(
            f"--burn-in {ppo_cfg.burn_in} + --chunk-len {ppo_cfg.chunk_len} = "
            f"{ppo_cfg.burn_in + ppo_cfg.chunk_len} does not fit in a rollout "
            f"of {args.rollout_steps - 1} buffer rows. RecurrentRolloutBuffer."
            f"chunk_starts drops any chunk that overruns the buffer, so this "
            f"would train on NOTHING while logging a normal-looking update."
        )
    _rows = args.rollout_steps - 1
    if _rows % ppo_cfg.chunk_len:
        raise SystemExit(
            f"--rollout-steps {args.rollout_steps} gives {_rows} buffer rows, "
            f"which is not a multiple of --chunk-len {ppo_cfg.chunk_len} "
            f"({_rows % ppo_cfg.chunk_len} left over per env). "
            f"RecurrentRolloutBuffer.chunk_starts DROPS a tail that does not "
            f"fill a whole chunk, silently and every epoch, so those rows are "
            f"collected at full cost and never trained on. Use "
            f"--rollout-steps {(_rows // ppo_cfg.chunk_len + 1) * ppo_cfg.chunk_len + 1}."
        )
    log.info("recurrent context: %d burn-in + %d gradient-carrying = %.2f s "
             "at %.0f Hz (%d chunks/minibatch)",
             ppo_cfg.burn_in, ppo_cfg.chunk_len,
             ppo_cfg.chunk_len / ppo_cfg.decision_hz, ppo_cfg.decision_hz,
             ppo_cfg.minibatch_chunks)
    if args.lane_approach is not None:
        reward_cfg.weights.lane_approach = float(args.lane_approach)
        log.info("lane_approach weight overridden to %g (%.3f for the full walk)",
                 args.lane_approach, args.lane_approach * 6.835)
    if args.lane_presence is not None:
        reward_cfg.weights.lane_presence = float(args.lane_presence)
        log.info("lane_presence weight overridden to %g (~%.1f per episode)",
                 args.lane_presence, args.lane_presence * 18000)
    log.info(
        "reward: alpha %s (%s)", args.alpha,
        f"annealed from {reward_cfg.zero_sum_alpha_start} over "
        f"{reward_cfg.zero_sum_anneal_steps} {args.anneal_clock}"
        if reward_cfg.zero_sum_anneal_steps > 0 else "constant",
    )
    policy = LanePolicy(model_cfg).to(args.device)

    # Load the BC prior INTO the policy, and freeze a copy as the KL reference.
    # Until now there was no reachable consumer for a BC checkpoint from this
    # entrypoint at all: the policy was built random, DualClipPPO was handed no
    # reference= so kl_ref_coef was dead code, and --bc-checkpoint only fed the
    # bc_policy ANCHOR (which anchor_launch_spec then rejects, because only
    # 'scripted' anchors can be played by the in-server bot). So every BC run so
    # far trained a prior that RL never saw.
    reference = None
    if args.init_from is not None:
        blob = torch.load(args.init_from, map_location=args.device)
        state = blob.get("policy", blob)
        # The SCREEN heads legitimately will not match a checkpoint trained on
        # the old 9x9 direction space: they went from 9 outputs to 96 and 54.
        # Everything else -- the entity encoder, the core, the button and
        # target heads, the critic -- is unchanged and is most of the prior, so
        # dropping those on the floor and starting from random would throw away
        # the part that works. Shape-mismatched entries are skipped LOUDLY.
        model_sd = policy.state_dict()
        reshaped = [k for k, v in state.items()
                    if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape)]
        loadable = {k: v for k, v in state.items() if k not in reshaped}
        missing, unexpected = policy.load_state_dict(loadable, strict=False)
        unexplained_missing = [k for k in missing if k not in reshaped]
        if unexplained_missing or unexpected:
            # Silently ignoring these is how a "BC-initialised" run ends up
            # random in exactly the heads that matter.
            raise SystemExit(
                f"--init-from {args.init_from} does not match the policy:\n"
                f"  missing={unexplained_missing}\n  unexpected={list(unexpected)}"
            )
        if reshaped:
            print(f"initialised policy from {args.init_from}; "
                  f"{len(reshaped)} tensor(s) LEFT RANDOM because their shape "
                  f"changed: {reshaped}")
        else:
            print(f"initialised policy from {args.init_from}")
        if args.kl_ref_coef > 0.0 and reshaped:
            # A KL anchor to a prior whose action heads are random noise would
            # pull the new heads TOWARD noise. Refuse rather than quietly
            # anchoring to nothing.
            raise SystemExit(
                "--kl-ref-coef > 0 with a checkpoint whose action heads do not "
                f"match ({reshaped}). The reference's screen heads would be "
                "randomly initialised, so the KL term would drag the policy "
                "toward noise. Use --kl-ref-coef 0 for the first run on a new "
                "action space."
            )
        if args.kl_ref_coef > 0.0:
            reference = LanePolicy(model_cfg).to(args.device)
            reference.load_state_dict(state, strict=True)
            reference.eval()
            print(f"KL reference frozen, coef={args.kl_ref_coef}")
    elif args.kl_ref_coef > 0.0:
        raise SystemExit("--kl-ref-coef needs --init-from: there is no prior to "
                         "pull toward, and the term would silently do nothing.")
    learner = DualClipPPO(policy, ppo_cfg, reference=reference,
                          train_step_source=lambda: loop.train_steps.value)

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
        actor_mode=args.actor_mode,
        agent_type=args.agent_type,
        league_dir=args.league_dir,
        opponent_mode=(LEAGUE if args.opponent == LEAGUE else ('scripted' if str(args.opponent).startswith('scripted') else SELF)),
    )

    # Resolved config + seed, written for reproducibility BEFORE anything can
    # fail -- a run that dies on its first update should still leave behind
    # exactly what was asked for.
    resolved = {
        "args": vars(args),
        "run_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(run_cfg).items()},
        "model_config": dataclasses.asdict(model_cfg),
        "ppo_config": dataclasses.asdict(ppo_cfg),
        # The RESOLVED reward, not just the flags that produced it: "alpha was
        # held at 0.5" and "alpha was annealed 0.5 -> 1.0" differ by two fields
        # of this object, and a run dir that does not carry them cannot say
        # afterwards which reward its numbers were measured under.
        "reward_config": dataclasses.asdict(reward_cfg),
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True, default=str))
    log.info("resolved config written to %s", run_dir / "resolved_config.json")

    # -- the frozen-opponent ladder --------------------------------------
    # anchors_for_run RAISES on a missing resource. That is the whole point:
    # the previous behaviour was a warning ("anchor bc_policy has no resource
    # configured; it will be skipped") followed by 40 eval reports that said
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
                adapter_factory_for=lambda: make_lane_adapters(
                        train_step_source=lambda: 0,
                        reward_cfg=reward_cfg),
                envs=args.anchor_envs,
                seed=args.seed,
            ),
            agent_id_fn=lambda: loop.agent_id(),
            # None means "use the config's own default", which is sized for
            # statistical power. Passing args.anchor_episodes unconditionally
            # is how a CLI default of 1 silently overrode that and left the
            # evaluator unable to resolve anything under ~29 CS.
            config=AnchorEvalConfig(
                rotate=not args.anchor_all_per_eval,
                **({} if args.anchor_episodes is None
                   else {"episodes_per_anchor": args.anchor_episodes}),
            ),
        )

    if args.opponent == NO_ENEMY and args.alpha != 0.0:
        log.warning(
            "--opponent none with --alpha %.2f: the zero-sum term subtracts "
            "%.2f x an IDLE champion's reward. That champion does nothing, so "
            "the subtraction is near-constant -- it shifts the return without "
            "expressing any competition, and its variance is pure noise in "
            "every advantage. --alpha 0 is the honest setting for a no-enemy "
            "phase.", args.alpha, args.alpha)

    actor_pool = None
    if args.actor_mode == "process":
        from .procactor import ActorSpec, ProcessActorPool

        actor_pool = ProcessActorPool(
            [
                ActorSpec(
                    actor_id=i,
                    envs_per_actor=args.envs_per_actor,
                    port_base=args.port_base,
                    model_cfg=model_cfg,
                    run_dir=run_dir,
                    device=args.device,
                    reward_cfg=reward_cfg,
                    end_on_death=True if args.end_on_death is None else args.end_on_death,
                    opponent=args.opponent,
                    seed=args.seed,
                    rollout_steps=args.rollout_steps,
                    gamma=ppo_cfg.gamma,
                    gae_lambda=ppo_cfg.gae_lambda,
                    policy_key=SELF,
                    league=(args.opponent == LEAGUE),
                )
                for i in range(args.num_actors)
            ],
            queue_capacity=args.queue_capacity,
        )

    loop = TrainingLoop(
        run_cfg,
        learner,
        evaluator=Evaluator(anchors=anchors),
        anchor_eval=anchor_evaluator,
        gpu=GpuProbe(args.device),
        actor_pool=actor_pool,
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
        driver, actor, reward_contexts, labels, _opp = _build_driver_for_actor(
            actor_idx, args.envs_per_actor, args.port_base, model_cfg, run_dir,
            args.device, loop.train_steps, reward_cfg,
            True if args.end_on_death is None else args.end_on_death,
            args.opponent, args.seed, league=(args.opponent == LEAGUE),
        )
        built_drivers.append(driver)
        return driver, actor, reward_contexts, labels

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
