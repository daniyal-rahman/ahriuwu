"""Actors as PROCESSES, because as threads they cannot use the machine.

Why this exists
===============
``run.ActorLoop`` starts each actor with ``threading.Thread``, so every actor
and the learner share one interpreter and one GIL.  The work an actor does is
``ObservationBuilder.build`` -- pure Python, which *holds* the GIL for its whole
duration and is ~55% of a decision even after the 2026-09-13 cut.  Amdahl on a
0.55 serial fraction caps a threaded design at ``1/0.55 ~= 1.8`` cores, and that
is exactly what was measured (``lanerl/gil_probe.py``):

    1 actor  x 12 envs    0.84 cores of 16
    2 actors x  6 envs    1.13
    4 actors x  3 envs    1.81          <- the predicted ceiling, reached
    4 actors x  6 envs    1.60

Going from 1 to 2 actors burned 35% more CPU and produced LESS throughput: a
GIL convoy, not work.  The control that makes this unambiguous rather than
suggestive: 4 CPU-bound Python threads in one process measure 1.00 cores, the
same as 1.  So 14 of 16 cores were unreachable, and no amount of tuning inside
one interpreter could reach them.

The design
==========
One OS process per actor.  Each child builds its **own** ``VecDriver``, its own
game-server instances on its own disjoint port block, and its own policy -- all
of which the threaded design already did per actor, which is why the seam is
clean.  ``collect(actor_id, payload, version) -> Rollout`` is unchanged; it just
runs somewhere else now.

``spawn``, not ``fork``
-----------------------
The parent has a live CUDA context by the time actors start, and CUDA does not
survive ``fork`` -- a forked child inherits a context it cannot use, and the
failure is a hang or an opaque driver error rather than an exception naming the
cause.  ``spawn`` re-imports and rebuilds in the child, which is why everything
crossing the boundary has to be picklable, which is why :class:`ActorSpec`
exists instead of the closure the threaded path passes around.

What crosses the boundary
-------------------------
* **down**: ``(version, payload, train_step)`` on a per-actor queue, latest
  wins.  The payload is the policy state dict on CPU.
* **up**: a ``Rollout`` whose buffer has been moved to CPU
  (``RecurrentRolloutBuffer.to``).  At T=128, B=24 that is ~15 MB; sent through
  ``torch.multiprocessing``, whose reducer puts CPU tensors in shared memory and
  passes a file descriptor, so it is not 15 MB of pickle per rollout.
* **errors**: the traceback as a STRING.  An exception object may reference
  things that do not pickle (a CUDA tensor, a socket, a driver), and an actor
  whose death is itself unreportable is the failure mode this project has paid
  for most.

``train_step`` comes down with the parameters rather than from a shared counter
because that is the only thing that keeps the env-side anneal consistent with
the weights being acted on -- the same reason ``TrainingLoop`` publishes them
together.
"""
from __future__ import annotations

import logging
import os
import queue as _queue
import signal
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.multiprocessing as mp

log = logging.getLogger("lanerl_train.procactor")


def _use_robust_sharing() -> None:
    """Leave torch's sharing strategy alone unless explicitly overridden.

    This used to force ``file_system``, to fix a rare loss: torch's default
    ``file_descriptor`` strategy unlinks the backing /dev/shm file right after
    passing the fd, and under load that races, throwing inside the child's
    multiprocessing feeder THREAD where it is logged and swallowed. One
    rollout then never arrives and nothing counts it. Measured rate on run
    rl-0913d: 1 in 1,281.

    ``file_system`` traded that for something far worse. With several spawned
    processes, one process's ``resource_tracker`` unlinks a segment another
    still has mapped, ``MapAllocator::close`` throws a c10::Error, and the
    actor dies on SIGABRT::

        terminate called after throwing an instance of 'c10::Error'
          Exception raised from close at ATen/MapAllocator.cpp:545
        actor process lanerl-actor-3 exited with code -6

    Run rl-league-0913 died that way inside five minutes. A rare, non-fatal,
    now-instrumented loss beats a frequent fatal one, so the default is back.

    ``LANERL_SHARING_STRATEGY`` overrides it for anyone who wants to
    experiment, deliberately and with this note in front of them.
    """
    want = os.environ.get("LANERL_SHARING_STRATEGY")
    if not want:
        return
    try:
        mp.set_sharing_strategy(want)
        log.warning("sharing strategy forced to %r via LANERL_SHARING_STRATEGY; "
                    "'file_system' has been observed to abort actors with a "
                    "c10::Error from MapAllocator::close", want)
    except Exception:
        log.warning("could not set sharing strategy %r", want, exc_info=True)


#: How long a child waits for its first parameters before deciding the parent is
#: gone.  Generous: the parent may still be booting its own servers.
FIRST_PARAM_TIMEOUT_S = 600.0


@dataclass
class ActorSpec:
    """Everything a child needs to rebuild an actor from nothing.

    Must be picklable -- under ``spawn`` the child shares no memory with the
    parent, so anything not in here does not exist on the other side.  In
    particular there is no ``train_step_source`` callable: that value arrives
    with each parameter push instead.
    """

    actor_id: int
    envs_per_actor: int
    port_base: int
    model_cfg: Any
    run_dir: Path
    device: str
    reward_cfg: Any
    end_on_death: bool
    opponent: str
    seed: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    policy_key: str
    league: bool = False
    log_level: int = logging.INFO


class _TrainStepBox:
    """A child-local stand-in for ``run.TrainStepCounter``.

    The threaded path hands actors ``loop.train_steps`` directly.  A child
    process cannot hold that object, so it holds this and the parent refreshes
    it on every parameter push.  Same contract: callable, returns an int.
    """

    def __init__(self, value: int = 0) -> None:
        self._value = int(value)

    def __call__(self) -> int:
        return self._value

    def set(self, value: int) -> None:
        # Monotonic for the same reason TrainStepCounter is: a schedule that
        # can go backwards is a schedule that can be replayed.
        if int(value) > self._value:
            self._value = int(value)


def _drain_latest(q: "mp.Queue", block_timeout: Optional[float]) -> Optional[tuple]:
    """The freshest item on ``q``, discarding anything staler behind it.

    Parameters are a LATEST-WINS channel, not a work queue.  Consuming them in
    order would make an actor collect against weights the learner has already
    replaced -- which is precisely the staleness the learner then rejects the
    rollout for.
    """
    item = None
    if block_timeout is not None:
        try:
            item = q.get(timeout=block_timeout)
        except _queue.Empty:
            return None
    while True:
        try:
            item = q.get_nowait()
        except _queue.Empty:
            return item


def _load_opponent(holder: Dict[str, Any], opp: Optional[dict],
                   live_payload: Mapping[str, Any], device: str) -> None:
    """Put the sampled opponent's weights on the red-side policy.

    ``opp`` is ``None`` or ``{"id":..., "path":...}``. A ``None`` path means
    the league drew "latest", so red gets the LIVE weights and the game is an
    exact mirror -- which is what every game was before the league was wired,
    so that path has to stay byte-identical.

    Checkpoints are loaded from disk rather than shipped down the queue: a
    snapshot is ~55 MB and the pool holds up to 30 of them, while the child
    runs on the same node as the file. Cached by id so a repeated draw is free
    -- PFSP deliberately draws the same hard opponent often, so without the
    cache this would reload tens of MB on most updates.
    """
    actor = holder.get("actor")
    if actor is None:
        return
    want = (opp or {}).get("id") or "__latest__"
    path = (opp or {}).get("path")
    if holder.get("loaded_id") == want:
        return
    if not path:
        if live_payload:
            actor.policy.load_state_dict(live_payload["policy"])
        holder["loaded_id"] = want
        # CLEARED, not left: red now carries the live weights, so this really
        # is a mirror. A stale id here would attribute a genuine self-match to
        # whichever checkpoint happened to be loaded last, and feed a 0.5 into
        # that opponent's win rate forever.
        actor.opponent_id = None
        return
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
        sd = blob.get("policy", blob) if isinstance(blob, dict) else blob
        actor.policy.load_state_dict(sd)
        holder["loaded_id"] = want
        # Stamped on the ACTOR because that is what _opponent_of can reach
        # from the driver; without it every league game is attributed to
        # "self" and its result is discarded as a mirror draw.
        actor.opponent_id = want
        log.info("red side is now league opponent %s", want)
    except Exception:
        # Fall back to the live mirror rather than silently keeping whatever
        # weights red happened to have -- an opponent nobody can name is worse
        # than a known one, and it would poison the win-rate table.
        log.error("could not load league opponent %s from %s; using live weights",
                  want, path, exc_info=True)
        if live_payload:
            actor.policy.load_state_dict(live_payload["policy"])
        holder["loaded_id"] = "__latest__"
        actor.opponent_id = None


def _actor_main(
    spec: ActorSpec,
    param_q: "mp.Queue",
    out_q: "mp.Queue",
    err_q: "mp.Queue",
    stop_ev: Any,
) -> None:
    """Child entry point.  Module-level so ``spawn`` can import it by name."""
    built_drivers: List[Any] = []
    sent = {"n": 0}
    try:
        logging.basicConfig(
            level=spec.log_level,
            format=f"%(asctime)s %(levelname)s [actor{spec.actor_id}] %(name)s: %(message)s",
        )
        # Torch defaults to as many intra-op threads as there are cores. With
        # one process per actor that oversubscribes the box by n_actors-fold,
        # and the forward here is tiny -- the whole point is to spend cores on
        # DIFFERENT actors, not on more threads inside one.
        torch.set_num_threads(1)
        _use_robust_sharing()

        # Turn SIGTERM into an exception so the `finally` below actually runs.
        # ProcessActorPool.shutdown escalates to Process.terminate() for a
        # child still busy inside collect(), and SIGTERM's DEFAULT disposition
        # kills the interpreter outright -- no finally, no env.close(), and the
        # actor's game servers survive it holding their ports.
        def _on_sigterm(signum, frame):
            raise SystemExit(f"actor {spec.actor_id} received SIGTERM")

        signal.signal(signal.SIGTERM, _on_sigterm)

        # Imported inside the child: under spawn these pull in torch, CUDA and
        # the whole training stack, and doing it at module scope would pay that
        # cost in the parent and in every unrelated importer of this module.
        from .lane_wiring import make_collect_fn
        from .__main__ import _build_driver_for_actor

        train_step = _TrainStepBox()

        opp_holder: Dict[str, Any] = {"actor": None, "loaded_id": None}

        def build(actor_idx: int):
            built = _build_driver_for_actor(
                actor_idx,
                spec.envs_per_actor,
                spec.port_base,
                spec.model_cfg,
                Path(spec.run_dir),
                spec.device,
                train_step,
                spec.reward_cfg,
                spec.end_on_death,
                spec.opponent,
                spec.seed,
                league=spec.league,
            )
            # built[4] is the red-side policy when running a league. Held here
            # because make_collect_fn only threads through the first four
            # elements, and the weights have to be swapped from the parameter
            # loop below rather than at build time.
            if len(built) > 4:
                opp_holder["actor"] = built[4]
            # The child OWNS these servers, and nothing else can reach them:
            # in the threaded path `__main__` keeps a `built_drivers` list and
            # closes them in its finally, but that list lives in the parent and
            # stays empty here. Without this the game-server subprocesses are
            # orphaned when the actor exits -- still running, still holding
            # their ports, so the NEXT run fails to bind and looks like a
            # server that would not start.
            built_drivers.append(built[0])
            return built

        collect = make_collect_fn(
            build, spec.policy_key, spec.rollout_steps, spec.gamma, spec.gae_lambda
        )

        first = True
        while not stop_ev.is_set():
            got = _drain_latest(param_q, FIRST_PARAM_TIMEOUT_S if first else 0.0)
            if got is None and first:
                raise TimeoutError(
                    f"actor {spec.actor_id} waited {FIRST_PARAM_TIMEOUT_S:.0f}s for its "
                    f"first parameters and got none; the parent never published or died"
                )
            if got is not None:
                version, payload, step, opp = got
                train_step.set(step)
                first = False
                _load_opponent(opp_holder, opp, payload, spec.device)
            if first:
                continue
            rollout = collect(spec.actor_id, payload, version)
            if rollout is None:
                continue
            rollout.actor_id = spec.actor_id
            rollout.param_version = version
            # A per-actor sequence number, so the PARENT can tell a rollout
            # that was never sent from one that was sent and vanished. The
            # shared-memory send can fail inside the feeder thread, which logs
            # and swallows it -- observed once in 1,281 on run rl-0913d, with
            # nothing anywhere counting the loss. A gap in this sequence is the
            # only way to see it.
            sent["n"] += 1
            rollout.actor_seq = sent["n"]
            # To CPU before it crosses: the parent has its own CUDA context and
            # cannot receive a tensor living in this one.
            if rollout.data is not None and hasattr(rollout.data, "to"):
                rollout.data.to("cpu")
            out_q.put(rollout)
    except BaseException as exc:  # noqa: BLE001 - deliberately everything
        # An ORDERLY stop is not a death. shutdown() sets stop_event and then
        # escalates to SIGTERM for a child parked in a blocking put() or socket
        # read; reporting that on the error queue would make every clean
        # shutdown raise ActorFailure in the parent and turn a finished run
        # into a failed one.
        orderly = isinstance(exc, SystemExit) and stop_ev.is_set()
        if orderly:
            log.info("actor %d stopping: %s", spec.actor_id, exc)
        else:
            # As a STRING. A pickled exception can drag a socket or a CUDA
            # tensor along with it and fail to cross, turning a reportable
            # death into a silent one.
            try:
                err_q.put(
                    (spec.actor_id, f"{type(exc).__name__}: {exc}", traceback.format_exc())
                )
            except Exception:
                pass
            log.error("actor %d died: %s", spec.actor_id, exc, exc_info=True)
            try:
                out_q.put_nowait(None)  # wake a learner blocked on get()
            except Exception:
                pass
    finally:
        # On EVERY exit path, including the stop_event one. A terminated actor
        # that leaves its servers running holds its whole port block, and the
        # next run's failure to bind shows up as "the server would not start".
        for driver in built_drivers:
            try:
                driver.env.close()
            except Exception:
                log.error("actor %d: error closing its server instances",
                          spec.actor_id, exc_info=True)


class ProcessActorPool:
    """The parent-side handle on a set of actor processes.

    Mirrors what ``TrainingLoop`` expects of ``ActorLoop``: it is started once,
    it publishes parameters, rollouts arrive on a queue, and an actor death
    surfaces rather than hanging the run.
    """

    def __init__(
        self,
        specs: List[ActorSpec],
        queue_capacity: int = 2,
        ctx: Optional[Any] = None,
    ) -> None:
        # spawn: see the module docstring -- the parent's CUDA context does not
        # survive fork.
        _use_robust_sharing()
        self.ctx = ctx or mp.get_context("spawn")
        self.specs = list(specs)
        self.out_queue: "mp.Queue" = self.ctx.Queue(maxsize=max(1, queue_capacity))
        self.error_queue: "mp.Queue" = self.ctx.Queue()
        self.stop_event = self.ctx.Event()
        self.param_queues: List["mp.Queue"] = [self.ctx.Queue(maxsize=4) for _ in specs]
        self.procs: List[Any] = []

    def start(self) -> None:
        for spec, pq in zip(self.specs, self.param_queues):
            p = self.ctx.Process(
                target=_actor_main,
                args=(spec, pq, self.out_queue, self.error_queue, self.stop_event),
                name=f"lanerl-actor-{spec.actor_id}",
                daemon=True,
            )
            p.start()
            self.procs.append(p)
        log.info("started %d actor PROCESSES (pids %s)",
                 len(self.procs), [p.pid for p in self.procs])

    def publish(self, version: int, payload: Mapping[str, Any], train_step: int,
                opponents: Optional[List[Optional[dict]]] = None) -> None:
        """Push parameters to every actor, newest-wins, never blocking.

        A full queue means that actor has not picked up the last few versions;
        dropping the oldest is right, because it is the STALEST and would only
        produce a rollout the learner then rejects.
        """
        cpu = {k: _to_cpu(v) for k, v in dict(payload).items()}
        for i, pq in enumerate(self.param_queues):
            # One opponent PER ACTOR, so the league mixture is realised across
            # actors rather than every actor playing the same draw.
            opp = opponents[i] if opponents and i < len(opponents) else None
            while True:
                try:
                    pq.put_nowait((version, cpu, int(train_step), opp))
                    break
                except _queue.Full:
                    try:
                        pq.get_nowait()
                    except _queue.Empty:
                        break

    def stop(self) -> None:
        """Ask the actors to stop, without waiting.

        Separate from :meth:`shutdown` so the parent can signal first and drain
        the rollout queue second: an actor parked in a blocking ``put()`` on a
        full queue cannot see the stop flag until something makes room, and
        draining before signalling just lets it refill and block again. Setting
        the flag first turns a 30-second join timeout into an immediate exit.
        """
        self.stop_event.set()

    def raise_if_any_died(self) -> None:
        if self.stop_event.is_set():
            # We asked them to stop. A child that exits from SIGTERM leaves a
            # non-zero exit code behind, and reporting that as a death would
            # turn every clean shutdown into a failed run.
            return
        try:
            actor_id, msg, tb = self.error_queue.get_nowait()
        except _queue.Empty:
            # An actor can also die without reporting -- SIGKILL, an OOM, a
            # segfault in the CUDA driver. exitcode is the only witness then.
            for p in self.procs:
                if p.exitcode not in (None, 0):
                    raise RuntimeError(
                        f"actor process {p.name} exited with code {p.exitcode} and "
                        f"reported nothing. That is a kill or a native crash, not a "
                        f"Python exception -- check the actor's own log in run_dir."
                    )
            return
        raise RuntimeError(f"actor {actor_id} died: {msg}\n{tb}")

    def shutdown(self, timeout: float = 30.0) -> None:
        self.stop_event.set()
        for p in self.procs:
            p.join(timeout=timeout)
        for p in self.procs:
            if p.is_alive():
                log.warning("actor %s did not exit in %.0fs; terminating", p.name, timeout)
                p.terminate()
                p.join(timeout=10.0)
                if p.is_alive():
                    p.kill()


def _to_cpu(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().to("cpu")
    if isinstance(v, dict):
        return {k: _to_cpu(x) for k, x in v.items()}
    return v
