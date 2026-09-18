"""Deterministic in-process sharding for the Tier-1 one-step differential.

The one-step harness resets from the oracle at every pair, so pair ranges are
independent after the deterministic wave-state replay.  Threads are used
instead of forked Python workers: forking a live JAX runtime is unsupported,
while separate spawned processes would each compile and allocate another copy
of the simulator.  Threads share the one JIT cache and accelerator allocation.

This primarily overlaps Python injection/matching with JAX dispatch.  It is
not expected to scale linearly when ``tick`` itself saturates the accelerator;
the returned timings make that visible rather than promising a speedup.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .one_step import (
    OneStepResult,
    merge_one_step_results,
    run_one_step_differential,
)

__all__ = [
    "ParallelProgress", "ParallelRunStats",
    "partition_pair_ranges", "run_parallel_one_step_differential",
]


@dataclass(frozen=True, slots=True)
class ParallelProgress:
    attempted_pairs: int
    total_pairs: int
    elapsed_s: float
    pairs_per_s: float


@dataclass(frozen=True, slots=True)
class ParallelRunStats:
    workers: int
    pair_ranges: Tuple[Tuple[int, int], ...]
    attempted_pairs: int
    compared_pairs: int
    skipped_pairs: int
    elapsed_s: float

    @property
    def attempted_pairs_per_s(self) -> float:
        return self.attempted_pairs / self.elapsed_s if self.elapsed_s else math.inf

    @property
    def compared_pairs_per_s(self) -> float:
        return self.compared_pairs / self.elapsed_s if self.elapsed_s else math.inf


def partition_pair_ranges(n_pairs: int, workers: int) -> List[Tuple[int, int]]:
    """Return balanced, contiguous, non-empty ``[start, stop)`` ranges."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if n_pairs < 0:
        raise ValueError("n_pairs must be >= 0")
    workers = min(workers, n_pairs)
    if workers == 0:
        return []
    q, r = divmod(n_pairs, workers)
    out = []
    start = 0
    for worker in range(workers):
        width = q + int(worker < r)
        out.append((start, start + width))
        start += width
    return out


def run_parallel_one_step_differential(
    trace,
    patch=None,
    max_pairs: Optional[int] = None,
    *,
    action_log=None,
    route_table=None,
    terrain=None,
    workers: int = 2,
    progress_every: int = 250,
    progress_callback: Optional[Callable[[ParallelProgress], None]] = None,
) -> Tuple[OneStepResult, ParallelRunStats]:
    """Evaluate independent pair shards and merge in original trace order.

    The result has the same counters and ordered error samples as
    :func:`run_one_step_differential`.  ``progress_callback`` may be called by
    worker threads and therefore must itself be thread-safe.
    """
    n_pairs = max(0, len(trace) - 1)
    if max_pairs is not None:
        n_pairs = min(n_pairs, max(0, max_pairs))
    ranges = partition_pair_ranges(n_pairs, workers)
    started = time.perf_counter()

    if not ranges:
        elapsed = time.perf_counter() - started
        empty = OneStepResult()
        return empty, ParallelRunStats(
            workers=0, pair_ranges=(), attempted_pairs=0, compared_pairs=0,
            skipped_pairs=0, elapsed_s=elapsed)

    lock = threading.Lock()
    shard_done = [0] * len(ranges)

    def worker_progress(shard: int, done: int, _total: int) -> None:
        with lock:
            shard_done[shard] = done
            aggregate_done = sum(shard_done)
            elapsed = time.perf_counter() - started
            if progress_callback is not None:
                progress_callback(ParallelProgress(
                    attempted_pairs=aggregate_done,
                    total_pairs=n_pairs,
                    elapsed_s=elapsed,
                    pairs_per_s=(aggregate_done / elapsed if elapsed else math.inf),
                ))

    def run_shard(shard: int, bounds: Tuple[int, int]) -> OneStepResult:
        start, stop = bounds
        return run_one_step_differential(
            trace, patch=patch, max_pairs=max_pairs,
            pair_start=start, pair_stop=stop,
            progress_every=progress_every,
            progress_callback=lambda done, total: worker_progress(
                shard, done, total),
            action_log=action_log,
            route_table=route_table,
            terrain=terrain,
        )

    if len(ranges) == 1:
        parts = [run_shard(0, ranges[0])]
    else:
        # executor.map preserves input order, which preserves serial ordering
        # of FieldStats.errors. merge_one_step_results also sorts the one
        # explicitly timestamped series as a defensive check.
        with ThreadPoolExecutor(
                max_workers=len(ranges), thread_name_prefix="one-step") as pool:
            parts = list(pool.map(
                lambda item: run_shard(item[0], item[1]), enumerate(ranges)))

    result = merge_one_step_results(parts)
    elapsed = time.perf_counter() - started
    stats = ParallelRunStats(
        workers=len(ranges), pair_ranges=tuple(ranges),
        attempted_pairs=n_pairs, compared_pairs=result.n_ticks,
        skipped_pairs=result.n_ticks_skipped, elapsed_s=elapsed,
    )
    return result, stats
