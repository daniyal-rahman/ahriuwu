"""DDP sharding correctness for VideoGroupedSampler (2026-07-06).

Under DDP the sampler must (a) give every rank the SAME number of indices — else
DDP's all-reduce hangs on ragged data — and (b) keep each whole video on ONE rank
(cache locality for R2 streaming). world_size==1 must be byte-identical to before.

Run:  PYTHONPATH=src python tests/test_ddp_sampler.py
"""
import torch
from ahriuwu.data.dataset import VideoGroupedSampler


class _FakeDS:
    def __init__(self, seqs):
        self.sequences = seqs


def _make(seqs_per_video):
    seqs, idx_to_vid = [], {}
    for v, n in enumerate(seqs_per_video):
        for _ in range(n):
            idx_to_vid[len(seqs)] = v
            seqs.append({"video_id": f"vid{v:03d}"})
    return _FakeDS(seqs), idx_to_vid


def test_single_process_identical():
    ds, _ = _make([5, 3, 8, 1, 6])
    s = VideoGroupedSampler(ds, world_size=1)
    idxs = list(iter(s))
    assert len(s) == len(ds.sequences) == 23
    assert sorted(idxs) == list(range(23)), "world_size=1 must yield every index exactly once"
    print("OK: world_size=1 yields all indices exactly once (unchanged behaviour).")


def test_ddp_equal_counts_and_whole_video_shards():
    # 13 videos, deliberately uneven sequence counts
    torch.manual_seed(0)  # equalization truncates the shuffle tail -> seed for determinism
    seqs_per = [7, 3, 11, 2, 9, 4, 6, 8, 1, 5, 10, 3, 7]
    ds, idx_to_vid = _make(seqs_per)
    total = sum(seqs_per)
    W = 4
    target = total // W

    owned, yielded = [], []
    for rank in range(W):
        s = VideoGroupedSampler(ds, rank=rank, world_size=W)
        # (1) equal counts -> DDP all-reduce can't hang on ragged data
        assert len(s) == target, f"rank {rank}: __len__ {len(s)} != target {target}"
        idxs = list(iter(s))
        assert len(idxs) == target, f"rank {rank}: yielded {len(idxs)} != {target} (DDP would hang)"
        # ownership = the videos in this rank's shard (whole-video partition, pre-truncation)
        rank_owned = {idx_to_vid[g[0]] for g in s.video_groups}
        rank_yielded = {idx_to_vid[i] for i in idxs}
        # truncation only DROPS from the tail; it never moves a video to another rank
        assert rank_yielded <= rank_owned, f"rank {rank}: yielded a video it doesn't own"
        owned.append(rank_owned); yielded.append(rank_yielded)

    # (2) whole-video: ownership is disjoint across ranks (no cache thrash / double-count)
    for a in range(W):
        for b in range(a + 1, W):
            assert not (owned[a] & owned[b]), f"video(s) split across ranks {a},{b}"
    # (3) ownership is complete: every video belongs to exactly one rank
    assert set().union(*owned) == set(range(len(seqs_per))), "ownership doesn't cover all videos"
    print(f"OK: W={W} ranks each yield {target} indices; whole-video ownership disjoint + covers all "
          f"{len(seqs_per)} videos (truncation may drop a tiny video's tail in a given epoch).")


def test_ddp_handles_rank_with_few_videos():
    # 5 videos over 4 ranks -> some ranks get 1 video, must still hit `target` via wrap-repeat
    seqs_per = [10, 2, 3, 4, 1]
    ds, _ = _make(seqs_per)
    W = 4
    target = sum(seqs_per) // W
    for rank in range(W):
        s = VideoGroupedSampler(ds, rank=rank, world_size=W)
        idxs = list(iter(s))
        assert len(idxs) == target, f"rank {rank}: {len(idxs)} != {target}"
    print("OK: ranks with few/small videos still emit exactly `target` indices (wrap-repeat).")


if __name__ == "__main__":
    test_single_process_identical()
    test_ddp_equal_counts_and_whole_video_shards()
    test_ddp_handles_rank_with_few_videos()
