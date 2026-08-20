"""Replay-data dataset: latents + memory-derived actions + computed rewards.

Each match contributes three on-disk artifacts produced by
``scripts/aggregation/pipeline.py``:

* Packed latents (``<latents_dir>/<match_id>.pt``) — same format the YT
  pretokenize pipeline emits.
* ``labels.json`` — per-frame memory snapshots (gold/level/hp, action,
  visible_heroes, movement).
* ``clicks.json`` — gt-tagged click and cast events (used for binary
  ability-press flags; one frame per cast).

Contract with the pretokenize step (NOT runtime-checked):
  ``frame_indices`` MUST be sorted strictly ascending (no duplicates,
  no permutation). The slice-based latent loading in ``__getitem__``
  assumes ``frame_to_idx[run_start + k] == frame_to_idx[run_start] + k``.
  Pretokenize writes in PNG-number order, so this holds for any pack
  produced by our own tooling.

The dataset emits per sequence:

* ``latents``: (T, C, H, W)
* ``actions``: ``{movement: (T, 2), Q W E R D F item B: (T,) long}``
* ``rewards``: (T,)

Action mapping
--------------
* **Q/W/E/R** — set on the frame matching the cast's ``game_t`` for any
  cast event whose ``slot`` is one of those letters.
* **D/F** — set the same way; if ``labels.summoner_slots`` is present we
  trust it; otherwise we fall back to the cast's ``slot`` field.
* **B** — set on cast events with ``slot == "B"`` (the recall stream
  pipeline already emits). Falls back to ``label.action.type == "recall"``
  if no cast event lands on that frame.
* **item** — not tracked by the pipeline; constant 0.
* **movement** (x, y) — screen-space target of the most recent movement CLICK
  (``clicks.json:clicks``), projected with that click's own frame camera and
  held forward until the next click. See :meth:`_parse_movement_clicks`.
* **movement_event** — bool, True exactly on the frames where a new click
  landed. This is the gate target for the sticky-categorical movement head.

Rewards come from :func:`ahriuwu.rewards.reward.compute_episode_reward`.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import ABILITY_KEYS
from ..rewards.reward import RewardConfig, compute_episode_reward
from .lane_opponent import resolve_lane_opponent

# Aux-state supervision targets (all in [0,1]); order is the StateHead output order.
STATE_TARGETS = ("own_hp_frac", "own_level", "enemy_hp_frac", "enemy_visible")

# Garen v1 action mapping: clicks.json cast spell_name (or label.action.spell)
# -> action key. TP / super-recall / any unmapped spell are intentionally ignored.
_SPELL_TO_KEY = {
    "GarenQ": "Q", "GarenW": "W", "GarenR": "R",
    # GarenE and GarenECancel are inconsistent per-match aliases for "E used"
    # (verified mutually exclusive across matches) -> both map to E.
    "GarenE": "E", "GarenECancel": "E",
    "SummonerFlash": "Flash", "SummonerDot": "Ignite",
    "recall": "Recall",
}
_STRIDE_ITEM_ID = 6631  # Stridebreaker — the one item-active the labels log (sparse).

MOVEMENT_SOURCES = ("clicks", "cursor")

# ── Screen projection (mirrors scripts/aggregation/pipeline.py:project) ──
# Constants are read per-match from labels["projection"] where present; these
# are the values the shipped corpus was built with.
_FLOOR_Y = 52.0            # ground-plane height; not written to labels.json
_DEFAULT_PROJ = {"fov_v_deg": 40.0, "tilt_deg": 56.0, "cam_y": 1912.0}
_VZ_MIN = 10.0             # project() rejects points at/behind this depth


class _Projection:
    """The pipeline's fixed-orientation ground-plane camera model.

    ``project`` reproduces ``pipeline.project`` (minus the viewport clamp);
    ``invert`` recovers the camera (cx, cz) from one world->screen pair.

    Why the inversion is exact even though ``cy`` is unobservable: writing
    ``dy = FLOOR_Y - cy``, ``dx = wx - cx``, ``dz = wz - cz``, both screen
    coordinates depend only on the ratios ``dx/vz`` and ``vy/vz`` where
    ``vy, vz`` are linear in ``(dy, dz)``. Scaling ``(dy, dx, dz)`` by a common
    alpha leaves both ratios unchanged, so ANY assumed ``cy`` yields a camera
    that reproduces ``project()`` exactly for every point on the floor plane.
    Verified empirically: reprojecting each frame's other heroes with the
    recovered camera reproduces the pipeline's stored ``visible_heroes[*].screen``
    to within 1 px (pure int-truncation), on every match sampled.
    """

    __slots__ = ("tan_h", "tan_v", "cos_t", "sin_t", "dy", "w", "h")

    def __init__(self, labels: dict):
        proj = labels.get("projection") or {}
        w, h = labels["screen_resolution"]
        self.w, self.h = float(w), float(h)
        fov_v = math.radians(float(proj.get("fov_v_deg", _DEFAULT_PROJ["fov_v_deg"])))
        tilt = math.radians(float(proj.get("tilt_deg", _DEFAULT_PROJ["tilt_deg"])))
        cam_y = float(proj.get("cam_y", _DEFAULT_PROJ["cam_y"]))
        fov_h = 2 * math.atan(math.tan(fov_v / 2) * self.w / self.h)
        self.tan_h, self.tan_v = math.tan(fov_h / 2), math.tan(fov_v / 2)
        self.cos_t, self.sin_t = math.cos(tilt), math.sin(tilt)
        self.dy = _FLOOR_Y - cam_y

    def invert(self, wx: float, wz: float, px: float, py: float) -> tuple[float, float]:
        """(cx, cz) such that ``project(wx, wz, cx, cz) == (px, py)``.

        ``px, py`` are in pixels; pass the pixel CENTRE (+0.5) since
        ``pipeline.project`` int-truncates. Always solvable: the denominator
        ``kv*cos - sin`` is bounded away from 0 for any on-screen ``py``.
        """
        sx, sy = px / self.w, py / self.h
        kv = (0.5 - sy) * 2 * self.tan_v
        den = kv * self.cos_t - self.sin_t
        dz = self.dy * (self.cos_t + kv * self.sin_t) / den
        vz = -self.dy * self.sin_t + dz * self.cos_t
        dx = (sx - 0.5) * 2 * self.tan_h * vz
        return wx - dx, wz - dz

    def project_norm(self, wx: float, wz: float, cx: float, cz: float) -> tuple[float, float]:
        """World -> screen in NORMALIZED [0,1]-ish coords, NOT clamped.

        Unlike ``pipeline.project`` this never returns None: off-viewport
        commands (edge-of-screen walks, minimap clicks — 24% of frames in the
        shipped labels, audit finding 2) keep a real, signed coordinate instead
        of being thrown away. Depth is floored at ``_VZ_MIN`` so a point behind
        the camera degrades to a far-edge direction rather than a sign flip.
        """
        dx, dz = wx - cx, wz - cz
        vy = self.dy * self.cos_t + dz * self.sin_t
        vz = -self.dy * self.sin_t + dz * self.cos_t
        if vz < _VZ_MIN:
            vz = _VZ_MIN
        return (0.5 + (dx / vz) / self.tan_h * 0.5,
                0.5 - (vy / vz) / self.tan_v * 0.5)


def load_outcomes(manifest_path: Path | str) -> dict[str, bool]:
    """Read ``garen_win`` per ``match_id`` from a manifest JSON.

    Raises if any match entry is missing ``garen_win`` — silently defaulting
    a missing outcome to False would flip the sign of the terminal reward.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    out: dict[str, bool] = {}
    missing: list[str] = []
    for m in manifest.get("matches", []):
        mid = m.get("match_id")
        if mid is None:
            continue
        if "garen_win" not in m:
            missing.append(mid)
            continue
        out[mid] = bool(m["garen_win"])
    if missing:
        raise ValueError(
            f"manifest at {manifest_path} is missing `garen_win` for "
            f"{len(missing)} match(es): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return out


class ReplayLatentSequenceDataset(Dataset):
    """Sequences of (latents, actions, rewards) drawn from replay matches."""

    def __init__(
        self,
        latents_dir: Path | str,
        labels_root: Path | str,
        outcomes: dict[str, bool] | None = None,
        manifest_path: Path | str | None = None,
        sequence_length: int = 64,
        stride: int = 1,
        reward_config: RewardConfig | None = None,
        max_cache_size: int = 2,
        cache_path: str | Path | None = None,
        movement_source: str = "clicks",
    ):
        # max_cache_size is per-worker — DataLoader fork-spawns each worker
        # with its own cache copy. With VideoShuffleSampler the access
        # pattern is roughly linear within each video, so a 2-deep LRU
        # gives near-100% hit rate without ballooning RAM at high
        # num_workers (each .pt file is ~210MB).
        if outcomes is None and manifest_path is None:
            raise ValueError("Provide either `outcomes` dict or `manifest_path`")
        if movement_source not in MOVEMENT_SOURCES:
            raise ValueError(
                f"movement_source={movement_source!r} not in {MOVEMENT_SOURCES}")
        self.latents_dir = Path(latents_dir)
        self.labels_root = Path(labels_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.reward_config = reward_config or RewardConfig()
        self.max_cache_size = max_cache_size
        self.movement_source = movement_source
        # Matches that asked for click-events but had no clicks.json and fell
        # back to the legacy (drift-contaminated) cursor target. Reported at
        # the end of _index so a run never silently mixes the two.
        self.click_fallback_matches: list[str] = []

        self.outcomes: dict[str, bool] = (
            dict(outcomes) if outcomes is not None else load_outcomes(manifest_path)
        )

        # LRU cache for packed latent arrays (heavy)
        self._latent_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._latent_cache_order: list[str] = []

        # Per-match parsed actions + rewards (light, kept for all loaded matches)
        self.match_data: dict[str, dict] = {}

        # Precomputed sequence index built at construction time
        self.sequences: list[dict] = []

        # Dumb optional cache of the expensive index (label/reward parse + the
        # per-.pt frame_indices reads, ~tens of GB). Keyed only by
        # (latents_dir, seq_len, stride, #matches); rebuild if any differ. No
        # auto-discovery/invalidation by design — delete the file to force rebuild.
        if not self._load_index_cache(cache_path):
            self._index()
            self._save_index_cache(cache_path)

    # ─────────────────────── index cache (dumb) ───────────────────────

    def _cache_meta(self) -> dict:
        # "schema" invalidates caches when _parse_match's output shape changes
        # (2 = added aux state targets); bump it on any md-dict change.
        #   3 = cursor dead-band denoise in _parse_movement
        #   4 = click-event movement target + movement_event; enemy_visible
        #       gated on screen!=None; GarenQAttack counted as an auto-attack
        return {"latents_dir": str(self.latents_dir),
                "seq_len": self.sequence_length, "stride": self.stride,
                "movement_source": self.movement_source,
                "matches": self._match_fingerprint(),
                "schema": 4}

    def _match_fingerprint(self) -> str:
        """Cheap fingerprint of WHICH latent packs are in latents_dir.

        The index cache used to key only on (dir, seq_len, stride, schema), so
        adding/removing a .pt silently reused a stale index (audit R3). Names
        only — no file loading, so this costs one directory listing.
        """
        stems = sorted({p.stem for p in self.latents_dir.glob("*.pt")
                        if p.stem != "index"}
                       | {p.stem for p in self.latents_dir.glob("*.npz")})
        # hashlib, NOT builtin hash(): str hashing is salted per process, which
        # would make the cache key differ on every run and never hit.
        h = hashlib.sha1("\n".join(stems).encode()).hexdigest()[:16]
        return f"{len(stems)}:{h}"

    def _load_index_cache(self, cache_path) -> bool:
        """Load match_data + sequences from a prior build. Returns True on a
        usable hit (skips the whole ~expensive _index)."""
        if not cache_path or not Path(cache_path).exists():
            return False
        try:
            c = torch.load(cache_path, weights_only=False)
        except Exception as e:  # corrupt/partial cache -> just rebuild
            warnings.warn(f"dataset cache read failed ({e}); rebuilding")
            return False
        if c.get("meta") != self._cache_meta():
            print(f"  [dscache] {cache_path}: params differ from request -> rebuild")
            return False
        self.match_data = c["match_data"]
        self.sequences = c["sequences"]
        self.click_fallback_matches = [
            m for m, md in self.match_data.items() if not md.get("movement_from_clicks")
        ] if self.movement_source == "clicks" else []
        print(f"  [dscache] HIT {cache_path}: {len(self.match_data)} matches, "
              f"{len(self.sequences)} sequences (skipped label/reward parse)")
        if self.click_fallback_matches:
            print(f"  [movement] WARNING: {len(self.click_fallback_matches)} matches "
                  f"use the LEGACY cursor target (no clicks.json): "
                  f"{sorted(self.click_fallback_matches)}")
        return True

    def _save_index_cache(self, cache_path) -> None:
        if not cache_path:
            return
        try:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            torch.save({"meta": self._cache_meta(), "match_data": self.match_data,
                        "sequences": self.sequences}, tmp)
            tmp.replace(p)
            print(f"  [dscache] wrote {cache_path} ({len(self.sequences)} sequences)")
        except Exception as e:  # cache is best-effort; never fail the run over it
            warnings.warn(f"dataset cache write failed ({e}); continuing")

    # ───────────────────────── indexing ─────────────────────────

    def _index(self) -> None:
        """Walk available matches, parse labels/clicks once, build sequence list."""
        index_path = self.latents_dir / "index.pt"
        if index_path.exists():
            index = torch.load(index_path, weights_only=True)
            match_ids = list(index.keys())
            frame_indices_by_match = {mid: index[mid].numpy() for mid in match_ids}
        else:
            match_ids = []
            frame_indices_by_match = {}
            seen: set[str] = set()
            # .pt is preferred (raw tensors); .npz only consulted for matches
            # that don't have a .pt next to them.
            for path in sorted(self.latents_dir.glob("*.pt")):
                mid = path.stem
                if mid == "index" or mid in seen:
                    continue
                data = torch.load(path, weights_only=True)
                frame_indices_by_match[mid] = data["frame_indices"].numpy()
                match_ids.append(mid)
                seen.add(mid)
            for path in sorted(self.latents_dir.glob("*.npz")):
                mid = path.stem
                if mid in seen:
                    continue
                with np.load(path) as data:
                    frame_indices_by_match[mid] = data["frame_indices"].copy()
                match_ids.append(mid)
                seen.add(mid)

        skipped: list[tuple[str, str]] = []
        for mid in match_ids:
            if mid not in self.outcomes:
                skipped.append((mid, "not in outcomes manifest"))
                continue
            labels_path = self.labels_root / mid / "labels.json"
            if not labels_path.exists():
                skipped.append((mid, "missing labels.json"))
                continue

            md = self._parse_match(mid, labels_path)
            if md is None:
                skipped.append((mid, "labels parse returned no frames"))
                continue
            n_latent = len(frame_indices_by_match[mid])
            n_label = md["frame_count"]
            if n_latent != n_label:
                warnings.warn(
                    f"{mid}: latent frame count ({n_latent}) != label frame count "
                    f"({n_label}); using min({n_latent}, {n_label}) and dropping the rest"
                )
            self.match_data[mid] = md
            self._index_match(mid, frame_indices_by_match[mid], n_label)

        n_matches = len(self.match_data)
        n_seqs = len(self.sequences)
        print(f"ReplayLatentSequenceDataset: {n_matches} matches, {n_seqs} sequences "
              f"(seq_len={self.sequence_length}, stride={self.stride}, "
              f"movement_source={self.movement_source})")
        if self.click_fallback_matches:
            print(f"  [movement] WARNING: {len(self.click_fallback_matches)}/{n_matches} "
                  f"matches have no clicks.json and fell back to the LEGACY "
                  f"cursor target: {sorted(self.click_fallback_matches)}")
        for mid, why in skipped[:5]:
            print(f"  skipped {mid}: {why}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more skipped")

    def _index_match(
        self,
        match_id: str,
        frame_indices: np.ndarray,
        n_match_frames: int,
    ) -> None:
        """Append (start_frame, start_idx) entries for all valid windows in a match."""
        if len(frame_indices) < self.sequence_length:
            return
        usable = min(len(frame_indices), n_match_frames)
        frame_to_idx = {int(frame_indices[i]): i for i in range(len(frame_indices))}
        frame_nums = sorted(int(f) for f in frame_indices[:usable])

        # Walk contiguous runs and emit windows.
        run_start = frame_nums[0]
        run_len = 1
        for i in range(1, len(frame_nums)):
            if frame_nums[i] == frame_nums[i - 1] + 1:
                run_len += 1
            else:
                self._emit_windows(match_id, run_start, run_len, frame_to_idx)
                run_start = frame_nums[i]
                run_len = 1
        self._emit_windows(match_id, run_start, run_len, frame_to_idx)

    def _emit_windows(
        self,
        match_id: str,
        run_start: int,
        run_len: int,
        frame_to_idx: dict[int, int],
    ) -> None:
        if run_len < self.sequence_length:
            return
        for off in range(0, run_len - self.sequence_length + 1, self.stride):
            start_frame = run_start + off
            self.sequences.append({
                "video_id": match_id,
                "start_frame": start_frame,
                "start_idx": frame_to_idx[start_frame],
            })

    # ───────────────────────── per-match parsing ─────────────────────────

    def _parse_match(self, match_id: str, labels_path: Path) -> Optional[dict]:
        with open(labels_path) as f:
            labels = json.load(f)
        frames = labels.get("frames") or []
        T = len(frames)
        if T == 0:
            return None

        # Pipeline invariant: labels.frames[i].frame == i for every i. We rely
        # on this so that start_frame indexes both label-derived tensors and
        # latent frame numbers consistently. Hard-fail if it's violated.
        for i in range(min(T, 64)):  # spot-check the first 64 to avoid full O(T) cost
            f_idx = frames[i].get("frame")
            if f_idx is not None and f_idx != i:
                raise ValueError(
                    f"{match_id}: labels.frames[{i}].frame={f_idx!r} (expected {i}). "
                    "Pipeline invariant violated; dataset slicing assumes 1:1."
                )

        garen_won = self.outcomes[match_id]
        rewards = compute_episode_reward(labels, garen_won, self.reward_config)

        # clicks.json is read once and shared by the movement + ability parsers.
        clicks_path = labels_path.parent / "clicks.json"
        click_doc: Optional[dict] = None
        if clicks_path.exists():
            with open(clicks_path) as f:
                click_doc = json.load(f)

        movement, movement_event, from_clicks = self._parse_movement(
            match_id, labels, frames, click_doc)
        if self.movement_source == "clicks" and not from_clicks:
            self.click_fallback_matches.append(match_id)
        abilities = self._parse_abilities(labels, frames, match_id, click_doc)
        state, state_mask = self._parse_state(labels, frames)

        return {
            "rewards": rewards,
            "movement": movement,
            "movement_event": movement_event,
            "movement_from_clicks": from_clicks,
            "abilities": abilities,
            "state": state,
            "state_mask": state_mask,
            "frame_count": T,
        }

    def _parse_state(self, labels: dict, frames: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-frame aux-state targets (T, len(STATE_TARGETS)) + validity mask.

        own_hp_frac / own_level from champion_stats; enemy_hp_frac from the
        resolved lane opponent's visible_heroes entry (masked while unseen);
        enemy_visible is 1/0 whenever an opponent resolves at all. Unlabeled /
        placeholder frames (YT) mask everything — the aux loss sees no signal.

        NOTE on ``visible_heroes``: despite the name it is a MEMORY list, not a
        visibility list — ``pipeline.py`` deliberately includes every hero so
        off-screen stats stay readable, and only ``screen`` is None for them.
        Keying ``enemy_visible`` on mere membership made the target 1 on
        1,052,748 frames (29.6% of all frames, 54.5% of the positives) where
        the enemy is nowhere in the input, and unmasked ``enemy_hp_frac`` on
        them too — an unlearnable residual driven into the shared agent tokens
        (audit finding 4). Both are now gated on ``screen is not None``.

        Residual, NOT fixed here: ``screen is not None`` means "inside the view
        frustum", not "rendered" — fog-of-war units still get coordinates
        (audit finding 15). Closing that needs a per-hero fog flag that
        ``pipeline.py`` does not currently read from memory.
        """
        T = len(frames)
        S = len(STATE_TARGETS)
        state = torch.zeros((T, S), dtype=torch.float32)
        mask = torch.zeros((T, S), dtype=torch.float32)
        try:
            opp = resolve_lane_opponent(labels)
        except Exception:
            opp = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                continue
            cs = lab.get("champion_stats") or {}
            hp, hpm, lvl = cs.get("hp"), cs.get("hp_max"), cs.get("level")
            if hp is not None and hpm:
                state[i, 0] = float(hp) / float(hpm)
                mask[i, 0] = 1.0
            if lvl:
                state[i, 1] = float(lvl) / 18.0
                mask[i, 1] = 1.0
            if opp:
                vh_hp = vh_hpm = None
                seen = False
                for vh in lab.get("visible_heroes") or []:
                    if vh.get("name") == opp:
                        seen = vh.get("screen") is not None
                        vh_hp, vh_hpm = vh.get("hp"), vh.get("hp_max")
                        break
                state[i, 3] = 1.0 if seen else 0.0
                mask[i, 3] = 1.0
                if seen and vh_hp is not None and vh_hpm:
                    state[i, 2] = float(vh_hp) / float(vh_hpm)
                    mask[i, 2] = 1.0
        return state, mask

    # ───────────────────────── movement target ─────────────────────────

    def _parse_movement(
        self,
        match_id: str,
        labels: dict,
        frames: list[dict],
        click_doc: Optional[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Dispatch to the click-EVENT target (default) or the legacy cursor one.

        Returns ``(movement (T,2) float32, movement_event (T,) bool,
        built_from_clicks)``.
        """
        if self.movement_source == "clicks":
            out = self._parse_movement_clicks(match_id, labels, frames, click_doc)
            if out is not None:
                return out[0], out[1], True
            warnings.warn(
                f"{match_id}: movement_source='clicks' but no usable clicks.json; "
                "falling back to the legacy cursor.screen target (43% of whose "
                "transitions are camera drift — see docs/DATA_AUDIT_2026-08-12.md)."
            )
        movement = self._parse_movement_cursor(labels, frames)
        event = self._movement_events_from_cursor_world(frames)
        return movement, event, False

    def _parse_movement_clicks(
        self,
        match_id: str,
        labels: dict,
        frames: list[dict],
        click_doc: Optional[dict],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Movement target rebuilt from the raw click EVENTS in clicks.json.

        ``label.cursor.screen`` — the previous target — is not a cursor: the
        pipeline re-projects a *held* world point through the *current* camera
        every frame, so it sweeps across the screen with no player input. 43.2%
        of its bin transitions are that drift, and a phantom >1-cell jump fires
        at 89.7% of attack endings (audit findings 1 and 6).

        ``clicks.json:clicks`` is the actual command stream: one entry per
        change of the engine's "last commanded destination", each with a
        ``game_t`` and a world ``(x, z)``. Here we

        1. map each click's ``game_t`` to a frame with the SAME rounding the
           cast mapping uses (``round((gt - gt0) * fps)``);
        2. project its world point with the camera **of that frame** (recovered
           by inverting ``pipeline.project`` on champion_world -> champion_screen);
        3. HOLD that screen coordinate — a fixed number, not a re-projection —
           until the next click.

        So the target is piecewise-constant by construction: it can only change
        on a frame where the player actually issued a command, and
        ``movement_event`` marks exactly those frames.

        Returns None when there is no usable click stream (caller falls back).
        """
        clicks = (click_doc or {}).get("clicks") or []
        T = len(frames)
        if T == 0 or not clicks:
            return None

        proj = _Projection(labels)
        cx, cz = self._recover_cameras(frames, proj)
        if cx is None:
            return None

        # Identical timebase to the cast mapping in _parse_abilities: gt0 is the
        # first frame's game time, step is one frame, round (not truncate) for
        # symmetric frame-boundary quantization. Verified against the clicks'
        # own hero_x/hero_z: |champion_world[i] - click.hero_xz| is minimised at
        # offset 0 (p50 8.6 world units, vs 22.5 at -1 and 25.8 at +2).
        gt0 = float(frames[0]["gt"])
        step = 1.0 / float(labels["fps"])

        # (frame -> world) for every in-range click. Sorted by time first so
        # "later click on the same frame wins" is true regardless of file order.
        events: dict[int, tuple[float, float]] = {}
        n_out_of_range = 0
        for c in sorted(clicks, key=lambda c: c.get("game_t", c.get("game_time", 0.0))):
            gt = c.get("game_t")
            if gt is None:
                gt = c.get("game_time")
            wx, wz = c.get("x"), c.get("z")
            if gt is None or wx is None or wz is None:
                continue
            i = int(round((float(gt) - gt0) / step))
            if 0 <= i < T:
                events[i] = (float(wx), float(wz))
            else:
                n_out_of_range += 1

        if not events:
            return None
        if n_out_of_range:
            # Expected: the memory recorder outlives the PNG recorder on ~19
            # truncated matches (audit finding 7). Not an error.
            warnings.warn(
                f"{match_id}: {n_out_of_range}/{len(clicks)} clicks fall outside "
                "the frame record and were dropped")

        movement = torch.full((T, 2), 0.5, dtype=torch.float32)
        event = torch.zeros(T, dtype=torch.bool)
        held_x = held_y = 0.5  # pre-first-click: screen centre (no command yet)
        for i in range(T):
            e = events.get(i)
            if e is not None:
                x, y = proj.project_norm(e[0], e[1], cx[i], cz[i])
                # Clamp to the viewport: off-screen commands keep their
                # DIRECTION (an edge bin) instead of being dropped, which is
                # what the label pipeline did to 24% of frames. Clamping also
                # keeps the dynamics' movement action-embedding in-range.
                held_x = min(max(x, 0.0), 1.0)
                held_y = min(max(y, 0.0), 1.0)
                event[i] = True
            movement[i, 0], movement[i, 1] = held_x, held_y
        return movement, event

    @staticmethod
    def _recover_cameras(
        frames: list[dict], proj: "_Projection"
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Per-frame camera (cx, cz) inverted from champion_world/champion_screen.

        Frames with no champion_screen (~0.05%: champion off-viewport) are
        filled from the nearest frame that has one. Returns (None, None) if the
        match has no usable frame at all.
        """
        T = len(frames)
        cx = np.full(T, np.nan)
        cz = np.full(T, np.nan)
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                continue
            cs, cw = lab.get("champion_screen"), lab.get("champion_world")
            if not cs or not cw or len(cs) != 2 or len(cw) != 2:
                continue
            # +0.5: pipeline.project int-truncates, so the stored pixel is the
            # floor of the true coordinate; the pixel centre is the best estimate.
            cx[i], cz[i] = proj.invert(float(cw[0]), float(cw[1]),
                                       float(cs[0]) + 0.5, float(cs[1]) + 0.5)
        ok = np.flatnonzero(np.isfinite(cx))
        if ok.size == 0:
            return None, None
        if ok.size < T:  # nearest-valid fill (ffill then bfill)
            idx = np.searchsorted(ok, np.arange(T)).clip(0, ok.size - 1)
            prev = np.maximum(idx - 1, 0)
            take = np.where(np.abs(ok[idx] - np.arange(T))
                            <= np.abs(ok[prev] - np.arange(T)), ok[idx], ok[prev])
            cx, cz = cx[take], cz[take]
        return cx, cz

    def _parse_movement_cursor(self, labels: dict, frames: list[dict]) -> torch.Tensor:
        """LEGACY (schema-3) target: per-frame (x, y) from label.cursor.screen.

        Kept only so the new click-event target can be compared against what
        every checkpoint before schema 4 was trained on. ``cursor.screen`` is a
        held world point re-projected each frame, so this target drifts without
        player input; see :meth:`_parse_movement_clicks`.
        """
        T = len(frames)
        screen_w, screen_h = labels["screen_resolution"]
        screen_w, screen_h = float(screen_w), float(screen_h)
        movement = torch.full((T, 2), 0.5, dtype=torch.float32)
        last_xy: Optional[tuple[float, float]] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                if last_xy is not None:
                    movement[i, 0], movement[i, 1] = last_xy
                continue
            cs = (lab.get("cursor") or {}).get("screen")
            if cs and len(cs) == 2:
                new_xy = (cs[0] / screen_w, cs[1] / screen_h)
                # Dead-band: the mem-read "held command location" drifts every
                # frame (~60% of deltas are <1% of screen — attack-target drift
                # + read noise). A NEW command means the location moved; jitter
                # under 1% is the same command and must not fabricate movement
                # transitions (it teaches the gate noise near bin boundaries).
                if last_xy is None or abs(new_xy[0] - last_xy[0]) > 0.01 \
                        or abs(new_xy[1] - last_xy[1]) > 0.01:
                    last_xy = new_xy
            else:
                # Old-schema fallback. Drop once all data is re-recorded /
                # backfilled to write `cursor`.
                hs = (lab.get("movement") or {}).get("heading_screen")
                if hs and len(hs) == 2:
                    last_xy = (hs[0] / screen_w, hs[1] / screen_h)
            if last_xy is not None:
                movement[i, 0], movement[i, 1] = last_xy
        return movement

    @staticmethod
    def _movement_events_from_cursor_world(frames: list[dict]) -> torch.Tensor:
        """Approximate command events for matches with NO clicks.json.

        ``cursor.world`` is the held command point BEFORE projection, so a
        change in it means a new command was issued — this is the same signal
        the audit used as its ground truth when attributing transitions. It is
        weaker than the click stream: it also fires when a cast starts/ends
        (audit finding 6) and while a targeted unit moves. Only used on the 13
        matches that ship without clicks.json.
        """
        T = len(frames)
        event = torch.zeros(T, dtype=torch.bool)
        prev: Optional[tuple] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                continue
            cw = (lab.get("cursor") or {}).get("world")
            if not cw or len(cw) != 2:
                continue
            cur = (cw[0], cw[1])
            if prev is not None and cur != prev:
                event[i] = True
            prev = cur
        return event

    def _parse_abilities(
        self,
        labels: dict,
        frames: list[dict],
        match_id: str,
        click_doc: Optional[dict],
    ) -> dict[str, torch.Tensor]:
        """Per-frame binary action flags for the Garen v1 action space.

        clicks.json casts map by spell_name -> key (Q/W/E/Ecancel/R/Flash/Ignite/
        Recall); AA from label.action.type transitions; Stride from inventory
        `lf` jumps. Falls back to label.action.spell when clicks.json is absent.
        Unmapped spells (TP, super-recall, ...) are ignored.
        """
        T = len(frames)
        abilities: dict[str, torch.Tensor] = {
            k: torch.zeros(T, dtype=torch.long) for k in ABILITY_KEYS
        }
        if T == 0:
            return abilities

        # gt0 is the timebase reference. Pipeline writes gt for every frame;
        # cast game_t is anchored to it.
        gt0 = float(frames[0]["gt"])
        step = 1.0 / float(labels["fps"])

        casts: list[dict] = (click_doc or {}).get("casts") or []

        if casts:
            n_no_time = n_unmapped = n_out_of_range = 0
            for c in casts:
                # Explicit None-check, not `or` — game_t == 0.0 is valid.
                gt = c.get("game_t")
                if gt is None:
                    gt = c.get("game_time")
                if gt is None:
                    n_no_time += 1
                    continue
                key = _SPELL_TO_KEY.get(c.get("spell_name"))
                if key is None:
                    n_unmapped += 1  # TP / super-recall / etc. — intentionally dropped
                    continue
                # round, not int-truncate, for symmetric frame-boundary quantization
                i = int(round((float(gt) - gt0) / step))
                if 0 <= i < T:
                    abilities[key][i] = 1
                else:
                    n_out_of_range += 1
            if n_no_time or n_out_of_range:
                warnings.warn(
                    f"{match_id}: {n_no_time} no-time, {n_out_of_range} "
                    f"out-of-range casts dropped ({n_unmapped} unmapped ignored)"
                )
        else:
            warnings.warn(
                f"{match_id}: no clicks.json casts; mapping "
                "QWER/Flash/Ignite/Recall from label.action.spell (lossy)."
            )
            self._fill_from_action_spell(frames, abilities)

        # AA — attack-move / auto-attack initiation isn't in the cast stream;
        # take it from label.action.type transitions into "attack".
        self._fill_aa_from_attack(frames, abilities)
        # Stride active — sparse signal from the item's `lf` (last-fired) jumps.
        self._fill_stride_from_inventory(frames, abilities)
        return abilities

    @staticmethod
    def _is_attack_frame(lab: dict) -> bool:
        """Is this frame an auto-attack (including empowered autos)?

        ``pipeline.classify_spell`` tests ``"basicattack" in name`` first, so
        ``GarenBasicAttack``/``GarenBasicAttack2`` land on ``type == "attack"``
        — but ``GarenQAttack`` (the Q-empowered auto, a real right-click on a
        unit) matches the champion+slot branch and is filed as ``"ability"``,
        and ``GarenCritAttack`` falls through to ``"other"``. That undercounts
        real attack commands by ~21% (audit finding 10).

        Re-derive from ``action.spell`` here rather than in the pipeline: the
        shipped labels.json already have the wrong ``type`` baked in, and
        re-running the pipeline is impossible (the .rofl files were deleted).
        The Q *press* itself still comes from clicks.json casts, so this does
        not double-count it.
        """
        act = lab.get("action") or {}
        if act.get("type") == "attack":
            return True
        spell = act.get("spell")
        return bool(spell) and spell.lower().endswith("attack")

    @classmethod
    def _fill_aa_from_attack(
        cls,
        frames: list[dict],
        abilities: dict[str, torch.Tensor],
    ) -> None:
        """abilities['AA'][i] = 1 on the frame the champion enters an attack."""
        prev_attack = False
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                prev_attack = False
                continue
            is_attack = cls._is_attack_frame(lab)
            if is_attack and not prev_attack:
                abilities["AA"][i] = 1
            prev_attack = is_attack

    @staticmethod
    def _fill_stride_from_inventory(
        frames: list[dict],
        abilities: dict[str, torch.Tensor],
    ) -> None:
        """abilities['Stride'][i] = 1 when Stridebreaker's `lf` (last-fired
        game-time) jumps up — i.e. the active was used. Sparse (~2-18/game) but
        the only item-active the labels reliably log (pots/tiamat/ward have no
        usable signal). `lf` is held across unlabeled gaps so a gap isn't read
        as a use.
        """
        prev_lf: Optional[float] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            lf = None
            if lab:
                for it in (lab.get("inventory") or []):
                    if it and it.get("id") == _STRIDE_ITEM_ID:
                        lf = it.get("lf")
                        break
            if lf is not None and prev_lf is not None and lf > prev_lf + 1e-6:
                abilities["Stride"][i] = 1
            if lf is not None:
                prev_lf = lf

    @staticmethod
    def _fill_from_action_spell(
        frames: list[dict],
        abilities: dict[str, torch.Tensor],
    ) -> None:
        """Fallback (no clicks.json): map label.action.spell -> key on entry."""
        prev_spell: Optional[str] = None
        for i, fr in enumerate(frames):
            lab = fr.get("label")
            if not lab:
                prev_spell = None
                continue
            spell = (lab.get("action") or {}).get("spell")
            if spell and spell != prev_spell:
                key = _SPELL_TO_KEY.get(spell)
                if key:
                    abilities[key][i] = 1
            prev_spell = spell

    # ───────────────────────── latent loading ─────────────────────────

    def _load_latents(self, match_id: str) -> tuple[np.ndarray, np.ndarray]:
        if match_id in self._latent_cache:
            self._latent_cache_order.remove(match_id)
            self._latent_cache_order.append(match_id)
            return self._latent_cache[match_id]

        while len(self._latent_cache) >= self.max_cache_size:
            evicted = self._latent_cache_order.pop(0)
            del self._latent_cache[evicted]

        pt_path = self.latents_dir / f"{match_id}.pt"
        if pt_path.exists():
            data = torch.load(pt_path, weights_only=True)
            latents = data["latents"].numpy()
            frame_indices = data["frame_indices"].numpy()
        else:
            npz_path = self.latents_dir / f"{match_id}.npz"
            with np.load(npz_path, mmap_mode=None) as data:
                latents = data["latents"].copy()
                frame_indices = data["frame_indices"].copy()

        self._latent_cache[match_id] = (latents, frame_indices)
        self._latent_cache_order.append(match_id)
        return latents, frame_indices

    # ───────────────────────── Dataset API ─────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        match_id = seq["video_id"]
        start_frame = seq["start_frame"]
        start_idx = seq["start_idx"]
        T = self.sequence_length

        latents_arr, _ = self._load_latents(match_id)
        latents = torch.from_numpy(latents_arr[start_idx:start_idx + T].copy())

        md = self.match_data[match_id]
        sl = slice(start_frame, start_frame + T)
        rewards = md["rewards"][sl]
        movement = md["movement"][sl]
        # movement_event: True exactly on the frames a new command was issued.
        # This is the supervision target for the sticky-categorical gate — the
        # gate must NOT be inferred from "did the bin change", which conflates
        # camera drift with commands and drops every command that quantizes
        # into the same bin.
        actions = {"movement": movement, "movement_event": md["movement_event"][sl]}
        for k in ABILITY_KEYS:
            actions[k] = md["abilities"][k][sl]
        # cursor_valid gates real-action vs. no_action_embed per frame. Replays
        # (NA1_*) carry real actions -> valid everywhere (matches legacy behavior).
        # Unlabeled YT matches carry no actions -> invalid, so embed_actions
        # substitutes the learned no_action_embed for movement (paper: unlabeled
        # video is modeled without action conditioning). Abilities default to the
        # no-press class 0, ~99% correct since casts are <1% of frames.
        actions["cursor_valid"] = torch.full(
            (T,), match_id.startswith("NA1_"), dtype=torch.bool
        )

        return {
            "latents": latents,
            "actions": actions,
            "rewards": rewards,
            "state": md["state"][sl],
            "state_mask": md["state_mask"][sl],
            "video_id": match_id,
            "start_frame": start_frame,
        }
