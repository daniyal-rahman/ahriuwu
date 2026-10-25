"""Versioned v2 replay examples, reusing v4 observations only after parity checks."""
from __future__ import annotations

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data import champion_table as CT
from . import convert as C, observation as RO
from .ability_metadata import ability_evidence
from .event_inputs import SELF_FIELDS, ABILITY_FIELDS, previous_received_events, visible_cast_ages, encode_ages
from .event_labels import label_events, POINT_TARGET
from .events import BUTTON_INDEX as B, ordinary_move
from .history import received_cast_ages
from .labels import screen_cell
from .event_targets import pointer_candidates, grid_candidates
from .sources import SOURCE_CONTRACT
from . import fog


CONTRACT = 'replay-lane-events-v2'
MODES = {'self': 0, 'unit': 1, 'point': 2, 'direction': 3}


@lru_cache(maxsize=256)
def catalog(path):
    path = Path(path)
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def spell_metadata(name, ranks, catalog_dir):
    result = np.full((len(ranks), 4, len(ABILITY_FIELDS) + 4), np.nan, np.float32)
    modes = np.full(4, -1, np.int8)
    data, digest = catalog(str(Path(catalog_dir) / (name + '.json')))
    if data is None:
        return result, modes, digest
    for slot in range(4):
        evidence = ability_evidence(data, slot)
        mode = MODES.get(evidence['raw_mode'])
        if mode is not None:
            modes[slot] = mode
            result[:, slot, :4] = np.eye(4)[mode]
        # A rank cooldown reference is not an activation/readiness estimate.
        cooldown = evidence['raw_cooldown_by_rank'] or {}
        for rank in range(1, 6):
            value = cooldown.get(str(rank))
            if value is not None and np.isfinite(value) and value >= 0:
                result[ranks[:, slot] == rank, slot, 4 + ABILITY_FIELDS.index('base_cooldown')] = value / 300.
    return result, modes, digest


def co_movement_targets(actions, times, labels, anchors, axis, normal):
    source = np.full(len(times), -1, np.int32)
    for i, a in enumerate(actions):
        k = int(np.floor((a.t - times[0]) / .1 + 1e-9))
        if ordinary_move(a) and 0 <= k < len(times) and B['q'] <= labels.button[k] <= B['r']:
            source[k] = i
    valid = np.zeros(len(times), bool)
    x, y = np.zeros((2, len(times)), np.int32)
    delta_t = np.full(len(times), np.nan, np.float32)
    for k in np.flatnonzero(source >= 0):
        a = actions[source[k]]
        cast = actions[labels.source[k]]
        delta_t[k] = a.t - cast.t
        if not labels.valid[k] or not np.isfinite(anchors[k]).all():
            continue
        if a.point is None or not np.isfinite(a.point).all():
            continue
        delta = np.asarray(a.point) - anchors[k]
        x[k], y[k], clipped = screen_cell(float(delta @ axis), float(delta @ normal))
        valid[k] = not clipped and bool(grid_candidates()[y[k], x[k]])
    return source, valid, x, y, delta_t


def convert_events(g, row, original_npz, catalog_dir):
    me, opponent = g.index(g.me), g.index(g.opponent)
    team, lane = g.units[me].team, C.ROLE_LANE[row['role']]
    frame = RO.frames(g, lane)[team]
    own_xy = np.stack([g.x[:, me], g.y[:, me]], -1)
    casts = [(a.t, a.slot) for a in g.actions if a.kind == 'cast' and a.slot in range(6)]
    timers, _ = C.received_spell_state(g.t, g.cooldown_events, casts, n_slots=6)
    entities, pad, self_vec, global_vec, slots, visible = RO.build(
        g, frame, C._self_columns('modern-world-v4'), row['role'], 0, True, timers)
    cols = C._self_columns('modern-world-v4')
    with np.load(original_npz) as original:
        oi = original['obs_index'].copy()
        np.testing.assert_array_equal(oi, np.maximum(np.arange(len(g.t)) - 1, 0))
        np.testing.assert_array_equal(g.t, original['t'])
        np.testing.assert_allclose(entities[oi], original['entities'], rtol=0, atol=1e-6, equal_nan=True)
        np.testing.assert_array_equal(pad[oi], original['entity_pad_mask'])
        selected_self = [cols[n] for n in SELF_FIELDS]
        np.testing.assert_allclose(self_vec[oi][:, selected_self], original['self_vec'][:, selected_self],
                                   rtol=0, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(global_vec[oi, :2], original['global_vec'][:, :2], rtol=0, atol=1e-6, equal_nan=True)
    ids = np.asarray([u.id for u in g.units])
    selected = slots[oi]
    unit_ids = np.where(selected >= 0, ids[np.maximum(selected, 0)], -1)
    scope, _, _ = C._scope(g, own_xy, RO.corridor(lane), True)
    scope &= np.isfinite(own_xy).all(1) & np.isfinite(g.hp[:, me])
    scope &= np.arange(len(g.t)) > 0
    labels = label_events(g.t, g.actions, own_xy[oi], unit_ids, np.asarray(frame.axis),
                          np.asarray(frame.normal), scope, g.me)
    dead = ~g.alive[:, me]
    deaths = g.t[dead & ~np.r_[False, dead[:-1]]]
    history, previous_target = previous_received_events(
        g.t[oi], g.actions, own_xy[oi], unit_ids, np.asarray(frame.axis), np.asarray(frame.normal), reset_times=deaths)
    per_entity_age = visible_cast_ages(g, slots, visible, oi)
    entity_features = np.concatenate([entities[oi], per_entity_age, previous_target[..., None]], -1)
    champion_ids = np.asarray([CT.index_of(u.subtype) if u.family == 'champion' else 0 for u in g.units])
    entity_ids = np.where(selected >= 0, champion_ids[np.maximum(selected, 0)], 0)
    cols = C._self_columns('modern-world-v4')
    own_age = encode_ages(received_cast_ages(g.t, casts, g.summoner_slot)[oi, :4])
    compact_self = np.concatenate([self_vec[oi][:, [cols[n] for n in SELF_FIELDS]], own_age, history], -1)
    global_features = np.column_stack([global_vec[oi, 0], global_vec[oi, 1], np.full(len(g.t), team)])
    own_meta, modes, own_digest = spell_metadata(row['champion'], g.spell_rank[oi, 0], catalog_dir)
    # Enemy rank receipts have not been qualified as player-observable.
    enemy_meta, _, enemy_digest = spell_metadata(g.opponent_name, np.full((len(g.t), 4), np.nan), catalog_dir)
    button_modes = np.array([0, POINT_TARGET, 1, *modes, 0, 0], np.int8)
    candidates = np.asarray(pointer_candidates(entities[oi], pad[oi], np.asarray(CT.index_of(row['champion'])), labels.button))
    eligible = candidates[np.arange(len(g.t)), np.maximum(labels.target_row, 0)]
    bad_unit = (labels.aim_mode == 1) & ~eligible
    bad_cell = (labels.aim_mode == POINT_TARGET) & ~np.asarray(grid_candidates())[labels.screen_y, labels.screen_x]
    labels = labels._replace(aim_valid=labels.aim_valid & ~bad_unit & ~bad_cell)
    # Retain every received sub-frame event for collision/reissue experiments.
    raw_events = [dict(a._asdict()) for a in g.actions]
    co_source, co_valid, co_x, co_y, co_delta = co_movement_targets(
        g.actions, g.t, labels, own_xy[oi], np.asarray(frame.axis), np.asarray(frame.normal))
    co_after, co_order_valid = np.zeros((2, len(g.t)), bool)
    for k in np.flatnonzero(co_source >= 0):
        move, cast = g.actions[co_source[k]], g.actions[labels.source[k]]
        co_order_valid[k] = move.t != cast.t or (move.source_order >= 0 and cast.source_order >= 0
                                                and move.source_order != cast.source_order)
        co_after[k] = (move.t, move.source_order) > (cast.t, cast.source_order)
    arrays = dict(anchor_xy=own_xy[oi], frame_axis=np.asarray(frame.axis), frame_normal=np.asarray(frame.normal),
                  t=g.t, obs_index=oi, entities=entity_features.astype(np.float16), pad=pad[oi],
                  self=compact_self.astype(np.float32), global_=global_features.astype(np.float32),
                  entity_ids=entity_ids.astype(np.int16), unit_ids=unit_ids, own_id=np.int16(CT.index_of(row['champion'])),
                  enemy_id=np.int16(CT.index_of(g.opponent_name)), patch=np.int8(0),
                  own_metadata=own_meta.astype(np.float16), enemy_metadata=enemy_meta.astype(np.float16),
                  button_modes=button_modes, button=labels.button, valid=labels.valid,
                  aim_mode=labels.aim_mode, aim_valid=labels.aim_valid, target_row=labels.target_row,
                  screen_x=labels.screen_x, screen_y=labels.screen_y, angle=labels.angle,
                  clipped=labels.clipped, event_source=labels.source, co_move_source=co_source,
                  co_move_valid=co_valid, co_move_x=co_x, co_move_y=co_y, co_move_delta=co_delta, co_move_after=co_after, co_move_order_valid=co_order_valid)
    counts = dict(valid=int(labels.valid.sum()), events=int(((labels.button != 0) & labels.valid).sum()),
                  aims=int(labels.aim_valid.sum()), clipped=int(labels.clipped.sum()),
                  co_move=int((co_source >= 0).sum()), co_move_aim=int(co_valid.sum()),
                  buttons={n: int(((labels.button == b) & labels.valid).sum()) for n, b in B.items()})
    meta = dict(contract=CONTRACT, source_contract=SOURCE_CONTRACT, profile='modern-world-v4',
                fog=fog.provenance(), game=row['game'], base=row['base'], split=row['split'],
                champion=row['champion'], role=row['role'], source=g.provenance,
                observation_parity=True, observation_lag=1, aim_anchor='delayed-observation-own-position', event_order='receipt-time-native-update-index',
                self_fields=list(SELF_FIELDS), metadata_fields=list(ABILITY_FIELDS), counts=counts,
                collisions=labels.collisions, own_catalog_sha256=own_digest, enemy_catalog_sha256=enemy_digest,
                metadata_limits='raw explicit root modes; effect ranges unknown; enemy ranks unknown; no readiness masks',
                input_limits='previous received event, not issued command; full cast history filtered at receipt visibility')
    return arrays, meta, raw_events


def save_events(out, arrays, metadata, raw_events):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_path, meta_path, events_path = (Path(str(out) + suffix) for suffix in ('.npz', '.json', '.events.json.gz'))
    if data_path.exists() or meta_path.exists():
        raise FileExistsError(out)
    np.savez_compressed(data_path, **arrays)
    with gzip.open(events_path, 'wt') as f:
        json.dump(raw_events, f)
    meta_path.write_text(json.dumps(metadata, indent=2))


def artifact_hashes(prefix):
    result = {}
    for suffix in ('.npz', '.json', '.events.json.gz'):
        with open(str(prefix) + suffix, 'rb') as f:
            result[suffix] = hashlib.sha256(f.read()).hexdigest()
    return result
