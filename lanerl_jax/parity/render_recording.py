"""Convert source-server diagnostic recordings to the shared replay format.

Reads actual server rows, routes, swing clocks and missiles. Does not simulate
forward. Screen-click commands are overlaid at the same decision boundary.
The dump exposes held targets, not the separate melee swing victim; the latter
is left unknown rather than inferred from a target that may have changed.
"""
import argparse
import bisect
import json
from pathlib import Path

import numpy as np

from .trace import HASH_RE, parse_stream, StatQ
from .diagnostic_identity import net_id_to_entity
from .policy_driver import PolicyActionLog, WIRE_MT_TO_SIM
from ..sim.profiles import PROFILES
from ..sim.state import Kind, Team
from ..sim.orders import OrderKind
from ..replay import render_metadata


def sampled_snapshots(path, hz):
    """Stream one selected tick at a time, bounding memory for multi-GB logs."""
    bucket, wanted, next_ms = [], False, 0.
    with Path(path).open() as f:
        for line in f:
            match = HASH_RE.search(line)
            if match:
                if bucket:
                    yield parse_stream(bucket)[0]
                t = int(match[1])
                wanted = t >= next_ms
                if wanted:
                    next_ms = t + 1000. / hz - 1.
                bucket = [line] if wanted else []
            elif wanted and ("LANERL_STATEROW " in line or "LANERL_INTERNAL " in line):
                bucket.append(line)
        if bucket:
            yield parse_stream(bucket)[0]


def convert(log_path, observations, actions, out, hz=10., label="Source-server replay"):
    wire = []
    with Path(observations).open() as f:
        for line in f:
            wire.append(json.loads(line))
    times = [o["t"] for o in wire]
    action_log = PolicyActionLog.load(actions)
    slots, n, m, rows = {}, 128, 128, []
    last_position = np.zeros((n, 2), np.float32)
    unknown_missile_sources = 0
    for snap in sampled_snapshots(log_path, hz):
        at = min(bisect.bisect_left(times, snap.t_ms), len(times)-1)
        if at and abs(times[at-1]-snap.t_ms) < abs(times[at]-snap.t_ms):
            at -= 1
        if abs(times[at]-snap.t_ms) > 35:
            continue
        current_wire = {int(u["id"]): u for u in wire[at]["u"]}
        entities = net_id_to_entity(snap)
        internals = {a.net_id: a for a in snap.ai_internals}
        for nid in list(slots):
            if nid not in entities:
                del slots[nid]
        for nid, e in entities.items():
            if nid not in slots:
                if e.kind == "Champion":
                    slots[nid] = 0 if e.team == 100 else 1
                else:
                    free = next((i for i in range(2, n) if i not in slots.values()), None)
                    if free is None:
                        raise ValueError("replay entity capacity exceeded")
                    slots[nid] = free
        d = {k: np.zeros(n, np.float32) for k in
             ("x", "y", "hp", "max_hp", "aa_windup", "aa_cooldown")}
        d.update({k: np.zeros(n, np.int64) for k in
                  ("kind", "team", "model", "spawn_seq", "cs", "deaths", "level",
                   "is_attacking", "has_auto_attacked", "alive", "waypoint_key", "n_waypoints")})
        d.update(target=np.full(n, -1), aa_target=np.full(n, -1),
                 waypoints=np.zeros((2, 256, 2), np.float32),
                 spell_cooldown=np.zeros((2, 4)), spell_level=np.zeros((2, 4), int),
                 q_active=np.zeros(2, bool), w_active=np.zeros(2, bool), e_active=np.zeros(2, bool),
                 order_kind=np.zeros(2, int), order_target=np.full(2, -1),
                 order_resolved=np.zeros(2, bool),
                 order_x=np.zeros(2), order_y=np.zeros(2), button=np.zeros(2, int),
                 t_ms=np.asarray(snap.t_ms),
                 missile_alive=np.zeros(m, bool), missile_x=np.zeros(m), missile_y=np.zeros(m),
                 missile_tx=np.full(m, -1), missile_source=np.full(m, -1),
                 missile_source_seq=np.full(m, -1, np.int64), missile_damage=np.zeros(m))
        d['x'][:], d['y'][:] = last_position[:, 0], last_position[:, 1]
        for nid, e in entities.items():
            slot, a = slots[nid], internals[nid]
            if (a.q_x, a.q_y) != (e.q_x, e.q_y):
                raise ValueError("diagnostic identity/position mismatch")
            team = 0 if e.team == 100 else 1
            kind = Kind.CHAMPION if e.kind == "Champion" else (Kind.TURRET if "Turret" in e.kind else Kind.LANE_MINION)
            subtype = WIRE_MT_TO_SIM.get(int(current_wire.get(nid, {}).get("mt", 0)), 0)
            profile = (kind, subtype if kind == Kind.LANE_MINION else -1, team)
            d['model'][slot] = PROFILES.index(profile) if profile in PROFILES else 0
            d['kind'][slot], d['team'][slot], d['spawn_seq'][slot] = kind, team, nid
            d['x'][slot], d['y'][slot] = e.x, e.y
            d['hp'][slot], d['max_hp'][slot] = e.q_hp / StatQ, e.q_max_hp / StatQ
            d['alive'][slot] = not e.dead
            d['target'][slot] = slots.get(a.target_net_id, -1)
            d['is_attacking'][slot], d['has_auto_attacked'][slot] = a.is_attacking, a.has_auto_attacked
            d['aa_windup'][slot], d['aa_cooldown'][slot] = a.q_aa_windup / StatQ, a.q_aa_cooldown / StatQ
            if slot < 2:
                if len(a.waypoints) > 256:
                    raise ValueError("replay waypoint capacity exceeded")
                d['waypoint_key'][slot], d['n_waypoints'][slot] = a.waypoint_key, len(a.waypoints)
                if a.waypoints:
                    d['waypoints'][slot, :len(a.waypoints)] = np.asarray(a.waypoints) / 16.
                c = e.champ
                d['cs'][slot], d['deaths'][slot], d['level'][slot] = c.minions_killed, c.deaths, c.level
                d['spell_level'][slot] = [s[0] for s in c.spells]
                d['spell_cooldown'][slot] = [max(0, s[1]) / StatQ for s in c.spells]
                for name, buff in (("q_active", "GarenQ"), ("w_active", "GarenW"), ("e_active", "GarenE")):
                    d[name][slot] = buff in e.ai.buffs
        from lanerl_rl.constants import BUTTON_INDEX
        for side, orders in enumerate((action_log.blue, action_log.red)):
            order = orders[at] if at < len(orders) else {"t": "noop"}
            button = order.get('button', order.get('t', 'noop'))
            d['button'][side] = BUTTON_INDEX.get(button, BUTTON_INDEX['noop'])
            d['order_x'][side] = order.get('x', d['x'][side])
            d['order_y'][side] = order.get('y', d['y'][side])
            d['order_kind'][side] = (1 if button in ('move', 'attack_move') else
                {'q': OrderKind.CAST_Q, 'w': OrderKind.CAST_W,
                 'e': OrderKind.CAST_E, 'r': OrderKind.CAST_R}.get(button, 0))
        d['click_x'], d['click_y'] = d['order_x'].copy(), d['order_y'].copy()
        if len(snap.missile_internals) > m:
            raise ValueError("replay missile capacity exceeded")
        for i, missile in enumerate(snap.missile_internals):
            d['missile_alive'][i] = True
            d['missile_x'][i], d['missile_y'][i] = missile.q_x / 16., missile.q_y / 16.
            d['missile_tx'][i] = slots.get(missile.target_net_id, -1)
            d['missile_source'][i] = slots.get(missile.owner_net_id, -1)
            d['missile_source_seq'][i] = missile.owner_net_id
            d['missile_damage'][i] = missile.q_damage / StatQ
            unknown_missile_sources += int(d['missile_source'][i] < 0)
        last_position = np.stack([d['x'], d['y']], axis=-1)
        rows.append(d)
    if not rows:
        raise ValueError("no server frames found")
    from ..sim.terrain_jax import map1_terrain
    terrain = map1_terrain()
    meta = dict(label=label, controller="recorded source-server episode", environment="source-server",
        melee_victim_recorded=False,
        view="omniscient diagnostic; hidden state is not an actor input", **render_metadata(),
        terrain=dict(min_x=terrain.min_x, min_y=terrain.min_y, cell_size=terrain.cell_size),
        source_log=str(log_path), hz=hz, unknown_missile_sources=unknown_missile_sources,
        limitations="Quantized server dump. Melee swing victim and R cast timer unavailable; held targets shown. Clicks show commands, not inferred hits.")
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data, walkable=np.asarray(terrain.walkable), metadata=json.dumps(meta))
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hz', type=float, default=10.)
    p.add_argument('--label', default='Source-server replay')
    p.add_argument('--tag', default='policy')
    p.add_argument('--actions', type=Path)
    a = p.parse_args()
    if a.hz <= 0 or not np.isfinite(a.hz):
        p.error('hz must be positive')
    convert(a.recording / a.tag / 'instance000.log', a.recording / f'{a.tag}_obs.jsonl',
            a.actions or a.recording / f'{a.tag}_policy_actions.json', a.out, a.hz, a.label)


if __name__ == '__main__':
    main()
