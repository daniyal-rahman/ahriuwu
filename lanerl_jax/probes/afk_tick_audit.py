"""DIAG E69: exact E68 replay, tick damage attribution and timed physical clicks.

The diagnostic AST only adds outputs to tick's final return. Production code,
arithmetic and ordering are untouched. Instrumented continuations must match
ordinary env_step on every state leaf before counterfactuals are interpreted.
"""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.sim.step import env_apply, env_step, tick
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.parity.policy_driver import load_params, _lane_frames
from lanerl_jax.train.actions import orders_from
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.scripted_policy import cell_for_offset
from lanerl_jax.train.wave_scenario import prepare_scenario_bank, park_afk_opponent
from lanerl_jax.train.replay_audit import serialize_replay_state
from lanerl_jax.train.run_manifest import file_sha256, git_provenance


def diagnostic_tick():
    original = ast.parse(inspect.getsource(tick))
    tree = copy.deepcopy(original)
    ret = tree.body[0].body[-1]
    assert isinstance(ret, ast.Return)
    extra = ast.parse("dict(start=aa.start, hit=aa.hit, hit_target=hit_target, "
                      "melee=landed, aa_damage=aa.damage, missile=ms.damage_ij, "
                      "damage=dmg_ij, killer=killer, died=died, hp_before=state.hp)",
                      mode='eval').body
    ret.value = ast.Tuple(elts=[ret.value, extra], ctx=ast.Load())
    stripped = copy.deepcopy(tree)
    stripped.body[0].body[-1].value = stripped.body[0].body[-1].value.elts[0]
    assert ast.dump(stripped) == ast.dump(original)
    ast.fix_missing_locations(tree)
    namespace = dict(tick.__globals__)
    exec(compile(tree, inspect.getsourcefile(tick), 'exec'), namespace)
    return namespace['tick']


def max_tree_error(a, b):
    errors = []
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype
        errors.append(float(np.max(np.abs(x.astype(float)-y.astype(float)), initial=0)))
    return max(errors, default=0.)


def main():
    spec = json.loads(Path('experiments', sys.argv[1]+'.json').read_text())
    out = Path('/mnt/nfs/shared')/spec['id']
    out.mkdir(exist_ok=False)
    stage = Path('/mnt/nfs/shared')/(spec['id']+'-staged-'+os.environ['SLURM_JOB_ID'])
    stage.mkdir()
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT, stage/'routes')
    source = Path(spec['checkpoint'])
    assert file_sha256(source) == spec['checkpoint_sha256']
    shutil.copyfile(source, stage/'checkpoint.msgpack')
    shutil.copyfile(source.parent/'manifest.json', stage/'manifest.json')
    sim = SimConfig.training(route_artifact=stage/'routes').replace(step_ticks=6)
    policy, params, _ = load_params(str(stage/'checkpoint.msgpack'))
    assert policy.cfg.observation_interface == 'viewport-structured-v3'
    bank = park_afk_opponent(prepare_scenario_bank(sim, out/'bank', [-45], 1007))
    state = jax.tree.map(lambda x: x[1], bank)
    frames = _lane_frames()

    @jax.jit
    def act(s, c, key):
        # Exactly wave_replay's batched two-player sampling and key split.
        key, ak = jax.random.split(key)
        obs = [build_observation(s, t, frames[t], params=sim.params,
               horizon_s=600., vision=sim.vision) for t in (0, 1)]
        obs = jax.tree.map(lambda a, b: jnp.stack([a, b]), *obs)
        lg, c = policy.apply(params, obs.entities, obs.entity_pad_mask,
                             obs.self_vec, obs.global_vec, c)
        a, _, _ = _sample(lg, ak, ~obs.entity_pad_mask)
        return tuple(x.at[1].set(0) for x in a), c, key

    decode = jax.jit(lambda a, s: orders_from(a, s, None, frames[0],
        snap_moves=False, params=sim.params, vision=sim.vision, drop_unwalkable_moves=True))
    step = jax.jit(lambda s, o: env_step(s, o, sim))
    instrument = diagnostic_tick()

    @jax.jit
    def detailed(s, o, target):
        def one(s, _):
            nxt, d = instrument(s, sim.params, delta_ms=sim.delta_ms,
                lane_path=sim.lane_path, minion_hp=sim.minion_hp,
                enable_call_for_help=sim.enable_call_for_help,
                enable_collision=sim.enable_collision,
                collision_terrain=sim.collision_terrain,
                defer_collision_terrain=sim.defer_collision_terrain,
                route_table=sim.route_table, terrain=sim.terrain, vision=sim.vision)
            row = dict(t_ms=nxt.t_ms, tick=nxt.tick, hp_before=d['hp_before'][target],
                hp=nxt.hp[target], alive=nxt.alive[target], spawn_seq=nxt.spawn_seq[target],
                cs=nxt.cs[0], cd=nxt.aa_cooldown[0], windup=nxt.aa_windup[0],
                aa_target=nxt.aa_target[0], target=nxt.target[0],
                start=d['start'][0], hit=d['hit'][0], hit_target=d['hit_target'][0],
                aa_damage=d['aa_damage'][0], damage=d['damage'][:, target],
                missile=d['missile'][:, target], killer=d['killer'][target],
                died=d['died'][target],
                distance=jnp.hypot(nxt.x[target]-nxt.x[0], nxt.y[target]-nxt.y[0]),
                own_xy=jnp.stack([nxt.x[0], nxt.y[0]]),
                target_xy=jnp.stack([nxt.x[target], nxt.y[target]]))
            return nxt, row
        return jax.lax.scan(one, env_apply(s, o, sim), None, length=6)

    @jax.jit
    def click(s, target, wait):
        dx, dy = s.x[target]-s.x[0], s.y[target]-s.y[0]
        sx, sy = cell_for_offset(dx*frames[0].axis[0]+dy*frames[0].axis[1],
                                dx*frames[0].normal[0]+dy*frames[0].normal[1])
        wx, wy = cell_for_offset(0., 0.)
        return (jnp.where(wait, 1, 2).astype(jnp.int32),
                jnp.where(wait, wx, sx), jnp.where(wait, wy, sy))

    ref = dict(np.load(spec['reference_trace']))
    carry, key = policy.initial_carry((2,)), jax.random.key(7)
    bases, errors = {}, {}
    indices = {case['base_index'] for case in spec['cases']}

    def compare(name, got, expected):
        error = float(np.max(np.abs(np.asarray(got).astype(float)-np.asarray(expected).astype(float)), initial=0))
        errors[name] = max(errors.get(name, 0.), error)
        if error > 1e-5:
            raise AssertionError(f'E68 reproduction failed at {i}: {name}, error={error}')

    for i in range(len(ref['t_ms'])):
        for name in ref:
            if hasattr(state, name):
                value = getattr(state, name)
                if name == 'waypoints': value = value[:2]
                compare(name, value, ref[name][i])
        if i == len(ref['t_ms'])-1: break
        if i in indices:
            bases[i] = (state, carry, key)
            (out/f'base_{i}.msgpack').write_bytes(serialize_replay_state(dict(state=state, carry=carry, key=key)))
        a, carry, key = act(state, carry, key)
        o = decode(a, state)
        for name, value in zip(('button', 'screen_x', 'screen_y'), a): compare(name, value, ref[name][i])
        for name in ('kind', 'target', 'x', 'y'): compare('order_'+name, getattr(o, name), ref['order_'+name][i])
        for name in ('q', 'w', 'e'): compare(name+'_active', getattr(state.buffs, name).active[:2], ref[name+'_active'][i])
        state = jax.block_until_ready(step(state, o))
        if (i+1) % 200 == 0: print('REPRODUCE', i+1, flush=True)
    reproduction = dict(max_errors=errors, final_cs=int(state.cs[0]), final_deaths=int(state.deaths[0]),
                        frames=len(ref['t_ms']), checkpoint_sha256=file_sha256(source))
    (out/'reproduction.json').write_text(json.dumps(reproduction, indent=2))
    print('E68 REPRODUCTION PASSED', json.dumps(reproduction), flush=True)

    results = []
    for case in spec['cases']:
        base, bc, bk = bases[case['base_index']]
        target, length = case['target'], case['decisions']
        seq = int(base.spawn_seq[target])
        variants = [('baseline', -1)]
        variants += [('early', t) for t in case['attack_indices']]
        variants += [('delay', t) for t in case['attack_indices']]
        for mode, attack_index in variants:
            s, c, k = base, bc, bk
            chunks, order_rows, instrument_error = [], [], 0.
            forced_count = directed_count = wait_count = wait_moves = 0
            for j in range(length):
                a, c, k = act(s, c, k)
                same = bool(s.alive[target]) and int(s.spawn_seq[target]) == seq
                wait = mode == 'delay' and j < attack_index
                force = same and mode != 'baseline' and (j >= attack_index or wait)
                if force:
                    f = click(s, target, wait)
                    a = tuple(x.at[0].set(y) for x, y in zip(a, f))
                o = decode(a, s)
                forced_count += int(force and not wait)
                directed_count += int(force and not wait and int(o.target[0]) == target)
                wait_count += int(force and wait)
                wait_moves += int(force and wait and int(o.kind[0]) == 1)
                nxt, d = jax.block_until_ready(detailed(s, o, jnp.int32(target)))
                if mode == 'baseline':
                    ordinary = jax.block_until_ready(step(s, o))
                    instrument_error = max(instrument_error, max_tree_error(nxt, ordinary))
                    assert instrument_error < 1e-5, instrument_error
                    ix = case['base_index']+j+1
                    for name in ('x', 'y', 'hp', 'cs', 'aa_cooldown', 'aa_windup', 'aa_target', 'missile_alive'):
                        compare('branch_'+name, getattr(nxt, name), ref[name][ix])
                chunks.append(jax.tree.map(np.asarray, d))
                order_rows.append(dict(index=case['base_index']+j, seconds=(float(s.t_ms)-120000)/1000,
                    button=int(a[0][0]), order_kind=int(o.kind[0]), order_target=int(o.target[0]),
                    force=force, wait=wait, pre_cd=float(s.aa_cooldown[0]), pre_windup=float(s.aa_windup[0])))
                s = nxt
            data = {name: np.concatenate([d[name] for d in chunks]) for name in chunks[0]}
            tag = f"unit{target}_{mode}_{attack_index}"
            np.savez_compressed(out/(tag+'.npz'), **data)
            events = []
            for t in range(len(data['t_ms'])):
                if data['start'][t] or data['hit'][t] or data['damage'][t].sum() > 0 or data['died'][t]:
                    events.append({name: data[name][t].tolist() for name in data})
            death = np.flatnonzero(data['died'])
            row = dict(target=target, spawn_seq=seq, base_index=case['base_index'], mode=mode,
                attack_index=attack_index, attack_seconds=(float(base.t_ms)-120000)/1000+attack_index*.1 if attack_index>=0 else None,
                cs_gain=int(s.cs[0]-base.cs[0]), target_killed_by=int(data['killer'][death[0]]) if len(death) else None,
                target_death_seconds=(float(data['t_ms'][death[0]])-120000)/1000 if len(death) else None,
                target_cs=bool(len(death) and data['killer'][death[0]] == 0),
                instrument_max_error=instrument_error, forced_clicks=forced_count,
                directed_clicks=directed_count, wait_clicks=wait_count, wait_moves=wait_moves,
                events=events, orders=order_rows)
            (out/(tag+'.json')).write_text(json.dumps(row, indent=2))
            results.append({name: value for name, value in row.items() if name not in ('events', 'orders')})
            print('BRANCH', json.dumps(results[-1]), flush=True)
    report = dict(spec=spec, source=git_provenance(), reproduction=reproduction, results=results,
        limits='Two selected incidents, not aggregate performance. Same initial recurrent history and random numbers; policy observes changed branch states. Early holds directed clicks from chosen time; delay uses repeated near-self move clicks first. These are multi-action physical options, not single-action effects or C# parity proof. Same-tick damage attribution follows production simulator approximation. Raw target IDs are diagnostic only.')
    (out/'result.json').write_text(json.dumps(report, indent=2))
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
