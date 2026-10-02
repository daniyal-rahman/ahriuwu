"""E85: real-wave action search and credit, frozen parameters throughout.

All interventions pass through normal screen clicks at 10 Hz. Alternative
branches are diagnostic, never treated as on-policy PPO samples.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.obs.builder import build_observation, NORM_AD
from lanerl_jax.parity.policy_driver import load_params, _lane_frames
from lanerl_jax.sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.sim.step import env_step
from lanerl_jax.train.actions import orders_from
from lanerl_jax.train.ppo import gae
from lanerl_jax.train.replay_audit import serialize_replay_state
from lanerl_jax.train.run_manifest import file_sha256, git_provenance
from lanerl_jax.train.scripted_policy import cell_for_offset, _minion_view
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.vec_train import VecConfig, _relative_reward
from lanerl_jax.train.wave_scenario import prepare_scenario_bank, park_afk_opponent


def main():
    spec = json.loads(Path('experiments', sys.argv[1] + '.json').read_text())
    out = Path('/mnt/nfs/shared') / spec['id']
    out.mkdir(exist_ok=False)
    scratch = Path('/scratch') / (spec['id'] + '-' + os.environ['SLURM_JOB_ID'])
    scratch.mkdir()
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT, scratch / 'routes')
    source = Path(spec['checkpoint'])
    assert file_sha256(source) == spec['checkpoint_sha256']
    shutil.copyfile(source, scratch / 'checkpoint.msgpack')
    shutil.copyfile(source.parent / 'manifest.json', scratch / 'manifest.json')
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_compilation_cache_dir', '/scratch/lanerl-jax-compilation-cache')
    sim = SimConfig.training(route_artifact=scratch / 'routes').replace(step_ticks=6)
    bank = park_afk_opponent(prepare_scenario_bank(sim, out / 'bank', [-45, -5, 35, 75], 1007))
    policy, params, _ = load_params(str(scratch / 'checkpoint.msgpack'))
    frame = _lane_frames()[0]
    cfg = VecConfig(cs_only=True, xp_scale=0., health_loss_gold=0., tower_damage_gold=0.,
                    death_loss_gold=0., tower_damage_personal=True)
    started = time.monotonic()
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(True))
    report = dict(spec=spec, source=git_provenance(), status='running', rows=[])

    def save():
        tmp = out / 'result.json.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(out / 'result.json')

    def bound():
        if stopping or time.monotonic() - started > spec['max_seconds']:
            report['status'] = 'interrupted'
            save()
            raise RuntimeError('Bound reached; partial diagnostic is not a completed result')

    def observe(s):
        return build_observation(s, 0, frame, params=sim.params, horizon_s=600., vision=sim.vision)

    def act(s, c, k):
        o = observe(s)
        lg, nc = policy.apply(params, o.entities, o.entity_pad_mask, o.self_vec, o.global_vec, c)
        nk, ak = jax.random.split(k)
        a, _, _ = _sample(lg, ak, ~o.entity_pad_mask)
        return jnp.stack(a), nc, nk, lg, o

    def decode(a, s):
        return orders_from(tuple(jnp.stack([x, jnp.zeros_like(x)]) for x in a),
            s, None, frame, snap_moves=False, params=sim.params, vision=sim.vision,
            drop_unwalkable_moves=True)

    def choose(m, a, b):
        return jax.tree.map(lambda x, y: jnp.where(m.reshape(m.shape + (1,) * (x.ndim-m.ndim)), x, y), a, b)

    def collect_one(s, c, k, threshold):
        a, nc, nk, lg, o = act(s, c, k)
        valid, ds, dn, hp, _, armor = _minion_view(o)
        distance = jnp.hypot(ds, dn)
        damage = o.self_vec[10] * NORM_AD * 100 / (100 + armor)
        possible = valid & (hp <= damage * 1.5) & (distance <= 500) & (distance >= 30)
        eligible = possible.any() & (s.t_ms >= threshold) & (s.t_ms < 215000) & s.alive[0]
        live = s.t_ms < 240000
        nxt = env_step(s, decode(a, s), sim)
        reward, _ = _relative_reward(s, nxt, cfg)
        data = dict(entities=o.entities, mask=o.entity_pad_mask, self=o.self_vec,
                    global_=o.global_vec, action=a, value=lg.value,
                    reward=jnp.where(live, reward[0], 0.), done=nxt.t_ms >= 240000,
                    cs=jnp.where(live, nxt.cs[0] - s.cs[0], 0))
        return choose(live, nxt, s), jnp.where(live, nc, c), nk, eligible, data

    @jax.jit
    def capture_chunk(r, t0):
        def step(r, t):
            s, c, k, cs, cc, ck, found_at = r
            ns, nc, nk, eligible, d = jax.vmap(collect_one)(s, c, k, thresholds)
            take = eligible & (found_at < 0)
            return (ns, nc, nk, choose(take, s, cs), choose(take, c, cc),
                    choose(take, k, ck), jnp.where(take, t, found_at)), d
        return jax.lax.scan(step, r, jnp.arange(100) + t0)

    n = spec['cases']
    thresholds = jnp.asarray([128000, 145000, 165000, 185000], jnp.float32)[jnp.arange(n) % 4]
    ids = jnp.arange(n) % len(bank.t_ms)
    s = jax.tree.map(lambda x: x[ids], bank)
    c = policy.initial_carry((n,))
    k = jax.random.split(jax.random.key(85001), n)
    r = (s, c, k, s, c, k, jnp.full(n, -1, jnp.int32))
    history = []
    save()
    for i in range(13):
        bound()
        r, data = jax.block_until_ready(capture_chunk(r, i * 100))
        history.append(jax.tree.map(np.asarray, data))
    hist = {key: np.concatenate([d[key] for d in history]) for key in history[0]}
    np.testing.assert_array_equal(hist['reward'], hist['cs'])
    assert np.asarray(r[0].t_ms >= 240000).all(), 'capture must finish every original episode'
    _, _, _, states, carries, keys, found_at = r
    selected = np.flatnonzero(np.asarray(found_at) >= 0)
    assert len(selected) >= spec['minimum_cases'], ('Insufficient eligible cases', len(selected))
    # Save complete observation prefixes plus the untouched on-policy trajectory.
    np.savez_compressed(out / 'histories.npz', **hist, found_at=np.asarray(found_at))
    (out / 'cases.msgpack').write_bytes(serialize_replay_state(
        dict(state=states, carry=carries, key=keys, found_at=found_at)))

    @jax.jit
    def replay_prefix(entities, masks, self_vecs, global_vecs, stops):
        def step(c, xs):
            t, e, m, sv, gv = xs
            _, nc = policy.apply(params, e, m, sv, gv, c)
            return jnp.where((t < stops)[:, None], nc, c), None
        return jax.lax.scan(step, policy.initial_carry((n,)),
            (jnp.arange(entities.shape[0]), entities, masks, self_vecs, global_vecs))[0]

    replayed = replay_prefix(*(jnp.asarray(hist[key]) for key in
        ('entities', 'mask', 'self', 'global_')), found_at)
    prefix_error = float(jnp.max(jnp.abs(replayed[selected] - carries[selected])))
    assert prefix_error < 1e-4, ('Full-history carry mismatch', prefix_error)
    actual_gae, _ = gae(jnp.asarray(hist['reward']), jnp.asarray(hist['value']),
                       jnp.asarray(hist['done']), jnp.zeros(n), .99, .95)
    report.update(cases=len(selected), prefix_max_error=prefix_error,
                  original_frozen_cs=hist['cs'].sum(0).tolist())
    print('CAPTURE COMPLETE', len(selected), 'prefix error', prefix_error, flush=True)
    save()

    def menu_one(s, c, k):
        _, _, _, lg, o = act(s, c, k)
        valid, ds, dn, *_ = _minion_view(o)
        angle = jnp.arange(20) * (2 * jnp.pi / 20)
        mx, my = jax.vmap(cell_for_offset)(250 * jnp.cos(angle), 250 * jnp.sin(angle))
        tx, ty = jax.vmap(cell_for_offset)(ds, dn)
        # Two independent but identical natural-policy controls, then six buttons,
        # twenty movement directions, twelve visible minion target centres.
        base = jnp.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 0, 0],
                            [4, 0, 0], [5, 0, 0], [6, 48, 27], [7, 0, 0]], jnp.int32)
        movement = jnp.stack([jnp.ones(20, jnp.int32), mx, my], -1)
        target = jnp.stack([jnp.full(12, 2, jnp.int32), tx, ty], -1)
        rng = jax.random.fold_in(k, 85002)
        random_keys = jax.random.split(rng, spec['policy_samples'])
        samples = jax.vmap(lambda key: jnp.stack(_sample(lg, key, ~o.entity_pad_mask)[0]))(random_keys)
        actions = jnp.concatenate([base, movement, target, samples])
        allowed = jnp.concatenate([jnp.ones(28, bool), valid, jnp.ones(spec['policy_samples'], bool)])
        return actions, allowed

    actions, allowed = jax.jit(jax.vmap(menu_one))(states, carries, keys)
    menu_end = 40
    np.savez_compressed(out / 'candidate_actions.npz', actions=np.asarray(actions), allowed=np.asarray(allowed))

    def branch_one(s, c, k, forced_action, force_steps, t):
        a, nc, nk, lg, o = act(s, c, k)
        # Repeated option repeats the same screen cell, without any input delay,
        # target-ID lock, altered physics, or suppressed normal AA cooldown.
        a = jnp.where(t < force_steps, forced_action, a)
        live = s.t_ms < 240000
        nxt = env_step(s, decode(a, s), sim)
        cs = jnp.where(live, nxt.cs[0] - s.cs[0], 0.)
        done = nxt.t_ms >= 240000
        d = jnp.stack([cs, lg.value, done.astype(jnp.float32)])
        return choose(live, nxt, s), jnp.where(live, nc, c), nk, d

    @jax.jit
    def branch_chunk(r, forced_action, force_steps, t0):
        def step(r, t):
            ns, nc, nk, d = jax.vmap(branch_one, in_axes=(0, 0, 0, 0, 0, None))(
                *r, forced_action, force_steps, t)
            return (ns, nc, nk), d
        return jax.lax.scan(step, r, jnp.arange(100) + t0)

    @jax.jit
    def final_values(s, c, k):
        return jax.vmap(lambda si, ci, ki: act(si, ci, ki)[3].value)(s, c, k)

    def forks(case_ids, chosen, repetitions, seed, repeat_counts, chunks):
        count, options = chosen.shape[:2]
        idx = np.repeat(case_ids, repetitions * options)
        state = jax.tree.map(lambda x: x[idx], states)
        carry = carries[idx]
        # Every action branch has the same random stream within case/replicate.
        seed_ids = np.repeat(np.asarray(case_ids)[:, None] * 100 + np.arange(repetitions), options, axis=1).reshape(-1)
        kk = jax.vmap(lambda z: jax.random.fold_in(jax.random.key(seed), z))(jnp.asarray(seed_ids))
        fa = jnp.asarray(np.repeat(chosen[:, None], repetitions, axis=1).reshape(-1, 3))
        repeat_counts = np.broadcast_to(repeat_counts, (count, options))
        fs = jnp.asarray(np.repeat(repeat_counts[:, None], repetitions, axis=1).reshape(-1))
        rr, parts = (state, carry, kk), []
        for chunk in range(chunks):
            bound()
            rr, d = jax.block_until_ready(branch_chunk(rr, fa, fs, chunk * 100))
            parts.append(np.asarray(d))
        data = np.concatenate(parts)
        values = np.asarray(final_values(*rr))
        rewards, vs, done = (data[:, :, i] for i in range(3))
        discount = .99 ** np.arange(len(rewards))[:, None]
        mc = (rewards * discount).sum(0)
        boot = mc + .99 ** len(rewards) * values * (1 - done[-1])
        advantages, _ = gae(jnp.asarray(rewards), jnp.asarray(vs), jnp.asarray(done), jnp.asarray(values), .99, .95)
        shape = (count, repetitions, options)
        metrics = dict(cs=rewards.sum(0).reshape(shape), cs_8s=rewards[:80].sum(0).reshape(shape),
                       discounted_return=mc.reshape(shape), boot_return=boot.reshape(shape),
                       gae=np.asarray(advantages[0]).reshape(shape), value=vs[0].reshape(shape))
        for value in metrics.values():
            assert np.isfinite(value).all()
        return metrics

    all_discovery, all_validation = [], []
    for start in range(0, len(selected), 4):
        case_ids = selected[start:start + 4]
        aa = np.asarray(actions)[case_ids]
        # Last group padded so compilation uses one discovery/validation shape.
        real_count = len(case_ids)
        if real_count < 4:
            case_ids = np.pad(case_ids, (0, 4-real_count), mode='edge')
            aa = np.asarray(actions)[case_ids]
        force = np.ones(aa.shape[1], np.int32)
        force[:2] = 0
        discovery = forks(case_ids, aa, spec['discovery_seeds'], 85003, force, 1)
        for val in discovery.values():
            np.testing.assert_array_equal(val[:, :, 0], val[:, :, 1])
        score = discovery['cs_8s'].mean(1)
        score = np.where(np.asarray(allowed)[case_ids], score, -np.inf)
        best_menu = np.argmax(score[:, 2:menu_end], axis=1) + 2
        best_policy = np.argmax(score[:, menu_end:], axis=1) + menu_end
        chosen_ids = np.stack([np.zeros(4, int), best_menu, best_policy, best_menu, np.ones(4, int)], -1)
        chosen = np.take_along_axis(aa, chosen_ids[..., None], axis=1)
        validation = forks(case_ids, chosen, spec['validation_seeds'], 85004,
                           np.asarray([0, 1, 1, 10, 0]), 13)
        for val in validation.values():
            np.testing.assert_array_equal(val[:, :, 0], val[:, :, 4])
        all_discovery.append({k: v[:real_count] for k, v in discovery.items()})
        all_validation.append({k: v[:real_count] for k, v in validation.items()})
        for j in range(real_count):
            case = int(case_ids[j])
            t = int(found_at[case])
            delta8 = validation['cs_8s'][j] - validation['cs_8s'][j, :, :1]
            delta_full = validation['cs'][j] - validation['cs'][j, :, :1]
            report['rows'].append(dict(case=case, decision=t, case_seconds=float(states.t_ms[case]/1000-120),
                chosen_ids=chosen_ids[j].tolist(), chosen_actions=chosen[j].tolist(),
                original_on_policy_gae=float(actual_gae[t, case]), original_value=float(hist['value'][t, case]),
                validation_cs_delta_mean=delta_full.mean(0).tolist(), validation_8s_delta_mean=delta8.mean(0).tolist(),
                validation_cs_deltas=delta_full.tolist(), validation_gae_mean=validation['gae'][j].mean(0).tolist(),
                validation_discounted_return_mean=validation['discounted_return'][j].mean(0).tolist(),
                sampled_policy_discovery_useful_fraction=float((score[j, menu_end:] > score[j, 0]).mean()),
                menu_discovery_useful_fraction=float((score[j, 2:menu_end] > score[j, 0]).mean()),
                aa_cooldown=float(states.aa_cooldown[case, 0]), aa_windup=float(states.aa_windup[case, 0])))
        save()
        print('VALIDATED', min(start+4, len(selected)), '/', len(selected), flush=True)
    np.savez_compressed(out / 'discovery.npz', **{k: np.concatenate([d[k] for d in all_discovery]) for k in all_discovery[0]})
    np.savez_compressed(out / 'validation.npz', **{k: np.concatenate([d[k] for d in all_validation]) for k in all_validation[0]})
    rows = report['rows']
    useful = [r for r in rows if r['validation_8s_delta_mean'][1] >= .5 and r['validation_cs_delta_mean'][1] >= 0]
    report.update(status='complete', elapsed_seconds=time.monotonic()-started,
        validated_menu_cases=len(useful), search_signal_gate=len(useful) >= spec['useful_case_gate'],
        validation_arm_names=['natural', 'best_menu_one_action', 'best_sampled_policy_one_action', 'best_menu_repeated_1s', 'natural_duplicate'],
        mean_cs_delta=np.mean([r['validation_cs_delta_mean'] for r in rows], axis=0).tolist(),
        limitations='Opportunity-conditioned real cases, not optimal-policy/global perfect-CS evidence. Discovery ranks 8s total CS using two continuations; four independent seeds validate selected actions and remaining-episode CS. Sampled-policy useful-action fraction is a noisy Monte Carlo discovery estimate INCLUDING physical ground attack-move acquisition, not exact aggregate mass or a certified probability. Menu coverage is a frozen intervention, not a trained menu policy. GAE on forced branches is diagnostic; only saved original trajectories are on-policy. Repeated arm repeats the same screen cell at native cadence for1s; it is a separate sequence intervention, not one-action credit, and introduces no latency. Actor information sufficiency is not established by this study.')
    save()
    print('RESULT', json.dumps({k: report[k] for k in ('status', 'cases', 'validated_menu_cases', 'search_signal_gate', 'mean_cs_delta')}), flush=True)
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
