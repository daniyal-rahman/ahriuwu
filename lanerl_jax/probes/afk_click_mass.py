"""DIAG LEARN-AFK-19: exact click mass on saved E59 caster cases; no simulation.

Run under ops/login_capped.sh. One frozen forward pass over 16 saved states;
the saved recurrent carries are valid ONLY for the original E59 checkpoint.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import msgpack_restore

from lanerl_jax.obs.builder import build_observation, NORM_DIST
from lanerl_jax.obs.fog import visible_to
from lanerl_jax.obs.vision import map1_vision
from lanerl_jax.parity.policy_driver import _lane_frames
from lanerl_jax.sim.init import init_lane, lane_params
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
from lanerl_jax.train.replay_audit import restore_replay_state
from lanerl_jax.train.run_manifest import file_sha256, git_provenance
from lanerl_jax.train.scripted_policy import screen_grid
from lanerl_jax.train.click_proposals import proposal_cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    jax.config.update('jax_default_matmul_precision', 'highest')
    report = json.loads((a.cases.parent / 'result.json').read_text())
    assert file_sha256(a.checkpoint) == report['checkpoint_sha256'], 'carry/checkpoint mismatch'
    n = report['cases']
    manifest = json.loads((a.checkpoint.parent / 'manifest.json').read_text())
    cfg = PolicyConfig(**manifest['config']['train']['policy'])
    assert cfg.observation_interface == 'viewport-structured-v3'
    policy = LanePolicy(cfg)
    params = msgpack_restore(a.checkpoint.read_bytes())['params']
    template = dict(state=init_lane(seed=0), carry=policy.initial_carry((n,)),
                    key=jax.random.split(jax.random.key(0), n), target=jnp.zeros(n, jnp.int32))
    saved = restore_replay_state(template, a.cases.read_bytes())
    state = jax.tree.map(jnp.asarray, saved['state'])
    frame, profiles, vision = _lane_frames()[0], lane_params(), map1_vision()
    observe = jax.jit(jax.vmap(lambda s: build_observation(
        s, 0, frame, params=profiles, horizon_s=600., vision=vision)))
    obs = observe(state)
    logits, _ = jax.jit(policy.apply)(params, obs.entities, obs.entity_pad_mask,
        obs.self_vec, obs.global_vec, jnp.asarray(saved['carry']))
    value_error = float(np.max(np.abs(np.asarray(logits.value) - report['initial_value'])))
    assert value_error < 1e-4, ('saved history value mismatch', value_error)
    pb, px, py = [np.asarray(jax.nn.softmax(x)) for x in
                  (logits.button, logits.screen_x, logits.screen_y)]
    visible = np.asarray(jax.jit(jax.vmap(lambda s: visible_to(
        0, s.x, s.y, s.kind, s.team, s.alive, vision)))(state))
    ds, dn, clickable = screen_grid()  # Y,X layout
    dx = ds * float(frame.axis[0]) + dn * float(frame.normal[0])
    dy = ds * float(frame.axis[1]) + dn * float(frame.normal[1])
    radii = np.asarray(profiles['collision_radius'])[np.asarray(state.model)]
    rows = []
    for i in range(n):
        x, y = np.asarray(state.x[i]), np.asarray(state.y[i])
        d2 = (x[None, None, :] - (x[0]+dx[..., None]))**2
        d2 += (y[None, None, :] - (y[0]+dy[..., None]))**2
        eligible = visible[i] & np.asarray(state.alive[i]) & (np.arange(len(x)) != 0)
        hit = (d2 <= radii[i]**2) & eligible
        picked = np.argmin(np.where(hit, d2, np.inf), axis=-1)
        target = int(saved['target'][i])
        target_cells = hit.any(-1) & (picked == target) & clickable
        joint = py[i, :, None] * px[i, None, :]
        mass = float(joint[target_cells].sum())
        e = np.asarray(obs.entities[i])
        # A hypothetical 10% uniform proposal over observed enemy minions;
        # no low-HP ranking or privileged selection is involved.
        candidates = np.flatnonzero(~np.asarray(obs.entity_pad_mask[i]) &
                                    (e[:, 5] > .5) & (e[:, 11] > .5))
        hits = 0
        for slot in candidates:
            err = (ds-e[slot, 1]*NORM_DIST)**2 + (dn-e[slot, 2]*NORM_DIST)**2
            flat = int(np.argmin(np.where(clickable, err, np.inf)))
            hits += bool(target_cells.flat[flat])
        proposal_mass = hits / len(candidates) if len(candidates) else mass
        implemented_cells, implemented_valid = proposal_cells(obs.entities[i], obs.entity_pad_mask[i])
        implemented_cells = np.asarray(implemented_cells)[np.asarray(implemented_valid)]
        implemented_hits = int(target_cells[implemented_cells % 54, implemented_cells // 54].sum())
        implemented_mass = implemented_hits / len(implemented_cells) if len(implemented_cells) else mass
        cursor_button = float(pb[i, 1] + pb[i, 2])
        rows.append(dict(case=i, target_cells=int(target_cells.sum()),
            aa_cooldown=float(state.aa_cooldown[i,0]), aa_windup=float(state.aa_windup[i,0]),
            cursor_button_probability=cursor_button, attack_move_probability=float(pb[i, 2]),
            conditional_direct_target=mass, direct_target_probability=cursor_button*mass,
            candidate_count=len(candidates), candidate_cells_hitting_target=hits,
            hypothetical_10pct_mixture_direct_target=cursor_button*(.9*mass+.1*proposal_mass),
            implemented_candidate_count=len(implemented_cells), implemented_cells_hitting_target=implemented_hits,
            implemented_10pct_mixture_direct_target=cursor_button*(.9*mass+.1*implemented_mass)))
    # Reuse already-computed E59 counterfactuals; no additional game rollout.
    forks = np.load(a.cases.parent / 'forks.npz')
    for label, length in (('two_seconds',20),('remaining',len(forks['cs']))):
        cs = forks['cs'][:length].sum(0).reshape(n,-1,2).mean(1)
        reward = (forks['reward'][:length] * (.99**np.arange(length))[:,None]).sum(0).reshape(n,-1,2).mean(1)
        for i,row in enumerate(rows):
            row[label+'_directed_delta_cs'] = float(cs[i,1]-cs[i,0])
            row[label+'_directed_delta_return'] = float(reward[i,1]-reward[i,0])
    out = dict(source=git_provenance(), checkpoint_sha256=file_sha256(a.checkpoint),
        cases_sha256=file_sha256(a.cases), value_reproduction_max_error=value_error, rows=rows,
        limitations='16 selected E59 cases, not representative gameplay. Direct clicks only: ground attack-move may auto-acquire. Hypothetical proposal mass is not a trained policy or a gameplay gain.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: float(np.mean([r[k] for r in rows])) for k in
        ('target_cells','cursor_button_probability','conditional_direct_target',
         'direct_target_probability','hypothetical_10pct_mixture_direct_target')}))


if __name__ == '__main__':
    main()
