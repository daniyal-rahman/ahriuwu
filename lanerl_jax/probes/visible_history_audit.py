"""DIAG E71: visible-only associations against offline E68 identity labels.

This reconstructs entity tables and self position from recorded snapshots; it
does not claim to reconstruct unrecorded HUD/buff fields or policy history.
Unit IDs and spawn sequence are used only by the independent accuracy scorer.
No environment stepping, parameter updates, or model export.
"""
import json
import os
from pathlib import Path
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.obs.builder import build_observation, NORM_DIST, NORM_XY
from lanerl_jax.obs.visible_history import append_visible_history, empty_visible_history, PAST_SAMPLES
from lanerl_jax.parity.policy_driver import _lane_frames
from lanerl_jax.sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.sim.init import init_lane
from lanerl_jax.train.run_manifest import git_provenance, file_sha256


def main():
    spec = json.loads(Path('experiments', sys.argv[1]+'.json').read_text())
    provenance = git_provenance()
    out = Path('/mnt/nfs/shared')/spec['id']
    out.mkdir(exist_ok=False)
    stage = Path('/scratch')/(spec['id']+'-'+os.environ['SLURM_JOB_ID'])
    stage.mkdir()
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT, stage/'routes')
    shutil.copyfile(spec['reference_trace'], stage/'trace.npz')
    sim = SimConfig.training(route_artifact=stage/'routes').replace(step_ticks=6)
    template = init_lane()
    frame = _lane_frames()[0]
    trace = dict(np.load(stage/'trace.npz'))
    fields = ('x','y','hp','max_hp','alive','kind','team','model','level',
              'spell_level','spell_cooldown','t_ms','r_cast_ms')

    def observe(row):
        state = template.replace(**row)
        obs = build_observation(state, 0, frame, params=sim.params, horizon_s=600., vision=sim.vision)
        return obs.entities, obs.entity_pad_mask, obs.self_vec[:2], obs.slot_unit

    batch = jax.jit(jax.vmap(observe))
    chunks = []
    for start in range(0, len(trace['t_ms']), 32):
        indices = np.minimum(np.arange(start, start+32), len(trace['t_ms'])-1)
        row = {name: jnp.asarray(trace[name][indices]) for name in fields}
        chunks.append(jax.tree.map(np.asarray, jax.block_until_ready(batch(row))))
    e, mask, own_xy, unit = [np.concatenate([c[i] for c in chunks])[:len(trace['t_ms'])] for i in range(4)]

    @jax.jit
    def histories(e, mask, xy):
        def one(memory, row):
            features, memory = append_visible_history(*row, memory)
            return memory, features
        return jax.lax.scan(one, empty_visible_history(), (e, mask, xy))[1]

    augmented = np.asarray(histories(jnp.asarray(e), jnp.asarray(mask), jnp.asarray(own_xy)))
    features = augmented[..., 16:].reshape(len(e), 32, PAST_SAMPLES, 4)
    np.testing.assert_array_equal(augmented[..., :16], e)
    assert np.isfinite(augmented).all()
    assert not features[mask].any(), 'history leaked through an absent current row'
    position = e[..., 1:3] + own_xy[:, None, :]*(NORM_XY/NORM_DIST)
    seq = np.take_along_axis(trace['spawn_seq'], np.maximum(unit, 0), axis=1)
    # Diagnostic labels only: neither array was passed to append_visible_history.
    identity = np.where(unit >= 0, seq.astype(np.int64)*trace['x'].shape[1]+unit, -1)
    by_lag = []
    for lag in range(1, PAST_SAMPLES+1):
        match = ((identity[lag:, :, None] == identity[:-lag, None, :])
                 & (identity[lag:, :, None] >= 0))
        has_truth = match.any(-1)
        previous = match.argmax(-1)
        hp = np.take_along_axis(e[:-lag, :, 3], previous, axis=1)
        xy = np.take_along_axis(position[:-lag], previous[..., None], axis=1)
        expected = np.concatenate([hp[..., None], xy-position[lag:]], axis=-1)
        claimed = features[lag:, :, lag-1, 3] > .5
        correct = has_truth & np.isclose(features[lag:, :, lag-1, :3], expected, atol=1e-5, rtol=1e-5).all(-1)
        minion = e[lag:, :, 5] > .5
        by_lag.append(dict(lag=lag, claimed=int(claimed.sum()), correct=int((claimed&correct).sum()),
            minion_claimed=int((claimed&minion).sum()), minion_correct=int((claimed&correct&minion).sum()),
            minion_available=int((has_truth&minion).sum()),
            minion_recovered=int((claimed&correct&minion).sum())))
    total = sum(r['claimed'] for r in by_lag)
    correct = sum(r['correct'] for r in by_lag)
    precision = correct/max(total, 1)
    coverage = by_lag[0]['minion_recovered']/max(by_lag[0]['minion_available'], 1)
    cases = []
    for target, index in ((15,792),(15,808),(19,1125),(19,1129)):
        rows = np.flatnonzero(unit[index] == target)
        assert len(rows) == 1, (target,index,rows)
        row = int(rows[0])
        cases.append(dict(target=target, frame=index, seconds=(float(trace['t_ms'][index])-120000)/1000,
            known=features[index,row,:,3].tolist(), hp=features[index,row,:,0].tolist(),
            current_hp=float(e[index,row,3])))
    passed = precision >= spec['minimum_precision'] and coverage >= spec['minimum_one_step_coverage']
    report = dict(spec=spec, source=provenance, trace_sha256=file_sha256(stage/'trace.npz'),
        frames=len(e), claimed_history_samples=total, correct_history_samples=correct,
        incorrect_history_samples=total-correct, precision=precision,
        consecutive_visible_minion_coverage=coverage, by_lag=by_lag, cases=cases, passed=passed,
        limits='One recorded game. Identity labels score history content offline only; actor history uses position/type matching. Unknown histories are masked rather than filled from hidden state. Entity tables and self position reconstructed, not full unrecorded HUD/buff or GRU state. Association correctness does not demonstrate farming improvement.')
    np.savez_compressed(out/'visible_histories.npz', entities=e, mask=mask, own_xy=own_xy,
                        history=features, diagnostic_identity=identity)
    (out/'result.json').write_text(json.dumps(report, indent=2))
    print('HISTORY AUDIT', json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError('visible association quality gate failed; do not start learning experiment')
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
