"""Frozen E39/E40 75-second scenario recording; LEARN-PAIR-07, no training."""
import json, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from lanerl_jax.sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.sim.step import env_step
from lanerl_jax.parity.policy_driver import load_params, _lane_frames
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.train.wave_scenario import prepare_scenario_bank, START_MS
from lanerl_jax.train.actions import orders_from, _screen_to_centred_lane
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.run_manifest import file_sha256, git_provenance
from lanerl_jax.replay import state_snapshot, render_metadata, summarize
from lanerl_rl.constants import N_SCREEN_X, N_SCREEN_Y


def main():
    spec=json.loads(Path('experiments',sys.argv[1]+'.json').read_text())
    out=Path('/mnt/nfs/shared')/spec['id'];out.mkdir(exist_ok=False)
    stage=Path('/mnt/nfs/shared')/(spec['id']+'-staged-'+os.environ['SLURM_JOB_ID'])
    stage.mkdir();shutil.copytree(DEFAULT_ROUTE_ARTIFACT,stage/'routes')
    source=Path(spec['checkpoint'])
    shutil.copyfile(source,stage/'checkpoint.msgpack')
    shutil.copyfile(source.parent/'manifest.json',stage/'manifest.json')
    sim=SimConfig.training(route_artifact=stage/'routes').replace(step_ticks=6)
    policy,params,_=load_params(str(stage/'checkpoint.msgpack'))
    assert not policy.cfg.click_mask, 'This probe mirrors the unmasked E40 policy only'
    bank=prepare_scenario_bank(sim,out/'bank',[spec['offset']],1007)
    frames=_lane_frames()
    @jax.jit
    def act(state,carry,key):
        ob=[build_observation(state,t,frames[t],params=sim.params,horizon_s=600.,vision=sim.vision) for t in (0,1)]
        obs=jax.tree.map(lambda a,b:jnp.stack([a,b]),*ob)
        lg,carry=policy.apply(params,obs.entities,obs.entity_pad_mask,obs.self_vec,obs.global_vec,carry)
        action,_,_=_sample(lg,key,~obs.entity_pad_mask)
        return action,carry
    decode=jax.jit(lambda a,s:orders_from(a,s,None,frames[0],snap_moves=False,
        params=sim.params,vision=sim.vision,drop_unwalkable_moves=True))
    step=jax.jit(lambda s,o:env_step(s,o,sim))
    @jax.jit
    def snapshot(s,o,a):
        ds,dn=_screen_to_centred_lane((a[1]+.5)/N_SCREEN_X,(a[2]+.5)/N_SCREEN_Y)
        side=jnp.where(s.team[:2]==0,1.,-1.)
        axis,normal=frames[0].axis,frames[0].normal
        cursor=SimpleNamespace(x=s.x[:2]+side*ds*axis[0]+dn*normal[0],y=s.y[:2]+side*ds*axis[1]+dn*normal[1])
        return state_snapshot(s,o,a,cursor)
    for low in (0,1):
        dest=out/f'low_team_{low}';dest.mkdir()
        state=jax.tree.map(lambda x:x[low],bank)
        carry=policy.initial_carry((2,));key=jax.random.key(spec['seed']);rows=[]
        assert np.allclose(np.asarray(state.hp[:2]/state.max_hp[:2]),[.7,1.] if low==0 else [1.,.7])
        while float(state.t_ms)<START_MS+spec['seconds']*1000:
            key,ak=jax.random.split(key)
            action,carry=act(state,carry,ak);orders=decode(action,state)
            rows.append(jax.tree.map(np.asarray,snapshot(state,orders,action)))
            state=jax.block_until_ready(step(state,orders))
            if len(rows)%100==0:print('RECORD',low,len(rows),float(state.t_ms),flush=True)
        rows.append(jax.tree.map(np.asarray,snapshot(state,orders,tuple(jnp.zeros(2,jnp.int32) for _ in range(3)))))
        data={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        meta=dict(label=f'E40 u{spec["update"]} | 75s mirror | low HP team {low}',
            controller='frozen policy on both champions',environment='jax-wave-scenario',
            checkpoint=str(source),checkpoint_sha256=file_sha256(stage/'checkpoint.msgpack'),
            source=git_provenance(),seed=spec['seed'],hz=10,seconds=spec['seconds'],start_seconds=120.,
            observation_horizon_s=600.,view='omniscient diagnostic; not actor input',
            action_alignment='pre-action; terminal frame has unsent NOOP placeholder',
            terrain=dict(min_x=float(sim.terrain.min_x),min_y=float(sim.terrain.min_y),cell_size=float(sim.terrain.cell_size)),
            **render_metadata())
        meta['summary']=summarize(data)
        np.savez_compressed(dest/'trace.npz',**data,walkable=np.asarray(sim.terrain.walkable),metadata=np.asarray(json.dumps(meta)))
        (dest/'trace.json').write_text(json.dumps(meta,indent=2))
        print('REPLAY READY',str(dest),flush=True)
    print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
