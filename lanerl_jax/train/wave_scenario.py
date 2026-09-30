"""E39 reset distribution: level-three tower starts with asymmetric health.

Only initial conditions change. Subsequent combat, regeneration and spawning
use the production simulator. This is a finite-horizon curriculum task.
"""
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from ..sim.init import init_lane, spawn_minion, TOP_OUTER_TURRET, TOP_LANE_PATH
from ..sim.state import Kind, MI_SLICE
from ..sim.profiles import profile_id
from ..sim.orders import Orders, OrderKind
from ..sim.step import env_step

START_MS = 120000.

def point_on_path(path, distance):
    path = np.asarray(path, float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    i = int(np.clip(np.searchsorted(cumulative, distance, side='right')-1, 0, len(lengths)-1))
    return path[i]+(path[i+1]-path[i])*((distance-cumulative[i])/lengths[i]), i+1


def turret_distance(path, tower):
    path = np.asarray(path, float); delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    t = np.clip(np.sum((np.asarray(tower)-path[:-1])*delta,axis=1)/lengths**2,0,1)
    projected = path[:-1]+t[:,None]*delta
    i = np.argmin(np.linalg.norm(projected-tower,axis=1))
    return float(np.r_[0.,np.cumsum(lengths)][i]+t[i]*lengths[i])


def raw_state(sim, seed, offset=0.):
    s = init_lane(seed=seed)
    s = s.replace(t_ms=jnp.float32(START_MS), next_spawn_ms=jnp.float32(START_MS+30000),
                  xp=s.xp.at[:2].set(sim.params['xp_to_reach_level'][3]))
    # Move both starts equally toward/away from the centre; retain native map geometry.
    for team in (1,0):
        path = np.asarray(TOP_LANE_PATH)[::1 if team==0 else -1]
        distance = turret_distance(path, TOP_OUTER_TURRET[team])+100.+offset
        xy,_ = point_on_path(path,distance)
        xy=xy.astype(np.float32)
        s=s.replace(x=s.x.at[team].set(xy[0]), y=s.y.at[team].set(xy[1]),
            collision_x=s.collision_x.at[team].set(xy[0]), collision_y=s.collision_y.at[team].set(xy[1]),
            waypoints=s.waypoints.at[team,0].set(xy), n_waypoints=s.n_waypoints.at[team].set(1),
            waypoint_key=s.waypoint_key.at[team].set(1))
    # Match normal spawn insertion order: red then blue for each minion.
    for k in range(6):
        for team in (1,0):
            path=np.asarray(TOP_LANE_PATH)[::1 if team==0 else -1]
            distance=turret_distance(path,TOP_OUTER_TURRET[team])+100.+offset-250.-260.*k
            xy,next_vertex=point_on_path(path,distance)
            model=profile_id(Kind.LANE_MINION,0 if k<3 else 1,team)
            free=np.flatnonzero(~np.asarray(s.alive)[MI_SLICE])[0]+MI_SLICE.start
            s=spawn_minion(s,team,model,sim.params['max_hp'][model],jnp.asarray(path),spawn_xy=xy)
            s=s.replace(lane_waypoint_key=s.lane_waypoint_key.at[free].set(next_vertex))
    return s


def health_pair(s):
    # Recent damage delays passive naturally; it resumes under normal mechanics.
    s=s.replace(t_ms=jnp.float32(START_MS), ms_since_damaged=s.ms_since_damaged.at[:2].set(0.))
    return [s.replace(hp=s.hp.at[:2].set(s.max_hp[:2]*jnp.asarray(r)))
            for r in ((.7,1.),(1.,.7))]


def prepare_scenario_bank(sim, out: Path, offsets, seed):
    out.mkdir(parents=True,exist_ok=True)
    noop=Orders(jnp.zeros(2,jnp.int8),jnp.zeros(2),jnp.zeros(2),jnp.full(2,-1,jnp.int8))
    # One real step installs level-three stats/ranks/passives and minion routing.
    step=jax.jit(lambda s: env_step(s,noop,sim))
    states=[]; rows=[]
    for i,offset in enumerate(offsets):
        s=jax.block_until_ready(step(raw_state(sim,seed+i,float(offset))))
        for low,pair in enumerate(health_pair(s)):
            assert np.all(np.asarray(pair.level[:2])==3)
            assert np.all(np.asarray(pair.cs[:2])==0)
            assert np.all(np.asarray(pair.spell_cooldown[:2])==0)
            assert int(np.asarray((pair.kind==Kind.LANE_MINION)&pair.alive).sum())==12
            states.append(pair)
            rows.append(dict(index=len(states)-1,offset=float(offset),low_hp_team=low,
                level=np.asarray(pair.level[:2]).tolist(),hp=np.asarray(pair.hp[:2]).tolist(),
                max_hp=np.asarray(pair.max_hp[:2]).tolist(),spell_level=np.asarray(pair.spell_level[:2]).tolist(),
                xy=np.stack([pair.x[:2],pair.y[:2]],-1).tolist()))
    (out/'setup.json').write_text(json.dumps(rows,indent=2))
    return jax.tree.map(lambda *xs:jnp.stack(xs),*states)


def park_afk_opponent(bank):
    """Park red at its fountain with no held order; simulator remains unchanged."""
    from ..sim.init import CHAMPION_SPAWN
    xy=jnp.asarray(CHAMPION_SPAWN[1],jnp.float32)
    def one(s):
        return s.replace(x=s.x.at[1].set(xy[0]),y=s.y.at[1].set(xy[1]),
            collision_x=s.collision_x.at[1].set(xy[0]),collision_y=s.collision_y.at[1].set(xy[1]),
            waypoints=s.waypoints.at[1,0].set(xy),n_waypoints=s.n_waypoints.at[1].set(1),
            waypoint_key=s.waypoint_key.at[1].set(1),target=s.target.at[1].set(-1),
            hp=s.hp.at[1].set(s.max_hp[1]))
    return jax.vmap(one)(bank)
