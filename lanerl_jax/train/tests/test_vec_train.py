"""The vectorised trainer's invariants: its on-device relative reward is the
collector path's `server_train.relative_reward`; the loop runs for gru and mlp
cores and a scripted opponent; and the rollout's log-probs equal the learner's
recomputation (actor/learner agreement, the silent-ratio bug class)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim.config import SimConfig
from lanerl_jax.sim.init import init_lane
from lanerl_jax.train.policy import PolicyConfig
from lanerl_jax.train.ppo import PPOConfig, factored_log_prob
from lanerl_jax.train.vec_train import VecConfig, _relative_reward, make_vec_train


def _bank(k=2):
    return jax.tree.map(lambda *a: jnp.stack(a), *(init_lane(seed=i) for i in range(k)))


def _sim():
    return SimConfig.training().replace(step_ticks=6)


def test_relative_reward_matches_collector_path():
    from lanerl_jax.train import server_train as st
    from lanerl_jax.train.reward import lane_corridor_distance
    cfg = VecConfig(gold_scale=20.0, xp_scale=0.008, enemy_scale=1.0)
    prev = init_lane(seed=0)
    nxt = prev.replace(gold=prev.gold.at[0].add(30.).at[1].add(10.),
                       xp=prev.xp.at[0].add(77.).at[1].add(51.),
                       x=prev.x.at[0].add(400.), y=prev.y.at[1].add(-300.))
    r, terms = _relative_reward(prev, nxt, cfg)
    def stats(s):
        pot = -lane_corridor_distance(s.x[:2], s.y[:2]) / 10000.
        return np.stack([np.asarray(s.cs[:2], np.float32), np.ones(2, np.float32),
                         np.asarray(pot), np.asarray(s.xp[:2], np.float32),
                         np.asarray(s.gold[:2], np.float32)], -1)
    st.RELATIVE_REWARD.update(mode="relative", gold_scale=20.0, xp_scale=0.008, enemy_scale=1.0)
    sb, sa = stats(prev), stats(nxt)
    ref, ref_terms = st.relative_reward(sb, sa, sb[::-1], sa[::-1])
    np.testing.assert_allclose(np.asarray(r), np.asarray(ref), rtol=1e-5, atol=1e-6)
    for k in ("cs", "xp", "approach"):
        np.testing.assert_allclose(np.asarray(terms[k]), np.asarray(ref_terms[k]), rtol=1e-5, atol=1e-6)
    assert float(jnp.abs(terms["death"]).sum()) == 0.0
    # zero-sum in a mirror: the gold and xp terms sum to zero over the two champions
    assert abs(float((terms["cs"] + terms["xp"]).sum())) < 1e-5


@pytest.mark.parametrize("core,opponent", [("gru", "mirror"), ("mlp", "mirror"), ("gru", "lasthit"), ("gru", "frozen"), ("gru", "afk")])
def test_loop_runs_and_actor_learner_agree(core, opponent):
    pcfg = PolicyConfig(core=core, core_norm=(core == "gru"), core_residual=(core == "gru"),
                        d_model=32, n_layers=1, ffn_dim=32, ctx_dim=32, core_dim=32,
                        mlp_hidden=32, mlp_layers=1)
    cfg = VecConfig(n_envs=2, rollout_steps=6, n_updates=2, n_minibatches=2, episode_s=2.0,
                    opponent=opponent, ppo=PPOConfig.standard(decision_hz=10.0), policy=pcfg)
    if opponent == "afk":
        cfg = cfg._replace(xp_scale=0., tower_damage_gold=900., tower_damage_personal=True, health_loss_gold=100.)
    opponent_params = None
    if opponent == "frozen":
        reference = make_vec_train(cfg._replace(opponent="mirror"), _sim(), _bank())
        opponent_params = reference["init_params"](jax.random.key(17))
    built = make_vec_train(cfg, _sim(), _bank(), opponent_params=opponent_params)
    runner = built["initial_runner"](jax.random.key(0))
    runner2, tr, batch = jax.jit(built["rollout"])(runner)
    if opponent == "afk":
        assert np.all(np.asarray(tr.action[0][:,:,1]) == 0)
        assert batch["carry0"].shape[0] == cfg.n_envs
    lg = built["loss"].forward(runner.params, batch)
    lp = factored_log_prob((lg.button, lg.screen_x, lg.screen_y), batch["action"],
                           batch["uses_screen"], click_mask=batch.get("click_mask"))
    np.testing.assert_allclose(np.asarray(lp), np.asarray(batch["log_prob"]), rtol=1e-4, atol=1e-5)
    # Deadlines are floored at two rollouts from the start, so no done inside
    # the first rollout by design; four more updates (2.4 s of game at 10 Hz)
    # must end every 2-s episode at least once; `cs_episodes` counts only FULL
    # episodes (the staggered first one is excluded), so run 8 updates = 4.8 s.
    runner3, m = jax.jit(built["run_chunk"], static_argnums=1)(runner2, 8)
    assert float(np.asarray(m["cs_episodes"]).sum()) > 0, "no full episode ended in 4.8 s of game"
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "post_kl", "reward",
              "reward_cs", "reward_xp", "reward_approach", "lane_dist", "cs_episodes"):
        assert k in m and np.isfinite(np.asarray(m[k])).all(), k
    assert int(runner3.step) == 8 * cfg.n_envs * cfg.learn_agents * cfg.rollout_steps
    if core == "gru":
        # the carry is zeroed on done, both for the env and the batch's carry0
        assert batch["carry0"].shape == (cfg.n_envs * cfg.learn_agents, 32)


def test_stagger_warmup_preserves_weights_and_full_episode_contract():
    from lanerl_jax.train.wave_scenario_train import warmup_staggered_runner
    pcfg=PolicyConfig(core='gru',core_norm=True,core_residual=True,
        d_model=16,n_layers=1,n_heads=4,ffn_dim=32,ctx_dim=32,core_dim=32,
        mlp_hidden=32,mlp_layers=1)
    cfg=VecConfig(n_envs=4,rollout_steps=6,n_minibatches=2,episode_s=2.,
        opponent='afk',policy=pcfg,stagger_initial=True)
    built=make_vec_train(cfg,_sim(),_bank())
    initial=built['initial_runner'](jax.random.key(13))
    warmed,info=warmup_staggered_runner(built,initial,cfg)
    for field in ('params','opt_state','step'):
        for before,after in zip(jax.tree.leaves(getattr(initial,field)),
                               jax.tree.leaves(getattr(warmed,field))):
            np.testing.assert_array_equal(before,after)
    assert info['untrained_environment_decisions']==5*6*4
    assert info['clock_max_ms']>info['clock_min_ms']+100
    np.testing.assert_array_equal(warmed.deadline_ms,2000.)
    # Every future reset is a normal full game; no artificial warmup terminal
    # can enter learning. Real recurrent carry must agree with recomputation.
    _,tr,batch=jax.jit(built['rollout'])(warmed)
    np.testing.assert_array_equal(tr.done,tr.done_full)
    lg=built['loss'].forward(warmed.params,batch)
    lp=factored_log_prob((lg.button,lg.screen_x,lg.screen_y),batch['action'],batch['uses_screen'])
    np.testing.assert_allclose(lp,batch['log_prob'],rtol=1e-4,atol=1e-5)
    assert np.asarray(warmed.carry).any()


def test_relative_reward_afk_death_cost_is_once_per_event():
    p = init_lane(seed=0)
    dead = p.replace(alive=p.alive.at[0].set(False), hp=p.hp.at[0].set(0),
                     deaths=p.deaths.at[0].add(1))
    cfg = VecConfig(death_loss_gold=300., health_loss_gold=100.)
    base, base_terms = _relative_reward(p, dead, cfg._replace(death_loss_gold=0.))
    reward, terms = _relative_reward(p, dead, cfg)
    np.testing.assert_allclose(reward-base, [-15., 0.])
    np.testing.assert_allclose(terms['death'], [-15., 0.])
    np.testing.assert_allclose(terms['health'], base_terms['health'])
    np.testing.assert_array_equal(base_terms['death'], [0., 0.])
    # Continued dead frames and respawn never incur another death charge.
    respawn = p.replace(deaths=dead.deaths)
    for before, after in ((dead, dead), (dead, respawn)):
        np.testing.assert_array_equal(_relative_reward(before, after, cfg)[1]['death'], [0., 0.])
    # The event belongs only to the champion who died, including red.
    red_dead = p.replace(deaths=p.deaths.at[1].add(1))
    np.testing.assert_array_equal(_relative_reward(p, red_dead, cfg)[1]['death'], [0., -15.])


def test_relative_reward_afk_health_and_tower():
    from lanerl_jax.sim.init import TOP_OUTER_TURRET
    p=init_lane(seed=0)
    xy=np.stack([p.x,p.y],-1);u=int(np.argmin(np.sum((xy-TOP_OUTER_TURRET[1])**2,-1)))
    cfg=VecConfig(health_loss_gold=100.,tower_damage_gold=900.)
    q=p.replace(hp=p.hp.at[0].add(-p.max_hp[0]*.1).at[u].add(-p.max_hp[u]/6))
    _,terms=_relative_reward(p,q,cfg)
    np.testing.assert_allclose(terms['health'][0],-.5,atol=1e-5)
    np.testing.assert_allclose(terms['tower'][0],7.5,atol=1e-5)
    growth=p.replace(hp=p.hp.at[0].add(100),max_hp=p.max_hp.at[0].add(100))
    assert float(_relative_reward(p,growth,cfg)[1]['health'][0])==0
    dead=p.replace(alive=p.alive.at[0].set(False),hp=p.hp.at[0].set(0))
    assert float(_relative_reward(dead,p,cfg)[1]['health'][0])==0
    np.testing.assert_allclose(_relative_reward(p,dead,cfg)[1]['health'][0],-5.)


def test_relative_reward_personal_tower_and_no_xp():
    from lanerl_jax.sim.init import TOP_OUTER_TURRET
    p=init_lane(seed=0)
    xy=np.stack([p.x,p.y],-1)
    u=int(np.argmin(np.sum((xy-TOP_OUTER_TURRET[1])**2,-1)))
    cfg=VecConfig(xp_scale=0.,tower_damage_gold=900.,tower_damage_personal=True)
    q=p.replace(hp=p.hp.at[u].add(-300),xp=p.xp.at[0].add(500))
    r,terms=_relative_reward(p,q,cfg)
    assert float(terms['tower'][0])==0. and float(terms['xp'][0])==0.
    q=q.replace(champion_tower_damage=q.champion_tower_damage.at[0,u].add(100))
    _,terms=_relative_reward(p,q,cfg)
    np.testing.assert_allclose(terms['tower'][0],45*100/float(p.max_hp[u]),rtol=1e-6)
    assert float(terms['tower'][1])==0.


def test_relative_reward_personal_tower_attribution():
    from lanerl_jax.sim.combat import effective_champion_tower_damage
    # Rows: buff, blue Garen, red Garen, blue minion, orphan projectile.
    # Victims: champs0/1, red tower2, blue tower3, red minion4.
    damage=np.zeros((5,5),np.float32)
    damage[1,2]=80;damage[3,2]=90;damage[2,3]=200;damage[1,4]=50
    hp=np.array([100,100,100,30,100],np.float32)
    kind=np.array([1,1,3,3,2]);team=np.array([0,1,1,0,1]);alive=np.ones(5,bool)
    def attribution(d):
        return effective_champion_tower_damage(d,np.cumsum(d,axis=0),hp,kind,team,alive,np)
    got=attribution(damage)
    np.testing.assert_array_equal(got,[[0,0,80,0,0],[0,0,0,30,0]])
    damage[1,2]=0 # same minion hit, no Garen damage
    assert attribution(damage)[0,2]==0
    damage[1,2]=150 # overkill never earns more than HP removed
    assert attribution(damage)[0,2]==100
    alive[2]=False
    assert attribution(damage)[0,2]==0


def test_relative_reward_personal_tower_live_hit():
    from lanerl_jax.sim.init import TOP_OUTER_TURRET, lane_params
    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.step import tick
    p=init_lane(seed=0)
    xy=np.stack([p.x,p.y],-1)
    u=int(np.argmin(np.sum((xy-TOP_OUTER_TURRET[1])**2,-1)))
    live=jnp.zeros_like(p.alive).at[0].set(True).at[u].set(True)
    p=p.replace(alive=live,x=p.x.at[0].set(p.x[u]+60),y=p.y.at[0].set(p.y[u]),
        collision_x=p.collision_x.at[0].set(p.x[u]+60),collision_y=p.collision_y.at[0].set(p.y[u]),
        target=p.target.at[0].set(u),aa_target=p.aa_target.at[0].set(u),
        is_attacking=p.is_attacking.at[0].set(True),aa_windup=p.aa_windup.at[0].set(.001),
        aa_cooldown=p.aa_cooldown.at[0].set(1.0))
    q=jax.jit(lambda s:tick(s,lane_params(load_patch())))(p)
    removed=float(p.hp[u]-q.hp[u]);assert removed>0
    np.testing.assert_allclose(q.champion_tower_damage[0,u],removed,rtol=1e-5)
    # Telemetry has no effect on any physical state field.
    r=jax.jit(lambda s:tick(s,lane_params(load_patch())))(p.replace(
        champion_tower_damage=jnp.ones_like(p.champion_tower_damage)*123))
    np.testing.assert_array_equal(q.hp,r.hp)
