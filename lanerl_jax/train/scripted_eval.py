"""Run the scripted players (`scripted_policy.py`) through the trainer's wiring.

Same path as ``trainer.make_train``'s ``_env_step``: ``build_observation`` for
blue (unit 0, blue frame) and red (unit 1, red frame) with the training
``SimConfig``'s params and ``horizon_s = 600``, then ``orders_from(action,
state, obs.slot_unit, frame)`` with the BLUE frame for both (as the trainer
does; the decoder mirrors red itself), then ``env_step`` under
``SimConfig.training()`` (routed Moves, deferred terrain repair, TOP waves).
Episodes are exactly 600 s from ``init_lane()``, no reset inside.

The sim and the scripted players are both deterministic and ``init_lane``'s
seed feeds nothing, so a batch of identical envs would be one env copied.
Diversity is injected ONLY at the action boundary: env 0 of every batch is the
unperturbed player; envs 1.. start after a uniform 0-4 s idle in the fountain
and replace each decision with a NOOP with probability ``--drop`` (default
5%, a "missed input"). Both sides are perturbed independently.

Instrumentation is read from the state, never fed to the players: per decision
and champion, the attacked unit's true distance/HP/post-mitigation damage and
the champion's swing clock at order time, and afterwards whether the target
died and whether the kill was credited (``cs`` rose). ``_classify`` turns that
into last-hit attempts, successes and a failure breakdown.

Usage::

    python -m lanerl_jax.train.scripted_eval --envs 16 --seeds 0 1 2 \
        --matchups lasthit:lasthit lasthit:noop brawler:brawler noop:noop
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl.constants import BUTTON_INDEX, DECISION_HZ

from ..obs.builder import HP_BAR_STEPS, NORM_DIST, build_observation
from ..obs.frame import delta_to_lane, make_lane_frame
from ..sim.combat import growth_sum
from ..sim.config import SimConfig
from ..sim.init import TOP_OUTER_TURRET, init_lane
from ..sim.orders import OrderKind
from ..sim.state import Kind, Team
from ..sim.step import env_step
from .actions import orders_from
from .reward import lane_corridor_distance
from .scripted_policy import (CHAMP_ATTACK_RANGE, MINION_COLLISION_RADIUS,
                              PLAYERS)
from .trainer import BLUE_NEXUS, RED_NEXUS

EPISODE_S = 600.0
REACH = CHAMP_ATTACK_RANGE + MINION_COLLISION_RADIUS


def make_runner(blue_name, red_name, n_envs, sim, *, drop=0.05,
                max_delay_s=4.0, chunk=600):
    blue_act, red_act = PLAYERS[blue_name], PLAYERS[red_name]
    params = sim.params
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    red_frame = make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                                TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS)
    frames = (frame, red_frame)

    def obs_of(state):
        b = build_observation(state, 0, frame, params=params, horizon_s=EPISODE_S, vision=sim.vision)
        r = build_observation(state, 1, red_frame, params=params, horizon_s=EPISODE_S, vision=sim.vision)
        return b, r

    def one(state, key, delay_ms, perturb):
        b, r = obs_of(state)
        kb, kr, kd = jax.random.split(key, 3)
        ab = blue_act(b, kb)
        ar = red_act(r, kr)
        action = tuple(jnp.stack([x, y]).astype(jnp.int32) for x, y in zip(ab, ar))
        drop_now = perturb & ((jax.random.uniform(kd, (2,)) < drop)
                              | (state.t_ms < delay_ms))
        button = jnp.where(drop_now, BUTTON_INDEX["noop"], action[0])
        action = (button,) + action[1:]
        slot_unit = jnp.stack([b.slot_unit, r.slot_unit])
        orders = orders_from(action, state, slot_unit, frame, params=params, vision=sim.vision)
        nxt = env_step(state, orders, sim)

        # ---- instrumentation (state-side, never shown to the players) ----
        n = state.kind.shape[0]
        is_att = orders.kind == OrderKind.ATTACK
        u = jnp.where(is_att, orders.target.astype(jnp.int32), -1)
        uc = jnp.clip(u, 0, n - 1)
        me = jnp.arange(2)
        dist = jnp.hypot(state.x[uc] - state.x[me], state.y[uc] - state.y[me])
        ad = (params["attack_damage"][state.model[me]]
              + params["ad_per_level"][state.model[me]] * growth_sum(state.level[me], jnp))
        dmg = ad * 100.0 / (100.0 + params["armor"][state.model[uc]])
        # does the unit the order names look like what the policy read in
        # the slot it chose? (enemy lane minion, same offset, same bar)
        slot = jnp.argmax(slot_unit == u[:, None], axis=1)
        feats = jnp.stack([b.entities[slot[0]], r.entities[slot[1]]])
        ds0, dn0 = delta_to_lane(frame, state.x[uc[0]] - state.x[0], state.y[uc[0]] - state.y[0])
        ds1, dn1 = delta_to_lane(red_frame, state.x[uc[1]] - state.x[1], state.y[uc[1]] - state.y[1])
        tds = jnp.stack([ds0, ds1]) / NORM_DIST
        tdn = jnp.stack([dn0, dn1]) / NORM_DIST
        thp = jnp.round(state.hp[uc] / state.max_hp[uc] * HP_BAR_STEPS) / HP_BAR_STEPS
        enemy_minion = ((state.kind[uc] == Kind.LANE_MINION)
                        & (state.team[uc] != state.team[me]) & state.alive[uc])
        mismatch = is_att & ~(enemy_minion
                              & (jnp.abs(feats[:, 1] - tds) < 1e-4)
                              & (jnp.abs(feats[:, 2] - tdn) < 1e-4)
                              & (jnp.abs(feats[:, 3] - thp) < 1e-4))
        # enemy minions that died this step within 800 of the champion
        died = state.alive & ~nxt.alive & (state.kind == Kind.LANE_MINION)
        d_all = jnp.hypot(state.x[None, :] - state.x[me][:, None],
                          state.y[None, :] - state.y[me][:, None])
        opp = died[None, :] & (state.team[None, :] != state.team[me][:, None]) \
            & (d_all < 800.0)
        rec = dict(
            attack=is_att, unit=u, seq=state.spawn_seq[uc],
            mismatch=mismatch, dist=dist, hp=state.hp[uc], dmg=dmg,
            cd=state.aa_cooldown[me], winding=state.is_attacking[me],
            died=state.alive & ~nxt.alive,
            swinging_on=jnp.where(nxt.is_attacking[me],
                                  nxt.aa_target[me].astype(jnp.int32), -1),
            landed_on=jnp.where(nxt.has_auto_attacked[me]
                                & ~state.has_auto_attacked[me],
                                nxt.aa_target[me].astype(jnp.int32), -1),
            cs_up=nxt.cs[me] > state.cs[me],
            alive=state.alive[me], post_champ_alive=nxt.alive[me],
            in_lane=lane_corridor_distance(state.x[:2], state.y[:2]) <= 0.0,
            opp_died_near=opp.sum(-1),
            opp_died_in_reach=(opp & (d_all <= REACH)).sum(-1),
            opp_died_in_400=(opp & (d_all <= 400.0)).sum(-1),
            button=button,
            cx=state.x[me], cy=state.y[me],
            n_enemy_vis=(~jnp.stack([b.entity_pad_mask, r.entity_pad_mask])[:, 13:25]).sum(-1),
        )
        return nxt, rec

    vone = jax.vmap(one)

    @jax.jit
    def run_chunk(states, key, delay_ms, perturb):
        def body(carry, _):
            st, k = carry
            k, sk = jax.random.split(k)
            keys = jax.random.split(sk, n_envs)
            st, rec = vone(st, keys, delay_ms, perturb)
            return (st, k), rec
        (states, key), recs = jax.lax.scan(body, (states, key), None, length=chunk)
        return states, key, recs

    return run_chunk


def run_matchup(blue, red, n_envs, seed, sim, *, drop=0.05, chunk=600,
                run_s=EPISODE_S, run_chunk=None):
    if run_chunk is None:
        run_chunk = make_runner(blue, red, n_envs, sim, drop=drop, chunk=chunk)
    base = init_lane()
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (n_envs,) + a.shape), base)
    k = jax.random.key(seed)
    k, kd = jax.random.split(k)
    delay = jax.random.uniform(kd, (n_envs,), minval=0.0, maxval=4000.0)
    perturb = jnp.arange(n_envs) > 0
    delay = jnp.where(perturb, delay, 0.0)
    n_steps = int(round(run_s * DECISION_HZ))
    assert n_steps % chunk == 0
    recs = []
    t0 = time.perf_counter()
    for _ in range(n_steps // chunk):
        states, k, rec = run_chunk(states, k, delay, perturb)
        recs.append(jax.tree.map(np.asarray, rec))
    wall = time.perf_counter() - t0
    rec = {kk: np.concatenate([r[kk] for r in recs], 0) for kk in recs[0]}
    return states, rec, wall


def _classify(rec, e, c):
    """Walk one champion's decisions; return attempt counters."""
    T = rec["attack"].shape[0]
    g = lambda k, t: rec[k][t, e, c]  # noqa: E731
    out = Counter()
    reasons = Counter()
    pend = None

    def close(p, how):
        out[how] += 1
        if how != "fail":
            return
        # primary reason, in order of precedence
        if p["mismatch"]:
            r = "slot->unit mismatch"
        elif p["landed"] and p["survived_hit"]:
            r = "hit landed, target survived (damage mis-estimated)"
        elif p["hp_gt_dmg"] and not p["swung"]:
            r = "damage mis-estimated at order (true hp > dmg)"
        elif p["swung"]:
            r = "swing in flight, target taken first"
        elif p["oor"]:
            r = "target out of range at order"
        elif p["not_ready"]:
            r = "swing not ready (cooldown/windup) until target died"
        else:
            r = "never swung (other)"
        reasons[r] += 1
        for flag in ("mismatch", "oor", "not_ready", "hp_gt_dmg", "swung", "landed"):
            if p[flag]:
                reasons["flag:" + flag] += 1

    for t in range(T):
        if g("attack", t):
            u, q = int(g("unit", t)), int(g("seq", t))
            if pend is None or (pend["u"], pend["q"]) != (u, q):
                if pend is not None:
                    close(pend, "retargeted")
                out["attempts"] += 1
                pend = dict(u=u, q=q, mismatch=bool(g("mismatch", t)),
                            oor=bool(g("dist", t) > REACH),
                            not_ready=bool(g("cd", t) > 0 or g("winding", t)),
                            hp_gt_dmg=bool(g("hp", t) > g("dmg", t)),
                            swung=False, landed=False, survived_hit=False)
        if pend is not None:
            u = pend["u"]
            if int(g("swinging_on", t)) == u:
                pend["swung"] = True
            dead = bool(rec["died"][t, e, u])
            if int(g("landed_on", t)) == u:
                pend["landed"] = True
                pend["swung"] = True
                if not dead:
                    pend["survived_hit"] = True
            if dead:
                close(pend, "success" if g("cs_up", t) else "fail")
                pend = None
                continue
        if pend is not None and g("alive", t) and not g("post_champ_alive", t):
            close(pend, "champion died")
            pend = None
    if pend is not None:
        close(pend, "open at end")
    return out, reasons


def summarise(states, rec, name):
    cs = np.asarray(states.cs[:, :2])
    n_envs = cs.shape[0]
    rows = []
    for c, side in enumerate(("blue", "red")):
        tot, rs = Counter(), Counter()
        for e in range(n_envs):
            o, r = _classify(rec, e, c)
            tot += o
            rs += r
        deaths = (rec["alive"][:, :, c] & ~rec["post_champ_alive"][:, :, c]).sum(0)
        rows.append(dict(
            matchup=name, side=side,
            cs10_mean=float(cs[:, c].mean()), cs10_min=int(cs[:, c].min()),
            cs10_max=int(cs[:, c].max()), cs10_env0=int(cs[0, c]),
            cs_per_min=float(cs[:, c].mean() / (rec['attack'].shape[0] / DECISION_HZ / 60)),
            attempts=int(tot["attempts"]), successes=int(tot["success"]),
            fails=int(tot["fail"]), retargeted=int(tot["retargeted"]),
            champ_died_during=int(tot["champion died"]),
            deaths_mean=float(deaths.mean()),
            in_lane_frac=float(rec["in_lane"][:, :, c].mean()),
            opp_minion_deaths_near_mean=float(rec["opp_died_near"][:, :, c].sum(0).mean()),
            opp_minion_deaths_in_reach_mean=float(rec["opp_died_in_reach"][:, :, c].sum(0).mean()),
            opp_minion_deaths_in_400_mean=float(rec["opp_died_in_400"][:, :, c].sum(0).mean()),
            mismatch_orders=int(rec["mismatch"][:, :, c].sum()),
            attack_orders=int(rec["attack"][:, :, c].sum()),
            fail_reasons=dict(rs),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--matchups", nargs="+",
                    default=["lasthit:lasthit", "lasthit:noop",
                             "brawler:brawler", "brawler:noop", "noop:noop"])
    ap.add_argument("--drop", type=float, default=0.05)
    ap.add_argument("--run-s", type=float, default=EPISODE_S,
                    help="smoke tests only; CS@10 needs the full 600")
    ap.add_argument("--chunk", type=int, default=600)
    ap.add_argument("--no-route-table", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print("device:", jax.devices()[0], flush=True)
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for rel in ("train/scripted_policy.py", "train/scripted_eval.py",
                "train/actions.py", "obs/builder.py"):
        print("loaded", rel, hashlib.sha256((root / rel).read_bytes()).hexdigest()[:12],
              flush=True)
    sim = (SimConfig.training(route_artifact=None) if a.no_route_table
           else SimConfig.training())
    print("sim:", json.dumps(sim.describe())[:400], flush=True)
    all_rows = []
    for m in a.matchups:
        blue, red = m.split(":")
        runner = make_runner(blue, red, a.envs, sim, drop=a.drop, chunk=a.chunk)
        for seed in a.seeds:
            states, rec, wall = run_matchup(blue, red, a.envs, seed, sim, drop=a.drop,
                                           chunk=a.chunk, run_s=a.run_s,
                                           run_chunk=runner)
            rows = summarise(states, rec, m)
            for r in rows:
                r["seed"] = seed
                r["wall_s"] = wall
                print(json.dumps(r), flush=True)
            all_rows += rows
    if a.out:
        with open(a.out, "w") as f:
            json.dump(all_rows, f, indent=1)


if __name__ == "__main__":
    main()
