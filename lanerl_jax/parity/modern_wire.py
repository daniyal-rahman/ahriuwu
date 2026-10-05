"""Explicit 26.19 observation reconstruction; never a resumable sim snapshot."""
from dataclasses import fields
import numpy as np
from ..modern.data.champions import IDS


def validate(frame, names):
    champs = {u['tm']: u for u in frame.get('u', []) if u.get('k') == 'Champion'}
    for team, name in zip((100, 200), names):
        m = champs.get(team, {}).get('modern', {})
        if len(m.get('castCounts', [])) != 4:
            raise ValueError('modern wire requires four authoritative cast counters')
        if (m.get('patch'), m.get('schema'), m.get('id')) != ('26.19', 2, IDS[name]):
            raise ValueError(f'modern wire requires schema 2, patch 26.19, {name} on team {team}')
    return champs


def reconstruct(state, frame, names):
    champs = validate(frame, names)
    c = {f.name: np.asarray(getattr(state.champion, f.name)).copy() for f in fields(state.champion)}
    buffs = state.buffs
    q, qe, e, ee, w, we, haste, he = [np.zeros_like(state.hp) for _ in range(8)]
    silence, casting = np.zeros_like(state.hp), np.zeros_like(state.hp)
    for i, team in enumerate((100, 200)):
        u = champs[team]; m = u['modern']; garen = m['id'] == 86
        for dst, src, scale in (
            ('mana','mana',1), ('max_mana','maxMana',1), ('shield','shield',1),
            ('shield_ms','shieldTime',1000), ('dash_ms','jumpTime',1000)):
            c[dst][i] = m[src] * scale
        c['ap'][i] = u['ap']
        # CC bits are sufficient for observation/status gates, not remaining duration.
        c['stun_ms'][i] = float(m['stunned'])
        silence[i] = float(m['silenced']); casting[i] = float(m['casting'])
        if garen:
            c['garen_w_stacks'][i] = m['kills']
            q[i], qe[i] = m['q'] > 0, max(0, 4.5 - m['q'])
            e[i], ee[i] = m['e'] > 0, m['eElapsed']
            w[i], we[i] = m['w'] > 0, max(0, 4 - m['w'])
            haste[i] = m['haste'] > 0
            he[i] = max(0, 1.4 + .55 * (u['sl'][0] - 1) - m['haste'])
        else:
            for dst, src, scale in (
                ('jax_stacks','passiveStacks',1), ('jax_stack_ms','passiveTime',1000),
                ('jax_w_ms','w',1000), ('jax_e_ms','e',1000), ('jax_e_dodges','dodges',1),
                ('jax_r_ms','r',1000), ('jax_r_cast_ms','rPending',1000),
                ('jax_r_armor','rArmor',1), ('jax_r_hits','rHits',1)):
                c[dst][i] = m[src] * scale
    buffs = buffs.replace(
        q=buffs.q.replace(active=q.astype(bool), elapsed_s=qe),
        e=buffs.e.replace(active=e.astype(bool), elapsed_s=ee),
        w=buffs.w.replace(active=w.astype(bool), elapsed_s=we),
        q_haste=buffs.q_haste.replace(active=haste.astype(bool), elapsed_s=he),
        w_passive=np.zeros_like(buffs.w_passive))
    return state.replace(champion=state.champion.replace(**c), buffs=buffs,
                         silenced_ms=silence, r_cast_ms=casting)
