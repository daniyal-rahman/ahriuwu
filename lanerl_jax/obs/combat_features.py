"""Opt-in Tencent-documented combat features for the current no-items lane.

Reference: LEARN-AFK-29; this is feature alignment, not full Tencent parity.
Visible units use public stats/type/level and direct current/max HP, as requested
for the simulator reference baseline. No enemy attack clocks, target identities
or history enter the actor. HP and own readiness are simulator observations: a video
implementation must estimate it before this interface can be used there.
"""
import jax.numpy as jnp

from .frame import to_lane
from ..sim.combat import growth_sum
from ..sim.spells import status_of, Q_HASTE_MULTIPLIER

COMBAT_INTERFACE = 'viewport-structured-v6-combat'
ENTITY_FEATURES = (
    'hp_points', 'max_hp', 'attack_damage', 'attack_range',
    'attack_speed', 'move_speed', 'kill_income', 'distance_to_self',
    'lane_s', 'lane_n', 'level',
)
SELF_FEATURES = (
    'hp_points', 'max_hp', 'attack_range', 'attack_speed', 'move_speed',
    'attack_available', 'q_active', 'w_active', 'e_active',
    'q_rank', 'w_rank', 'e_rank', 'r_rank',
    'q_available', 'w_available', 'e_available', 'r_available',
    'move_available', 'recall_available',
)
COMBAT_ENTITY_DIM = 16 + len(ENTITY_FEATURES)
COMBAT_SELF_DIM = 16 + len(SELF_FEATURES)
# Fountain profiles intentionally have zero attack period. Saturate the public
# rate at 10/s instead of feeding infinity into the policy, including the AFK
# team's otherwise unused forward pass. Ordinary lane units are below this cap.
MAX_OBS_ATTACK_SPEED = 10.


def append_combat_features(obs, state, me, frame, params):
    """Augment an already visibility-filtered v3 observation.

Stats refer to the current fixed-profile, no-items simulator. Max HP is a
public profile/level statistic; current HP is read directly for visible units.
The original v3 rounded fraction remains in the unchanged base columns.
Q movement haste is visibly active.
Attack availability excludes target/range checks: it describes own readiness,
not whether attacking any particular minion will work on the next sim tick.
"""
    if obs.entities.shape[-1] != 16 or obs.self_vec.shape[-1] != 16:
        raise ValueError('combat features require the original v3 observation')
    u = jnp.maximum(obs.slot_unit, 0)
    growth = growth_sum(state.level, jnp)
    p = lambda key: params[key][state.model]
    ad = p('attack_damage') + p('ad_per_level') * growth
    speed = jnp.minimum(MAX_OBS_ATTACK_SPEED,
        (1 + p('attack_speed_per_level') / 100 * growth)
        / jnp.maximum(p('attack_period'), 1e-6))
    movement = p('move_speed') * jnp.where(
        state.buffs.q_haste.active, Q_HASTE_MULTIPLIER, 1.)
    s, n = to_lane(frame, state.x[u], state.y[u])
    distance = jnp.sqrt((state.x[u]-state.x[me])**2 + (state.y[u]-state.y[me])**2)
    extra = jnp.stack([
        state.hp[u] / 3000.,
        state.max_hp[u] / 3000., ad[u] / 200., p('attack_range')[u] / 1000.,
        speed[u] / 2., movement[u] / 1000., p('gold_on_death')[u] / 300.,
        distance / 3000., s / 3000., n / 3000., state.level[u] / 18.,
    ], -1).astype(obs.entities.dtype)
    extra = jnp.where(obs.entity_pad_mask[:, None], 0., extra)
    status = status_of(state)
    attack_available = (status.can_attack[me] & (state.aa_cooldown[me] <= 0)
                        & ~state.is_attacking[me])
    own = jnp.concatenate([jnp.stack([
        state.hp[me] / 3000., state.max_hp[me] / 3000.,
        p('attack_range')[me] / 1000., speed[me] / 2., movement[me] / 1000.,
        attack_available.astype(obs.self_vec.dtype),
        state.buffs.q.active[me], state.buffs.w.active[me], state.buffs.e.active[me],
    ]), state.spell_level[me] / jnp.array([5., 5., 5., 3.]),
        status.can_cast[me].astype(obs.self_vec.dtype),
        jnp.stack([state.alive[me] & (state.recall_windup_ms[me] <= 0) & (state.r_cast_ms[me] <= 0),
                   status.may_cast[me]]).astype(obs.self_vec.dtype)]).astype(obs.self_vec.dtype)
    return obs._replace(entities=jnp.concatenate([obs.entities, extra], -1),
                        self_vec=jnp.concatenate([obs.self_vec, own]))
