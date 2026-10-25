"""Unweighted event CE plus button-conditioned target and optional auxiliary losses."""
import jax
import jax.numpy as jnp

from .event_labels import UNIT_TARGET, POINT_TARGET, ANGLE_TARGET
from .events import BUTTON_INDEX as B, BUTTONS
from .targets import gaussian_target


INPUTS = ('entities', 'pad', 'self', 'global_', 'entity_ids', 'own_id', 'enemy_id',
          'own_metadata', 'enemy_metadata', 'patch')


def make_event_forward(policy):
    def forward(params, batch, teacher=True):
        inputs = tuple(jnp.swapaxes(batch[k], 0, 1) for k in INPUTS)
        buttons = jnp.swapaxes(batch['button'], 0, 1)

        @jax.checkpoint
        def step(carry, values):
            x, b = values[:-1], values[-1]
            logits, carry = policy.apply(params, *x, carry, target_button=b if teacher else None)
            return carry, logits
        _, logits = jax.lax.scan(step, policy.initial_carry((batch['button'].shape[0],)), inputs + (buttons,))
        return jax.tree.map(lambda v: jnp.swapaxes(v, 0, 1), logits)
    return forward


def grid_ce(logits, x, y, sigma=1., nx=96, ny=54):
    qx, qy = gaussian_target(x, nx, sigma), gaussian_target(y, ny, sigma)
    lp = jax.nn.log_softmax(logits, -1).reshape(logits.shape[:-1] + (ny, nx))
    return -jnp.einsum('...y,...x,...yx->...', qy, qx, lp)


def make_event_loss(policy, *, co_move_weight=0., auxiliary_weight=0., sigma=1., cast_aim_weight=1.):
    forward = make_event_forward(policy)

    def ce(logits, index):
        return -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), index[..., None], -1)[..., 0]

    def loss(params, batch):
        lg = forward(params, batch)
        target = batch.get('button_target', jax.nn.one_hot(batch['button'], len(BUTTONS)))
        button = -jnp.sum(target * jax.nn.log_softmax(lg.button, -1), -1)
        point = grid_ce(lg.grid, batch['screen_x'], batch['screen_y'], sigma,
                        policy.cfg.grid_x, policy.cfg.grid_y)
        pointer = ce(lg.pointer, jnp.maximum(batch['target_row'], 0))
        angle_index = jnp.mod(jnp.floor((jnp.nan_to_num(batch['angle']) + jnp.pi) /
                                        (2 * jnp.pi) * policy.cfg.n_angles).astype(jnp.int32), policy.cfg.n_angles)
        angle = .5 * ce(lg.angle, angle_index) + .25 * ce(lg.angle, (angle_index - 1) % policy.cfg.n_angles)
        angle += .25 * ce(lg.angle, (angle_index + 1) % policy.cfg.n_angles)
        aim = jnp.where(batch['aim_mode'] == UNIT_TARGET, pointer,
                        jnp.where(batch['aim_mode'] == ANGLE_TARGET, angle, point))
        w = batch['weight']
        target_w = w * batch['aim_valid'] * (batch['aim_mode'] != 0)
        denom = jnp.maximum(w.sum(), 1.)
        cast = (batch['button'] >= B['q']) & (batch['button'] <= B['r'])
        aim_scale = jnp.where(cast, cast_aim_weight, 1.)
        weighted_aim = target_w * aim_scale * aim
        total = (w * button).sum() / denom + weighted_aim.sum() / denom
        info = dict(button_nll=(w * button).sum() / denom,
                    aim_nll=(target_w * aim).sum() / jnp.maximum(target_w.sum(), 1.),
                    button_accuracy=(w * (lg.button.argmax(-1) == batch['button'])).sum() / denom)
        for name, mode in [('unit', UNIT_TARGET), ('point', POINT_TARGET), ('angle', ANGLE_TARGET)]:
            mask = target_w * (batch['aim_mode'] == mode)
            info[name + '_loss_mass'] = (mask * aim_scale * aim).sum() / denom
            info[name + '_count'] = mask.sum()
        info['cast_aim_count'] = (target_w * cast).sum()
        info['cast_aim_loss_mass'] = (weighted_aim * cast).sum() / denom
        info['noncast_aim_loss_mass'] = (weighted_aim * ~cast).sum() / denom
        if co_move_weight:
            cast = (batch['button'] >= B['q']) & (batch['button'] <= B['r'])
            cw = w * cast
            positive = batch['co_move_source'] >= 0
            extra = ce(lg.co_move, positive.astype(jnp.int32))
            extra += batch['co_move_valid'] * grid_ce(lg.co_move_grid, batch['co_move_x'], batch['co_move_y'], sigma,
                                                      policy.cfg.grid_x, policy.cfg.grid_y)
            extra += batch['co_move_valid'] * ce(lg.co_move_order, (batch['co_move_delta'] >= 0).astype(jnp.int32))
            total += co_move_weight * (cw * extra).sum() / denom
        if auxiliary_weight:
            aux = jnp.zeros_like(w)
            for name, logits in [('own_future', lg.own_future), ('enemy_future', lg.enemy_future)]:
                aux += batch[name + '_valid'] * ce(logits, jnp.maximum(batch[name], 0))
            delta = lg.own_hp_future - jnp.nan_to_num(batch['own_hp_future'])
            huber = jnp.where(jnp.abs(delta) <= .1, .5 * delta ** 2, .1 * (jnp.abs(delta) - .05))
            aux += batch['own_hp_future_valid'] * huber
            total += auxiliary_weight * (w * aux).sum() / denom
        info['loss'] = total
        return total, info
    return loss
