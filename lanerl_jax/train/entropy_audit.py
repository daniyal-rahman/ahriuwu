"""Read-only action-entropy diagnostic; never initializes or updates a policy.

Run with --out pointing to a new ignored evidence directory. Compares the
actual PPO entropy to an independent derivative and exercises the actual
server adapter's unranked-R collapse. Alternative formulas are diagnostics,
not proposed or installed learner changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl.constants import BUTTONS, BUTTON_INDEX, N_SCREEN_X, N_SCREEN_Y
from .ppo import PPOConfig, factored_entropy
from .server_train import screen_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    command = shlex.join([sys.executable, '-m', 'lanerl_jax.train.entropy_audit', *sys.argv[1:]])
    (args.out / 'command.txt').write_text(command + '\n')
    (args.out / 'working-tree.patch').write_bytes(subprocess.check_output(
        ['git', 'diff', '--binary', 'HEAD'], cwd=root))
    source = {}
    # Capture every loaded local module, including this diagnostic and the
    # adapter's transitive imports. The dirty diff alone omits untracked code.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root) or path.suffix != '.py' or '.venv' in str(path):
            continue
        relative = path.relative_to(root)
        data = path.read_bytes()
        destination = args.out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        source[str(relative)] = hashlib.sha256(data).hexdigest()
    manifest = dict(command=command, head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(), source_sha256=source,
        python=sys.version, jax=jax.__version__, numpy=np.__version__,
        seeds='none: deterministic uniform-logit calculation',
        buttons=BUTTONS, screen_shape=[N_SCREEN_X, N_SCREEN_Y])
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    n = len(BUTTONS)
    screen = np.array([name in ('move', 'attack_move', 'r') for name in BUTTONS])
    cursor_h = math.log(N_SCREEN_X * N_SCREEN_Y)
    zero = jnp.zeros(n)
    x, y = jnp.zeros(N_SCREEN_X), jnp.zeros(N_SCREEN_Y)
    actual = lambda b: factored_entropy((b, x, y))
    entropy = float(actual(zero))
    gradient = np.asarray(jax.grad(actual)(zero))
    analytic = cursor_h / n * (screen.astype(float) - screen.mean())

    def independent(b):
        p = np.exp(b - np.max(b)); p /= p.sum()
        return -(p * np.log(p)).sum() + p[screen].sum() * cursor_h

    eps = 1e-5
    finite = np.array([(independent(np.eye(n)[i]*eps) -
                        independent(-np.eye(n)[i]*eps))/(2*eps) for i in range(n)])
    np.testing.assert_allclose(gradient, analytic, atol=2e-6)
    np.testing.assert_allclose(finite, analytic, atol=1e-8)
    np.testing.assert_allclose(entropy, math.log(n)+screen.mean()*cursor_h, atol=2e-6)

    # Actual adapter: this rank mask rejects R before accessing position/frame.
    commands = {json.dumps(screen_order((BUTTON_INDEX['r'], ix, iy),
                {'sl': [1, 1, 1, 0]}, None), sort_keys=True)
                for ix in range(N_SCREEN_X) for iy in range(N_SCREEN_Y)}
    assert commands == {'{"t": "noop"}'}, commands

    def h(b):
        lp = jax.nn.log_softmax(b)
        return -jnp.sum(jnp.exp(lp)*lp)

    def variant(b, kind):
        p = jax.nn.softmax(b)
        usage = jnp.sum(p[jnp.asarray(screen)])
        if kind == 'normalized_cursor':
            return h(b) + usage  # H(cursor)/maximum H(cursor) == 1 here.
        if kind == 'stop_gradient_usage':
            return h(b) + jax.lax.stop_gradient(usage)*cursor_h
        if kind == 'hud_mask_unranked_r':
            masked = b.at[BUTTON_INDEX['r']].set(-1e9)
            return actual(masked)
        raise ValueError(kind)

    variants = {kind: np.asarray(jax.grad(lambda b: variant(b, kind))(zero)).tolist()
                for kind in ('normalized_cursor', 'stop_gradient_usage', 'hud_mask_unranked_r')}
    result = dict(entropy=entropy, cursor_entropy=cursor_h, buttons=BUTTONS,
        actual_gradient=gradient.tolist(), analytic_gradient=analytic.tolist(),
        independent_finite_difference=finite.tolist(), entropy_coef=PPOConfig().entropy_coef,
        entropy_loss_gradient=(-PPOConfig().entropy_coef*gradient).tolist(),
        unranked_r_cursor_choices=N_SCREEN_X*N_SCREEN_Y,
        unranked_r_distinct_wire_commands=[json.loads(s) for s in commands],
        diagnostic_variant_button_gradients=variants,
        caveats=['Uniform logits, not measured trained-policy gradients.',
                 'Normalized cursor entropy still favors coordinate buttons.',
                 'Stopping usage gradients changes the regularizer gradient, not its scalar value.',
                 'Masking R removes unavailable R but retains move/A entropy bias.',
                 'No policy was initialized, trained, or modified.'],
        elapsed_s=time.monotonic()-started)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
