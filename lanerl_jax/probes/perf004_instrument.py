"""Probe-only naming of existing computations; no production arithmetic edits.

All inserted nodes are ``with jax.named_scope(...)``. Removing those nodes must
recover the original AST exactly. Tick labels follow all 26 documented phases,
plus initial stat preparation and final state assembly. The GPU canary also
checks lowered computation and output agreement before profiling.
"""
from __future__ import annotations

import ast
import copy
import inspect
import re
import textwrap

PREFIX = 'perf004_'
PHASES = {}


def _scope(label, statements):
    call = ast.Call(func=ast.Attribute(value=ast.Name(id='jax', ctx=ast.Load()),
                                     attr='named_scope', ctx=ast.Load()),
                    args=[ast.Constant(PREFIX + label)], keywords=[])
    node = ast.With(items=[ast.withitem(context_expr=call)], body=statements)
    ast.copy_location(node, statements[0])
    node.end_lineno = statements[-1].end_lineno
    node.end_col_offset = statements[-1].end_col_offset
    return node


class _RemoveScopes(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        call = node.items[0].context_expr
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'named_scope' and call.args
                and isinstance(call.args[0], ast.Constant)
                and str(call.args[0].value).startswith(PREFIX)):
            return node.body
        return node


def _target_names(node):
    if isinstance(node, ast.Assign):
        return {n.id for target in node.targets for n in ast.walk(target)
                if isinstance(n, ast.Name)}
    return set()


def _group(body, choose):
    out, pending, previous = [], [], None
    for statement in body:
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)):
            out.append(statement)
            continue
        label = choose(statement)
        if pending and label != previous:
            out.extend([_scope(previous, pending)] if previous else pending)
            pending = []
        pending.append(statement)
        previous = label
    if pending:
        out.extend([_scope(previous, pending)] if previous else pending)
    return out


def _rebuild(fn, transform):
    lines, start = inspect.getsourcelines(fn)
    source = textwrap.dedent(''.join(lines))
    original = ast.parse(source)
    tree = copy.deepcopy(original)
    transform(tree.body[0], source)
    stripped = _RemoveScopes().visit(copy.deepcopy(tree))
    if ast.dump(stripped, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise AssertionError(f'instrumentation changed arithmetic/control AST: {fn.__qualname__}')
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, start - 1)
    namespace = dict(fn.__globals__)
    exec(compile(tree, inspect.getsourcefile(fn), 'exec'), namespace)
    result = namespace[fn.__name__]
    result.__module__, result.__qualname__ = fn.__module__, fn.__qualname__
    return result


def instrument_tick(fn):
    def transform(node, source):
        PHASES.clear()
        markers = []
        for line, text in enumerate(source.splitlines(), 1):
            m = re.match(r'    # ---- (\d+)\. (.*?)\s*-*$', text)
            if m:
                number = int(m[1])
                label = f'tick_{number:02d}'
                markers.append((line, label))
                PHASES[label] = m[2].rstrip(' -')
        assert list(PHASES) == [f'tick_{i:02d}' for i in range(1, 27)]
        PHASES['tick_00'] = 'Profile stats and tick clock preparation'
        PHASES['tick_27'] = 'Final state assembly (including inline expressions)'

        def choose(statement):
            if isinstance(statement, ast.Return):
                return 'tick_27'
            labels = [label for line, label in markers if line <= statement.lineno]
            return labels[-1] if labels else 'tick_00'
        node.body = _group(node.body, choose)
    return _rebuild(fn, transform)


def _factory_transform(node, source):
    del source
    class Visitor(ast.NodeTransformer):
        def visit_FunctionDef(self, fn):
            fn = self.generic_visit(fn)
            whole = {'_obs': 'observe', '_apply': 'policy_actor',
                     '_click_mask': 'click_mask', '_batch': 'batch_layout',
                     'forward': 'learner_network', 'kl_to_prior': 'prior_kl',
                     'trunk_grad_norms': 'optional_trunk_diagnostics'}
            if fn.name in whole:
                fn.body = _group(fn.body, lambda _: whole[fn.name])
            elif fn.name == 'one':
                current = 'rollout_misc'
                def choose(statement):
                    nonlocal current
                    names = _target_names(statement)
                    labels = [('obs', 'observe'), ('logits', 'policy_actor'),
                              ('cm', 'click_mask'), ('action', 'sampling'),
                              ('orders', 'decode'), ('nxt', 'simulation'),
                              ('reward', 'reward'), ('done', 'reset_and_record')]
                    for name, label in labels:
                        if name in names:
                            # Later nxt writes are reset operations, not sim ticks.
                            if name != 'nxt' or current == 'decode':
                                current = label
                            break
                    return current
                fn.body = _group(fn.body, choose)
            elif fn.name == 'learn':
                current = 'bootstrap'
                def choose(statement):
                    nonlocal current
                    names = _target_names(statement)
                    for name, label in [('adv', 'gae'), ('batch', 'batch_layout'),
                                        ('params', 'ppo_update'), ('new_lg', 'post_forward'),
                                        ('new_lp', 'post_metrics'), ('r_var', 'learner_metrics')]:
                        if name in names:
                            current = label
                            break
                    return current
                fn.body = _group(fn.body, choose)
            elif fn.name == '_update_minbatch':
                current = 'ppo_grad'
                def choose(statement):
                    nonlocal current
                    names = _target_names(statement)
                    for name, label in [('grad_fn', 'ppo_grad'), ('grads', 'ppo_grad'),
                                        ('grad_norm', 'ppo_grad_norm'),
                                        ('updates', 'ppo_adam'), ('params', 'ppo_apply')]:
                        if name in names:
                            current = label
                            break
                    return current
                fn.body = _group(fn.body, choose)
            elif fn.name == '_update_epoch':
                # The inner gradient/optimizer scopes override this outer label.
                fn.body = _group(fn.body, lambda _: 'ppo_shuffle')
            return fn
    Visitor().visit(node)


def install():
    """Install labels only in this probe process. Return originals for inspection."""
    from lanerl_jax.sim import step, orders
    from lanerl_jax.train import learner, ppo, vec_train
    originals = {}
    def replace(module, name, fn):
        originals[f'{module.__name__}.{name}'] = getattr(module, name)
        setattr(module, name, fn)
    replace(step, 'tick', instrument_tick(step.tick))
    replace(orders, 'apply_orders', _rebuild(orders.apply_orders,
            lambda node, _: setattr(node, 'body', _group(node.body, lambda _: 'apply_orders'))))
    replace(learner, 'make_learner', _rebuild(learner.make_learner, _factory_transform))
    replace(ppo, 'update_epochs', _rebuild(ppo.update_epochs, _factory_transform))
    # These are from-import aliases in the factory's globals.
    replace(vec_train, 'make_learner', learner.make_learner)
    replace(vec_train, 'update_epochs', ppo.update_epochs)
    replace(vec_train, 'make_vec_train', _rebuild(vec_train.make_vec_train, _factory_transform))
    return originals
