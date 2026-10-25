"""Static BC inventory at immutable git refs; observations, not correctness gates."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


PREFIX = 'lanerl_jax/modern/bc/'


def git(*args):
    return subprocess.check_output(['git', *args], text=True)


def inventory(ref):
    commit = git('rev-parse', '--verify', ref + '^{commit}').strip()
    paths = git('ls-tree', '-r', '--name-only', commit, '--', PREFIX).splitlines()
    files = []
    for path in paths:
        if not path.endswith('.py'):
            continue
        source = git('show', commit + ':' + path)
        tree = ast.parse(source, filename=path)
        imports, constants, functions = [], {}, {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(dict(module='.' * node.level + (node.module or ''),
                                    names=[alias.name for alias in node.names], line=node.lineno))
            elif isinstance(node, ast.Import):
                imports.append(dict(module=None, names=[alias.name for alias in node.names],
                                    line=node.lineno))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(node.name, []).append(dict(line=node.lineno,
                    lines=node.end_lineno - node.lineno + 1))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                            'CONTRACT', 'SELF_FIELDS', 'ABILITY_FIELDS', 'BUTTONS'):
                        try:
                            constants[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        files.append(dict(path=path, lines=len(source.splitlines()),
            sha256=hashlib.sha256(source.encode()).hexdigest(),
            test='tests' in Path(path).parts, imports=imports,
            constants=constants, functions=functions))
    production = [f for f in files if not f['test']]
    tests = [f for f in files if f['test']]
    return dict(requested_ref=ref, commit=commit,
        production_files=len(production), production_lines=sum(f['lines'] for f in production),
        test_files=len(tests), test_lines=sum(f['lines'] for f in tests),
        largest=sorted(((f['path'], f['lines']) for f in production),
                       key=lambda pair: (-pair[1], pair[0]))[:10], files=files)


def compare(left, right):
    a = {f['path']: f for f in left['files']}
    b = {f['path']: f for f in right['files']}
    return [dict(path=p, left_lines=a[p]['lines'] if p in a else None,
                 right_lines=b[p]['lines'] if p in b else None,
                 status='added' if p not in a else 'removed' if p not in b else 'changed')
            for p in sorted(a.keys() | b.keys())
            if p not in a or p not in b or a[p]['sha256'] != b[p]['sha256']]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--refs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    snapshots = [inventory(ref) for ref in args.refs]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), snapshots=snapshots,
        comparisons=[dict(left=snapshots[0]['commit'], right=s['commit'],
                          files=compare(snapshots[0], s)) for s in snapshots[1:]],
        limits='Tracked Python files only; AST/import/size inventory is not execution coverage, '
               'a call graph, dead-code proof, or a correctness verdict. Git refs are resolved once.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    for snapshot in snapshots:
        print(json.dumps({k: snapshot[k] for k in ('commit', 'production_files',
                         'production_lines', 'test_files', 'test_lines')}))


if __name__ == '__main__':
    main()
