"""Name-free fingerprint of the traced modern-world tick, for no-op refactors (comments, names, moves).

TOOL (MODERN-025). Traces one ``step`` (with ``ops.modern.golden.chaos_orders``) for the full-map and
top-lane worlds and hashes every equation: primitive, input/output shapes, parameters and literals, with
variables numbered by first use. Equal hashes mean the same computation, so a refactor that keeps them
is behaviour-preserving without running a game. Takes about a minute per world on CPU, no compile.

    python -m ops.modern.jaxpr_fingerprint --out ref.json
    python -m ops.modern.jaxpr_fingerprint --against ref.json      # exit 1 and first differing eqn if changed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

ADDRESS = re.compile(r" at 0x[0-9a-f]+")     # object addresses in parameter reprs differ per process
WORLDS = {"full": dict(lanes=(0, 1, 2), jungle=True, objectives=True),
          "top": dict(lanes=(2,), jungle=False, objectives=False)}


def canon(jaxpr, out: list, ids: dict) -> None:
    from jax.extend import core as jcore

    def var(x):
        if isinstance(x, jcore.Literal):
            return f"lit:{x.val!r}:{x.aval}"
        ids.setdefault(id(x), f"v{len(ids)}")
        return f"{ids[id(x)]}:{x.aval}"

    for x in jaxpr.invars:
        var(x)
    for e in jaxpr.eqns:
        params = []
        for k in sorted(e.params):
            p = e.params[k]
            subs = p if isinstance(p, (tuple, list)) else (p,)
            if subs and all(isinstance(q, (jcore.ClosedJaxpr, jcore.Jaxpr)) for q in subs):
                for q in subs:
                    canon(q.jaxpr if isinstance(q, jcore.ClosedJaxpr) else q, out, ids)
                params.append(f"{k}=<{len(subs)} jaxpr>")
            elif k not in ("name", "debug_info", "source_info"):
                params.append(ADDRESS.sub("", f"{k}={p!r}")[:200])
        out.append(f"{e.primitive.name}({','.join(map(var, e.invars))})->{','.join(map(var, e.outvars))} "
                   + " ".join(params))
    out.append("ret " + ",".join(map(var, jaxpr.outvars)))


def fingerprint(world: str) -> list[str]:
    import jax
    from lanerl_jax.modern import world as MS
    from ops.modern.golden import build, chaos_orders
    cfg = build(world)
    s = MS.init_state(cfg)
    mid = cfg.lane_path[cfg.lane_path.shape[0] // 2]
    closed = jax.make_jaxpr(lambda s, k: MS.step(s, chaos_orders(s, k, mid), cfg))(s, jax.random.PRNGKey(0))
    out: list[str] = []
    canon(closed.jaxpr, out, {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS))
    ap.add_argument("--out")
    ap.add_argument("--against")
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    eqns = {w: fingerprint(w) for w in args.worlds}
    summary = {w: {"sha256": hashlib.sha256("\n".join(e).encode()).hexdigest(), "eqns": len(e)}
               for w, e in eqns.items()}
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "eqns": eqns}, f)
    if args.against:
        ref = json.load(open(args.against))
        bad = False
        for w, e in eqns.items():
            old = ref["eqns"].get(w)
            if old is None or old == e:
                continue
            bad = True
            k = next((i for i, (a, b) in enumerate(zip(old, e)) if a != b), min(len(old), len(e)))
            print(json.dumps({"world": w, "first_diff": k, "eqns_ref": len(old), "eqns_now": len(e),
                              "ref": old[k] if k < len(old) else None, "now": e[k] if k < len(e) else None}))
        print(json.dumps({"verdict": "changed" if bad else "identical"}))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
