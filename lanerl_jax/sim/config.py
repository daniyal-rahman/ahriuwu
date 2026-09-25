"""One step configuration object, shared by the trainer, the gates and eval.

`STRUCT-003`. Every consumer of the simulator used to assemble its step
configuration by hand -- route table, terrain, the collision mode, call for
help, lane path, the stat tables -- and the trainer's deferred-terrain mode
(`collision_terrain=False, defer_collision_terrain=True`) was run by nothing
else. That is how the two fountain turrets walked across the map in every RL
run for five days without a gate noticing (`COLL-004`). ``apply_orders``
also accepted a missing ``params`` and silently used a stale level-one AD for
E (``_ad_placeholder``, now deleted: ``params`` is required).

A :class:`SimConfig` holds everything that selects sim behaviour and is not
per-tick state. Build ONE with a named constructor and pass it through:

* :meth:`SimConfig.training` -- exactly what ``train/trainer.py`` steps.
* :meth:`SimConfig.gate` -- the policy-divergence gate (`PARITY-001`): the
  training configuration, one tick per call so every server tick is diffed.
  ``sim/tests/test_sim_config.py`` asserts it differs from training ONLY in
  an allow-list whose every entry cites a ledger row.
* :meth:`SimConfig.scripted` -- the scripted parity drivers (gate 3,
  tier 1.5/2, perturbation, benchmarks' warm-up): TOP lane waves with the
  INLINE terrain repair, which is the tick's default and NOT the training
  mode. Kept distinct and named so that difference is visible.
* :meth:`SimConfig.unit_test` -- the bare ``tick``/``step_decision``
  defaults the sim tests use (no waves, no routing).

The sim itself reads it only through :func:`lanerl_jax.sim.step.env_step`
(and its two halves ``env_apply``/``env_advance``). ``describe()`` is the
JSON-safe summary a run manifest records; ``fingerprint()`` also digests the
array contents, so two runs with the same fingerprint stepped the same sim.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .movement_jax import TICK_MS

__all__ = ["SimConfig", "DEFAULT_ROUTE_ARTIFACT", "ROUTE_PATHFINDING_RADIUS",
           "BEHAVIOUR_FIELDS"]

#: The Map1 local-route artifact production trains on (`PATH-001`/`PATH-006`).
DEFAULT_ROUTE_ARTIFACT = (Path(__file__).resolve().parents[2] / "data" /
                          "jax_routes" / "map1_garen_r35_o50_v2")

#: Garen's pathfinding radius; the artifact is baked for it (``r35``).
ROUTE_PATHFINDING_RADIUS = 35.0

#: The fields the sim READS. Everything else on the object is provenance for
#: ``describe()`` and never changes what a step computes.
BEHAVIOUR_FIELDS = (
    "params", "route_table", "terrain", "vision", "lane_path", "minion_hp",
    "collision_terrain", "defer_collision_terrain", "enable_collision",
    "enable_call_for_help", "step_ticks", "delta_ms",
)

_SCALAR_FIELDS = ("collision_terrain", "defer_collision_terrain",
                  "enable_collision", "enable_call_for_help", "step_ticks",
                  "delta_ms")


def _top_lane_path():
    import jax.numpy as jnp

    from .init import TOP_LANE_PATH
    return jnp.asarray(np.array(TOP_LANE_PATH, np.float32))


def _load_routes(route_artifact, route_table, terrain):
    """``(route_table, terrain, artifact_path_or_None, table_digest_or_None)``."""
    if route_artifact is not None and route_table is not None:
        raise ValueError("pass a route artifact path OR a loaded route_table, "
                         "not both: the recorded provenance would be ambiguous")
    if route_table is not None:
        if terrain is None:
            raise ValueError("route_table requires a TerrainGrid")
        return route_table, terrain, None, None
    if route_artifact is None:
        return None, terrain, None, None
    path = Path(route_artifact)
    if not path.exists():
        raise FileNotFoundError(
            f"route artifact not found: {path}. Build it with: "
            "python -m lanerl_jax.data.local_route_artifact "
            f"--out {path} --radius 35 --offset-radius 50")
    from ..data.local_route_artifact import load_local_route_artifact
    from .terrain_jax import map1_terrain

    art = load_local_route_artifact(path, pathfinding_radius=ROUTE_PATHFINDING_RADIUS)
    m = art.manifest
    digest = m.table_sha256
    if getattr(m, "run_length_sha256", ""):
        digest += ":" + m.run_length_sha256
    from ..data.chase_routes import derive_chase_routes
    import jax.numpy as jnp
    landmarks, hops = derive_chase_routes(art)
    routed = art.as_jax()._replace(chase_landmark=jnp.asarray(landmarks),
                                  chase_next_hop=jnp.asarray(hops))
    digest += ":chase-landmarks-v1:" + _digest((landmarks, hops))
    return (routed, terrain if terrain is not None else map1_terrain(),
            str(path), digest)


def _leaves_equal(a, b) -> bool:
    """Content equality for a config field: None, scalar, array or pytree."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    import jax

    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    if ta != tb or len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        if x is y:
            continue
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.shape != ya.shape or xa.dtype != ya.dtype:
            return False
        if not np.array_equal(xa, ya):
            return False
    return True


def _digest(tree) -> Optional[str]:
    if tree is None:
        return None
    import jax

    h = hashlib.sha256()
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    h.update(str(treedef).encode())
    for leaf in leaves:
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclasses.dataclass(frozen=True, eq=False)
class SimConfig:
    """Everything that selects sim behaviour and is not per-tick state.

    Frozen; derive a variant with :meth:`replace`. Equality of two configs is
    :meth:`differing_fields` (content, not identity: the arrays are compared
    by value), because dataclass ``==`` on arrays is not a bool.
    """

    #: ``lane_params(patch)`` -- the profile stat tables the tick gathers.
    params: Mapping[str, Any]
    #: local-route table for Move orders, or ``None`` for the PATH-001
    #: two-point control. Requires ``terrain``.
    route_table: Any = None
    #: Map1 walkability grid used by routed Moves.
    terrain: Any = None
    #: Brush/wall vision grid; None is the synthetic radius-only test model.
    vision: Any = None
    #: ``(W, 2)`` lane waypoints; ``None`` runs without wave spawning.
    lane_path: Any = None
    minion_hp: Any = None
    collision_terrain: bool = True
    defer_collision_terrain: bool = False
    enable_collision: bool = True
    enable_call_for_help: bool = True
    #: server ticks per decision (`LANERL_STEP_TICKS`); 2 = 30 Hz off 60 Hz.
    step_ticks: int = 2
    delta_ms: float = TICK_MS
    # ---- provenance only: never read by the sim ---------------------------
    name: str = "custom"
    patch_root: Optional[str] = None
    route_artifact: Optional[str] = None
    route_digest: Optional[str] = None

    def __post_init__(self):
        if self.route_table is not None and self.terrain is None:
            raise ValueError("route_table requires a TerrainGrid")
        if self.params is None:
            raise TypeError("SimConfig requires params (lane_params(patch))")

    # ---- named constructors -------------------------------------------------
    @classmethod
    def unit_test(cls, patch=None) -> "SimConfig":
        """``tick``/``step_decision``'s own defaults: no waves, no routing,
        inline terrain repair, call for help on, 2 ticks per decision."""
        from .init import lane_params
        return cls(params=lane_params(patch), name="unit_test",
                   patch_root=_patch_root(patch))

    @classmethod
    def training(cls, patch=None, route_artifact=DEFAULT_ROUTE_ARTIFACT, *,
                 route_table=None, terrain=None) -> "SimConfig":
        """Exactly what ``train/trainer.py``'s env step runs.

        Routed Moves through ``route_artifact`` (``None`` = the PATH-001
        two-point control, ``run_train --no-route-table``), TOP lane waves,
        and the DEFERRED terrain repair: exact per-neighbour repair inside the
        collision sweep misses J1's throughput gate by two orders of magnitude,
        so training repairs after the dynamic sweep (``COLL-004`` is what this
        mode alone got wrong). An already-loaded ``route_table``/``terrain``
        may be passed instead of a path (then ``route_artifact`` must be None).
        """
        from .init import lane_params
        from ..obs.vision import map1_vision
        rt, ter, art, dig = _load_routes(route_artifact, route_table, terrain)
        return cls(params=lane_params(patch), route_table=rt, terrain=ter, vision=map1_vision(),
                   lane_path=_top_lane_path(),
                   collision_terrain=False, defer_collision_terrain=True,
                   enable_collision=True, enable_call_for_help=True,
                   step_ticks=2, delta_ms=TICK_MS, name="training",
                   patch_root=_patch_root(patch), route_artifact=art,
                   route_digest=dig)

    @classmethod
    def gate(cls, patch=None, route_artifact=DEFAULT_ROUTE_ARTIFACT, *,
             route_table=None, terrain=None) -> "SimConfig":
        """The policy-divergence gate (`PARITY-001`): the training
        configuration stepped ONE tick per call, so the gate can diff every
        server tick (orders are applied before the first of each decision's
        two ticks). ``step_decision`` is a ``lax.scan`` of the same ``tick``,
        so two one-tick calls are the same computation as one two-tick call.
        """
        return cls.training(patch, route_artifact, route_table=route_table,
                            terrain=terrain).replace(step_ticks=1, name="gate")

    @classmethod
    def scripted(cls, patch=None, route_artifact=None, *, route_table=None,
                 terrain=None, enable_call_for_help: bool = True) -> "SimConfig":
        """The scripted parity drivers and benchmark warm-up: TOP lane waves,
        ``step_decision``'s INLINE terrain repair (not the training mode),
        routed only when a table or artifact is given."""
        from .init import lane_params
        from ..obs.vision import map1_vision
        rt, ter, art, dig = _load_routes(route_artifact, route_table, terrain)
        return cls(params=lane_params(patch), route_table=rt, terrain=ter, vision=map1_vision(),
                   lane_path=_top_lane_path(),
                   enable_call_for_help=enable_call_for_help,
                   name="scripted", patch_root=_patch_root(patch),
                   route_artifact=art, route_digest=dig)

    # ---- derivation / comparison -------------------------------------------
    def replace(self, **changes) -> "SimConfig":
        return dataclasses.replace(self, **changes)

    def differing_fields(self, other: "SimConfig") -> set:
        """Names of the BEHAVIOUR fields whose content differs."""
        return {f for f in BEHAVIOUR_FIELDS
                if not _leaves_equal(getattr(self, f), getattr(other, f))}

    def same_behaviour(self, other: "SimConfig") -> bool:
        return not self.differing_fields(other)

    # ---- manifests ----------------------------------------------------------
    def describe(self) -> dict:
        """The non-array fields, JSON-serialisable, for run manifests."""
        out = {k: getattr(self, k) for k in _SCALAR_FIELDS}
        out["step_ticks"] = int(out["step_ticks"])
        out["delta_ms"] = float(out["delta_ms"])
        out.update(
            name=self.name,
            patch_root=self.patch_root,
            routed=self.route_table is not None,
            route_artifact=self.route_artifact,
            route_digest=self.route_digest,
            terrain=None if self.terrain is None else "map1",
            vision=None if self.vision is None else "map-grid-supercover-v1",
            lane_path=_describe_lane_path(self.lane_path),
            minion_hp=None if self.minion_hp is None else "custom",
            n_params=len(self.params),
        )
        return out

    def fingerprint(self) -> str:
        """sha256 over ``describe()`` and the array contents -- minus the
        name and the two filesystem paths, so the same sim fingerprints the
        same from any checkout (the params and route digests carry content).

        The route table is digested from its artifact manifest when it was
        loaded from one (hashing ~240 MB per call would be wasteful), else
        from its arrays.
        """
        d = self.describe()
        for k in ("name", "patch_root", "route_artifact"):
            d.pop(k)
        h = hashlib.sha256(json.dumps(d, sort_keys=True).encode())
        h.update(str(_digest(dict(self.params))).encode())
        h.update(str(_digest(self.lane_path)).encode())
        h.update(str(_digest(self.minion_hp)).encode())
        h.update(str(_digest(None if self.vision is None else tuple(self.vision))).encode())
        h.update(str(_digest(None if self.terrain is None
                             else tuple(self.terrain))).encode())
        if self.route_table is not None and self.route_digest is None:
            h.update(str(_digest(tuple(self.route_table))).encode())
        return h.hexdigest()[:16]


def _patch_root(patch) -> Optional[str]:
    return None if patch is None else str(getattr(patch, "root", None))


def _describe_lane_path(lane_path):
    if lane_path is None:
        return None
    from .init import TOP_LANE_PATH
    a = np.asarray(lane_path)
    if np.array_equal(a, np.asarray(TOP_LANE_PATH, np.float32)):
        return "TOP_LANE_PATH"
    return f"custom ({a.shape[0]} points)"
