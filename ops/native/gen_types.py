"""Generate C++ value types for the JAX NamedTuples the champion layer passes between modules.

Each NamedTuple class becomes a struct with its fields in the same order: a ``()`` leaf is a scalar, an array leaf
an ``Arr<T>`` (flattened), a nested NamedTuple a nested struct. ``visit(f)`` calls ``f`` on every member in JAX
flatten order, which is all the generic marshalling (``native/src/champ/marshal.hpp``) needs. Field dtypes come from
real instances built from the Garen-vs-Jax world; fields that are ``None`` there take ``OPTIONAL``.

    python -m ops.native.gen_types          # writes native/src/gen/types.hpp
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "native" / "src" / "gen" / "types.hpp"
CTYPE = {"float32": "float", "int32": "int32_t", "uint32": "uint32_t", "bool": "uint8_t"}
# (class name, field) -> C++ type of fields that are None in the instances (optional inputs).
OPTIONAL = {
    ("Attack", "natural_crit"): "Arr<uint8_t>", ("Defense", "dodge_basic"): "Arr<uint8_t>",
    ("Defense", "aoe_received_mult"): "Arr<float>", ("Defense", "received_mult_all"): "Arr<float>",
    ("RuneEvents", "own"): "Arr<int32_t>", ("KitCtx", "attack_target_kind"): "Arr<int32_t>",
    ("KitCtx", "rooted"): "Arr<uint8_t>", ("KitOut", "cleanse_slow"): "Arr<uint8_t>",
    ("KitOut", "attack_target"): "Arr<int32_t>", ("KitAttackMods", "cannot_crit"): "Arr<uint8_t>",
    ("KitAttackMods", "windup"): "Arr<float>", ("KitAttackMods", "period"): "Arr<float>",
    ("KitAttackMods", "uncancellable"): "Arr<uint8_t>", ("MinionDeaths", "unit"): "Arr<int32_t>",
    ("StructureEvents", "is_structure"): "Arr<uint8_t>",
    ("RuneEvents", "report"): "Report",          # empty (no packets) outside on_damage
    **{("EconomyInputs", f): "Arr<float>" for f in ("extra_gold", "extra_xp", "epic", "recall_channel",
                                                     "minion_gold_delta", "minion_xp_mult")},
}
SKIP_FIELDS: set = set()
EXCLUDE = {"ModernState"}                       # the env itself (gen_state.py); its members are generated


def qualname(cls) -> str:
    """Struct name: the class name, prefixed by its module when the name is ambiguous (``State``)."""
    mod = cls.__module__.replace("lanerl_jax.modern.", "")
    name = cls.__name__
    if name in ("State",):
        return mod.replace(".", "_").replace("effects_", "") + "_" + name
    return name


def is_nt(v) -> bool:
    return isinstance(v, tuple) and hasattr(v, "_fields")


def examples():
    """Instances of every interface type (built from a fresh Garen-vs-Jax world)."""
    import jax.numpy as jnp

    from lanerl_jax.modern import champions as K
    from lanerl_jax.modern import economy as E
    from lanerl_jax.modern import mechanics as M
    from lanerl_jax.modern import role_quest as Q
    from lanerl_jax.modern import wards as WD
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.champions import summoners as S
    from lanerl_jax.modern.core import damage as D
    from lanerl_jax.modern.core import types as W
    from lanerl_jax.modern.items import inventory as I
    from lanerl_jax.modern.items.catalog import catalog, zero_stats
    from lanerl_jax.modern.items.effects import core as IC
    from lanerl_jax.modern.runes.effects import core as RC
    from lanerl_jax.modern.world import units as U
    from lanerl_jax.modern.world import views as V
    from ops.modern.golden import build
    cfg = build("top")
    s = MS.init_state(cfg)
    c, n = 2, cfg.n_units
    now, dt = jnp.float32(cfg.dt), cfg.dt
    caps = M.capabilities(s.cc, now)
    static, st = V.static_stats(s, cfg, caps, now, dt)
    ctx = V.item_ctx(s, cfg, st, now, dt)
    kctx = V.kit_ctx(s, cfg, st, caps, now, dt)
    units = U.item_units(s)
    wu = U.units_view(s)
    fc, zc, ic = jnp.zeros((c,), bool), jnp.zeros((c,), jnp.float32), jnp.zeros((c,), jnp.int32)
    p = D.empty_packets(4)
    dfn = D.default_defense(n)._replace(dodge_basic=jnp.zeros((n,), bool), aoe_received_mult=jnp.ones((n,)),
                                        received_mult_all=jnp.ones((n,)))
    off = D.default_offense(n)
    res = D.resolve(p, off, dfn, s.hp, s.max_hp, s.shields, now)
    report = IC.Report(p, res, jnp.zeros((n,)), jnp.zeros((n,)))
    own = I.owned_counts(s.champ.inventory)
    attack = IC.Attack(fc, fc, ic, zc, fc, fc)
    ev = RC.rune_events(ctx, n, own=own)
    md = E.MinionDeaths(jnp.zeros((n,), bool), *([jnp.zeros((n,))] * 2), jnp.zeros((n,), jnp.int32),
                        *([jnp.zeros((n,))] * 2), jnp.zeros((n,), jnp.int32), jnp.full((n,), -1, jnp.int32),
                        jnp.arange(n, dtype=jnp.int32))
    sev = E.StructureEvents(jnp.zeros((n,), bool), jnp.arange(n, dtype=jnp.int32), *([jnp.zeros((n,))] * 2),
                            jnp.zeros((n,), jnp.int32), *([jnp.zeros((n,))] * 2), *([jnp.zeros((n,), bool)] * 3))
    einp = E.EconomyInputs(now, jnp.arange(c, dtype=jnp.int32), zc, zc, ic, zc, zc, report, IC.CC(
        jnp.zeros((c, n), bool), jnp.zeros((c, n), bool)), ic - 1, md, jnp.zeros((n,), bool), sev, zc, fc, fc, fc, fc,
        fc, fc, fc, fc, fc, zc, zc, zc, zc, zc, zc)
    eout = E.economy_step(s.econ, einp)
    summ, s_eff, s_out = S.step(s.summoners, ctx, wu, request=W.CastOrder(ic - 1, ic - 1, zc, zc), now=now,
                                dt=jnp.float32(dt), summoner_haste=zc, can_cast=~fc, channel_interrupted=fc,
                                quest_complete=fc, rooted=fc)
    wreq = WD.WardRequest(ic - 1, zc, zc)
    wards, wev = WD.ward_step(s.wards, now=now, dt=jnp.float32(dt), request=wreq, x=zc, y=zc, team=ic, alive=~fc,
                              level=ic + 1, trinket_id=ic + 3340, control_count=ic, grid=cfg.ward_grid, can_use=~fc,
                              trinket_haste=zc, hits=jnp.zeros((16,), jnp.int32), hitter=jnp.full((16,), -1, jnp.int32),
                              rune_pages=cfg.rune_pages, ward_visible=jnp.zeros((2, 16), bool))
    view, _ = WD.ward_view(s.wards, now=now, x=zc, y=zc, team=ic, alive=~fc, level=ic + 1)
    qs = Q.quest_step(s.econ.quest, Q.no_quest_events(c), now=now, dt=dt, in_lane=fc, alive=~fc, level=ic + 1,
                      recalled=fc)
    roots = [
        s, cfg.champion_base, zero_stats((c,)), st, p, s.shields, dfn, off, res, D.Vamp(zc, zc), ctx, units, wu,
        attack, IC.Cast(fc, ic, ic - 1), IC.CC(jnp.zeros((c, n), bool), jnp.zeros((c, n), bool)),
        IC.Kills(zc, zc, zc, fc, jnp.zeros((c, n), bool)), report, IC.AttackMods(fc, zc + 1),
        IC.neutral_defense(c), IC.neutral_debuffs(n), IC.shield_grants(zc), IC.no_effects(c, n),
        IC.StatusFlags(fc), IC.ActiveOut(fc, zc, ~fc, fc), IC.Owned(own, jnp.zeros(own.shape, bool)), ev, RC.no_outputs(c, len(catalog().ids)),
        kctx, K.core.no_out(c, n), K.core.neutral_defense(c), K.core.neutral_attack_mods(c), W.no_cc(c, n),
        W.no_dash(c), W.CastOrder(ic, ic, zc, zc), W.AttackLaunch(fc, ic, fc, fc, ic), md, sev, einp, eout,
        Q.no_quest_events(c), qs, s_out, wreq, wev, view]
    return roots


def collect(roots) -> dict:
    """class -> one instance; nested NamedTuples included."""
    found: dict = {}

    def walk(v):
        if is_nt(v):
            found.setdefault(type(v), v)
            for f in v._fields:
                walk(getattr(v, f))
    for r in roots:
        walk(r)
    return found


def field_type(cls, name, value, found) -> str | None:
    if (cls.__name__, name) in SKIP_FIELDS:
        return None
    if value is None:
        return OPTIONAL[(cls.__name__, name)]
    if is_nt(value):
        return qualname(type(value))
    if isinstance(value, tuple):
        raise TypeError(f"{cls.__name__}.{name}: plain tuple fields are not generated")
    a = np.asarray(value)
    t = CTYPE[str(a.dtype)] if str(a.dtype) in CTYPE else CTYPE["float32"]
    return t if a.ndim == 0 and not isinstance(value, float) else (t if a.ndim == 0 else f"Arr<{t}>")


def main() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    found = collect(examples())
    # Dependency order: nested types first.
    order, seen = [], set()

    def visit(cls):
        if cls in seen:
            return
        seen.add(cls)
        inst = found[cls]
        for f in inst._fields:
            v = getattr(inst, f)
            if is_nt(v):
                visit(type(v))
        order.append(cls)
    for cls in sorted(found, key=lambda k: qualname(k)):
        if cls.__name__ not in EXCLUDE:
            visit(cls)
    lines = ["// GENERATED by ops/native/gen_types.py from the JAX NamedTuples: do not edit.",
             "#pragma once", "#include <cstdint>", "", '#include "../champ/arr.hpp"', "", "namespace lanesim {", ""]
    for cls in order:
        inst = found[cls]
        mod = cls.__module__.replace("lanerl_jax.modern.", "")
        lines.append(f"// {mod}.{cls.__name__}")
        lines.append(f"struct {qualname(cls)} {{")
        members = []
        for f in inst._fields:
            t = field_type(cls, f, getattr(inst, f), found)
            if t is None:
                continue
            v = getattr(inst, f)
            shape = "" if v is None or is_nt(v) else f"  // {list(np.shape(v))}"
            fname = f + "_" if f in ("type", "default", "new", "delete", "union") else f
            lines.append(f"    {t} {fname}{{}};{shape}")
            members.append(fname)
        lines.append("    template <class F> void visit(F&& f) { " + " ".join(f"f({m});" for m in members) + " }")
        lines.append("};")
        lines.append("")
    lines.append("}  // namespace lanesim")
    OUT.write_text("\n".join(lines) + "\n")
    print(len(order), "types ->", OUT)


if __name__ == "__main__":
    main()
