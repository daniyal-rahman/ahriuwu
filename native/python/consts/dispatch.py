"""Item module coverage (COVERAGE plus ACTIVE_ITEMS): the dispatch skips a module no holder can hold (items/effects
``_each``). Rune tree coverage: the native dispatch skips a dormant tree (none of its runes on a page)."""


def consts():
    from lanerl_jax.modern.items import effects as IE
    out = {}
    for m in IE.MODULES:
        name = m.__name__.rsplit(".", 1)[-1]
        ids = sorted({*m.COVERAGE, *getattr(m, "ACTIVE_ITEMS", ())})
        out[f"dispatch.items.{name}"] = ids or [0]
    from lanerl_jax.modern.runes import effects as RE
    for m in RE.MODULES:
        out[f"dispatch.runes.{m.__name__.rsplit('.', 1)[-1]}"] = sorted(m.COVERAGE) or [0]
    return out
