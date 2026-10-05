"""Compare the modern stat pipeline with Riot match-v5 timeline champion stats.

TOOL (MODERN-016). For every participant-minute of the recorded 16.9 games it
rebuilds the inventory from the timeline's item events, then predicts max HP,
AD, AP, armor, MR, ability haste, attack speed and move speed through
``core.stat_pipeline.compose`` with 16.9 champion records, 16.9 item stats
and the player's stat shards, and compares with Riot's ``championStats``.

    python -m ops.modern.riot_stats_oracle extract --matches DIR --out lanerl_jax/modern/data/oracle/riot_16_9_frames.json.gz

``extract`` writes a compact, anonymised table (no player names or PUUIDs):
per participant-minute the observed stats, level, xp, gold, inventory and the
rune page, plus kill events with bounty/shutdown. Tests read that table.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter

STATS = ("healthMax", "attackDamage", "abilityPower", "armor", "magicResist", "abilityHaste",
         "attackSpeed", "movementSpeed", "health", "lifesteal", "omnivamp", "armorPen", "magicPen",
         "armorPenPercent", "magicPenPercent", "ccReduction", "powerMax")


def inventory_timeline(events, pid):
    """Item multiset after each event for participant ``pid``: list of (t_ms, Counter)."""
    inv = Counter()
    out = []
    for e in events:
        if e.get("participantId") != pid:
            continue
        t, kind = e["timestamp"], e["type"]
        if kind == "ITEM_PURCHASED":
            inv[e["itemId"]] += 1
        elif kind in ("ITEM_SOLD", "ITEM_DESTROYED"):
            if inv[e["itemId"]] > 0:
                inv[e["itemId"]] -= 1
        elif kind == "ITEM_UNDO":
            if e.get("beforeId") and inv[e["beforeId"]] > 0:
                inv[e["beforeId"]] -= 1
            if e.get("afterId"):
                inv[e["afterId"]] += 1
        else:
            continue
        out.append((t, +inv))
    return out


def extract(match_dir: str, out: str) -> None:
    rows, kills, meta = [], [], []
    for name in sorted(os.listdir(match_dir)):
        if not name.endswith("_timeline.json"):
            continue
        mid = name[:-len("_timeline.json")]
        tl = json.load(open(os.path.join(match_dir, name)))
        mt = json.load(open(os.path.join(match_dir, mid + ".json")))
        info = mt["info"]
        parts = {p["participantId"]: p for p in info["participants"]}
        events = [e for f in tl["info"]["frames"] for e in f["events"]]
        meta.append({"match": mid, "version": info["gameVersion"], "duration": info["gameDuration"]})
        for pid, p in parts.items():
            perks = p["perks"]
            runes = [s["perk"] for st in perks["styles"] for s in st["selections"]]
            shards = [perks["statPerks"]["offense"], perks["statPerks"]["flex"], perks["statPerks"]["defense"]]
            invs = inventory_timeline(events, pid)
            for f in tl["info"]["frames"]:
                pf = f["participantFrames"][str(pid)]
                t = f["timestamp"]
                inv = Counter()
                for te, c in invs:
                    if te <= t:
                        inv = c
                    else:
                        break
                rows.append([mid, pid, p["championName"], p["teamId"], p.get("teamPosition", ""), t,
                             pf["level"], pf["xp"], pf["totalGold"], pf["currentGold"],
                             [pf["championStats"].get(s) for s in STATS], sorted(inv.elements()), runes, shards])
        for e in events:
            if e["type"] == "CHAMPION_KILL":
                kills.append([mid, e["timestamp"], e["killerId"], e["victimId"], e.get("assistingParticipantIds", []),
                              e.get("bounty"), e.get("shutdownBounty"), e.get("killStreakLength")])
    payload = {"schema": "riot-match-v5-oracle/v1", "patch": "16.9",
               "source": "Riot match-v5 /matches/{id} and /timeline (americas), fetched 2026-10-01",
               "fields": ["match", "pid", "champion", "team", "position", "t_ms", "level", "xp", "total_gold",
                          "current_gold", "stats", "items", "runes", "shards"],
               "stat_names": list(STATS), "kill_fields": ["match", "t_ms", "killer", "victim", "assists",
                                                          "bounty", "shutdown", "streak"],
               "matches": meta, "frames": rows, "kills": kills}
    with gzip.open(out, "wt") as fh:
        json.dump(payload, fh)
    print(f"wrote {out}: {len(meta)} matches, {len(rows)} participant-minutes, {len(kills)} kills")


CHUNK = 2000
MULTIPLICATIVE = ("tenacity", "slow_resist", "percent_armor_pen", "percent_magic_pen")


def predict(payload: dict, client: dict, *, runes: bool = True, item_effects: bool = True) -> dict:
    """Predicted Riot-style stats for every participant-minute after t=0.

    Returns ``{"rows": [...], "obs": (R, S) array, "pred": {stat: (R,)}}``.
    Static parts come from 16.9 champion/item records and the shards; rune
    stats come from ``runes.effects.stats`` with every rune at its
    initial state (dynamic stacks 0), evaluated at the frame's game time,
    level, HP and inventory. Riot reports ``attackSpeed`` as 100 × (1 + bonus
    AS) (fits 65% of frames vs 14% for 100 × AS / base AS on champions whose
    AS ratio differs from base AS) and truncates every stat to an integer.
    """
    import jax.numpy as jnp
    import numpy as np

    from lanerl_jax.modern.core import stat_pipeline as SP
    from lanerl_jax.modern.items.catalog import STAT_FIELDS, ItemStats, catalog, combine_stats
    from lanerl_jax.modern.items.loadout import stat_shard_stats
    from lanerl_jax.modern.runes.catalog import rune_catalog
    f = {k: i for i, k in enumerate(payload["fields"])}
    s_idx = {k: i for i, k in enumerate(payload["stat_names"])}
    rows = [r for r in payload["frames"] if r[f["t_ms"]] > 0 and r[f["champion"]].lower() in client["champions"]]
    n = len(rows)
    base_rows = [client["champions"][r[f["champion"]].lower()] for r in rows]
    lv = np.asarray([r[f["level"]] for r in rows], np.float32)
    bonus = {k: np.zeros(n, np.float32) for k in STAT_FIELDS}
    for i, r in enumerate(rows):
        parts = [client["items"].get(str(it), {}) for it in r[f["items"]]]
        sh = stat_shard_stats(tuple(r[f["shards"]]), level=r[f["level"]], adaptive_to_ad=None, xp=np)
        parts.append({k: float(np.asarray(getattr(sh, k))) for k in STAT_FIELDS})
        for part in parts:
            for k, v in part.items():
                bonus[k][i] = 1 - (1 - bonus[k][i]) * (1 - v) if k in MULTIPLICATIVE else bonus[k][i] + v
    col = lambda key: jnp.asarray([b.get(key) or 0.0 for b in base_rows], jnp.float32)
    base = SP.ChampionBase(
        base_hp=col("base_hp"), hp_per_level=col("hp_per_level"), base_ad=col("base_ad"),
        ad_per_level=col("ad_per_level"), base_armor=col("base_armor"), armor_per_level=col("armor_per_level"),
        base_mr=col("base_mr"), mr_per_level=col("mr_per_level"), base_ms=col("base_ms"),
        attack_range=col("attack_range"), attack_speed=col("attack_speed"),
        attack_speed_ratio=col("attack_speed_ratio"), attack_speed_per_level=col("attack_speed_per_level"),
        windup_percent=jnp.full((n,), 0.3), windup_modifier=jnp.ones((n,)),
        hp_regen=jnp.zeros((n,)), hp_regen_per_level=jnp.zeros((n,)))
    static = ItemStats(**{k: jnp.asarray(v) for k, v in bonus.items()})
    total = static
    if runes:
        from lanerl_jax.modern.items.effects.core import Ctx
        from lanerl_jax.modern.runes import effects as RE
        from lanerl_jax.modern.runes.effects.core import rune_events
        cat = rune_catalog()
        page = np.zeros((n, len(cat.ids)), np.int32)
        icat = catalog()
        own = np.zeros((n, len(icat.ids)), np.int32)
        for i, r in enumerate(rows):
            for pid in list(r[f["runes"]]) + list(r[f["shards"]]):
                if pid in cat:
                    page[i, cat.row(pid)] += 1
            for it in r[f["items"]]:
                if it in icat:
                    own[i, icat.row(it)] += 1
        pre = SP.compose(base, jnp.asarray(lv), static)
        hp = jnp.asarray([r[f["stats"]][s_idx["health"]] or 0.0 for r in rows], jnp.float32)
        z = jnp.zeros((n,), jnp.float32)
        t = jnp.asarray([r[f["t_ms"]] / 1000.0 for r in rows], jnp.float32)
        ctx = Ctx(now=t, dt=jnp.float32(1 / 30), unit=jnp.arange(n, dtype=jnp.int32), team=jnp.zeros((n,), jnp.int32),
                  alive=hp > 0, level=jnp.asarray(lv), is_ranged=col("attack_range") > 300, x=z, y=z,
                  facing_x=z + 1, facing_y=z, moved=z, base_ad=pre.base_ad, bonus_ad=pre.bonus_ad, ap=pre.ap,
                  base_hp=pre.base_hp, max_hp=pre.max_hp, hp=hp, base_armor=pre.base_armor,
                  bonus_armor=pre.bonus_armor, base_mr=pre.base_mr, bonus_mr=pre.bonus_mr, mana=z, max_mana=z,
                  base_ms=col("base_ms"), move_speed=pre.move_speed, crit_chance=z, crit_damage=z + 2,
                  life_steal=z, bonus_attack_speed=pre.bonus_attack_speed, ability_haste=z, lethality=z,
                  heal_shield_power=z, attack_windup=z, in_combat=jnp.zeros((n,), bool),
                  in_shop=jnp.zeros((n,), bool))
        # Evaluate in chunks: rows act as independent "holders", and some hooks build
        # holder x holder arrays (ally searches), so one 40k-row batch would be 40k^2.
        parts = []
        for lo in range(0, n, CHUNK):
            sl = slice(lo, min(lo + CHUNK, n))
            c = Ctx(*(v[sl] if hasattr(v, "shape") and v.shape[:1] == (n,) else v for v in ctx))
            c = c._replace(unit=jnp.arange(c.level.shape[0], dtype=jnp.int32))
            ev = rune_events(c, 1, game_time=t[sl], own=jnp.asarray(own[sl]))
            dyn = RE.stats(RE.init(c.level.shape[0], 1), jnp.asarray(page[sl]), c, ev)
            if item_effects:
                # Item passives that are stats (Rabadon's %AP, Sterak's, Overlord's ...), each
                # at its initial state, through items.effects.dynamic_stats.
                from lanerl_jax.modern.items import effects as IE
                dyn = combine_stats(dyn, IE.dynamic_stats(IE.init(c.level.shape[0], 1), jnp.asarray(own[sl]), c))
            parts.append(dyn)
        sizes = [min(CHUNK, n - lo) for lo in range(0, n, CHUNK)]
        dyn = ItemStats(*(jnp.concatenate([jnp.broadcast_to(jnp.asarray(getattr(d, k), jnp.float32), (m,))
                                           for d, m in zip(parts, sizes)]) for k in ItemStats._fields))
        total = combine_stats(static, dyn)
    st = SP.compose(base, jnp.asarray(lv), total)
    obs = np.asarray([[np.nan if v is None else v for v in r[f["stats"]]] for r in rows], float)
    pred = {"healthMax": st.max_hp, "attackDamage": st.base_ad + st.bonus_ad, "abilityPower": st.ap,
            "armor": st.base_armor + st.bonus_armor, "magicResist": st.base_mr + st.bonus_mr,
            "movementSpeed": st.move_speed, "attackSpeed": 100.0 * (1.0 + st.bonus_attack_speed),
            "magicPen": total.magic_pen, "armorPenPercent": 100.0 * total.percent_armor_pen,
            "lifesteal": 100.0 * total.life_steal, "omnivamp": 100.0 * total.omnivamp,
            "ccReduction": 100.0 * st.tenacity}
    return {"rows": rows, "obs": obs, "stat_index": s_idx, "fields": f,
            "pred": {k: np.asarray(v, np.float64) for k, v in pred.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--matches", required=True)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.matches, args.out)


if __name__ == "__main__":
    main()
