"""Economy and role-quest tables (lanerl_jax/modern/economy.py, role_quest.py) for native/src/champ/econ.

Python-side arithmetic of the JAX module (e.g. the ambient payment per tick) is done here in float64, exactly as
the JAX code does before the value meets a float32 array.
"""


def consts():
    from lanerl_jax.modern import economy as E
    e = E.econ()
    t = E._tables()
    out = {f"econ.economy.{k}": t[k] for k in ("need", "kill_xp", "share", "death", "base_gold", "split")}
    out["econ.economy.minion_xp_radius"] = t["minion_xp_radius"]
    out.update({f"econ.economy.const.{k}": float(v) for k, v in e["constants"].items()})
    out.update({f"econ.economy.bounty.{k}": float(v) for k, v in e["bounty"].items()})
    out["econ.economy.ambient_per"] = (E._const("ai_AmbientGoldAmount") / E._const("ai_AmbientGoldInterval")
                                       * E.AMBIENT_TICK)
    out["econ.economy.assist_window"] = float(e["assist_window"])
    out["econ.economy.first_blood_bonus"] = float(e["first_blood_bonus"])
    out["econ.economy.level_difference_slope"] = float(e["level_difference_xp"][1])
    out["econ.economy.death_scaling_increment"] = float(e["death_scaling_increment"])
    out["econ.economy.death_scaling_cap"] = float(e["death_scaling_cap"]) - 1.0
    pts = e["death_scaling_points"]
    # (start, end, pct) per scaling point; end of the last is 1e9 as in time_increase_factor.
    out["econ.economy.death_scaling_points"] = [v for i, (s, p) in enumerate(pts)
                                                for v in (s, pts[i + 1][0] if i + 1 < len(pts) else 1e9, p)]
    return out
