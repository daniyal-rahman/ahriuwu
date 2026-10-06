"""Build ``26.19/economy_client.json`` (SR economy/progression) from 16.19.8230722 client data.

Follows the CLASSIC ``GameModeMapData`` of the cached Map11 bin to its experience curve, experience mod
data, death times and kill-gold/bounty config, and reads the CLASSIC ``mGameModeConstants``
(``classic-constants.json``, byte-identical to map11 ``{6cf687be}``). Hashed config fields keep their hash;
the runtime ``bounty.*`` names follow docs/modern/ECONOMY_PROGRESSION.md §6.1.

    python -m lanerl_jax.modern.data.build_economy [--research DIR] [--out PATH]
"""
from __future__ import annotations

import json
from pathlib import Path

from . import CLIENT_BUILD, PATCH, PATCH_DIR, build_main, sha256

CLASSIC = "Maps/Shipping/Map11/Modes/CLASSIC"
KILL_GOLD_CONFIG = "{4fd6b68d}"          # CLASSIC Configs entry with BaseGold/FirstBloodBonus
BOUNTY_NAMES = {                          # ECONOMY_PROGRESSION.md §6.1 (CDV value, INF name)
    "{11f10a40}": "early_assist_start", "{937cc95a}": "early_assist_end", "{1eacb90a}": "early_assist_mult",
    "{fd43d59f}": "max_above_base", "{907442e7}": "positive_buffer", "{fe1b406e}": "min_kill_gold",
    "{54ccd262}": "kill_gold_per_bounty", "{c29d06b9}": "gv_gold_per_bounty_positive",
    "{fa93507d}": "gv_gold_per_bounty_negative", "{ec211346}": "devalue_gold_per_bounty",
}
CONSTANTS = (
    "ai_StartingGold", "Gold_Max", "ai_AmbientGoldAmount", "ai_AmbientGoldInterval",
    "mission_AmbientGoldStartTime", "ai_ExpRadius2", "aiExp_timeForKillCreditAfterDeath",
    "aiExp_bonusExpLaneLevelStart", "aiExp_bonusExpLaneLevelDeltaMin", "aiExp_bonusExpLevelDeltaCap",
    "aiExp_bonusExpPercentPerLaneMinionLevelC1", "aiExp_bonusExpPercentPerLaneMinionLevelC1UBound",
    "aiExp_bonusExpPercentPerLaneMinionLevelC2", "gcd_PercentEXPBonusMinimum", "gcd_PercentEXPBonusMaximum",
    "gcd_PercentRespawnTimeModMinimum", "ai_levelUp_healthGainNetGain",
    "ai_levelUp_healthGainPercentMissingPenalty", "sp_HealthRegenPercent", "sp_ManaRegenPercent",
    "sp_RegenRadius", "sp_RegenTickInterval", "events_TimeForMultiKill", "events_TimerForBuildingKillCredit",
)


def build(research: Path) -> dict:
    map_path = research / "cdragon-16.19" / "map11.bin.json"
    const_path = research / "classic-constants.json"
    m = json.loads(map_path.read_text())
    mode = m[CLASSIC]
    if KILL_GOLD_CONFIG not in mode["Configs"]:
        raise RuntimeError("CLASSIC record no longer references the kill-gold config")
    curve, mods, death = m[mode["mExperienceCurveData"]], m[mode["mExperienceModData"]], m[mode["mDeathTimes"]]
    gold = m[KILL_GOLD_CONFIG]
    consts = {name: rec.get("mValue", 0.0 if "Float" in rec.get("__type", "") else 0)
              for group in json.loads(const_path.read_text())["mGroups"].values()
              for name, rec in group.get("mConstants", {}).items()}
    missing = [c for c in CONSTANTS if c not in consts]
    if missing:
        raise RuntimeError(f"CLASSIC constants missing: {missing}")
    bounty = {name: float(gold[h]) for h, name in BOUNTY_NAMES.items()}
    bounty["deferral_out_of_combat"] = float(gold["{a966473c}"]["{4b733ea3}"])
    return {
        "schema": "lanerl-client-sr-economy-v1", "patch": PATCH, "client_build": CLIENT_BUILD,
        "sources": {"map11.bin.json": sha256(map_path), "classic-constants.json": sha256(const_path)},
        "refs": {"experience_curve": mode["mExperienceCurveData"], "experience_mod": mode["mExperienceModData"],
                 "death_times": mode["mDeathTimes"], "kill_gold": KILL_GOLD_CONFIG},
        # Cumulative XP to reach level L = index L-2 (levels 2..30 in the array; SR uses 2..20).
        "xp_required": curve["mExperienceRequiredPerLevel"],
        "kill_xp": curve["mExperienceGrantedForKillPerLevel"],
        "shared_kill_xp_mult": curve["mExperienceGrantedMultForSharedKillPerLevel"],
        "level_difference_xp": curve["LevelDifferenceExperienceMultiplierPerLevel"],
        "minion_split_xp": mods["mPlayerMinionSplitXp"],
        "death_time_per_level": death["mTimeDeadPerLevel"],
        "death_scaling_start": death["mScalingStartTime"],
        "death_scaling_increment": death.get("mScalingIncrementTime", 30.0),
        "death_scaling_percent": death.get("mScalingPercentIncrease", 0.005),
        "death_scaling_cap": death.get("mScalingPercentCap", 1.5),
        "death_scaling_points": [[p["mStartTime"], p.get("mPercentIncrease", 0.0)] for p in death["mScalingPoints"]],
        "base_kill_gold": gold["BaseGold"],
        "first_blood_bonus": gold["FirstBloodBonus"],
        "assist_window": gold["AssistDurationOverride"],
        "bounty": bounty,
        "bounty_raw": {k: v for k, v in gold.items() if k.startswith("{")},
        "constants": {c: consts[c] for c in CONSTANTS},
    }


if __name__ == "__main__":
    build_main(__doc__, build, PATCH_DIR / "economy_client.json", lambda p, out: f"wrote {out}", indent=1)
