"""Season-4 Garen skill order and item path, with the reasoning attached.

Both tables are duplicated as defaults inside the C# bot (LanerlConfig.cs). This
module is the readable source of truth and the thing the tests check the numbers
against; the C# side reads the same values from a config file when one is given.
"""
from __future__ import annotations

from lanerl_bot import content

# Spell slots as the engine numbers them.
Q, W, E, R = 0, 1, 2, 3

# One entry per champion level, 1..18.
#
# WHY E FIRST. Checked against this server's own spell data rather than assumed:
#
#   Q (GarenQ + GarenQAttack, Content/.../Garen/Q.cs)
#       damage  30 + 25*(rank-1) + 1.4 AD, single target, on the next auto
#       cooldown 8s flat at every rank
#       per rank: +25 flat, single target
#
#   E (GarenE buff, Content/.../Buffs/Garen/GarenE.cs)
#       damage  10 + 12.5*(rank-1) + AD*(0.35 + 0.05*(rank-1)) per 500ms tick,
#               x0.75 against minions, for the buff's 3s -> about 6-7 ticks,
#               and it hits *every* enemy within 330 units
#       cooldown 13/12/11/10/9s
#       per rank: about +65 flat and +0.3 AD per target, times however many
#               targets are in the wave
#
#   W is a shield with no damage; R (Content/.../Characters/Garen/R.cs:28-33) is
#       175*rank flat -- 175/350/525 -- PLUS 28.57/33.33/40% of the target's
#       MISSING health, dealt as DAMAGE_TYPE_MAGICAL, on a 160/120/80s cooldown,
#       gated to levels 6/11/16 by the content data. There is no AD ratio: this
#       comment said "(+1.0 AD) execute damage" and the script has neither an AD
#       term nor a physical damage type.
#
# So E's marginal rank is worth several times Q's for anything involving a wave,
# which matches the Season-4 consensus of maxing Judgment first. R is taken on
# cooldown at 6/11/16 because CanLevelUpSpell gates it there anyway.
#
# One caveat the data forces and the wiki does not mention: GarenE's OnActivate
# calls SetStatus(CanAttack, false) for the buff's whole 3s duration. Spinning
# therefore *suppresses auto attacks*, which is exactly how this bot gets CS. The
# skill order still maxes E (it is the right order for a real game, and E is the
# right tool once the bot is asked to fight), but useEWaveclear defaults to off.
# Imported, not restated. This held its own copy (Q first) and drifted from
# the server's LanerlConfig.SkillOrder the moment that changed.
from lanerl_rl.constants import GAREN_SKILL_ORDER as _CANONICAL

GAREN_SKILL_ORDER = list(_CANONICAL)

# Season-4 Garen, by item id. Verified present in
# Content/LeagueSandbox-Default/Items/ with these prices.
#
# Item purchase IS implemented server-side: HandleBuyItem ->
# Champion.Shop.HandleItemBuyRequest, which checks gold, consumes owned recipe
# components and applies the item's stat modifiers. It does NOT check that the
# champion is at the shop, so the bot enforces that itself (buyRequiresFountain).
GAREN_BUILD_PATH = [
    1054,  # Doran's Shield      440  (+80 HP; starting gold is 475)
    2003,  # Health Potion        35
    2003,  # Health Potion        35
    1001,  # Boots of Speed      325
    3047,  # Ninja Tabi          375 (component price; server charges TotalPrice)
    3134,  # The Brutalizer      617
    3068,  # Sunfire Cape        850
    3035,  # Last Whisper       1065
]

ITEM_NAMES = {
    1054: "Doran's Shield",
    2003: "Health Potion",
    1001: "Boots of Speed",
    3047: "Ninja Tabi",
    3134: "The Brutalizer",
    3068: "Sunfire Cape",
    3035: "Last Whisper",
}


def skill_at(level: int) -> int:
    """Which slot gets the point granted at this champion level."""
    if not 1 <= level <= len(GAREN_SKILL_ORDER):
        raise ValueError(f"level out of range: {level}")
    return GAREN_SKILL_ORDER[level - 1]


def ranks_at(level: int) -> dict[int, int]:
    """Rank of each spell once `level` points have been spent in order."""
    out = {Q: 0, W: 0, E: 0, R: 0}
    for lv in range(1, level + 1):
        out[skill_at(lv)] += 1
    return out


def q_cooldowns() -> list[float]:
    return content.spell_cooldowns("GarenQ")


def e_cooldowns() -> list[float]:
    return content.spell_cooldowns("GarenE")


def r_cooldowns() -> list[float]:
    return content.spell_cooldowns("GarenR")
