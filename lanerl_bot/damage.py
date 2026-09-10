"""The last-hit model, in Python.

This is the reference implementation of the same decision the C# bot makes in
GameServerLib/Lanerl/LanerlAim.cs. Two copies of a rule is normally a smell; here it
is deliberate and the drift is tested rather than assumed:

  * the C# copy is what the frozen anchor actually plays, in-process, at 60Hz;
  * this copy is what the behaviour-cloning prior and any Python-side policy will
    reason with, and it is what the unit tests can exercise without booting a server.

tests/test_last_hit.py checks both against a table the server prints under
LANERL_SELFTEST=1, so if either side changes the rule the parity test fails.
"""
from __future__ import annotations

from dataclasses import dataclass

from lanerl_bot import content


def post_mitigation(damage: float, resist: float) -> float:
    """Stats.GetPostMitigationDamage, physical.

    Negative resist is mirrored rather than amplifying without bound, which is what
    the engine does (`mitigationPercent = 2 - mitigationPercent`).
    """
    if damage <= 0.0:
        return 0.0
    pct = 100.0 / (100.0 + resist)
    if resist < 0.0:
        pct = 2.0 - pct
    return damage * pct


def windup_seconds(
    attack_delay_offset_pct: float,
    attack_delay_cast_offset_pct: float,
    attack_speed_multiplier: float = 1.0,
) -> float:
    """Seconds from starting an auto attack to the damage landing.

    Mirrors Spell.Cast's auto-attack arm: the swing's total time comes from the
    unit's own AttackDelayOffsetPercent, the cast fraction from the global constant
    plus the variant's offset, and Spell.Update divides the wait by attack speed.
    """
    total = content.GCD_ATTACK_DELAY * (1.0 + attack_delay_offset_pct)
    cast = total * (content.GCD_ATTACK_DELAY_CAST_PERCENT + attack_delay_cast_offset_pct)
    return max(0.0, cast / max(0.01, attack_speed_multiplier))


def is_last_hit(
    target_hp: float,
    my_damage: float,
    incoming_damage: float = 0.0,
    time_to_land: float = 0.0,
    regen_per_second: float = 0.0,
    decline_when_wave_kills: bool = False,
) -> bool:
    """Should I start an auto attack at this minion right now?

    The two halves are deliberately asymmetric, and that asymmetry is the whole
    point of the rule:

      upper half -- "is my swing enough?" -- uses the *predicted* HP at landing, so
        the swing starts early enough to connect on a minion that is still dropping;

      lower half -- "will it die without me?" -- uses *current* HP by default.
        Declining a swing because the wave is predicted to finish the minion trades a
        guaranteed 20 gold for an auto attack that costs almost nothing. Measured over
        full 10-minute games on this server, declining cost roughly a quarter of
        CS@10, so it is off unless an anchor is specifically meant to play for wave
        control.
    """
    hp_at_landing = target_hp - incoming_damage + regen_per_second * time_to_land
    floor_hp = hp_at_landing if decline_when_wave_kills else target_hp
    return floor_hp > 0.0 and hp_at_landing <= my_damage


@dataclass
class LastHitModel:
    """Garen's last-hit model at a given level, against this server's minions.

    >>> m = LastHitModel(level=1)
    >>> round(m.damage_vs("Blue_Minion_Basic"), 2)   # 0 armor
    57.88
    >>> m.can_last_hit("Blue_Minion_Basic", hp=50)
    True
    >>> m.can_last_hit("Blue_Minion_Basic", hp=200)
    False
    """

    level: int = 1
    champion_name: str = "Garen"
    attack_speed_multiplier: float = 1.0
    bonus_attack_damage: float = 0.0

    @property
    def champ(self) -> content.ChampionStats:
        return content.champion(self.champion_name)

    @property
    def attack_damage(self) -> float:
        return self.champ.attack_damage(self.level) + self.bonus_attack_damage

    @property
    def windup(self) -> float:
        return windup_seconds(
            self.champ.attack_delay_offset_pct,
            self.champ.attack_delay_cast_offset_pct,
            self.attack_speed_multiplier,
        )

    def damage_vs(self, minion_name: str) -> float:
        return post_mitigation(self.attack_damage, content.unit(minion_name).armor)

    def travel_seconds(self, distance: float = 0.0, missile_speed: float = 0.0) -> float:
        """Melee resolves inside FinishCasting, so this is zero for Garen."""
        if self.champ.is_melee or missile_speed <= 1.0:
            return 0.0
        return distance / missile_speed

    def time_to_land(self, cooldown_remaining: float = 0.0, distance: float = 0.0,
                     missile_speed: float = 0.0) -> float:
        return cooldown_remaining + self.windup + self.travel_seconds(distance, missile_speed)

    def can_last_hit(
        self,
        minion_name: str,
        hp: float,
        incoming_damage: float = 0.0,
        cooldown_remaining: float = 0.0,
        decline_when_wave_kills: bool = False,
    ) -> bool:
        u = content.unit(minion_name)
        ttl = self.time_to_land(cooldown_remaining)
        return is_last_hit(
            target_hp=hp,
            my_damage=self.damage_vs(minion_name),
            incoming_damage=incoming_damage,
            time_to_land=ttl,
            regen_per_second=u.hp_regen / 5.0,
            decline_when_wave_kills=decline_when_wave_kills,
        )

    def autos_to_kill(self, minion_name: str) -> int:
        """How many autos a full-HP minion of this type takes."""
        u = content.unit(minion_name)
        d = self.damage_vs(minion_name)
        if d <= 0:
            return 10**6
        import math

        return math.ceil(u.base_hp / d)


def garen_q_damage(attack_damage: float, rank: int, target_armor: float) -> float:
    """GarenQAttack: 30 + 25*(rank-1) + 1.4 AD, physical.

    Taken from Content/.../Characters/Garen/Q.cs DealSpellDamage, not from a wiki --
    this server's numbers are what the anchor is measured against.
    """
    if rank <= 0:
        return 0.0
    raw = 30.0 + 25.0 * (rank - 1) + attack_damage * 1.4
    return post_mitigation(raw, target_armor)


def garen_e_tick_damage(attack_damage: float, rank: int, target_armor: float,
                        vs_minion: bool = True) -> float:
    """GarenE buff tick: 10 + 12.5*(rank-1) + AD*(0.35 + 0.05*(rank-1)), x0.75 vs minions.

    From Content/.../Buffs/Garen/GarenE.cs. Ticks every 500ms for the buff's 3s.
    """
    if rank <= 0:
        return 0.0
    raw = 10.0 + 12.5 * (rank - 1) + attack_damage * (0.35 + 0.05 * (rank - 1))
    if vs_minion:
        raw *= 0.75
    return post_mitigation(raw, target_armor)
