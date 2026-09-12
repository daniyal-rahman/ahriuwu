"""The two game configs must describe the SAME champion, and a mirror match.

There are exactly two server game configs in this repo and they had drifted
apart, so every scripted-anchor number the project quotes was measured on a
different champion than the one the RL agent plays.

    lanerl/cfg/garen1v1.json          launched by ALL training and eval servers
                                      (lanerl_train/paths.py default_game_config
                                      -> vec.py resolved_config), and by every
                                      lanerl/*.sh render/demo script.
    lanerl_bot/configs/garen1v1_bot.json
                                      launched by lanerl_bot/tests/conftest.py,
                                      lanerl_bot/bench/run_server.py and
                                      lanerl_audit/preflight.sh -- i.e. by the
                                      runs that produced the bronze/gold/diamond
                                      CS@10 anchors, the Q A/B (31.5 vs 14.8 CS)
                                      and the last-hit tables.

What the drift was (measured 2026-09-12, before this guard existed)
-------------------------------------------------------------------
The bot config carried ``"runes": {}`` and ``"talents": {}``. The train config
grants each champion 30 runes and a 16-entry mastery page, applied by
``Config.cs:104 LoadTalentsAndRunes`` -> ``Champion.cs AddStatModifier``:

    runes    9x 5245 + 3x 5335  -> FlatPhysicalDamageMod  +15.255
             9x 5317            -> FlatArmorMod            +9.0
             9x 5289            -> FlatSpellBlockMod       +12.06
    talents  4132 rank 1        -> Martial Mastery         +5.0 flat AD
             4122 rank 3        -> Brute Force             +0.55 AD per level
             4222 rank 3        -> Veteran's Scars         +36 flat HP
             4232 rank 1        -> Juggernaut              +3% max HP
             (the other 12 ids resolve to EmptyTalentScript and do nothing --
             Content/LeagueSandbox-Scripts/Talents/ implements only those four)

So the anchor champion had 57.88 AD at level 1 and the agent's champion has
57.88 + 15.255 + 5.0 = 78.135 (the live server reports 78.14): 35% more damage,
plus 9 armor, 12.06 MR and ~54 HP. A last-hit threshold, a trade outcome and a
CS@10 count are all functions of exactly those numbers, so none of the anchor
measurements were comparable to an agent measurement. This test re-derives the
stats from the config plus the item JSONs rather than comparing ids, so a page
edited to "the same number of runes, different runes" fails here too.

The mirror
----------
Both configs also gave blue ``Flash + Teleport`` and red ``Heal + Flash``: red
could heal in a trade and blue could not. That breaks two things at once.
Self-play assumes both sides are the same problem, and ``eval_vs_bot`` puts the
agent on BLUE and the scripted bot on RED -- so the *bot* held the combat
summoner in every anchor comparison.

Both sides now carry ``SummonerFlash`` + ``SummonerTeleport``, in that order.
Why that pair and not ``Flash + Heal``:

  * The observation has NO summoner feature at all -- no cooldown, no
    availability, nothing (see the self-vector layout in constants.py). A
    champion that can heal for ``75 + 15*level`` HP (SummonerHeal.cs
    PerformHeal; 90 HP at level 1, ~15% of Garen's 616 base) therefore has an
    HP jump with no observable cause. Mirroring does not fix that: it would
    hand BOTH sides unmodelled hidden state, and the reward is a function of
    HP. Teleport, by contrast, produces a POSITION jump, which the observation
    already handles explicitly and correctly (the displacement budget in
    constants.MAX_WALK_SPEED / BLINK_ALLOWANCE).
  * Teleport is inert in this sandbox in a way Heal is not: it is a 4 s channel
    onto a friendly unit (SummonerTeleport.cs OnSpellPostChannel -> TeleportTo)
    and ``LANERL_TOPONLY=1`` leaves one lane, so there is nowhere to go that
    walking does not reach.
  * Neither actor can cast a summoner today anyway -- the agent's cast action
    carries ``spell_slot`` 0..3 (env.py ServerCommand) and the scripted bot
    loops ``slot < 4`` (LanerlBot.cs) -- so this choice costs nothing now and
    removes the trap later. Summoners live in slots 4/5, so the ORDER matters
    as much as the set: the old configs would have made slot 4 mean "Flash" for
    blue and "Heal" for red.

Deliberately still different, and why
-------------------------------------
``forcedStart``: 120 in the train config, 1 in the bot config. This is the
number of seconds the server waits for a client before force-starting
(``Config.cs:94``, ``Game.cs:324-381``). It is consumed BEFORE ``IsRunning``,
so ``Update`` never runs and game time never advances during it -- it cannot
move a measured number, it only costs wall clock. 120 s is what the render and
demo scripts need for a real League client to finish loading (see the comment
in lanerl/run_game.sh); 1 s is what the test suite wants. Test speed, not
champion capability.

``players[*].name``: ``brian8544``/``prienten`` vs ``bluebot``/``redbot``. The
bot test suite and bench parse telemetry by name (``lanerl_bot/tests/
test_cs_baseline.py``, ``bench/sweep.py``), so these are load-bearing strings
in the bot config; they have no gameplay effect.

``_comment``: a pointer back to this test. ``Config.LoadConfig`` reads named
tokens only, so an extra top-level key is ignored by the server.

Everything else -- ``rank``, ``ribbon``, ``icon``, ``blowfishKey`` -- is
cosmetic lobby metadata and is asserted equal anyway, because there is no
reason for it to drift and a surprise there means someone edited one file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

#: 9x item 5245 (FlatPhysicalDamageMod 0.945) + 3x 5335 (2.25).
#: Pinned HERE rather than imported: `constants.RUNE_FLAT_AD` was deleted along
#: with the Python attack-damage derivation it served, because the server is
#: the source of truth for AD (it also applies a mastery page that no Python
#: copy modelled). This test is about the two CONFIGS matching each other,
#: which is a separate property and still worth guarding.
RUNE_FLAT_AD_EXPECTED = 15.255

from lanerl_rl import constants as C

# Resolve from THIS file, never from a mount literal. /mnt/nfs exists on both
# nodes but /srv/nfs only on danilogin, so a hardcoded /srv/nfs path SKIPS
# silently on desktop -- the node the suite actually runs on.
_REPO = Path(__file__).resolve().parents[2]
TRAIN_CFG = _REPO / "lanerl/cfg/garen1v1.json"
BOT_CFG = _REPO / "lanerl_bot/configs/garen1v1_bot.json"
ITEMS = _REPO.parent / "lanerl-vendor/LoLServer/Content/LeagueSandbox-Default/Items"

#: The only fields allowed to differ between the two configs. Anything else
#: that differs is drift -- see the module docstring for why each is here.
ALLOWED_CONFIG_DIFFS = {"_comment", "forcedStart", "name"}

#: Per-player fields that decide what the champion can DO. Two configs that
#: agree on these produce the same champion; a number measured under one is
#: comparable to a number measured under the other.
CAPABILITY_FIELDS = (
    "champion",
    "skin",
    "summoner1",
    "summoner2",
    "runes",
    "talents",
)

#: Summoners whose effect the observation cannot see. The obs carries no
#: summoner feature at all, so a spell that moves HP or damage is hidden state
#: on both sides of a mirror -- which is worse than an asymmetry, not better,
#: because the reward is a function of HP.
SUMMONERS_INVISIBLE_TO_THE_OBSERVATION = {
    "SummonerHeal",
    "SummonerBarrier",
    "SummonerDot",       # Ignite
    "SummonerExhaust",
    "SummonerBoost",     # Cleanse
}

requires_content = pytest.mark.skipif(
    not ITEMS.exists(), reason="vendored content tree not present"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _players(cfg: dict) -> list:
    players = cfg.get("players") or []
    assert len(players) == 2, f"expected a 1v1, got {len(players)} players"
    return players


def _rune_stats(runes: dict) -> dict:
    """The stat totals a rune page actually grants, off the item JSONs.

    Compared instead of the ids so that "same count, different runes" fails.
    """
    total: dict = {}
    for rune_id in runes.values():
        blob = json.loads((ITEMS / str(rune_id) / f"{rune_id}.json").read_text())
        data = blob.get("Values", blob).get("Data", {})
        for key, value in data.items():
            if isinstance(value, (int, float)) and value:
                total[key] = round(total.get(key, 0.0) + float(value), 6)
    return total


def _flatten(obj, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


# ---------------------------------------------------------------------------
# the two configs describe the same champion
# ---------------------------------------------------------------------------
def test_the_two_configs_differ_only_where_they_are_allowed_to():
    """A whole-file diff, so a NEW divergence fails even if nobody added a check."""
    train, bot = _flatten(_load(TRAIN_CFG)), _flatten(_load(BOT_CFG))

    assert set(train) == set(bot), (
        "the two configs no longer have the same keys: "
        f"only in train {sorted(set(train) - set(bot))}, "
        f"only in bot {sorted(set(bot) - set(train))}"
    )
    unexpected = {
        key: (train[key], bot[key])
        for key in train
        if train[key] != bot[key]
        and key.rsplit(".", 1)[-1] not in ALLOWED_CONFIG_DIFFS
    }
    assert not unexpected, (
        "lanerl/cfg/garen1v1.json and lanerl_bot/configs/garen1v1_bot.json have "
        f"drifted apart at {unexpected}. Every scripted anchor is measured with "
        "the bot config and every agent number with the train config, so a "
        "difference here makes those numbers incomparable. If the difference is "
        "deliberate, add the field to ALLOWED_CONFIG_DIFFS and say why in the "
        "module docstring."
    )


def test_both_configs_grant_the_same_champion_capability():
    """The field-by-field version of the above, with the better error message."""
    train_players, bot_players = _players(_load(TRAIN_CFG)), _players(_load(BOT_CFG))
    for train_p, bot_p in zip(train_players, bot_players):
        for field in CAPABILITY_FIELDS:
            assert train_p.get(field) == bot_p.get(field), (
                f"{field} differs: train {train_p.get('name')} has "
                f"{train_p.get(field)!r}, bot {bot_p.get('name')} has "
                f"{bot_p.get(field)!r}"
            )


def test_the_game_mechanics_blocks_are_identical():
    """COOLDOWNS/MANACOSTS/MINION_SPAWNS decide what game is being played."""
    train, bot = _load(TRAIN_CFG), _load(BOT_CFG)
    assert train["game"] == bot["game"]
    assert train["gameInfo"] == bot["gameInfo"]


@requires_content
def test_both_configs_grant_the_same_rune_stats_not_merely_the_same_ids():
    train, bot = _load(TRAIN_CFG), _load(BOT_CFG)
    for train_p, bot_p in zip(_players(train), _players(bot)):
        train_stats = _rune_stats(train_p.get("runes") or {})
        bot_stats = _rune_stats(bot_p.get("runes") or {})
        assert train_stats == bot_stats, (
            f"rune stats differ: train {train_stats} vs bot {bot_stats}"
        )
        assert train_stats, "the rune page grants nothing -- was it emptied?"


@requires_content
def test_both_configs_grant_the_same_flat_ad_from_runes():
    """Ties both configs to ``constants.RUNE_FLAT_AD``, the last-hit signal.

    ``lanerl_bot/damage.py LastHitModel`` defaults ``bonus_attack_damage=0``, so
    anything reasoning about the anchor in Python must now pass the rune AD --
    the bot config is no longer the rune-free one.
    """
    for cfg_path in (TRAIN_CFG, BOT_CFG):
        for player in _players(_load(cfg_path)):
            stats = _rune_stats(player.get("runes") or {})
            assert stats.get("FlatPhysicalDamageMod") == pytest.approx(
                RUNE_FLAT_AD_EXPECTED, abs=1e-6
            ), (
                f"{cfg_path.name}/{player.get('name')} gets "
                f"{stats.get('FlatPhysicalDamageMod')} flat AD from runes but "
                f"constants.py models {RUNE_FLAT_AD_EXPECTED}"
            )


# ---------------------------------------------------------------------------
# the match is a mirror
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_path", [TRAIN_CFG, BOT_CFG], ids=["train", "bot"])
def test_the_two_players_are_symmetric_in_capability(cfg_path: Path):
    """Self-play and side-canonicalisation both assume one problem, not two.

    ``lanerl_rl/frame.py`` reflects the red observation into blue's frame, so
    the policy cannot tell which side it is on -- it therefore cannot condition
    on having a different champion, and any asymmetry here is pure unobservable
    noise in the return.
    """
    blue, red = _players(_load(cfg_path))
    for field in CAPABILITY_FIELDS:
        assert blue.get(field) == red.get(field), (
            f"{cfg_path.name}: the two sides differ in {field} -- blue "
            f"{blue.get(field)!r}, red {red.get(field)!r}. The observation is "
            "side-canonicalised, so the policy cannot see this."
        )


@pytest.mark.parametrize("cfg_path", [TRAIN_CFG, BOT_CFG], ids=["train", "bot"])
def test_the_summoner_pair_is_identical_and_in_the_same_order(cfg_path: Path):
    """Slot 4 and slot 5 must mean the same spell on both sides.

    The set being equal is not enough: the server keys spells by slot
    (``PlayerConfig.Summoner1``/``Summoner2`` -> slots 4/5), so a swapped pair
    would make one cast id mean different things per side.
    """
    blue, red = _players(_load(cfg_path))
    assert (blue["summoner1"], blue["summoner2"]) == (
        red["summoner1"],
        red["summoner2"],
    )
    assert blue["summoner1"] != blue["summoner2"], "two copies of one summoner"


@pytest.mark.parametrize("cfg_path", [TRAIN_CFG, BOT_CFG], ids=["train", "bot"])
def test_no_equipped_summoner_is_invisible_to_the_observation(cfg_path: Path):
    """Heal was on red in both configs; it must not come back on either side.

    Mirroring a hidden-state summoner does not fix it. The observation carries
    no summoner feature, so ``75 + 15*level`` HP appearing mid-trade is an
    unexplainable jump in the reward for both agents.
    """
    for player in _players(_load(cfg_path)):
        equipped = {player["summoner1"], player["summoner2"]}
        banned = equipped & SUMMONERS_INVISIBLE_TO_THE_OBSERVATION
        assert not banned, (
            f"{cfg_path.name}/{player['name']} carries {sorted(banned)}, whose "
            "effect the observation cannot see. Add a summoner feature to the "
            "self vector before equipping one of these."
        )
