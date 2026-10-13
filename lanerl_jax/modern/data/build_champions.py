"""Build ``26.19/champions/<name>.json`` from the 16.19.8230722 champion bins.

Reads ``RESEARCH/cdragon-16.19/<name>.bin.json`` (CommunityDragon ``game/data/characters/<name>/<name>.bin.json``,
checksums in that directory's ``SHA256SUMS``). Each snapshot keeps the CharacterRecord stat fields plus the radii and
acquisition range the world needs, the Q/W/E/R/Passive root spells (``spellNames`` order, passive from
``mCharacterPassiveSpell``) and any second-form spells (``EXTRA_SLOTS``) with their calculations, cooldowns, mana and
data values, and a ``slots`` map from slot to spell path. CLIENT data only; rules and wiki values live in the kit modules.

    python -m lanerl_jax.modern.data.build_champions [--research DIR] [--out DIR] [NAME ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .champions import IDS
from . import PATCH, PATCH_DIR, RESEARCH

CHARACTER_FIELDS = (
    "armorPerLevelModifiable", "attackRangeModifiable", "attackSpeedModifiable", "attackSpeedPerLevelModifiable",
    "attackSpeedRatioModifiable", "baseArmorModifiable", "baseDamageModifiable", "baseHPModifiable", "baseMR",
    "baseMoveSpeedModifiable", "baseStaticHPRegenModifiable", "basicAttack", "damagePerLevelModifiable",
    "hpPerLevelModifiable", "hpRegenPerLevelModifiable", "mrPerLevel", "primaryAbilityResource",
    "acquisitionRange", "pathfindingCollisionRadius", "overrideGameplayCollisionRadius", "selectionRadius")
SPELL_FIELDS = (("calculations", "mSpellCalculations"), ("cooldown", "Cooldown"), ("mana", "manaValues"),
                ("values", "DataValues"))
# Second-form spells of shapeshifters, pinned under extra slot names (Jayce's Mercury Cannon; ``R2`` transforms back).
EXTRA_SLOTS = {"Jayce": {"Q2": "JayceShockBlastAbility/JayceShockBlast", "W2": "JayceHyperChargeAbility/JayceHyperCharge",
                         "E2": "JayceAccelerationGateAbility/JayceAccelerationGate",
                         "R2": "JayceStanceHtGAbility/JayceStanceGtH"}}


def snapshot(bin_path: Path, name: str) -> dict:
    b = json.loads(bin_path.read_text())
    root = b[f"Characters/{name}/CharacterRecords/Root"]
    prefix = f"Characters/{name}/Spells/"
    slots = {s: prefix + p for s, p in zip("QWER", root["spellNames"])}
    slots["Passive"] = root["mCharacterPassiveSpell"]
    slots.update({s: prefix + p for s, p in EXTRA_SLOTS.get(name, {}).items()})
    spells = {}
    for path in slots.values():
        m = b[path]["mSpell"]
        spells[path] = {dst: m[src] for dst, src in SPELL_FIELDS if m.get(src) is not None}
    return {"character": {k: root[k] for k in CHARACTER_FIELDS if k in root},
            "patch": PATCH,
            "slots": slots,
            "source": f"https://raw.communitydragon.org/16.19/game/data/characters/{name.lower()}/"
                      f"{name.lower()}.bin.json",
            "spells": spells}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--research", type=Path, default=RESEARCH)
    ap.add_argument("--out", type=Path, default=PATCH_DIR / "champions")
    ap.add_argument("names", nargs="*", default=sorted(IDS))
    args = ap.parse_args()
    for name in args.names:
        out = args.out / f"{name.lower()}.json"
        data = snapshot(args.research / "cdragon-16.19" / f"{name.lower()}.bin.json", name)
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(f"{name}: {len(data['spells'])} spells -> {out}")


if __name__ == "__main__":
    main()
