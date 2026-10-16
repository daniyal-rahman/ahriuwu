# Training roster (tentative, 2026-10-15)

Behaviour cloning can cover the whole champion pool; the simulated roster is a small slice chosen for **distinct
play patterns**, not a target count. Rule: one champion per way of playing (near-duplicates collapse to one
representative), and where two are close, prefer the one whose kit reuses mechanics the world already has (cheaper
to build and to run). Ornn is dropped (three unsourced kit values; Malphite and K'Sante cover tanks).

## Picks

| Role | Champion | Covers | Reuses |
|---|---|---|---|
| Top | Garen, Darius, Renekton, Nasus, Jax, Fiora, Malphite, K'Sante, Jayce (+ Aatrox, Camille, Yone, Ambessa) | Juggernauts, duelists, tanks, scaling, lane bully, ranged poke | Built |
| Jungle | Lee Sin | Early mobile ganker | Kit dashes, knockback |
| Jungle | Amumu | Tank engage, area CC | Skillshot self-dash, CC |
| Jungle | Graves | Ranged-attack jungler | Ranged attacks, dashes |
| Jungle (flex) | Jax | Duelist | Built |
| Mid | Ahri | Mobile mage, returning skillshot, charm | Skillshots, kit blinks |
| Mid | Lux | Long-range artillery / control | Skillshots, shields, root |
| Mid | Talon | Point-and-click assassin | Dashes, bleed, wall rays (wall hop) |
| Mid (flex) | Yone, Jayce, Malphite | Melee carry, poke, tank mid | Built |
| ADC | Caitlyn | Long-range lane bully, traps | Ranged attacks, self-dash (net) |
| ADC | Ezreal | Mobile skillshot poke, no CC | Kit blinks, skillshots |
| ADC | Jinx | Scaling hypercarry, resets | Ranged attacks, stance swap (weapon swap) |
| Support | Leona | Melee all-in engage | Stun, dash to target |
| Support | Blitzcrank | Hook | Pulls (displacement) |
| Support | Lulu | The one enchanter: shields, polymorph, ally ult | Needs ally targeting |
| Support (flex) | Lux, Malphite | Mage / tank support | In the list |

## Collapsed (one representative kept)

- Healing / shielding enchanters: Soraka, Sona, Nami, Yuumi, Milio, Janna -> Lulu.
- Hooks: Thresh, Nautilus, Pyke -> Blitzcrank.
- Artillery mages: Xerath, Vel'Koz -> Lux.
- Skillshot-poke marksmen: Varus, Lucian -> Ezreal.

Left out for unique, expensive mechanics: clones and pets (Zed, Shaco, Annie), Yuumi's attach, Viego's possession,
Aphelios's weapon rotation.

## World prerequisites

- The world has exactly two champions (`world.config.N_CHAMPIONS`). Mid and ADC picks can play 1v1 in their lane
  now; jungle and support picks need a world with more champions per team.
- Allied effects (Lulu, any heal or shield on an ally, Acceleration Gate on allies) need ally targeting in the
  action space; no champion can target an ally yet.
- Flex picks are a build, rune page and skill order plus the lane the champion is sent to.

Client data: `RESEARCH/cdragon-16.19/<name>.bin.json` for every pick (fetched 2026-10-15, `SHA256SUMS`).
