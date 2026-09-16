# League of Legends Mechanics — Concept Reference

## Purpose and the sharp edge

This document exists for a JAX reimplementation of a 1v1 Garen top-lane environment that is being brought to exact behavioral parity with a vendored C# server (LeagueSandbox, targeting **patch 4.20**, October 2014). That server — not this document, not the live game, not any wiki — is **the parity oracle**. Parity work means matching the server's actual behavior, bugs included.

The reason this document is still worth having is that the server is a fan reimplementation of a nine-plus-year-old patch, and it is demonstrably wrong in places, sometimes in ways that are easy to mistake for "that's just how League works." Three confirmed examples motivated this document:

- The server's turret basic attack applies a buff named `S5Test_TowerWrath`, but that buff's script is a complete no-op. Real League has escalating ("warming up") turret damage against champions; this server does not implement it at all, despite scaffolding that suggests someone intended to.
- `MinionModifiers` is declared and populated in the server's data, and never applied anywhere — by the code's own comment.
- Garen's passive keys off `UnitTag`, whose enum has no explicit flag values assigned, so the OR'd tags on cannon minions accidentally collide with the `Monster` tag — a genuine bug, not a design choice.

None of these are things a wiki read would predict. That's the point: **this document must never be read as "what to implement for parity."** It has exactly two jobs:

1. **Telling intent from bug.** If you know what a mechanic is *for*, you can recognize when the server is doing something no version of League ever did on purpose — versus something that's merely stale, patch-4.20-accurate behavior that looks weird from a 2026 vantage point.
2. **Feeding the eventual modern-patch migration.** Every place this document shows real League diverging from 2014-era League is a candidate line item for whenever this project moves off patch 4.20 parity and toward current-patch behavior.

If you're reading this while implementing or debugging the JAX environment: match the server. If you're reading this while planning what changes when the target patch moves forward: this is your list.

### How to read each entry

Every mechanic below is structured as:
- **What League does** — with the patch/era noted whenever it has changed.
- **Why it exists** — the design intent, i.e., what problem the mechanic solves.
- **What to watch for** — how a naive implementation (or a bug) gets it wrong.
- **Confidence** — `WELL-DOCUMENTED` (multiple/clear sources agree) or `UNCERTAIN` (sources are silent, vague, or conflict). A guess is never presented as fact; where I could not confirm something, I say so directly rather than filling the gap with a plausible-sounding number.

Numbers are largely avoided in favor of mechanisms, since exact numeric tuning is patch-specific and the server's target patch (4.20) predates most of the numbers on the current wiki by over a decade.

---

## 1. Basic-attack projectiles

This is the most load-bearing entry in this document: it's already implemented in the JAX environment, and it's where subtle timing bugs are most likely to silently break parity.

### Anatomy of an attack: windup, then either instant resolution or a projectile

A basic attack is not a single instantaneous event. The official wiki describes the attack timer as consisting of two phases: **windup** (the execution phase) and **cooldown/recovery** (during which the unit cannot attack again). Within the windup, there is a documented "grace period of one game-tick (0.033 seconds) before a champion's attack windup completes, during which the attack becomes inherently uncancellable" — implying that for most of the windup, the attack *can* still be cancelled (by a movement command, for instance), and only becomes locked in right at the very end.

Crucially, melee and ranged attacks resolve differently once windup finishes:
- **Non-projectile (melee, and hitscan-style) attacks hit simultaneously upon windup completion.** There is no separate travel phase.
- **Projectile (most ranged) attacks spawn a projectile when windup completes, and that projectile "follows and hits the target shortly after."** The travel time is additional time-to-damage on top of the windup, and it is governed by the champion's missile speed, not by attack speed.

**What League does:** Attack speed shortens windup (and the recovery/cooldown portion of the attack timer) — it does **not** change projectile travel time. Two champions with identical attack speed but different missile speeds will land damage at different total delays after the windup completes. This means "attack speed" and "how fast damage actually lands for a ranged auto" are two separate axes, and conflating them (e.g., scaling projectile travel time by attack speed) would be a parity bug in either direction — either matching or diverging from the server depending on what the server actually does, which is exactly the kind of thing worth checking against server behavior directly rather than assuming.
**Why it exists:** Separating windup (attack-speed-scalable) from travel time (fixed per champion) lets ranged champions have a consistent "reaction window" for their target to react to an incoming hit (dodge via blink/flash, block via shield) regardless of how much attack speed the attacker has stacked, while still letting attack speed meaningfully shorten how often attacks are thrown out.
**What to watch for:** A naive implementation might (a) treat all basic attacks as instant/hitscan, silently deleting the "outrun the arrow" and "the projectile is already committed" dynamics described below, or (b) apply attack-speed scaling to travel time, which the sourced formula split contradicts.
**Confidence:** WELL-DOCUMENTED for the windup/cooldown split and the "attack speed scales windup, not travel time" separation. [wiki.leagueoflegends.com/Basic_attack](https://wiki.leagueoflegends.com/en-us/Basic_attack), [wiki.leagueoflegends.com/Attack_speed](https://wiki.leagueoflegends.com/en-us/Attack_speed)

### Windup as a fraction of total attack time — and a documented formula change

The wiki describes two different formulas that have existed for computing a champion's windup percent (the share of the attack timer spent in windup rather than recovery):
1. An **older, constant-based formula**: `0.3 + attackOffset`, where `attackOffset` is a fixed per-champion value.
2. A **newer, time-based formula**: `attackCastTime ÷ attackTotalTime`.

I could not determine from the pages fetched which patch introduced the newer formula, or whether patch 4.20 (2014) used the old constant-based version — but given the timeline (the old formula is described as "the original way," and 4.20 is very early in League's history), it is plausible that 4.20 used the constant-based `0.3 + attackOffset` version. This is a real UNCERTAIN, not a confirmed fact.
**Confidence:** WELL-DOCUMENTED that both formulas have existed; UNCERTAIN which one governed patch 4.20 specifically, and UNCERTAIN what patch changed it. [wiki.leagueoflegends.com/Attack_speed](https://wiki.leagueoflegends.com/en-us/Attack_speed)

### Attack speed cap

The current wiki states the attack speed cap is **3.003 attacks/second**. For most of League's history — very likely including patch 4.20 — the cap was the long-standing **2.5 attacks/second**; the move to a higher, oddly-specific cap is a more recent systemic change. I was not able to pin down the exact patch of that transition in this session, so treat the *existence* of some hard attack-speed cap as WELL-DOCUMENTED and its *current specific value* as **not representative of 4.20** — this belongs on the modern-migration list, not the parity list.
**Confidence:** WELL-DOCUMENTED that a hard cap exists and is enforced; UNCERTAIN about the exact 4.20-era cap value (treated here as very likely 2.5 based on general knowledge of the game's history, not a source confirmed this session). [wiki.leagueoflegends.com/Attack_speed](https://wiki.leagueoflegends.com/en-us/Attack_speed)

### Target dies mid-flight

**What League does:** If the target of an already-launched ranged auto-attack projectile dies before the projectile lands (e.g., a minion dies to a tower shot, or to an ally's attack, in the interval between your projectile launching and arriving), the projectile's damage is lost — it does not redirect, and you do not get credit (no gold, no kill/last-hit credit) for a kill that happens that way. This is the mechanical basis of the extremely well-known "sniped last hit" frustration in laning: you can see your projectile in flight and still not get the CS if something else kills the minion first.
**Why it exists:** This is very likely an emergent consequence of projectile-based damage resolution (damage applies on impact against a specific target reference, and a dead target has no valid impact) rather than a deliberately designed feature — but the resulting behavior is a load-bearing part of skilled last-hitting (timing your attack so it lands *before* a turret shot or ally's hit would have killed the minion anyway).
**What to watch for:** A naive implementation might resolve damage at the moment the attack is declared/launched rather than at impact, which would silently make attacks always land even against targets that die in the interim — this changes the felt difficulty and outcome of contested last hits, and is exactly the kind of subtle timing question worth testing against the server directly.
**Confidence:** UNCERTAIN as a sourced claim — I could not find a wiki passage stating this directly in the pages fetched this session. It is presented here because it is extremely widely and consistently understood by the League playerbase and is implied by projectile-then-impact damage resolution as described by the wiki's own phrasing ("follows and hits the target"), but I want to flag plainly that I do not have a pinpoint citation for it, and it should be verified empirically against the server rather than assumed. [wiki.leagueoflegends.com/Basic_attack](https://wiki.leagueoflegends.com/en-us/Basic_attack)

### Attacker dies mid-flight

**What League does:** A projectile that has already been launched is generally understood to complete its flight and deal its damage even if the attacker is killed immediately afterward — once launched, the projectile is treated as an independent entity, not something that requires its source to remain alive.
**Why it exists:** Consistent with the windup being the "commitment point" of an attack (see the uncancellable-tail-of-windup mechanic above): once you're past that point, the attack is locked in, and this consistently extends past the point of projectile launch.
**What to watch for:** Whether the server actually implements this, or whether it (incorrectly) cancels in-flight projectiles when the source unit dies, is a very plausible and very easy corner-case bug to have — and also a very easy one for a naive JAX implementation to get wrong in either direction.
**Confidence:** UNCERTAIN as a sourced claim for the same reason as above — this reflects general, widely-shared community understanding of the game's mechanics rather than a specific citation retrieved this session. Treat it as a hypothesis to verify against server behavior, not a given.

### Homing vs. flying to a fixed point, and body-blocking

**What League does:** The wiki's own wording for ranged basic attacks is that the projectile "follows and hits the target" — i.e., it tracks the target unit rather than flying toward a fixed point computed at launch. This is corroborated indirectly by Yasuo's Wind Wall (E), whose tooltip is explicit that it "destroys all hostile **non-turret** projectiles" that collide with it — the explicit carve-out for turret shots (see the turret section below on turrets being hitscan/exempt) implies that champion, minion, and monster ranged basic attacks *are* modeled as real, collidable projectile objects, distinct from turret attacks.

League does **not** appear to model general physical body-blocking of basic-attack projectiles — i.e., a third unit simply standing in the geometric path between attacker and target does not intercept the shot the way it might in some other genre entries. Interception is instead handled by specific abilities that explicitly interact with "projectiles" as a category (Wind Wall being the clearest example), not by generic unit collision.
**Why it exists:** Target-following (rather than point-to-a-location) projectiles make ranged basic attacks reliable, skill-neutral tools rather than something that can be juked by simple sidestepping the way a skillshot ability can — this is a deliberate asymmetry between "auto-attacks are hard to dodge with movement alone" and "ability skillshots can be dodged," which is a foundational part of League's combat feel.
**What to watch for:** A naive implementation might (a) fly the projectile to the target's position-at-launch rather than tracking it, making attacks "dodgeable" by moving in a way real League does not support outside specific abilities, or (b) implement literal geometric body-blocking for basic attacks, which is not how real League works — this is precisely the kind of mechanic a fan server could plausibly add as a "reasonable-sounding" feature that isn't actually period- or game-accurate.
**Confidence:** WELL-DOCUMENTED for the "follows the target" wording and the Wind-Wall-implies-projectiles-are-real-objects evidence. UNCERTAIN / based on general knowledge (not independently re-confirmed this session with a direct citation) for the "no generic body-blocking exists" claim. [wiki.leagueoflegends.com/Attack_speed](https://wiki.leagueoflegends.com/en-us/Attack_speed), [wiki.leagueoflegends.com/Yasuo](https://wiki.leagueoflegends.com/en-us/Yasuo)

### When is damage calculated: launch or impact?

**What League does:** I was not able to confirm this cleanly from the sources fetched this session. This matters because it determines whether stat changes between launch and impact (e.g., the attacker gaining or losing AD/crit chance mid-flight from some interceding effect, or the target gaining/losing armor) affect the damage of an attack already in flight.
**Confidence:** UNCERTAIN — the wiki is not specific about this in the pages retrieved. This is exactly the kind of precise timing question the prompt asked me to flag rather than guess at, and it is a good candidate for direct empirical testing against server behavior (fire an attack, change a relevant stat before impact, see what damage lands) rather than assuming an answer either way.

---

## 2. Minion waves and the lane equilibrium

**What League does — the priority/retargeting rules underlying wave behavior:** (see Section 3 for the full detail) minions auto-target the nearest valid enemy in their priority class and trade with whatever they run into. Because both lanes' waves spawn simultaneously from their respective bases and walk toward each other at the same speed with (by default) equal composition, they tend to meet and begin trading at a roughly consistent point in the lane — and because equal compositions trade roughly evenly, that meeting point does not run away toward one tower on its own. The wiki states this plainly: **"Without player interference, a lane's wave is neither advancing or retreating in any significant amount."**

**Why it exists:** This baseline equilibrium is what makes wave manipulation a skill rather than a coin flip — because the *natural* state is a stalemate, any push or freeze you create is attributable to a deliberate action (killing minions faster/slower than "natural," or poking the enemy wave), which is what makes laning decisions legible and skill-expressive rather than random.

**The three named techniques the wiki documents**, each of which perturbs the natural equilibrium in a specific direction:
- **Freezing**: preventing your own wave from advancing — practically, only last-hitting the minimum needed to keep the wave from crashing into your own turret, so the wave stalls at a point deep on your side of the lane. This denies the opponent safe farm (they must walk into your side to get CS) while keeping you safe (you're near your own tower).
- **Pushing**: accelerating your wave's advance by killing enemy minions faster than they'd naturally die (often with extra hits beyond just the last-hit, or AoE), which unbalances the equilibrium in your favor and sends your wave toward the enemy turret.
- **Slow push**: a gentler version of pushing — letting your wave's minions kill the enemy wave somewhat faster than 1:1 over several cycles (by not diverting your own minions' damage, or by adding a little extra damage) so that your side accumulates a numbers advantage across multiple spawns, building toward a bigger wave that eventually crashes hard into the enemy tower.

**"Wave crash"** (not a separate wiki-cited term, but the natural consequence of the above): the moment a pushed or slow-pushed wave reaches the enemy turret, the turret itself begins killing the attacking wave (see Section 4's priority order — turrets prioritize minions well above champions by default), which resets the balance for the next cycle's meeting point.

**What to watch for:** A correct simulation of "lane equilibrium" is really a downstream consequence of getting minion aggro/targeting (Section 3) and relative minion damage/HP right — there isn't a separate "equilibrium" system to implement; it should just emerge. If the JAX/server minion combat doesn't reproduce a stable meeting point under no player interference, that's a sign something in minion damage, spawn timing, or targeting is off, not that "equilibrium" itself needs special-casing.

**Modern-vs-4.20 caveat:** The specific spawn cadence documented today (every 30s from 0:30, accelerating to every 25s from 14:00, then every 20s from 30:00) and other modern details like more frequent cannon/siege minions are near-certainly **not** what patch 4.20 had — wave timing and cannon-minion frequency have been revisited multiple times since 2014, including changes aimed at improving bot-lane pacing. I don't have a reliable, sourced version of the exact 4.20 spawn cadence, so treat any specific numbers here as current-patch only, and the spawn-timing details as a concrete item for the modern-migration list, not something to match now.

**Confidence:** WELL-DOCUMENTED for the equilibrium concept and the freeze/push/slow-push vocabulary and mechanism. UNCERTAIN for exact 4.20-era spawn timers/cannon-minion frequency. [wiki.leagueoflegends.com/Minion](https://wiki.leagueoflegends.com/en-us/Minion)

---

## 3. Minion aggro and target selection

**What League does — target priority:** Per the wiki, minions pick targets using a strict priority order, and once a target is chosen, a minion **only switches to a new target if the new candidate is strictly higher priority** than the current one (i.e., there's hysteresis — minions don't ping-pong between equal-priority targets):

1. Enemy champions currently attacking an allied champion
2. Enemy minions currently attacking an allied champion
3. Enemy minions currently attacking an allied minion
4. Enemy turrets currently attacking an allied minion
5. The closest enemy minion
6. The closest enemy champion

**"Call for Help":** separately from the standing priority list, minions respond to a "call for help" signal, which is raised when either (a) an enemy champion is standing in a minion's path with no other target currently in range, or (b) an enemy champion is dealing damage to something (via basic attacks or unit-targeted abilities) near the minion. When multiple call-for-help sources exist, minions prioritize the **closest champion**. Critically, **call-for-help signals are ignored while a minion is already attacking an enemy turret** — a minion committed to shooting a tower will not be pulled off that tower onto a nearby champion via this mechanism.

**Retargeting on lost sight:** minions re-evaluate between attack cycles, and if a minion loses sight of its current target, it either picks a new target or continues advancing if nothing is visible.

**Why it exists:** The priority order encodes "defend allies under attack first, then contest space, then farm" — it's what makes minions read as a coherent, if simple, local combat AI rather than random target selection, and it's what makes tactics like tanking minion aggro onto yourself (to protect a teammate, or to let a teammate freely trade) work as intended. The call-for-help-ignored-while-attacking-a-turret carve-out exists so that a losing turret dive doesn't get bailed out just because minions happen to be nearby shooting the tower — otherwise every tower dive would trivially pull the whole wave off the tower.

**What to watch for:** This is two separate systems (a static priority list re-evaluated on target loss, and a call-for-help override with its own trigger conditions and its own turret-attacking exception), and it's easy to accidentally collapse them into one, or to get the precedence between them wrong. In particular, whether a priority-1 event (an enemy champion directly attacking an allied *champion*, which is the top of the standing list) can pull a minion off a turret it's currently shooting, versus whether the "ignore CFH while attacking a turret" exception blocks *all* retargeting off a turret regardless of source, is genuinely ambiguous in the wording available — I did not find a source that disambiguates this specific interaction.

**Confidence:** WELL-DOCUMENTED for the priority list, the hysteresis rule, and the call-for-help triggers/turret exception as stated. UNCERTAIN for the precise interaction/precedence between the standing priority list and the call-for-help turret exception when both could apply simultaneously. [wiki.leagueoflegends.com/Minion](https://wiki.leagueoflegends.com/en-us/Minion)

---

## 4. Turret behavior

### Target priority

Per the wiki, turrets use their own strict priority order, and by default **champions are the lowest priority** — a turret will not voluntarily choose to shoot a champion over any minion/pet class unless something else forces the switch (see below):

1. Most pets (e.g., Tibbers, Voidlings)
2. Siege/super minions (and similar high-value minion-class units, e.g. Yorick's Dark Procession)
3. "Mist walkers" (similar special-unit class)
4. Melee minions
5. Caster minions (and similar, e.g. Wukong's clone)
6. Other special pets (e.g. Yorick's Maiden, Elise's spiderlings, Naafiri's packmates)
7. Champions

A turret holds its current target "until it dies, leaves the attack range of the turret, or stops being targetable."

### Aggro switching (champion-attacks-champion under tower)

**What League does:** If an enemy champion deals damage to an allied champion who is within 1400 range of a turret, the turret switches its target to that enemy champion — overriding the default minions-first priority. This is the mechanical basis of the standard laning rule "don't hit someone under their own tower unless you're prepared to eat a tower shot."
**Why it exists:** This is turret aggro as a deterrent against contesting lane dominance immediately under an enemy's own tower — it's what makes towers actually protective of the champion standing under them, not just protective against minion waves.
**Confidence:** WELL-DOCUMENTED. [wiki.leagueoflegends.com/Turret](https://wiki.leagueoflegends.com/en-us/Turret)

### Escalating damage — "Warming Up"

**What League does:** the wiki's own name for this mechanic is **"Warming Up"**: turret damage against a champion target increases by 50% per hit, stacking up to a cap of +150% bonus damage (i.e., three stacks). This resets 5 seconds after the turret's last hit on a champion, but **persists across a target switch** between champions — so diving with a second champion after the turret has already been "warmed up" on the first does not reset the ramp.
**Why it exists:** This punishes extended champion aggression under a tower (a 1-for-1 short trade is survivable; standing and fighting under a turret for several seconds becomes rapidly lethal), which is core to why turrets meaningfully deter dives and extended skirmishes rather than just chip damage.
**What to watch for:** This is **exactly the mechanic the vendored server's `S5Test_TowerWrath` buff is named for and does not implement** — the buff script is a confirmed no-op in the current server. Any future migration toward real turret behavior needs this mechanic added from scratch; it is not something to "fix" in the parity sense (the server's current no-op behavior *is* the 4.20-parity target for now), but it is the single clearest, most concrete item on the eventual modern-migration list.
**Confidence:** WELL-DOCUMENTED for the current mechanic's shape and name. UNCERTAIN whether "Warming Up" existed in this exact form (50%-per-hit, 150% cap, 5s reset, persists across target switch) at patch 4.20 — turret damage ramping has existed conceptually for a long time in League, but I could not confirm the specific numbers were the same in 2014. [wiki.leagueoflegends.com/Turret](https://wiki.leagueoflegends.com/en-us/Turret)

### Damage modifiers

**What League does (confirmed):** Turrets take **20% increased damage from melee champions** — this is a defensive modifier (how much damage the turret *takes*), intended to make melee-heavy dives more viable against an otherwise very tanky structure, balancing the fact that melee champions have to walk further into turret range to fight at all.

**Damage the turret deals to minions:** I could not find an explicit stated damage modifier specific to minions (as opposed to champions) in the sources fetched. What the wiki does document is the *practical survival pattern* this produces: **"Melee minions may sustain two turret attacks, and then one champion attack to last hit. Ranged minions may sustain one champion attack, then one turret attack, then another champion attack to last hit."** This describes an emergent outcome of turret AD vs. minion HP values, not a stated multiplicative modifier — so whether the server (or real League) applies an explicit "vs. minion" damage modifier, versus this pattern simply falling out of raw stat values, is UNCERTAIN.
**Confidence:** WELL-DOCUMENTED for the +20% melee-champion damage-taken modifier and for the melee/ranged minion survival pattern as stated. UNCERTAIN whether there is a separate explicit turret-damage-vs-minions modifier beyond raw AD numbers. [wiki.leagueoflegends.com/Turret](https://wiki.leagueoflegends.com/en-us/Turret)

### Turret stat growth over the game, and turret plating

**What League does (current):** Turret attack damage scales up over the course of the game (outer turrets roughly 182–350 AD, inner turrets roughly 187–427 AD, both scaling with game time), and outer turrets additionally *lose* armor and magic resistance during a mid-game window (roughly minutes 11–15, losing 10 armor and 15 MR per minute in that window) — turrets get squishier as the game goes on, which is a deliberate anti-stalling design (a turret that's just as tanky at minute 30 as minute 10 would make sieging impossible and games would never end).

Additionally, current League turrets have a **plating system**: outer/lane turrets have 5 destructible "plates," each worth local gold when destroyed, and turrets gain a stacking "Bulwark" bonus armor/MR effect as plates are lost.

**Modern-vs-4.20 caveat — this is important:** Turret plating is a **Season 9 (2019, roughly patch 9.13-era) addition**. Patch 4.20 (2014) predates it by about five years, so the vendored server almost certainly has (and should have, for 4.20 parity) **no turret plating system at all**. This is one of the clearest, highest-confidence items for "real League does this, but not at patch 4.20, and the server should not have it either." If the server *does* have anything plating-shaped, that would itself be a notable anachronism worth flagging separately, though I have no indication it does.
**Confidence:** WELL-DOCUMENTED for current turret stat scaling and the plating system's existence and current mechanics. WELL-DOCUMENTED (based on general, widely-known game history rather than a citation pulled this session) that plating postdates patch 4.20 and did not exist then. [wiki.leagueoflegends.com/Turret](https://wiki.leagueoflegends.com/en-us/Turret)

---

## 5. Last-hitting and CS: why gold is last-hit-only and XP is shared

**What League does:**
- **Gold** from killing a minion is awarded only to whichever unit lands the killing blow (last-hit) — no source retrieved this session spelled this out in so many words, but it is implicit throughout the wiki's gold documentation (which discusses minion gold purely in terms of "killing" minions, with no sharing/proximity language at all, in contrast to the XP page's explicit sharing language below) and is extremely well-established, uncontested general knowledge of the game.
- **Experience (XP)** from a minion's death, by contrast, is granted to **all champions within 1500 units of the minion when it dies, regardless of who dealt the killing blow.** When multiple champions are in range, the XP pool is split, but the total pool is inflated first (the wiki states shared minion XP generates "approximately 30% more" total XP before division) so that splitting it doesn't proportionally starve everyone the way an even split of a fixed pool would.
- Passive gold income (a small trickle every 10 seconds, independent of any minion activity) exists on top of this and is not last-hit-dependent at all.

**Why it exists (design-intent synthesis — not a directly sourced Riot statement, flagged accordingly):** last-hit-only gold makes gold the sharply individually-skill-differentiated resource — how well you last-hit, and the trade-offs you accept to get a last hit (walking into poke range, giving up a favorable trade to farm, etc.) is a continuously-repeated mechanical and decision-making skill test, and it's what lets two players in the same lane end up with very different item power despite similar XP. Proximity-shared XP, meanwhile, keeps teammates roughly level-synced even when only one of them can realistically secure kills — most obviously support champions in bot lane, who very often cannot or should not be the one landing the last hit, but also junglers assisting a lane gank, or a losing laner who's being zoned off CS but is still standing nearby. If XP required last-hits the way gold does, being zoned off farm would also mean falling behind in level (and therefore raw stats and ability points) on top of falling behind in items — a much harsher, more snowball-prone penalty. Splitting gold and XP into "last-hit-earned" and "proximity-shared" lets the game punish being zoned off CS primarily through the economy, while keeping level (and thus base survivability/combat viability) somewhat more forgiving.
**What to watch for:** Because gold requires the literal killing blow (a specific unit's attack/spell being the one that reduces the minion's HP below 0) while XP only requires being alive and in range at the moment of death, these two systems key off different data at different times — a bug that ties XP eligibility to "who last-hit" (rather than "who was in range"), or that ties gold to "who dealt damage to the minion recently" (rather than strictly "who got the killing blow"), would both be plausible, easy-to-make errors that look reasonable but are wrong.
**Confidence:** WELL-DOCUMENTED for the mechanical facts (last-hit gold; proximity-shared, pool-inflated XP; the 1500-unit range). UNCERTAIN / synthesized for the "why" — presented as commonly-understood design rationale rather than a directly sourced Riot design statement, since I could not retrieve a primary design-rationale source this session. [wiki.leagueoflegends.com/Gold_(League_of_Legends)](https://wiki.leagueoflegends.com/en-us/Gold_(League_of_Legends)), [wiki.leagueoflegends.com/Experience_(champion)](https://wiki.leagueoflegends.com/en-us/Experience_(champion))

---

## 6. Garen

General note: Garen's five-part kit shape (a passive sustain mechanic, a silence/gap-close/empowered-attack Q, a defensive W, an AoE spin E, and a true-damage execute R) has been stable since his original release and has never received a full ability-kit rework (no "VGU") — what's changed over the years has mostly been individual numbers and a small number of added sub-mechanics layered onto the existing shape, rather than the shape itself changing. This overall stability is WELL-DOCUMENTED from the pattern of the patch history available (which shows numeric tuning and targeted additions, not kit replacements).

### Passive — Perseverance

**What League does (current):** Garen regenerates a percentage of his maximum health periodically (every 0.5 seconds) whenever he hasn't recently taken damage. Taking damage from champions, epic monsters, turrets, or enemy abilities disables the regen for a fixed window (currently 8 seconds) before it can resume.
**Why it exists:** This is a sustain mechanic that specifically rewards *disengaging* and *winning trades cleanly* rather than continuous poke-trading — it makes Garen strong in patient, cooldown-driven skirmishes (bait a spell, punish, then reset) and comparatively weak against sustained poke that keeps re-triggering the shutoff window.
**What to watch for:** The shutoff is keyed on damage *source type* (champions/epic monsters/turrets/enemy abilities) — notably this reads as excluding self-inflicted damage and possibly excluding some non-champion, non-epic monster sources; getting the exact trigger set right matters for whether the passive behaves as "combat regen that shuts off" vs. "regen that's always on."
**Confidence:** WELL-DOCUMENTED for the current mechanic's shape. UNCERTAIN for the exact 4.20-era numbers (regen rate, shutoff duration) — not independently confirmed this session, and this general shape (sustain that disables on taking damage) is old enough it's plausible it existed in similar form in 2014, but I don't have a citation for the specific parameters at that patch. [wiki.leagueoflegends.com/Garen](https://wiki.leagueoflegends.com/en-us/Garen)

### Q — Decisive Strike

**What League does (current):** An active that cleanses existing slows and grants Garen bonus movement speed, and empowers his next basic attack to lunge at the target, deal bonus physical damage, and silence them briefly.
**Why it exists:** It's simultaneously an engage tool (slow-cleanse + speed to close a gap), a trading tool (bonus damage, and importantly the silence denies the opponent's response — especially valuable against champions who need to cast something to punish or peel), and (per general knowledge of the ability) a way to reliably start or finish a fight on Garen's terms given he otherwise has no hard CC.
**What to watch for:** Whether/how this interacts with the basic-attack system in Section 1 (it's described as empowering the *next basic attack*, i.e. it rides on top of the normal attack windup/projectile pipeline rather than being an independent instant-damage spell) matters for parity — if it's implemented as a separate instant-damage effect rather than routing through the same attack resolution as a normal auto, that's a plausible point of divergence.
**Confidence:** WELL-DOCUMENTED for the current mechanic. UNCERTAIN for exact 4.20-era numbers/behavior — not independently confirmed this session. [wiki.leagueoflegends.com/Garen](https://wiki.leagueoflegends.com/en-us/Garen)

### W — Courage

**What League does (current):** Has both a passive and an active component.
- **Passive:** Garen gains a stack of "Courage" for every kill (champions, monsters, or minions), up to 150 stacks, each stack giving a small amount of bonus armor and magic resist, capping at 30 bonus armor and 30 bonus MR total.
- **Active:** Reduces incoming damage for 4 seconds; for the first 0.75 seconds of that window, Garen additionally gets a shield and a large (60%) tenacity boost.

**Notable, sourced historical detail:** the patch history retrieved shows that the **shield + tenacity component of the active was added or redesigned in patch V10.4 (2020)** — meaning the "just damage reduction, no shield, no tenacity burst" version is very likely what existed at patch 4.20 in 2014, over five years earlier. This is a concrete, reasonably confident divergence point: **at 4.20, Garen's W active was almost certainly a flat percentage damage-reduction effect only.**

I was not able to confirm when the passive stacking-resistance mechanic was added; based on general knowledge (not a source confirmed this session) I believe it was added during a mid-2016 systemic itemization/tankiness pass affecting several bruiser/"juggernaut" champions, which would also postdate patch 4.20 — but I want to be explicit that **this specific claim is UNCERTAIN and unverified this session**, unlike the V10.4 shield/tenacity finding above, which is directly sourced from retrieved patch-history text.
**Why it exists:** The active is Garen's core damage-mitigation cooldown for surviving burst; the passive rewards him for successfully closing out fights (getting kills) with permanent-for-the-fight tankiness, reinforcing his identity as a snowbally duelist who gets harder to kill the more he's already winning.
**What to watch for:** If the server implements the modern shield+tenacity active, or the stacking passive, in a way that's period-accurate to patch 4.20's Courage, that would be a genuine anachronism worth flagging — this is exactly the kind of thing to check directly rather than assume either way.
**Confidence:** WELL-DOCUMENTED for the current mechanic and for the V10.4 shield/tenacity addition (directly sourced from patch history). UNCERTAIN for when the passive was added and for the pre-V10.4 (and specifically 4.20-era) shape of the active beyond "very likely no shield/tenacity burst." [wiki.leagueoflegends.com/Garen](https://wiki.leagueoflegends.com/en-us/Garen)

### E — Judgment

**What League does (current):** Garen spins rapidly for 3 seconds (7 spins at base, +1 additional spin per 25% bonus attack speed), unable to issue basic attacks during the spin but gaining ghosting (unstoppable-to-slows movement), dealing periodic physical damage to all nearby enemies per spin. The verbatim current tooltip and its notes make **no explicit mention of minions, monsters, or turrets** — it is written as a generic "nearby enemies" AoE effect with no stated exception for any unit class. It is explicitly documented elsewhere on the page as **not applying on-hit effects** (it's ability damage, not a basic attack, despite scaling with AD).

**Historical detail (sourced):** the ability had a **+150% bonus damage against non-epic monsters** modifier that was added in patch **V10.4 (2020)** and later **removed in V10.4→V25.11** — i.e. this jungle-clear-oriented monster bonus is entirely a modern-era addition that did not exist at patch 4.20, and has since been removed again as of the current patch cited in the history. This bonus was specifically about *monsters* (jungle camps), not minions.
**Why it exists:** E is Garen's primary AoE damage tool and the core of both his waveclear and his teamfight/skirmish damage; the "no on-hit" restriction keeps it from double-dipping with on-hit-triggered rune/item effects the way a normal attack-speed-scaling auto-attack build would, keeping its power budget centered on the ability's own numbers rather than compounding with on-hit itemization.
**What to watch for:** Because the tooltip makes no distinction for minions or turrets, the expectation (subject to the uncertainty below) is that Judgment should hit minions and turrets within its radius for the same per-tick value as anything else, with no special reduction — I found no source establishing a minion- or turret-specific damage modifier for this ability. If the server has one, it doesn't appear to be something real League's own tooltip documents, and is worth checking directly rather than assuming it's either a bug or an intentional (if undocumented) modifier.
**Confidence:** WELL-DOCUMENTED for the current mechanic, the no-on-hit rule, and the V10.4-added/since-removed monster bonus (all directly sourced from retrieved tooltip/patch-history text). UNCERTAIN whether any minion- or turret-specific damage modifier exists or ever existed for this ability — no source found either confirming or denying this, so "the wiki is not specific about this" applies directly here. [wiki.leagueoflegends.com/Garen](https://wiki.leagueoflegends.com/en-us/Garen)

### R — Demacian Justice

**What League does:** Deals **true damage** to the target, scaled off the target's *missing* health (i.e., it's a percentage-missing-health execute, not flat or max-health-scaled damage), on top of a flat base amount. The retrieved patch history (covering versions from V9.23 through the current V26.14) shows this ability described as true damage in **every single entry available**, with no patch showing a conversion to or from a different damage type — only numeric tuning (the most recent being a nerf to its base/scaling numbers in V26.14).
**Why it exists:** True damage means the ability's power is not diluted by the target building armor — as an execute-style finisher, this ensures Demacian Justice reliably closes out kills regardless of the target's defensive itemization, which is central to Garen's identity as a champion whose ultimate is a hard, resistance-ignoring answer to a low-health target trying to itemize or flee their way out of a kill.
**What to watch for:** True damage means resistances (armor/MR) should have **zero** effect on this ability's damage — a parity bug here would most plausibly show up as the damage being (incorrectly) reduced by the target's armor or magic resist.
**Confidence:** WELL-DOCUMENTED that R is true damage, based on consistent patch-history text covering 2019–2026 (V9.23 onward) with no contrary entries found. UNCERTAIN whether this held all the way back to the ability's original 2010 design and specifically at patch 4.20 (2014) — I found no direct source confirming the damage type that far back, though the complete absence of any "converted to true damage" patch note across the entire retrieved history is suggestive (a damage-type change would ordinarily be called out explicitly in patch notes as a significant rework) that it was true damage then too. Treat "always true damage, including at 4.20" as high-confidence-but-not-fully-verified rather than certain. [wiki.leagueoflegends.com/Garen](https://wiki.leagueoflegends.com/en-us/Garen)

---

## Sources consulted

- [League of Legends Wiki — Basic attack](https://wiki.leagueoflegends.com/en-us/Basic_attack)
- [League of Legends Wiki — Attack speed](https://wiki.leagueoflegends.com/en-us/Attack_speed)
- [League of Legends Wiki — Minion](https://wiki.leagueoflegends.com/en-us/Minion)
- [League of Legends Wiki — Turret](https://wiki.leagueoflegends.com/en-us/Turret)
- [League of Legends Wiki — Garen](https://wiki.leagueoflegends.com/en-us/Garen)
- [League of Legends Wiki — Experience (champion)](https://wiki.leagueoflegends.com/en-us/Experience_(champion))
- [League of Legends Wiki — Gold (League of Legends)](https://wiki.leagueoflegends.com/en-us/Gold_(League_of_Legends))
- [League of Legends Wiki — Yasuo](https://wiki.leagueoflegends.com/en-us/Yasuo) (Wind Wall, used as indirect evidence about projectile classification)

Notes on research limitations for whoever revisits this: `leagueoflegends.fandom.com` (the older community wiki, which sometimes retains more historical/legacy detail than the current official wiki) returned HTTP 402 to every fetch attempt this session and could not be consulted; `web.archive.org` was unreachable from this environment; and general web search was unavailable for the second half of this research pass (session search budget was exhausted), which is why several historical-patch questions (pre-2019 Garen numbers, exact 4.20 spawn timers, the exact patch of the attack-speed-cap change) are marked UNCERTAIN rather than answered — they would be worth another pass with those sources available rather than assumed.
