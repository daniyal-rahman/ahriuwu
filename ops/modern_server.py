"""TOOL: materialize the modern champion overlay in a NEW isolated server tree.

Never edits the vendor checkout. The source tree carries all existing lane
patches; modern engine replacements are exact-match guarded. See patches README.
"""
from pathlib import Path
import argparse
import json
import shutil
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lanerl_jax.data.modern import champion, stat, cooldowns
ROOT=Path(__file__).resolve().parents[1]
VENDOR=Path('/srv/nfs/projects/lanerl-vendor/LoLServer')


def materialize(dest, refresh=False):
    dest=Path(dest)
    if dest.resolve()==VENDOR.resolve() or VENDOR.resolve() in dest.resolve().parents:
        raise SystemExit("Refusing to materialize inside the shared vendor tree")
    if dest.exists() and not (refresh and (dest/"modern-build.json").exists()): raise SystemExit(f'Refusing to overwrite existing server: {dest}')
    shutil.copytree(VENDOR,dest,dirs_exist_ok=refresh,ignore=shutil.ignore_patterns('.git','bin','obj','Archives','LeagueSandbox-Default'))
    shutil.copytree(VENDOR/'Content/LeagueSandbox-Default',dest/'Content/LeagueSandbox-Default',dirs_exist_ok=refresh)
    changes={}
    def edit(rel,old,new,count=None):
        p=dest/rel;s=p.read_text(encoding='utf-8-sig')
        if not s.count(old) or (count is not None and s.count(old)!=count): raise RuntimeError(f'Baseline drift: {rel}: {old[:80]}')
        changes.setdefault(rel,hashlib.sha256(p.read_bytes()).hexdigest());p.write_text(s.replace(old,new))
    ai='GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs'
    edit(ai,'public ICharScript CharScript { get; private set; }','public ICharScript CharScript { get; private set; }\n        public LeagueSandbox.GameServer.Lanerl.ModernChampion ModernChampion { get; set; }',1)
    edit(ai,'            ApiEventManager.OnHitUnit.Publish(this, damageData);','''            if (target is ObjAIBase defender && defender.ModernChampion?.Dodge(this) == true)
            { ModernChampion?.AttackDodged(); return; }
''',1)
    edit(ai,'            target.TakeDamage(damageData, IsNextAutoCrit);','            ModernChampion?.OnHit(damageData);\n            ApiEventManager.OnHitUnit.Publish(this, damageData);\n            target.TakeDamage(damageData, IsNextAutoCrit);',1)
    edit(ai,'            return (!IsDead\n                && MovementParameters != null)','            return !IsDead && ((MovementParameters != null)',1)
    edit(ai,'|| Status.HasFlag(StatusFlags.Suppressed)));','|| Status.HasFlag(StatusFlags.Suppressed))));',1)
    unit='GameServerLib/GameObjects/AttackableUnits/AttackableUnit.cs'
    edit(unit,'            ApiEventManager.OnPreTakeDamage.Publish(damageData.Target, damageData);','''            ApiEventManager.OnPreTakeDamage.Publish(damageData.Target, damageData);
            (this as ObjAIBase)?.ModernChampion?.Incoming(damageData);
            postMitigationDamage = damageData.PostMitigationDamage;''',1)
    edit(unit,'                IsDead = true;\n                _death = new DeathData','                IsDead = true;\n                (attacker as ObjAIBase)?.ModernChampion?.KilledUnit();\n                _death = new DeathData',1)
    spell='GameServerLib/GameObjects/Spell/Spell.cs'
    edit(spell,'                ApiEventManager.OnLaunchAttack.Publish(CastInfo.Owner, this);','                CastInfo.Owner.ModernChampion?.AttackStarted();\n                ApiEventManager.OnLaunchAttack.Publish(CastInfo.Owner, this);',1)
    # Costs are dynamic: a Counter Strike recast costs zero. Do not mutate the
    # shared SpellData object (two Jaxes can have different E phases).
    edit(spell,'SpellData.ManaCost[CastInfo.SpellLevel]','((CastInfo.SpellSlot < 4 && CastInfo.Owner.ModernChampion != null) ? CastInfo.Owner.ModernChampion.ManaCost(CastInfo.SpellSlot) : SpellData.ManaCost[CastInfo.SpellLevel])')
    edit(spell,'            _attackType = AttackType.ATTACK_TYPE_RADIAL;\n            var stats = CastInfo.Owner.Stats;','''            if (CastInfo.SpellSlot < 4 && CastInfo.Owner.ModernChampion != null && !CastInfo.Owner.ModernChampion.CanCast(CastInfo.SpellSlot, unit)) return false;
            _attackType = AttackType.ATTACK_TYPE_RADIAL;
            var stats = CastInfo.Owner.Stats;''',1)
    edit(spell,'if (SpellData.CantCancelWhileWindingUp)','if (SpellData.CantCancelWhileWindingUp || (CastInfo.IsAutoAttack && CastInfo.Owner.ModernChampion?.Empowered == true && !CastInfo.Targets[0].Unit.IsDead))',1)
    edit(ai,'&& AutoAttackSpell.State == SpellState.STATE_CASTING && !AutoAttackSpell.SpellData.CantCancelWhileWindingUp)','&& AutoAttackSpell.State == SpellState.STATE_CASTING && !AutoAttackSpell.SpellData.CantCancelWhileWindingUp && ModernChampion?.Empowered != true)',1)
    edit(spell,'var castRange = GetCurrentCastRange();','var castRange = GetCurrentCastRange();\n                if (targetingType == TargetingType.Target && CastInfo.Owner.ModernChampion != null && unit != null) castRange += unit.CollisionRadius;',1)
    episode='GameServerLib/Lanerl/LanerlEpisode.cs'
    edit(episode,'            // buffs first: removing them runs their OnDeactivate, which is what','            champ.ModernChampion?.Detach();\n            // buffs first: removing them runs their OnDeactivate, which is what',1)
    edit(episode,'            champ.Stats.CurrentMana = champ.Stats.ManaPoints.Total;','            champ.Stats.CurrentMana = champ.Stats.ManaPoints.Total;\n            if (champ.ModernChampion != null) champ.CharScript.OnActivate(champ);',1)
    hooks='GameServerLib/Lanerl/LanerlHooks.cs'
    edit(hooks,'                if (spent >= _cfg.SkillOrder.Length) continue;','                var skillOrder = ch.Model == "Jax" ? ModernChampion.JaxSkillOrder : _cfg.SkillOrder;\n                if (spent >= skillOrder.Length) continue;',1)
    edit(hooks,'for (int i = spent; i < _cfg.SkillOrder.Length; i++)','for (int i = spent; i < skillOrder.Length; i++)',1)
    edit(hooks,'byte slot = (byte)_cfg.SkillOrder[i];','byte slot = (byte)skillOrder[i];',1)
    edit(hooks,'            if (_selfTest) { RunSelfTest(game); return; }','            if (Environment.GetEnvironmentVariable("LANERL_MODERN_SELFTEST") == "1") { ModernChampionSelfTest.Run(game); return; }\n            if (_selfTest) { RunSelfTest(game); return; }',1)
    control='GameServerLib/Lanerl/LanerlControl.cs'
    edit(control,'TargetId = hostile?.NetId ?? 0, HasTarget = hostile != null });','TargetId = (slot == 0 && champ.Model == "Jax" ? hit : hostile)?.NetId ?? 0, HasTarget = (slot == 0 && champ.Model == "Jax" ? hit : hostile) != null });',1)
    edit(control,'                    sb.Append(",\\\"dead\\\":")', '                    sb.Append(",\\\"modern\\\":").Append(Newtonsoft.Json.JsonConvert.SerializeObject(ch.ModernChampion?.Snapshot));\n                    sb.Append(",\\\"dead\\\":")',1)
    shutil.copy2(ROOT/'lanerl/patches/modern-champions/ModernChampion.cs',dest/'GameServerLib/Lanerl/ModernChampion.cs')
    shutil.copy2(ROOT/'lanerl/patches/modern-champions/ModernChampionSelfTest.cs',dest/'GameServerLib/Lanerl/ModernChampionSelfTest.cs')
    scripts=dest/'Content/LeagueSandbox-Scripts'
    # Replaced champion buffs must not keep registering historical passives.
    for name in ('Garen','Jax'):
        for old_buff in (scripts/'Buffs'/name).glob('*.cs'):
            old_buff.unlink()  # only inside this tool's owned isolated copy
    imports='''using System.Numerics;
using GameServerCore.Scripting.CSharp;
using LeagueSandbox.GameServer.Scripting.CSharp;
using LeagueSandbox.GameServer.Lanerl;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits.AI;
using LeagueSandbox.GameServer.GameObjects.SpellNS;
'''
    spellnames={'Garen':['GarenQ','GarenW','GarenE','GarenR'],'Jax':['JaxLeapStrike','JaxEmpowerTwo','JaxCounterStrike','JaxRelentlessAssault']}
    for name in spellnames:
        char=imports+f'''namespace CharScripts {{ public class CharScript{name} : ICharScript {{
ObjAIBase owner;
public void OnActivate(ObjAIBase unit, Spell spell=null) {{owner=unit;owner.ModernChampion=new ModernChampion(owner);}}
public void OnUpdate(float diff) {{owner.ModernChampion.Update(diff);}}
}} }}
'''
        (scripts/f'Characters/{name}/CharScript{name}.cs').write_text(char)
        for i,(slot,spellname) in enumerate(zip('QWER',spellnames[name])):
            fallback='public static Spell RestoreBasicAttack(ObjAIBase owner) => owner.GetNewAutoAttack();' if spellname=='GarenQ' else ''
            code=imports+f'''namespace Spells {{ public class {spellname} : ISpellScript {{
AttackableUnit target;
public SpellScriptMetadata ScriptMetadata {{get;}} = new SpellScriptMetadata {{ TriggersSpellCasts=true, CastTime={'.435f' if name=='Garen' and slot=='R' else '0f'} }};
public void OnSpellPreCast(ObjAIBase owner, Spell spell, AttackableUnit unit, Vector2 start,Vector2 end) {{target=unit;}}
public void OnSpellPostCast(Spell spell) {{spell.CastInfo.Owner.ModernChampion.Cast({i},target);}}
{fallback}
}} }}
'''
            (scripts/f'Characters/{name}/{slot}.cs').write_text(code)
            p=dest/f'Content/LeagueSandbox-Default/Spells/{spellname}/{spellname}.json';data=json.loads(p.read_text());d=data['Values']['SpellData']
            for k in range(7):
                d['Cooldown'+('' if k==0 else str(k))]=cooldowns(name,slot)[min(max(k-1,0),2 if slot=='R' else 4)]
                d['ManaCost'+('' if k==0 else str(k))]=0 if name=='Garen' else [50,30,40+10*k,100][i]
                if slot=='Q' and name=='Jax' or slot=='R' and name=='Garen': d['CastRange'+('' if k==0 else str(k))]=700 if name=='Jax' else 400
            d['OverrideCastTime']=.435 if name=='Garen' and slot=='R' else 0
            d['DelayCastOffsetPercent']=-1 if not (name=='Garen' and slot=='R') else d.get('DelayCastOffsetPercent',0)
            p.write_text(json.dumps(data,indent=2)+'\n')
        p=dest/f'Content/LeagueSandbox-Default/Stats/{name}/{name}.json';data=json.loads(p.read_text());d=data['Values']['Data']
        fields={'BaseHP':'baseHPModifiable','HPPerLevel':'hpPerLevelModifiable','BaseDamage':'baseDamageModifiable','DamagePerLevel':'damagePerLevelModifiable','Armor':'baseArmorModifiable','ArmorPerLevel':'armorPerLevelModifiable','SpellBlock':'baseMR','SpellBlockPerLevel':'mrPerLevel','MoveSpeed':'baseMoveSpeedModifiable','AttackRange':'attackRangeModifiable','AttackSpeedPerLevel':'attackSpeedPerLevelModifiable','BaseStaticHPRegen':'baseStaticHPRegenModifiable','HPRegenPerLevel':'hpRegenPerLevelModifiable'}
        for key,source in fields.items(): d[key]=stat(name,source)
        d.update(PathfindingCollisionRadius=35,GameplayCollisionRadius=65,AttackDelayOffsetPercent=.625/stat(name,'attackSpeedModifiable')-1)
        d['AttackDelayCastOffsetPercent']=champion(name)['character']['basicAttack']['mAttackDelayCastOffsetPercent']
        d.update(BaseMP=339 if name=='Jax' else 0,MPPerLevel=52 if name=='Jax' else 0,BaseStaticMPRegen=1.64 if name=='Jax' else 0,MPRegenPerLevel=.14 if name=='Jax' else 0)
        p.write_text(json.dumps(data,indent=2)+'\n')
    (scripts/'Characters/Jax/BasicAttack.cs').write_text(imports+'''namespace Spells {
public class JaxBasicAttack : ISpellScript {
public SpellScriptMetadata ScriptMetadata {get;} = new SpellScriptMetadata {TriggersSpellCasts=true,IsDamagingSpell=true};
}
public class JaxBasicAttack2 : JaxBasicAttack {}
public class JaxCritAttack : JaxBasicAttack {}
}
''')
    (scripts/'Buffs/Garen/ModernGarenShred.cs').write_text('''using GameServerCore.Enums;
using GameServerCore.Scripting.CSharp;
using LeagueSandbox.GameServer.Scripting.CSharp;
using LeagueSandbox.GameServer.GameObjects;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits;
using LeagueSandbox.GameServer.GameObjects.SpellNS;
using LeagueSandbox.GameServer.GameObjects.StatsNS;
namespace Buffs { public class ModernGarenShred : IBuffGameScript {
public BuffScriptMetaData BuffMetaData {get;set;} = new BuffScriptMetaData {BuffAddType=BuffAddType.REPLACE_EXISTING};
public StatsModifier StatsModifier {get;} = new StatsModifier();
public void OnActivate(AttackableUnit unit,Buff buff,Spell spell) {StatsModifier.Armor.PercentBonus=-.25f;unit.AddStatModifier(StatsModifier);}
} }
''')
    (dest/'modern-build.json').write_text(json.dumps({'patch':'26.19','source':str(VENDOR),'engine_baseline_sha256':changes},indent=2))
    return dest


def export_patch(dest, output):
    """Export only source/content changes, never build products or run settings."""
    import difflib
    dest=Path(dest)
    roots=['GameServerLib','Content/LeagueSandbox-Scripts/Characters/Garen',
           'Content/LeagueSandbox-Scripts/Characters/Jax','Content/LeagueSandbox-Scripts/Buffs/Garen',
           'Content/LeagueSandbox-Scripts/Buffs/Jax']
    paths=set()
    for root in roots:
        for tree in (VENDOR,dest):
            paths.update(str(p.relative_to(tree)) for p in (tree/root).rglob('*.cs') if not {'obj','bin'}.intersection(p.parts))
    for name,spells in {'Garen':['GarenQ','GarenW','GarenE','GarenR'],
                        'Jax':['JaxLeapStrike','JaxEmpowerTwo','JaxCounterStrike','JaxRelentlessAssault']}.items():
        paths.add(f'Content/LeagueSandbox-Default/Stats/{name}/{name}.json')
        paths.update(f'Content/LeagueSandbox-Default/Spells/{spell}/{spell}.json' for spell in spells)
    chunks=[]
    for rel in sorted(paths):
        old,new=VENDOR/rel,dest/rel
        a=old.read_bytes().decode().splitlines(keepends=True) if old.exists() else []
        b=new.read_bytes().decode().splitlines(keepends=True) if new.exists() else []
        if a!=b:
            for line in difflib.unified_diff(a,b,fromfile='a/'+rel if old.exists() else '/dev/null',tofile='b/'+rel if new.exists() else '/dev/null'):
                chunks.append(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n')
    Path(output).write_text(''.join(chunks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');p.add_argument('--refresh',action='store_true');a=p.parse_args();print(materialize(a.destination,a.refresh))
