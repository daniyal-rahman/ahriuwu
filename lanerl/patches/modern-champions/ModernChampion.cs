// Patch 26.19 lane combat. Data and exclusions: lanerl_jax/data/modern_26_19.
using System;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using GameServerCore.Enums;
using GameServerLib.GameObjects.AttackableUnits;
using LeagueSandbox.GameServer.API;
using LeagueSandbox.GameServer.GameObjects;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits.AI;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits.Buildings;
using LeagueSandbox.GameServer.GameObjects.SpellNS;
using LeagueSandbox.GameServer.GameObjects.StatsNS;
using static LeagueSandbox.GameServer.API.ApiFunctionManager;

namespace LeagueSandbox.GameServer.Lanerl
{
    public sealed class ModernChampion
    {
        readonly ObjAIBase owner;
        readonly Random rng = new Random(2619);
        public static readonly int[] JaxSkillOrder = {2,0,1,1,1,3,1,2,1,2,3,2,2,0,0,3,0,0};
        public object Snapshot => new {patch="26.19", id=garen?86:24, mana=owner.Stats.CurrentMana, maxMana=owner.Stats.ManaPoints.Total, q,haste,w,shield,shieldTime,e,eElapsed,spinTicks,spinCount,dodges,passiveStacks,passiveTime,r,rArmor,rHits,jumpTime,kills};
        public void Detach() {owner.RemoveStatModifier(modifier);}
        public float Tenacity => garen && shieldTime>0?.6f:0;
        readonly bool garen;
        readonly StatsModifier modifier = new StatsModifier();
        readonly Dictionary<AttackableUnit, int> spinHits = new Dictionary<AttackableUnit, int>();
        float q, haste, w, shield, shieldTime, e, eElapsed, ePower;
        float passiveTime, passiveStacks, passiveFalloff, r, rArmor, rHitTime;
        int spinTicks, spinCount, dodges, rHits, kills;
        AttackableUnit jumpTarget;
        float jumpTime, rPending;
        public static float GarenRegenPercent(int lv) => 1.5f+.2f*Math.Min(lv-1,5)+.8f*Math.Clamp(lv-6,0,7)+.4f*Math.Max(lv-13,0);
        public bool Empowered => garen ? q>0 : w>0;
        bool dead;
        public ModernChampion(ObjAIBase unit)
        {
            owner = unit; garen = owner.Model == "Garen";
            owner.AddStatModifier(modifier);
        }
        int Rank(int slot) => owner.Spells[(short)slot].CastInfo.SpellLevel;
        float BonusAD => owner.Stats.AttackDamage.Total-owner.Stats.AttackDamage.BaseValue;
        float AP => owner.Stats.AbilityPower.Total;
        float CD(int slot)
        {
            float[][] g = {new[]{8f,8,8,8,8},new[]{22f,19.5f,17,14.5f,12},new[]{9f,8.25f,7.5f,6.75f,6},new[]{120f,100,80}};
            float[][] j = {new[]{8f,7.5f,7,6.5f,6},new[]{7f,6,5,4,3},new[]{17f,15,13,11,9},new[]{110f,100,90}};
            return (garen?g:j)[slot][Math.Max(0,Rank(slot)-1)] * (1+owner.Stats.CooldownReduction.Total);
        }
        public float ManaCost(int slot) => garen || (slot==2 && e>0) ? 0 : new[]{50f,30,40+10*Rank(2),100}[slot];
        public bool CanCast(int slot, AttackableUnit target)
        {
            if(Rank(slot)<=0 || owner.IsDead || rPending>0 || (slot==0 && jumpTime>0)) return false;
            if(slot==2 && e>0) return eElapsed>=1;
            if(!garen && slot==1 && w>0) return false;
            if(!garen && slot==0) return target!=null && target!=owner && !target.IsDead && !(target is ObjBuilding) && !(target is BaseTurret) && Vector2.Distance(owner.Position,target.Position)<=700+target.CollisionRadius;
            if(garen && slot==3) return target is Champion && !target.IsDead && target.Team!=owner.Team && Vector2.Distance(owner.Position,target.Position)<=400+target.CollisionRadius;
            return true;
        }
        public void Cast(int slot, AttackableUnit target)
        {
            owner.Spells[(short)slot].SetCooldown(CD(slot),true);
            if(garen)
            {
                if(slot==0) { foreach(var buff in owner.GetBuffs().ToArray()) if(buff.BuffType==BuffType.SLOW)buff.DeactivateBuff(); q=4.5f; haste=1.4f+.55f*(Rank(0)-1); owner.CancelAutoAttack(true); }
                if(slot==1) { w=4; shieldTime=.75f; shield=45+20*Rank(1)+.18f*(owner.Stats.HealthPoints.Total-owner.Stats.HealthPoints.BaseValue); }
                if(slot==2) {
                    if(e>0) EndE();
                    else { e=3; eElapsed=0; spinCount=0; spinHits.Clear(); spinTicks=7+(int)Math.Floor(Math.Max(0,owner.Stats.AttackSpeedMultiplier.Total-1)/.25f); ePower=1+3*Rank(2)+(.37f+.03f*Rank(2))*owner.Stats.AttackDamage.Total; owner.Spells[2].SetCooldown(1,true); }
                }
                if(slot==3 && target!=null && !target.IsDead) Damage(target,50+75*Rank(3)+(.2f+.05f*Rank(3))*(target.Stats.HealthPoints.Total-target.Stats.CurrentHealth),DamageType.DAMAGE_TYPE_TRUE,false);
            }
            else
            {
                if(slot==0) { jumpTarget=target; jumpTime=Math.Max(.001f,Vector2.Distance(owner.Position,target.Position)/1400); owner.CancelAutoAttack(true); }
                if(slot==1) { w=10; owner.CancelAutoAttack(true); owner.Spells[1].SetCooldown(0,true); }
                if(slot==2) { if(e>0) EndE(); else {e=2;eElapsed=0;dodges=0;owner.Spells[2].SetCooldown(1,true);} }
                if(slot==3) rPending=.25f;
            }
            UpdateModifier();
        }
        List<AttackableUnit> Enemies(float radius) => GetUnitsInRange(owner.Position,radius,true).Where(t=>t.Team!=owner.Team && !t.IsDead && !(t is BaseTurret) && !(t is ObjBuilding)).ToList();
        void Damage(AttackableUnit target,float amount,DamageType type,bool aoe)
        {
            target.TakeDamage(owner,amount,type,aoe?DamageSource.DAMAGE_SOURCE_SPELLAOE:DamageSource.DAMAGE_SOURCE_SPELL,false);
        }
        public bool Dodge(AttackableUnit attacker)
        {
            if(garen || e<=0 || owner.IsDead || attacker is BaseTurret) return false;
            dodges++; return true;
        }
        public void AttackStarted()
        {
            if(garen) return;
            passiveStacks=Math.Min(8,passiveStacks+1);passiveFalloff=2.5f;UpdateModifier();
        }
        public void AttackDodged() { if(garen) {q=0;UpdateModifier();} }
        public void OnHit(DamageData data)
        {
            var target=data.Target;
            if(garen && q>0)
            {
                float amount=30*Rank(0)+1.5f*owner.Stats.AttackDamage.Total;
                data.Damage=amount;data.PostMitigationDamage=target.Stats.GetPostMitigationDamage(amount,DamageType.DAMAGE_TYPE_PHYSICAL,owner);
                q=0;AddBuff("Silence",1.5f*(1-((target as ObjAIBase)?.ModernChampion?.Tenacity ?? 0)),1,owner.Spells[0],target,owner,false);
            }
            if(!garen)
            {
                if(w>0) ConsumeW(target);
                if(Rank(3)>0) {
                    rHits++;rHitTime=2.5f;
                    if(rHits >= (r>0?2:3)) {rHits=0;Damage(target,(20+55*Rank(3)+.6f*AP)*(target is BaseTurret || target is ObjBuilding?.5f:1),DamageType.DAMAGE_TYPE_MAGICAL,false);}
                }
            }
            UpdateModifier();
        }
        void ConsumeW(AttackableUnit target)
        {
            if(w<=0)return;
            w=0;owner.Spells[1].SetCooldown(CD(1),true);
            Damage(target,(15+35*Rank(1)+.6f*AP)*(target is BaseTurret || target is ObjBuilding?.5f:1),DamageType.DAMAGE_TYPE_MAGICAL,false);
        }
        public void Incoming(DamageData data)
        {
            if(data.Attacker is Champion || data.Attacker is BaseTurret) passiveTime=0;
            if(data.DamageType!=DamageType.DAMAGE_TYPE_TRUE)
            {
                if(garen && w>0) data.PostMitigationDamage*=1-(.21f+.04f*Rank(1));
                if(!garen && e>0 && data.Attacker is Champion && data.DamageSource==DamageSource.DAMAGE_SOURCE_SPELLAOE) data.PostMitigationDamage*=.75f;
            }
            float absorbed=Math.Min(shield,data.PostMitigationDamage);shield-=absorbed;data.PostMitigationDamage-=absorbed;
        }
        public void KilledUnit() {if(garen) {kills=Math.Min(150,kills+1);UpdateModifier();}}
        void EndE()
        {
            e=0;owner.Spells[2].SetCooldown(CD(2),true);
            if(!garen && !owner.IsDead)
                foreach(var t in Enemies(375)) {
                    Damage(t,(10+30*Rank(2)+.7f*AP+.04f*t.Stats.HealthPoints.Total)*(1+.2f*Math.Min(5,dodges)),DamageType.DAMAGE_TYPE_MAGICAL,true);
                    AddBuff("Stun",1-((t as ObjAIBase)?.ModernChampion?.Tenacity ?? 0),1,owner.Spells[2],t,owner,false);
                }
            UpdateModifier();
        }
        void Spin()
        {
            var targets=Enemies(325); var nearest=targets.OrderBy(t=>Vector2.DistanceSquared(owner.Position,t.Position)).FirstOrDefault();
            float crit = rng.NextDouble()<owner.Stats.CriticalChance.Total ? 1+.3f*(owner.Stats.CriticalDamage.Total-1) : 1;
            foreach(var t in targets) {
                Damage(t,ePower*crit*(t==nearest?1.25f:1),DamageType.DAMAGE_TYPE_PHYSICAL,true);
                spinHits.TryGetValue(t,out int count);spinHits[t]=++count;
                if(t is Champion && (count==6 || count==7 || (count>7 && (count-7)%6==0))) AddBuff("ModernGarenShred",6,1,owner.Spells[2],t,owner,false);
            }
            spinCount++;
        }
        public void Update(float milliseconds)
        {
            float dt=milliseconds*.001f;
            if(owner.IsDead) {
                if(!dead) {if(e>0) EndE();if(!garen && w>0)owner.Spells[1].SetCooldown(CD(1),true);q=haste=w=shield=shieldTime=r=rPending=jumpTime=passiveStacks=rHits=0;UpdateModifier();}
                dead=true;return;
            }
            dead=false;passiveTime+=dt;
            if(rPending>0) {
                rPending=Math.Max(0,rPending-dt);
                if(rPending==0) {
                    var targets=Enemies(375); int count=targets.Count(t=>t is Champion);
                    foreach(var t in targets) Damage(t,25+75*Rank(3)+AP,DamageType.DAMAGE_TYPE_MAGICAL,true);
                    if(count>0) {r=8+dt;rArmor=30+15*Rank(3)+.4f*BonusAD+(count-1)*(15+5*Rank(3)+.1f*BonusAD);}
                }
            }
            if(garen && passiveTime>=8) {
                float pct=GarenRegenPercent(owner.Stats.Level)/500;
                owner.Stats.CurrentHealth=Math.Min(owner.Stats.HealthPoints.Total,owner.Stats.CurrentHealth+owner.Stats.HealthPoints.Total*pct*dt);
            }
            q=Math.Max(0,q-dt);haste=Math.Max(0,haste-dt);shieldTime=Math.Max(0,shieldTime-dt);if(shieldTime==0)shield=0;
            if(w>0) {w=Math.Max(0,w-dt);if(!garen && w==0)owner.Spells[1].SetCooldown(CD(1),true);}
            r=Math.Max(0,r-dt);rHitTime=Math.Max(0,rHitTime-dt);if(rHitTime==0)rHits=0;
            if(passiveStacks>0) {passiveFalloff-=dt;if(passiveFalloff<=0){passiveStacks--;passiveFalloff=.35f;}}
            if(e>0) {
                if(garen && spinCount<spinTicks && eElapsed>=3f*spinCount/spinTicks)Spin();
                eElapsed+=dt;e=Math.Max(0,e-dt);if(e==0)EndE();
            }
            if(jumpTime>0) {
                if(jumpTarget==null || jumpTarget.IsDead)jumpTime=0;
                else {owner.SetPosition(Vector2.Lerp(owner.Position,jumpTarget.Position,Math.Min(1,dt/jumpTime)));jumpTime=Math.Max(0,jumpTime-dt);
                    if(jumpTime==0 && jumpTarget.Team!=owner.Team) {Damage(jumpTarget,25+40*Rank(0)+BonusAD,DamageType.DAMAGE_TYPE_PHYSICAL,false);ConsumeW(jumpTarget);}}
            }
            UpdateModifier();
        }
        void UpdateModifier()
        {
            owner.RemoveStatModifier(modifier);
            modifier.Range.FlatBonus=(garen?q>0:w>0)?50:0;
            modifier.MoveSpeed.PercentBonus=garen && haste>0?.35f:0;
            modifier.Armor.FlatBonus=garen?kills*.2f:(r>0?rArmor:0);
            modifier.MagicResist.FlatBonus=garen?kills*.2f:(r>0?.6f*rArmor:0);
            modifier.AttackSpeed.PercentBaseBonus=garen?0:passiveStacks*(.05f+.015f*((owner.Stats.Level-1)/3));
            modifier.Tenacity.FlatBonus=garen && shieldTime>0?.6f:0;
            owner.AddStatModifier(modifier);
            owner.SetStatus(StatusFlags.Ghosted,(garen && e>0)||jumpTime>0);
            owner.SetStatus(StatusFlags.CanMove,jumpTime<=0);
            owner.SetStatus(StatusFlags.CanAttack,!(garen && e>0) && jumpTime<=0 && rPending<=0);
        }
    }
}
