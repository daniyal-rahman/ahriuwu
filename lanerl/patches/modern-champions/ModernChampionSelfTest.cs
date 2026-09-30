// PROBE entry point, invoked only by LANERL_MODERN_SELFTEST=1.
using System;
using System.Linq;
using System.Numerics;
using GameServerCore.Enums;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits.AI;
using Newtonsoft.Json.Linq;
namespace LeagueSandbox.GameServer.Lanerl
{
    public static class ModernChampionSelfTest
    {
        static void Equal(string name,float got,float want,float tolerance=.015f)
        {if(Math.Abs(got-want)>tolerance)throw new Exception($"{name}: {got} != {want}");Console.WriteLine($"MODERN_ASSERT {name} got={got:R} expected={want:R}");}
        static float Field(Champion c,string field) => JObject.FromObject(c.ModernChampion.Snapshot).Value<float>(field);
        public static void Run(Game game)
        {
            var g=game.ObjectManager.GetAllChampions().Single(c=>c.Model=="Garen");
            var j=game.ObjectManager.GetAllChampions().Single(c=>c.Model=="Jax");
            void Reset()
            {
                foreach(var c in new[]{g,j}) {
                    foreach(var b in c.GetBuffs().ToArray())b.DeactivateBuff();
                    c.ModernChampion.Detach();c.ModernChampion=new ModernChampion(c);
                    c.Stats.CurrentHealth=c.Stats.HealthPoints.Total;c.Stats.CurrentMana=c.Stats.ManaPoints.Total;
                    for(short slot=0;slot<4;slot++){c.Spells[slot].CastInfo.SpellLevel=1;c.Spells[slot].SetCooldown(0,true);}
                }
                g.SetPosition(new Vector2(5000,10000));j.SetPosition(new Vector2(5100,10000));
                // Refresh broadphase without a physics step: fixture positions
                // must remain exact for the independent damage assertions.
                game.Map.CollisionHandler.GetType().GetMethod("UpdateQuadTree",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                    .Invoke(game.Map.CollisionHandler,null);
            }
            foreach(var pair in new[]{(6,2.5f),(7,3.3f),(13,8.1f),(14,8.5f),(18,10.1f)}) Equal("garen_regen_level_"+pair.Item1,ModernChampion.GarenRegenPercent(pair.Item1),pair.Item2);
            Reset();g.ModernChampion.Cast(0,null);j.ModernChampion.Cast(2,null);g.AutoAttackHit(j);
            Equal("dodge_q_hp",j.Stats.CurrentHealth,650);Equal("dodge_q_consumed",Field(g,"q"),0);Equal("dodge_count",Field(j,"dodges"),1);
            if(j.HasBuff("Silence"))throw new Exception("dodged Q silenced Jax");
            j.ModernChampion.Update(1000);j.ModernChampion.Cast(2,null);
            Equal("counterstrike_damage",g.Stats.CurrentHealth,690-(40+.04f*690)*1.2f/1.32f);
            Equal("counterstrike_stun",g.GetBuffWithName("Stun").Duration,1);
            Reset();g.ModernChampion.Cast(1,null);
            for(int i=0;i<2;i++)g.TakeDamage(j,100,DamageType.DAMAGE_TYPE_PHYSICAL,DamageSource.DAMAGE_SOURCE_SPELL,false);
            Equal("w_reduction_and_shield",g.Stats.CurrentHealth,690-(200/1.38f*.75f-65));
            Reset();j.Stats.CurrentHealth=450;g.ModernChampion.Cast(3,j);
            Equal("garen_r_true",j.Stats.CurrentHealth,275);
            Reset();j.ModernChampion.Cast(1,null);j.ModernChampion.Cast(0,g);j.ModernChampion.Update(100);
            Equal("jax_q_w",g.Stats.CurrentHealth,690-65/1.38f-50/1.32f);Equal("jax_w_consumed",Field(j,"w"),0);
            Reset();for(int i=0;i<3;i++)j.AutoAttackHit(g);
            Equal("jax_r_third_hit",g.Stats.CurrentHealth,690-3*68/1.38f-75/1.32f);
            Reset();j.ModernChampion.Cast(3,null);j.ModernChampion.Update(250);
            Equal("jax_r_active",g.Stats.CurrentHealth,690-100/1.32f);Equal("jax_r_armor",j.Stats.Armor.Total,81);Equal("jax_r_mr",j.Stats.MagicResist.Total,59);
            for(int i=0;i<2;i++)j.AutoAttackHit(g);
            Equal("jax_r_second_hit",g.Stats.CurrentHealth,690-175/1.32f-136/1.38f);
            Reset();g.ModernChampion.Cast(2,null);for(int i=0;i<180;i++)g.ModernChampion.Update(1000f/60);
            Equal("garen_spin_ticks",Field(g,"spinCount"),7);
            Equal("garen_shred",j.Stats.Armor.Total,27);
            Console.WriteLine("MODERN COMBAT SELFTEST PASSED");game.SetToExit=true;
        }
    }
}
