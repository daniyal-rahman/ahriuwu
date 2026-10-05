"""TOOL: CPU/GPU cost of one full modern item tick (docs/modern/ITEMS_IMPLEMENTATION.md).

    ops/login_capped.sh 8G 2 .venv-jax/bin/python -m ops.modern.items_bench [BATCH]
"""
import sys, time, jax, jax.numpy as jnp
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import inventory as I, effects as E
from lanerl_jax.modern.items.effects import runtime as R
from lanerl_jax.modern.tests import item_harness as H
B = int(sys.argv[1]) if len(sys.argv) > 1 else 64
n=66
u=H.units(H.champions(x1=150.)+[dict(x=100.*k,y=200,team=k%2) for k in range(n-2)])
inv=I.inventory_from_ids([[3071,3053,3078,6631,3047,6333],[3075,3068,3065,2504,3111,3083]])
own=I.owned_counts(inv); item=I.inventory_stats(inv)
ctx=H.ctx(base_ad=60., bonus_ad=item.attack_damage, max_hp=600+item.health, in_combat=True)
dfn=D.default_defense(n)._replace(unit_class=u.cls); off=D.default_offense(n)._replace(unit_class=u.cls)
base=D.packets(jnp.ones(8,bool),jnp.arange(8)%2,(jnp.arange(8)+1)%2,100.,D.PHYSICAL,D.BASIC_ATTACK)
def f(st,hp,sh,status,now):
    c=ctx._replace(now=now,hp=hp[:2])
    return R.item_tick(st,own,c,u._replace(hp=hp),attack=H.attack(hit=(True,True),target=(1,0)),cast=H.cast(started=(True,False)),
        request=jnp.asarray([6631,0],jnp.int32),base_packets=base,base_offense=off,base_defense=dfn,hp=hp,max_hp=u.max_hp,
        shields=sh,status=status,kills=H.kills(n),holder_stats=item)
st=E.init(2,n); hp=jnp.full((n,),3000.,jnp.float32)
j=jax.jit(f)
t=time.time(); o=j(st,hp,D.init_shields(n),R.init_status(n),jnp.float32(1.)); jax.block_until_ready(o.hp); print('single compile+first %.1fs'%(time.time()-t))
t=time.time()
for k in range(30): o=j(o.state,o.hp,o.shields,o.status,jnp.float32(1.+k/30))
jax.block_until_ready(o.hp); print('single per tick %.2f ms overflow %d'%((time.time()-t)/30*1000, int(o.packet_overflow)))
vj=jax.jit(jax.vmap(f,in_axes=(0,0,0,0,None)))
bt=lambda x: jnp.broadcast_to(x,(B,)+x.shape)
S=jax.tree_util.tree_map(bt,st); HP=bt(hp); SH=jax.tree_util.tree_map(bt,D.init_shields(n)); ST=jax.tree_util.tree_map(bt,R.init_status(n))
t=time.time(); o=vj(S,HP,SH,ST,jnp.float32(1.)); jax.block_until_ready(o.hp); print('vmap%d compile+first %.1fs'%(B,time.time()-t))
t=time.time()
for k in range(10): o=vj(o.state,o.hp,o.shields,o.status,jnp.float32(1.+k/30))
jax.block_until_ready(o.hp); dt=(time.time()-t)/10; print('vmap%d per tick %.1f ms (%.3f ms/env) device %s'%(B,dt*1000,dt*1000/B,jax.devices()[0]))
