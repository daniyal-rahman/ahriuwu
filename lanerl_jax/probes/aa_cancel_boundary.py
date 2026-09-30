"""LEARN-PAIR-08: inspect suppression on the final windup tick (NumPy only)."""
import json
import numpy as np
from lanerl_jax.sim.autoattack import step_autoattack

def main():
    rows=[]
    for windup,can_attack in ((.050,False),(.005,False),(.005,True)):
        f=lambda x:np.array([x],np.float32)
        b=lambda x:np.array([x],bool)
        result=step_autoattack(f(.9),f(windup),b(True),b(False),in_range=b(True),
            can_attack=b(can_attack),has_target=b(True),attack_period=f(1.6),
            windup_time=f(.3),attack_damage=f(70),target_resist=f(0),xp=np)
        rows.append(dict(windup_s=windup,can_attack=can_attack,hit=bool(result.hit[0]),damage=float(result.damage[0])))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
