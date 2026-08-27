#!/usr/bin/env python3
"""Which label fields are LIVE, HELD, or GARBAGE?

Run this before trusting any label field. Two failure classes it catches:

HELD-read-as-live has caused three separate bugs here -- the movement target (a
frozen click point consumed as a per-frame command), the cursor (a reprojected
world anchor read as a mouse position), and the same suspicion on enemy gold.
The test is the same each time: how often does the field CHANGE between
consecutive frames?

GARBAGE found by this sweep 2026-08-27: `champion_stats.gold` is a single
constant -3.7744e+22 for an entire game, and a DIFFERENT constant per hero
inside visible_heroes (0.0 / 1.4594e+31 / -5.2098e+16 / 2.2036e+12). That is a
wrong memory offset reinterpreted as a float. `gold_total` is the correct field
(500.0 -> 3218.5, and 500 is League's starting gold). The training path uses
gold_total only, so this is a hazard rather than an active bug -- but it proves
the recorder has at least one bad offset, so field semantics should be measured
and not assumed.

A held value that a consumer treats as live has now caused three separate bugs
here: the movement target (frozen click point read as a per-frame command), the
cursor (a reprojected world anchor read as a mouse position), and possibly the
enemy's gold. The test that would have caught all three is the same one: how
often does the field CHANGE between consecutive frames?

  repeat ~= 100%  -> held/stale, or genuinely constant. Check the consumer.
  repeat  low     -> live per-frame read.

Champion gold is the control: it MUST tick (~1.9 g/s passive) in every game.
"""
import json, glob, os
import numpy as np
ROOT="/srv/nfs/datasets/lol_replays_16_9_772"
games=sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/NA1_*"))[:6]

def walk(d, prefix="", out=None, depth=0):
    if out is None: out={}
    if depth>3: return out
    if isinstance(d, dict):
        for k,v in d.items():
            key=f"{prefix}.{k}" if prefix else k
            if isinstance(v,(int,float)) and not isinstance(v,bool): out[key]=float(v)
            elif isinstance(v,list) and v and all(isinstance(x,(int,float)) for x in v[:3]):
                for i,x in enumerate(v[:2]): out[f"{key}[{i}]"]=float(x)
            elif isinstance(v,dict): walk(v,key,out,depth+1)
    return out

acc={}
for g in games:
    try: fr=json.load(open(f"{ROOT}/{g}/labels.json"))["frames"]
    except Exception: continue
    seq={}
    for f in fr[2000:5000]:
        for k,v in walk(f.get("label") or {}).items(): seq.setdefault(k,[]).append(v)
    for k,v in seq.items():
        if len(v)<100: continue
        a=np.array(v); rep=float((a[1:]==a[:-1]).mean())
        acc.setdefault(k,[]).append(rep)
rows=sorted(((np.mean(v),k,len(v)) for k,v in acc.items() if len(v)>=3), reverse=True)
print(f"{'field':46s} {'repeat%':>8s}  reading")
for rep,k,n in rows[:26]:
    tag = ("HELD - check the consumer" if rep>0.97 else
           "sticky" if rep>0.80 else
           "live")
    print(f"{k[:46]:46s} {100*rep:7.1f}%  {tag}")
