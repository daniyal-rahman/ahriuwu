"""Probe: minion spawns, deaths and lifetimes per side and per MODEL (melee /
caster / cannon) on a trace; run on the JAX and server traces of the same
scripted game to see whose waves die faster and which minion type differs."""
import sys, numpy as np, json
for p in sys.argv[1:]:
    z = np.load(p, allow_pickle=True); meta = json.loads(str(z["metadata"]))
    kind = z["kind"][0]; team = z["team"]; alive = z["alive"]; hp = z["hp"]; cs = z["cs"]; model = z["model"]; hz = meta.get("hz", 10.)
    models = meta.get("profiles") or meta.get("models") or {}
    mins = np.where(kind == 0)[0]; ch = [i for i in range(len(kind)) if kind[i] == 1]; blue = min(ch, key=lambda i: team[0, i])
    f0 = int(np.argmax(cs[:, blue] > 0)); T = alive.shape[0]
    print(f"== {p.split('/')[3]}: frames {T}, first CS frame {f0}; model codes seen {sorted(set(int(m) for m in np.unique(model[:, mins])))}; profile keys {list(models)[:8] if isinstance(models, dict) else models}")
    for f in (f0 - 100, f0 - 12, f0, f0 + 50, f0 + 200):
        if 0 <= f < T:
            al = (alive[f, mins] > 0) & (hp[f, mins] > 0)
            print(f"   frame {f}: alive blue {int((al & (team[f, mins] == 0)).sum())} red {int((al & (team[f, mins] == 1)).sum())} cs {int(cs[f, blue])}")
    stats = {}
    for i in mins:
        al = (alive[:, i] > 0) & (hp[:, i] > 0)
        starts = np.where(al[1:] & ~al[:-1])[0] + 1
        if al[0]: starts = np.concatenate([[0], starts])
        ends = np.where(al[:-1] & ~al[1:])[0]
        for s0 in starts:
            t = int(team[s0, i]); mdl = int(model[s0, i])
            if t not in (0, 1): continue
            e = ends[ends >= s0]; key = (t, mdl); st = stats.setdefault(key, {"spawned": 0, "died": 0, "life": []})
            st["spawned"] += 1
            if len(e): st["died"] += 1; st["life"].append((e[0] - s0) / hz)
    for (t, mdl), st in sorted(stats.items()):
        lt = st["life"]; print(f"   team {t} model {mdl}: spawned {st['spawned']:3d} died {st['died']:3d} mean life {np.mean(lt):5.1f} s median {np.median(lt):5.1f}" if lt else f"   team {t} model {mdl}: spawned {st['spawned']}")
