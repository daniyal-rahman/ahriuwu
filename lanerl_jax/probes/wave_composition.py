"""Probe: steady-state wave composition per side and model (mean alive minions
per frame, 120-450 s after the first last-hit) on each trace, plus spawn
cadence (new alive slots per 30 s). Robust to slot recycling in the renderers."""
import sys, numpy as np, json
LAB = {2: "melee", 3: "melee", 4: "caster", 5: "caster", 6: "cannon", 7: "cannon"}
for p in sys.argv[1:]:
    z = np.load(p, allow_pickle=True); meta = json.loads(str(z["metadata"])); hz = meta.get("hz", 10.)
    kind = z["kind"][0]; team = z["team"]; alive = z["alive"]; hp = z["hp"]; cs = z["cs"]; model = z["model"]
    # minion = model code 2..7 in THAT frame (renderers disagree on kind codes: server 0, JAX 2)
    ch = [i for i in range(len(kind)) if kind[i] == 1]; blue = min(ch, key=lambda i: team[0, i])
    mins = np.arange(len(kind))
    f0 = max(0, int(np.argmax(cs[:, blue] > 0)) - int(12 * hz)); a, b = f0 + int(120 * hz), min(f0 + int(450 * hz), alive.shape[0])
    md = model[a:b][:, mins]; al = (alive[a:b][:, mins] > 0) & (hp[a:b][:, mins] > 0) & (md >= 2) & (md <= 7); tm = team[a:b][:, mins]
    print(f"== {p.split('/')[3]} steady state frames {a}-{b}")
    for t, name in ((0, "blue"), (1, "red")):
        parts = {lab: float((al & (tm == t) & (md == code)).sum(1).mean()) for code, lab in LAB.items() if (code % 2) == t}
        print(f"   {name}: mean alive {float((al & (tm == t)).sum(1).mean()):5.2f} = " + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()))
    # spawn cadence: new alive transitions per side per 30 s, whole window from f0
    for t, name in ((0, "blue"), (1, "red")):
        alf = (alive[f0:][:, mins] > 0) & (team[f0:][:, mins] == t) & (model[f0:][:, mins] >= 2) & (model[f0:][:, mins] <= 7)
        new = (alf[1:] & ~alf[:-1]).sum(1)
        per30 = [int(new[i:i + int(30 * hz)].sum()) for i in range(0, min(len(new), int(480 * hz)), int(30 * hz))]
        print(f"   {name} new-alive per 30 s: {per30}")
