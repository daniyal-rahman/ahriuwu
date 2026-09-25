"""ENT-02 / AA-007 measurement: retarget a champion's auto-attack mid-wind-up.

Blue Garen engages the first red wave; red Garen idles in the fountain. Every
swing start (atk 0->1 in consecutive 1-tick observations) consumes the next
scenario: after `j` further observations the attack order switches to another
enemy minion (in range, or out of range). The order sent on the observation
`j` after the start observation applies at server tick s+1+j (s = start tick).
"""
import json
import math
import argparse
from pathlib import Path

from lanerl_jax.parity.record import record_trace, _local_move_toward  # noqa: E402

STAGE = (3000.0, 12700.0)
IN_RANGE = 200.0          # centre distance; idealRange = 175 + ~48 minion cr
OUT_LO, OUT_HI = 330.0, 800.0


class Driver:
    def __init__(self, out):
        self.prev_atk = 0
        self.start_i = None
        self.scen = None
        self.N = None               # obs offset at which an undisturbed hit shows
        self.base_hp = None
        self.plan = [("none", 0), ("none", 0), ("in", 3), ("in", "N-2"),
                     ("in", "N-1"), ("in", "N"), ("out", 3), ("in", "N-1"),
                     ("in", 3), ("in", "N-1"), ("out", 3)]
        self.log = open(out / "driver.jsonl", "w")

    def _units(self, obs):
        return {u["id"]: u for u in obs.get("u", []) if "id" in u}

    def __call__(self, obs, i):
        if obs is None:
            return None
        us = self._units(obs)
        g = next((u for u in obs["u"] if u.get("k") == "Champion" and u["tm"] == 100), None)
        if g is None:
            return {"blue": {"t": "noop"}, "red": {"t": "noop"}}
        gx, gy = g["x"], g["y"]
        enemies = [u for u in obs["u"] if u.get("k") == "LaneMinion" and u["tm"] == 200
                   and u.get("vb", 0)]
        for u in enemies:
            u["_d"] = math.hypot(u["x"] - gx, u["y"] - gy)
        atk, tgt = g.get("atk", 0), g.get("tgt", 0)
        act = {"t": "noop"}
        note = None

        if atk == 1 and self.prev_atk == 0:
            # a swing started on the tick this obs follows
            self.start_i = i
            self.start_tgt = tgt
            self.base_hp = us.get(tgt, {}).get("hp")
            if self.plan:
                kind, j = self.plan[0]
                if isinstance(j, str):
                    if self.N is None:
                        j = None
                    else:
                        j = self.N + int(j[1:]) if len(j) > 1 else self.N
                if j is not None:
                    self.scen = (kind, j)
                    self.plan.pop(0)
                    note = f"swing_start scen={kind}@{j}"
                else:
                    self.scen = ("none", 0)
                    note = "swing_start scen=none(learn N)"
            else:
                self.scen = None
                note = "swing_start (plan done)"

        # learn N from an undisturbed swing
        if (self.scen is not None and self.scen[0] == "none" and self.start_i is not None
                and self.base_hp is not None):
            hp = us.get(self.start_tgt, {}).get("hp")
            if hp is not None and self.base_hp - hp >= 50 and self.N is None:
                self.N = i - self.start_i
                note = f"learned N={self.N}"

        if (self.scen is not None and self.scen[0] != "none"
                and i - self.start_i == self.scen[1]):
            kind = self.scen[0]
            if kind == "in":
                cands = [u for u in enemies if u["id"] != self.start_tgt and u["_d"] <= IN_RANGE]
            else:
                cands = [u for u in enemies if u["id"] != self.start_tgt
                         and OUT_LO <= u["_d"] <= OUT_HI]
            if cands:
                b = max(cands, key=lambda u: u["hp"])
                act = {"t": "attack", "id": int(b["id"])}
                note = (f"SWITCH {kind} j={self.scen[1]} from={self.start_tgt} "
                        f"to={b['id']} d={b['_d']:.0f} hpB={b['hp']}")
            else:
                note = f"no candidate for {kind}; retry next swing"
                self.plan.insert(0, (kind, self.scen[1]))
            self.scen = ("done", -1)
        elif act["t"] == "noop" and (tgt == 0 or tgt not in us):
            near = [u for u in enemies if u["_d"] <= 1200]
            if near:
                a = min(near, key=lambda u: u["_d"])
                act = {"t": "attack", "id": int(a["id"])}
                note = note or f"engage {a['id']} d={a['_d']:.0f}"
            elif math.hypot(gx - STAGE[0], gy - STAGE[1]) > 60 and i % 10 == 0:
                act = _local_move_toward(g, *STAGE)

        self.prev_atk = atk
        rel = {k: us[k]["hp"] for k in us if us[k].get("k") == "LaneMinion"
               and us[k]["tm"] == 200 and math.hypot(us[k]["x"] - gx, us[k]["y"] - gy) < 900}
        self.log.write(json.dumps({"i": i, "t": obs.get("t"), "atk": atk, "tgt": tgt,
                                   "gx": gx, "gy": gy, "act": act, "note": note,
                                   "hp": rel}) + "\n")
        if note:
            print(i, obs.get("t"), note, flush=True)
        return {"blue": act, "red": {"t": "noop"}}



def main():
    from lanerl_train import paths
    from lanerl_jax.parity.script_health import assert_all_scripts_loaded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=15000)
    parser.add_argument("--port-base", type=int, default=47310)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    driver = Driver(args.out)
    try:
        log = record_trace(
            args.out, decisions=args.decisions, port_base=args.port_base,
            step_ticks=1, tag="ent02",
            extra_env={"LANERL_AUTOBUY": "0", "LANERL_DECISION_TRACE": "1"},
            server_dir=paths.server_dir().parent.parent / "Trace" / "net6.0",
            config_path=Path(__file__).resolve().parents[2] / "lanerl/cfg/garen1v1_trace.json",
            driver=driver)
    finally:
        driver.log.close()
    assert_all_scripts_loaded(log)
    rows = [json.loads(line) for line in (args.out / "driver.jsonl").read_text().splitlines()]
    switches = [r for r in rows if (r.get("note") or "").startswith("SWITCH")]
    if not any("SWITCH in" in r["note"] for r in switches) or not any(
            "SWITCH out" in r["note"] for r in switches):
        raise RuntimeError("incomplete retarget coverage; extend --decisions")
    print("LOG", log)
    print("SWITCHES", len(switches))
    for row in switches:
        print(row["t"], row["note"])


if __name__ == "__main__":
    main()
