"""Convert garen_clicks_vtable.json output → format expected by pipeline_to_overlay.py.

- Filters to the winning click-dest addr (lowest avg_d from identify) so we don't
  include the noisy sibling.
- Caps to first 600s of game time (10 min).
- Output: C:\\tmp\\garen_clicks_first10min.json with shape {"events":[...]}
"""
import json, sys, os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"C:\tmp\garen_clicks_vtable.json")
    ap.add_argument("--output", default=r"C:\tmp\garen_clicks_first10min.json")
    ap.add_argument("--winner-addr", default=None,
                    help="Hex address of the real click-dest. If not given, uses the most-frequent address in the input.")
    ap.add_argument("--max-gt", type=float, default=600.0)
    args = ap.parse_args()

    d = json.load(open(args.input))
    clicks = d.get("clicks", [])
    print(f"input: {len(clicks)} total clicks")

    if not args.winner_addr:
        from collections import Counter
        c = Counter(cl["addr"] for cl in clicks)
        print("addr distribution:")
        for a, n in c.most_common():
            print(f"  {a}: {n}")
        winner = c.most_common(1)[0][0]
        print(f"winner (most-frequent): {winner}")
    else:
        winner = args.winner_addr

    out_events = []
    for cl in clicks:
        if cl["addr"] != winner: continue
        if cl["game_t"] > args.max_gt: continue
        out_events.append({
            "game_time": cl["game_t"],
            "x": cl["x"], "z": cl["z"],
            "addr": cl["addr"],
            "color": [255, 255, 0],
        })

    # Cast events from the spellbook cd_expire jumps.
    casts = d.get("casts", [])
    out_casts = []
    last_emit = {}
    for c in casts:
        if c["game_t"] > args.max_gt: continue
        slot = c["slot"]
        # Dedupe within 0.4s per slot (cd jumps can register on consecutive samples)
        if c["game_t"] - last_emit.get(slot, -10) < 0.4: continue
        last_emit[slot] = c["game_t"]
        out_casts.append({
            "game_time": c["game_t"], "slot": slot,
            "spell_name": c.get("spell_name"),
            "hero_x": c.get("hero_x"), "hero_z": c.get("hero_z"),
        })

    out = {"events": out_events, "casts": out_casts,
           "winner_addr": winner, "max_gt": args.max_gt}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}  ({len(out_events)} clicks, {len(out_casts)} casts, addr={winner})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
