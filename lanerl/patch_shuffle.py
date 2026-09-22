"""
Measure gate 1's order-dependence floor instead of arguing about it.

WHY
---
`ORDER-003` says part of the residual can never reach zero: the server
updates units one at a time (`ObjectManager.Update`, `foreach (var obj in
_objects.Values) obj.Update(diff)`), so what each unit sees depends on
where it happens to sit in that iteration. The port updates every unit from
one consistent snapshot. That is a design trade, not a bug, and it puts a
floor under every order-sensitive field.

The floor has only ever been ARGUED. It was quoted as a mechanism and used
to classify rows, but nobody measured how large it is -- and this project's
record on argued-but-unmeasured mechanisms is five refuted out of five.

So measure it the same way everything else here gets measured: run the
server against ITSELF with the update order permuted each tick, and score
shuffled-vs-normal with the identical metrics used for sim-vs-server.

    sim residual <= shuffled residual   ->  FLOOR. Stop investigating.
    sim residual >  shuffled residual   ->  a real defect, worth the forensics.

It also answers a question worth more than the floor itself: if shuffling
barely moves outcomes then update order does not matter, most of gate 1 can
close immediately, and the vectorised tick was never costing fidelity. If
shuffling moves outcomes a lot, the SERVER's own behaviour is
order-fragile -- and then no amount of simulator work can fix it, which is
a fact about the reference, not the port.

WHAT IS AND IS NOT SHUFFLED
---------------------------
Only the ORDER OF THE `Update(diff)` CALLS in the first loop. The
collection itself is untouched, because insertion order is load-bearing in
the rest of that method -- `oldObjectsCount` gates which objects get
`LateUpdate` this tick, and the add/remove queues splice against it. Spawn
sequence is also semantically real (a minion's place in its wave decides
where it stands), so permuting the collection would change the scenario
rather than the scheduling.

Off unless `LANERL_SHUFFLE_ORDER` is set to an integer seed, and the
permutation is a seeded Fisher-Yates over `(seed, tickIndex)`, so a shuffled
run is exactly reproducible and an unset run is byte-identical to the
canonical corpus. That neutrality is VERIFIED by re-recording and comparing
the `LANERL_STATEROW` sha1, not assumed.

A shuffled recording is NOT the canonical corpus and can never be hash-
compared to it -- that is the entire point of it. It gets its own output
directory, and the dump announces the seed once so a log cannot be mistaken
for a clean one.

    python -m lanerl.patch_shuffle
    python -m lanerl.patch_shuffle --verify
    python -m lanerl.patch_shuffle --uninstall
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parents[1] / "lanerl-vendor"
_SRV = _VENDOR / "LoLServer"
_OBJMGR = _SRV / "GameServerLib/ObjectManager.cs"

#: `ObjectManager.cs` does not import the Lanerl namespace, so every
#: reference from inside it must be fully qualified.
_MARKER = "Lanerl.LanerlShuffle.Enabled"

_ANCHOR = """            _currentlyInUpdate = true;

            // For all existing objects
            foreach (var obj in _objects.Values)
            {
                obj.Update(diff);
            }
"""

_REPLACE = '''            _currentlyInUpdate = true;

            // For all existing objects
            if (LeagueSandbox.GameServer.Lanerl.LanerlShuffle.Enabled)
            {
                // ORDER-003's floor, measured rather than argued. See
                // `lanerl/patch_shuffle.py`. Only the CALL ORDER moves; the
                // collection is untouched, because `oldObjectsCount` below
                // gates `LateUpdate` against insertion order and the add/
                // remove queues splice against it.
                var lanerlOrder =
                    LeagueSandbox.GameServer.Lanerl.LanerlShuffle.Permute(_objects.Values);
                for (int lanerlI = 0; lanerlI < lanerlOrder.Length; lanerlI++)
                {
                    lanerlOrder[lanerlI].Update(diff);
                }
            }
            else
            {
                foreach (var obj in _objects.Values)
                {
                    obj.Update(diff);
                }
            }
'''

_HELPER = _SRV / "GameServerLib/Lanerl/LanerlShuffle.cs"
_HELPER_SRC = '''using System;
using System.Collections.Generic;
using System.Linq;
using LeagueSandbox.GameServer.GameObjects;

namespace LeagueSandbox.GameServer.Lanerl
{
    /// <summary>
    /// Permute the per-tick object update order, to MEASURE how much of the
    /// parity residual is order-dependence rather than simulator error.
    ///
    /// `ObjectManager.Update` walks `_objects.Values` serially, so a unit's
    /// view of the world depends on whether its neighbours have already been
    /// updated this tick. The JAX port updates everything from one snapshot
    /// and therefore cannot reproduce that, which `ORDER-003` records as a
    /// design FLOOR. How big the floor is has never been measured -- it has
    /// been asserted, and on this project five asserted mechanisms have been
    /// refuted by the first direct measurement.
    ///
    /// Running the server against itself with this on gives the floor as a
    /// number in the same units as the parity report. A simulator residual at
    /// or below it is not a defect to chase.
    ///
    /// DETERMINISM IS THE CONTRACT. The permutation is seeded Fisher-Yates
    /// over (seed, tick), so a shuffled run reproduces exactly; and with the
    /// variable unset this type is never touched, so an ordinary run stays
    /// byte-identical to the canonical corpus. That is verified by hash, not
    /// assumed.
    /// </summary>
    public static class LanerlShuffle
    {
        private static readonly string _raw =
            Environment.GetEnvironmentVariable("LANERL_SHUFFLE_ORDER");

        public static readonly bool Enabled =
            !string.IsNullOrEmpty(_raw) && int.TryParse(_raw, out _);

        public static readonly int Seed =
            Enabled ? int.Parse(_raw) : 0;

        private static long _tick;
        private static bool _announced;

        /// <summary>
        /// A fresh permutation every tick. Splitmix64 on (Seed, tick, i) --
        /// counter-based rather than a carried RNG state, so the order for a
        /// given tick does not depend on how many objects existed before it.
        /// </summary>
        public static GameObject[] Permute(IEnumerable<GameObject> objects)
        {
            var arr = objects.ToArray();
            if (!_announced)
            {
                // Once, loudly: a shuffled log must never be mistaken for a
                // clean one, and the parity corpora are compared by hash.
                Console.WriteLine(
                    "LANERL_SHUFFLE_ORDER active seed=" + Seed +
                    " -- THIS RECORDING IS NOT THE CANONICAL CORPUS");
                _announced = true;
            }
            long t = _tick++;
            for (int i = arr.Length - 1; i > 0; i--)
            {
                int j = (int)(Mix((ulong)Seed, (ulong)t, (ulong)i) % (ulong)(i + 1));
                var tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            }
            return arr;
        }

        private static ulong Mix(ulong a, ulong b, ulong c)
        {
            ulong x = a * 0x9E3779B97F4A7C15UL
                    ^ b * 0xBF58476D1CE4E5B9UL
                    ^ c * 0x94D049BB133111EBUL;
            x ^= x >> 30; x *= 0xBF58476D1CE4E5B9UL;
            x ^= x >> 27; x *= 0x94D049BB133111EBUL;
            x ^= x >> 31;
            return x;
        }
    }
}
'''


def _apply(install: bool) -> None:
    src = _OBJMGR.read_text()
    if install:
        if _MARKER in src:
            print("  skip  ObjectManager hook: already present")
        else:
            n = src.count(_ANCHOR)
            if n != 1:
                raise SystemExit(
                    f"ANCHOR: expected 1 occurrence in {_OBJMGR.name}, found {n}. "
                    "Re-derive it rather than loosening it.")
            _OBJMGR.write_text(src.replace(_ANCHOR, _REPLACE))
            print("  ADD   ObjectManager hook")
        _HELPER.write_text(_HELPER_SRC)
        print("  ADD   LanerlShuffle.cs")
    else:
        if _MARKER in src:
            _OBJMGR.write_text(src.replace(_REPLACE, _ANCHOR))
            print("  DEL   ObjectManager hook")
        if _HELPER.exists():
            _HELPER.unlink()
            print("  DEL   LanerlShuffle.cs")


def verify() -> int:
    ok_hook = _OBJMGR.exists() and _MARKER in _OBJMGR.read_text()
    ok_help = _HELPER.exists()
    print(f"  {'OK  ' if ok_hook else 'MISS'}  ObjectManager hook")
    print(f"  {'OK  ' if ok_help else 'MISS'}  LanerlShuffle.cs")
    return 0 if (ok_hook and ok_help) else 1


def main() -> None:
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    _apply(install="--uninstall" not in sys.argv)
    print("\nNow rebuild -- see lanerl/patch_observability.py's docstring.")


if __name__ == "__main__":
    main()
