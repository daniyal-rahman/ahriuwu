using System;
using System.Globalization;

namespace LeagueSandbox.GameServer.Lanerl
{
    /// <summary>
    /// Emit the BRANCH the server took, not the state that resulted from it.
    ///
    /// WHY THIS EXISTS
    /// ---------------
    /// <see cref="LanerlStateDump"/> answers "what is true after this tick".
    /// That is the right primitive for determinism and reset-leakage checks,
    /// and it is the wrong one for finding a porting bug, because one wrong
    /// branch propagates into every field it touches and you are left working
    /// backwards from consequences.
    ///
    /// Measured cost of that, on this project:
    ///
    /// * One unported line -- <c>Spell.FinishCasting</c> ending with
    ///   <c>UpdateMoveOrder(OrderType.Hold, true)</c> for any non-InstantCast
    ///   cast, which an auto-attack is -- read for weeks as a 4,791-count
    ///   "move order" residual that looked like symmetric boundary jitter. It
    ///   was neither symmetric (4,769 against 22) nor jitter.
    /// * One rounding bug in the dump -- a melee minion's final windup tick
    ///   has 2.67e-5 s left and publishes as a flat 0 -- surfaced as 1,041
    ///   <c>aa_hit</c> misses PLUS 1,126 <c>hp</c> misses PLUS parts of three
    ///   boolean fields, and cost a day to collapse back to one mechanism.
    ///
    /// Both are one line of C#. Neither is visible in a state diff as one
    /// line of anything. A trace that says "server called FinishCasting on
    /// unit 7 at tick N" and a sim that did not is a one-step diagnosis.
    ///
    /// WHY NOT POLL, AS <c>LanerlHooks.TurretTrace</c> DOES
    /// -----------------------------------------------------
    /// That sibling samples every 250 ms and emits when a turret's target
    /// differs from the last sample. It therefore reports transitions but
    /// never reasons, and it cannot see a transition shorter than its own
    /// interval -- which is most of them, at a 16.67 ms tick. Reasons are the
    /// entire point here, so these emit from inside the branch.
    ///
    /// BEHAVIOUR NEUTRALITY IS THE CONTRACT
    /// -------------------------------------
    /// Off by default and gated on one env var, so an unflagged run is
    /// byte-identical: the parity corpus must not move because a diagnostic
    /// was added. The gate is a <c>static readonly bool</c> read once, like
    /// <see cref="LanerlStateDump.Enabled"/>, so a disabled call is a
    /// predictable branch on a cached field and nothing else. Every call site
    /// must be a statement that reads state and writes none.
    ///
    ///     LANERL_DECISION_TRACE=1
    ///
    /// emits, one line per branch taken:
    ///
    ///     LANERL_DECISION t=&lt;gametime&gt; k=&lt;branch&gt; id=&lt;netid&gt; &lt;detail&gt;
    ///
    /// <c>t</c> matches <see cref="LanerlStateDump"/>'s clock so the two
    /// streams join on it. <c>id</c> is a NetId: deliberately unhashed and
    /// unstable across resets, exactly as the state dump's internals are, and
    /// for the same reason -- this stream is a diagnostic, never an input to
    /// a determinism comparison.
    /// </summary>
    public static class LanerlDecisionTrace
    {
        public static readonly bool Enabled =
            Environment.GetEnvironmentVariable("LANERL_DECISION_TRACE") == "1";

        /// <summary>One branch. <paramref name="kind"/> should name the C#
        /// site, not its effect, so a reader can grep the server for it.</summary>
        public static void Emit(float t, string kind, uint netId, string detail = "")
        {
            if (!Enabled) return;
            Console.WriteLine(
                "LANERL_DECISION t=" + t.ToString("F0", CultureInfo.InvariantCulture) +
                " k=" + kind + " id=" + netId.ToString(CultureInfo.InvariantCulture) +
                (string.IsNullOrEmpty(detail) ? "" : " " + detail));
        }

        // NOTE: there is deliberately no `Emit(AttackableUnit, ...)` overload.
        // `GameObject._game` is protected, so a static helper cannot read the
        // clock off a unit. Every call site is inside a GameObject subclass
        // where `_game` IS in scope, so each passes `_game.GameTime`
        // explicitly. Widening `_game`'s visibility to buy one convenience
        // overload would be a real change to the vendored server in order to
        // add a diagnostic, which is the opposite of this file's contract.
    }
}
