# Reproducing the gate numbers

Every number in `JAX_FIDELITY_LEDGER.md` should be regenerable from a fresh
clone plus the vendored server. This says how, and — more usefully — says
which inputs are **not** in git and therefore have to be remade.

## What is and is not in the repository

| thing | in git? | how to get it |
|---|---|---|
| all analysis code (`lanerl_jax/parity/**`) | yes | clone |
| the ledger and these docs | yes | clone |
| the vendored C# instrumentation | **as diffs + an installer** | `lanerl/patch_*.py`, `lanerl/vendor_patches/` |
| the vendored server itself | no | `/srv/nfs/projects/lanerl-vendor/` (not ours) |
| **server recordings** (`lanerl_jax/runs/**`) | **NO** — `.gitignore:42` `runs/` | re-record, below |
| drill outputs | no, same rule | re-run, below |

The recordings are the evidence behind every residual count and they are
75–600 MB each, so they are deliberately not committed. That means **a fresh
clone cannot reproduce a single number without first re-recording**, and any
run you intend to quote has to be re-derived rather than downloaded. Plan for
the boot time.

## 0. Where to run it

`desktop` (the 16-core Slurm node) is the intended home. When it is down —
`sinfo` shows `down*` / `State=DOWN+NOT_RESPONDING` — everything below still
fits on the login node, but ONLY behind the cap:

    ops/login_capped.sh <mem> <cores> <cmd...>

Its load-bearing setting is `MemorySwapMax=0`: an over-budget job is
OOM-killed in seconds instead of quietly pushing eight other users' pages to
swap for hours. A bare `pytest` once held this node for 5h19m. Never run one.

## 1. Install and build the instrumentation

    python -m lanerl.patch_decision_trace      # builds the isolated Content-trace tree
    python -m lanerl.patch_observability       # AA-004 + CFH-002
    python -m lanerl.patch_observability --verify

Then rebuild the assemblies (both patches touch `GameServerLib`) — the exact
command is in `lanerl/patch_observability.py`'s docstring. Build to
`bin/Trace/net6.0`, never over `bin/Release`.

## 2. Re-record

    # idle corpus, the gate-1 canonical fixture
    python -m lanerl_jax.parity.tier1_full --out lanerl_jax/runs/tier1_full \
        --game-seconds 420

    # instrumented recording, for AA-004 / CFH-002
    #   must exceed 200 s of game time: waves clash at ~110 s, so a 120 s
    #   window exercises neither emit site and "passes" while proving nothing.

Then, on any recording whose numbers you will quote:

    python -c "from pathlib import Path; \
      from lanerl_jax.parity.script_health import check_script_load; \
      print(check_script_load(Path('<log>')))"

It must say `Loaded all`. A script that fails to compile does not stop the
server — it silently stops being the AI, and the trace still parses
(`METH-003`).

## 3. The gate numbers

**Gate 1, whole corpus.** ~50 min, ~0.9 GB:

    ops/login_capped.sh 7G 3 .venv-jax/bin/python -m lanerl_jax.parity.tier1_full \
        --existing-log lanerl_jax/runs/tier1_full/server/instance000.log \
        --chunk-pairs 500

`--chunk-pairs` is what makes this fit anywhere: without it the parse
materialises every `Entity` in a 575 MB log (~26 GB) even when you asked for a
slice, because `--max-pairs` bounded the loop and not the parse. Chunked and
unchunked reports are byte-identical (test in `test_trace_and_diff.py`).

**Gate 1, per-residual decomposition.** ~2 min per window:

    ops/login_capped.sh 8G 2 .venv-jax/bin/python -m lanerl_jax.parity.tier1_residual_drill \
        --existing-log <log> --from-ms 130000 --to-ms 205000 --max-pairs 1500

Add `--cfh-log <log>` for the pre-clear call-for-help ground truth. **Run more
than one window.** Every rate measured on a single window here has moved
substantially when widened, always downward, because the first window was
picked for showing the mechanism: CFH 93.8 → 81.8 → 72.3% (then refuted
entirely), move-order containment 100 → 69.0%, waypoints 75 → 55.5%, and the
AA-004 negative control 0/2 → 2/16.

**Gate 3.** ~10 min, boots a server:

    ops/login_capped.sh 6G 2 .venv-jax/bin/python -m lanerl_jax.parity.gate3_first_divergence \
        --decisions 18000 --save lanerl_jax/runs/g3_streams.json

Always pass `--save`. The interesting question only appears at the end of the
run, and every follow-up should be answered from the run that produced the
finding rather than from a new one that may not reproduce it.

## 4. Guards worth running

    python -m lanerl_jax.parity.provenance          # what source a run actually executed
    ops/login_capped.sh 10G 3 .venv-jax/bin/python -m pytest lanerl_jax/parity/tests -q

The parity suite takes ~4 min. One test — `test_last_hit_gate` — is gate 3
itself and is EXPECTED to fail while gate 3 is open; it is a gate, not a
regression.

## 5. Two habits this project keeps re-learning

**Quote the tier with the rate.** Tier 1 re-injects server state every tick, so
it removes accumulation by construction. The champion scores 100.00% on every
Tier-1 field and still free-runs up to 930 u away from the server's champion at
the same game time (`METH-005`, `PATH-007`).

**A signature consistent with a mechanism is not evidence the mechanism
fired.** Two attributions were made from signatures in one day and both were
wrong, each refuted only by emitting the state and looking (`CFH-002`,
`GIVE-001`). If a claim rests on a proxy, say so in the ledger row, and prefer
adding a dump field over arguing.
