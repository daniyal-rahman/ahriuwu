#!/usr/bin/env bash
# Run this BEFORE any experiment. It refuses to be quiet.
#
# What it proves, in order, stopping at the first failure:
#   1. this node's view of the NFS export resolves (both /srv/nfs and /mnt/nfs)
#   2. the conda env has what the harness imports
#   3. the server binary exists AND is newer than every .cs that feeds it
#   4. the Roslyn-compiled Content scripts all compile -- they are NOT part of
#      any build, so a broken one degrades to SpellScriptEmpty and the ability
#      silently does nothing
#   5. every LANERL_BOT_CONFIG referenced by the bench configs resolves; a
#      missing one is ignored by LanerlConfig.cs:161 and the arm runs defaults
#   6. the mechanical checker (check_silent_failures.py) finds no HIGH finding
#   7. the fast test suites are green
#
# Usage:  lanerl_audit/preflight.sh [--skip-boot] [--skip-tests]
# Exit:   0 = safe to run an experiment. Non-zero = do not launch anything.

set -euo pipefail

# --- resolve everything from this script, never from a literal mount ---------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(dirname "$HERE")"
PROJECTS="$(dirname "$REPO")"
VENDOR="$PROJECTS/lanerl-vendor"
SERVER="$VENDOR/LoLServer"
BIN="$SERVER/GameServerConsole/bin/Release/net6.0"
CONTENT="$SERVER/Content"
PY="${PREFLIGHT_PYTHON:-/home/dani/miniconda3/envs/ml/bin/python}"

SKIP_BOOT=0
SKIP_TESTS=0
ALLOW_MISSING_CFG=0
for a in "$@"; do
    case "$a" in
        --skip-boot)  SKIP_BOOT=1 ;;
        --skip-tests) SKIP_TESTS=1 ;;
        # only for running an experiment that touches none of the dangling
        # configs; it downgrades step 5 to a warning and never to silence.
        --allow-missing-config) ALLOW_MISSING_CFG=1 ;;
        *) echo "unknown flag: $a" >&2; exit 2 ;;
    esac
done

RED=$'\033[1;31m'; GRN=$'\033[1;32m'; YEL=$'\033[1;33m'; OFF=$'\033[0m'
STEP=0
ok()   { echo "${GRN}  ok  ${OFF} $*"; }
warn() { echo "${YEL} warn ${OFF} $*"; }
die()  { echo; echo "${RED}=== PREFLIGHT FAILED ===${OFF}"; echo "${RED}$*${OFF}"; echo;
         echo "Do not launch the experiment. Fix the above first."; exit 1; }
step() { STEP=$((STEP+1)); echo; echo "── [$STEP] $* ─────────────────────────────"; }

echo "preflight   node=$(hostname)   repo=$REPO"
echo "            python=$PY"

# ---------------------------------------------------------------------------
step "NFS export resolves on this node"
# /srv/nfs and /mnt/nfs are the SAME export seen from two nodes. A Slurm job
# whose --output names the wrong one dies with exit 53 and writes NO file.
for m in /srv/nfs /mnt/nfs; do
    if [ -d "$m/projects" ]; then
        ok "$m/projects present"
    else
        # /srv/nfs is the real mount on danilogin; desktop has only /mnt/nfs
        # (and on danilogin /mnt/nfs is a symlink TO /srv/nfs). So /mnt/nfs is the
        # ONE portable literal. Missing /srv/nfs on desktop is expected -- warn,
        # do not block, or the gate is unusable on the node we actually train on.
        warn "$m/projects does not resolve on $(hostname) (expected on desktop).
     Anything hardcoding $m -- including '#SBATCH --output=$m/...' -- dies
     SILENTLY here. Use /mnt/nfs, which resolves on both nodes."
    fi
done
[ -w "$REPO" ] || die "$REPO is not writable by $(id -un) on this node"
ok "repo writable"
AVAIL_KB=$(df -Pk "$REPO" | awk 'NR==2{print $4}')
if [ "$AVAIL_KB" -lt 10485760 ]; then
    warn "only $((AVAIL_KB/1024/1024)) GiB free on $(df -P "$REPO" | awk 'NR==2{print $6}') -- a run that fills the disk fails like everything else"
else
    ok "$((AVAIL_KB/1024/1024)) GiB free"
fi

# ---------------------------------------------------------------------------
step "conda env has what the harness imports"
[ -x "$PY" ] || die "$PY does not exist on this node (conda envs are per-node)"
"$PY" - <<'EOF' || die "the ml env is missing a package the harness imports"
import importlib, importlib.util, sys
need = ["numpy", "pytest", "json", "statistics"]
optional = ["torch"]          # only lanerl_rl needs it
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:", missing); sys.exit(1)
print("  ok   python", sys.version.split()[0],
      "| " + " ".join(f"{m}={importlib.import_module(m).__version__}"
                      for m in need if hasattr(importlib.import_module(m), "__version__")))
for m in optional:
    if importlib.util.find_spec(m) is None:
        print(f"  warn {m} absent -- lanerl_rl will not import")
EOF
ok "env usable"

# ---------------------------------------------------------------------------
step "server binary exists and is NEWER than its sources"
[ -d "$VENDOR/GameServer" ] && warn "lanerl-vendor/GameServer exists and is NOT the tree
       anything runs (it has no GameServerLib/Lanerl). setup_server.sh builds it.
       The live tree is lanerl-vendor/LoLServer."
DLL="$BIN/GameServerLib.dll"
[ -f "$DLL" ] || die "$DLL missing -- nothing can run. Build LoLServer (not GameServer)."
ok "$(basename "$DLL") $(date -r "$DLL" '+%Y-%m-%d %H:%M')"
STALE=0
for sub in GameServerLib GameServerCore GameServerConsole; do
    while IFS= read -r f; do
        echo "${RED}  STALE${OFF} $f is newer than the dll"
        STALE=1
    done < <(find "$SERVER/$sub" -name '*.cs' -newer "$DLL" \
                  -not -path '*/obj/*' -not -path '*/bin/*' 2>/dev/null)
done
[ "$STALE" -eq 0 ] || die "C# sources are newer than the built dll.
     Every measurement you take now is of the OLD binary. Rebuild:
       DOTNET_ROOT=$VENDOR/dotnet $VENDOR/dotnet/dotnet build -c Release \\
         $SERVER/GameServer.sln 2>&1 | tail -20
     and CHECK the exit status -- do not grep for 'Build succeeded'."
ok "no .cs newer than the dll"
[ -x "$BIN/GameServerConsole" ] \
    && ok "apphost present (control_smoke.py invokes it directly)" \
    || warn "apphost $BIN/GameServerConsole missing; lanerl_rl/control_smoke.py will fail"

# ---------------------------------------------------------------------------
step "LANERL_BOT_CONFIG targets all resolve"
# LanerlConfig.cs:161 -- `if (!IsNullOrEmpty(path) && File.Exists(path))`. A path
# that does not resolve is IGNORED, with no warning, and the bot runs its
# built-in defaults. An A/B then compares the default bot against itself.
MISSING_CFG=0
while IFS= read -r cfg; do
    # bench configs now name their targets repo-relative (they must not carry a
    # mount point), so resolve those against $REPO before testing them -- matching
    # only absolute paths here would have made this step silently vacuous.
    case "$cfg" in /*) abs="$cfg" ;; *) abs="$REPO/$cfg" ;; esac
    if [ -e "$abs" ]; then
        ok "$cfg"
    else
        echo "${RED} MISS ${OFF} $cfg  ->  $abs"
        MISSING_CFG=$((MISSING_CFG+1))
    fi
done < <(grep -rhoE '"LANERL_BOT_CONFIG"[[:space:]]*:[[:space:]]*"[^"]+"' \
             --exclude-dir=out --exclude-dir=__pycache__ \
             "$REPO/lanerl_bot/bench" 2>/dev/null \
         | sed -E 's/.*:[[:space:]]*"([^"]+)"$/\1/' | sort -u)
# bench/out/ is EXCLUDED on purpose: it holds recorded RESULTS, which name the
# config each arm ran under as provenance. Several of those arms are retired and
# their configs no longer exist, so scanning results made this step fail forever
# and blocked every future experiment -- a gate that always fires is not a gate.
# Live experiment inputs live in bench/*.json and bench/configs/, which are scanned.
if [ "$MISSING_CFG" -gt 0 ]; then
    MSG="$MISSING_CFG bot-config path(s) do not resolve on this node.
     Those arms will run the DEFAULT bot and the sweep will report seed noise as
     a tuning result. Create the files, or fix LanerlConfig.Load() to throw."
    [ "$ALLOW_MISSING_CFG" -eq 1 ] && warn "$MSG" || die "$MSG"
fi

# ---------------------------------------------------------------------------
step "Roslyn Content scripts compile"
if [ "$SKIP_BOOT" -eq 1 ]; then
    warn "--skip-boot: NOT verified. A broken Content script is invisible until a
       spell silently no-ops."
else
    N_CS=$(find "$CONTENT" -name '*.cs' -not -path '*/obj/*' | wc -l)
    echo "       $N_CS scripts; booting the server once to make Roslyn compile them"
    PORT=$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
    BOOTLOG="$HERE/.preflight_boot.log"
    set +e
    (
        cd "$BIN" || exit 90
        DOTNET_ROOT="$VENDOR/dotnet" \
        LANERL_HEADLESS=1 LANERL_FREERUN=1 LANERL_TOPONLY=1 LANERL_BOT=none \
        LANERL_EXIT_AT=5000 \
        timeout 180 "$VENDOR/dotnet/dotnet" ./GameServerConsole.dll \
            --config "$REPO/lanerl_bot/configs/garen1v1_bot.json" --port "$PORT"
    ) > "$BOOTLOG" 2>&1
    BOOT_RC=$?
    set -e
    if grep -q "Script compilation error" "$BOOTLOG"; then
        echo
        grep -n "Script compilation error" "$BOOTLOG" | head -20
        die "A Content script failed to compile. It is NOT a build error:
     CSharpScriptEngine.Load() drops the broken file, Package.LoadScripts()
     returns true on SomeCompiled, and Spell.cs:149 binds SpellScriptEmpty --
     the ability silently does nothing for the whole experiment."
    fi
    if ! grep -q "Game is ready" "$BOOTLOG"; then
        tail -25 "$BOOTLOG"
        die "server never reached 'Game is ready' (rc=$BOOT_RC). Log: $BOOTLOG"
    fi
    ok "all $N_CS Content scripts compiled; server reached 'Game is ready'"
    NOSCRIPT=$(grep -c "Could not find script" "$BOOTLOG" || true)
    if [ "${NOSCRIPT:-0}" -gt 0 ]; then
        warn "$NOSCRIPT 'Could not find script' WARNs -- those abilities are no-ops.
       Garen's own are worth checking:"
        grep -o "Could not find script: [A-Za-z.]*Garen[A-Za-z]*" "$BOOTLOG" \
            | sort -u | sed 's/^/       /' | head -10 || true
    fi
    # stamp the successful compile so check_silent_failures.py can detect drift
    touch "$HERE/.content_compile_ok"
fi

# ---------------------------------------------------------------------------
step "mechanical silent-failure checker (HIGH only)"
CHECK_JSON="$HERE/.preflight_findings.json"
if "$PY" "$HERE/check_silent_failures.py" --severity high --quiet --json "$CHECK_JSON"; then
    ok "no HIGH findings"
else
    "$PY" - "$CHECK_JSON" <<'EOF'
import collections, json, sys
rows = [f for f in json.load(open(sys.argv[1])) if f["severity"] == "high"]
for check, n in collections.Counter(f["check"] for f in rows).most_common():
    print(f"       {n:4d}  {check}")
    for f in [x for x in rows if x["check"] == check][:3]:
        print(f"             {f['where']}")
EOF
    die "check_silent_failures.py reported HIGH findings.
     Full list:  $PY lanerl_audit/check_silent_failures.py
     Rationale:  lanerl_audit/FINDINGS.md"
fi

# ---------------------------------------------------------------------------
step "fast test suites"
if [ "$SKIP_TESTS" -eq 1 ]; then
    warn "--skip-tests: the suites were NOT run"
else
    # pyproject.toml's testpaths now covers all four dirs, so a bare `pytest`
    # does collect these -- but it also collects the @pytest.mark.slow suite
    # (real server boots), which does not belong in a fast preflight gate.
    # Named explicitly here so the -m "not slow" filter is guaranteed to
    # apply per-suite, not left to whatever the default marker expression
    # happens to be.
    for suite in lanerl_bot/tests lanerl_rl/tests lanerl_train/tests; do
        [ -d "$REPO/$suite" ] || continue
        echo "       $suite"
        if ( cd "$REPO" && LANERL_SKIP_SERVER_TESTS=1 \
             "$PY" -m pytest "$suite" -q -m "not slow" -p no:cacheprovider ) ; then
            ok "$suite green"
        else
            die "$suite is RED. A bare \`pytest\` at the repo root does not collect it
     (pyproject.toml testpaths=[\"tests\"]), which is why nobody noticed."
        fi
    done
fi

echo
echo "${GRN}=== PREFLIGHT PASSED on $(hostname) ===${OFF}"
echo "Reminder: server tests that SKIP are not tests that PASS."
echo "Run them explicitly with the build present:"
echo "  $PY -m pytest lanerl_bot/tests -q          # ~4 min, boots real games"
