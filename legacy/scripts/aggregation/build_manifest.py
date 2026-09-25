#!/usr/bin/env python3
"""
Build a pipeline manifest by walking Riot's Masters+ leaderboard and filtering
each player's recent ranked-solo matches to (champion, role, patch).

Default flow:
  1. GET league-v4 master/grandmaster/challenger leaderboards → summonerIds.
  2. (Optional) intersect with --otp-list (case-insensitive name match) so the
     final pool is just champion-mains rather than all Masters+.
  3. summonerId → puuid via summoner-v4.
  4. For each puuid: pull last N ranked-solo (queue=420) match-ids.
  5. For each match: filter (champion + role + version startswith patch) and
     emit a manifest entry.

Usage:
    python scripts/aggregation/build_manifest.py \
        --api-key RGAPI-... \
        --champion Garen \
        --role TOP \
        --patch 16.9 \
        --out manifest.json \
        [--otp-list otps.txt]      # one summoner name per line
        [--max-games 300] \
        [--max-players 80] \
        [--matches-per-player 50] \
        [--region na1]

Manifest format (consumed by scripts/aggregation/pipeline.py):

    {
      "name": "...",
      "matches": [
        {
          "match_id": "NA1_5554195441",
          "game_id":  "5554195441",
          "champion": "Garen",
          "garen_team": "blue",
          "garen_slot": 0,
          "duration": 1900,
          "version": "16.9.772.8292",
          "summoner_name": "...",
          "kda": "10/3/8"
        },
        ...
      ]
    }
"""
import argparse, json, os, socket, sys, time, urllib.request, urllib.error
from collections import Counter
from urllib.parse import quote

DEFAULT_REGION = "na1"
ROUTING_BY_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "kr": "asia", "jp1": "asia",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}

UA = "Mozilla/5.0 (replay-pipeline/manifest-builder)"


class RateLimiter:
    """Cheap rate limiter — Riot's dev key is 20 req/1s and 100 req/2min.
    We just enforce a min interval; if a 429 comes back we sleep the
    Retry-After. Good enough for the ~10k requests this builder makes."""
    def __init__(self, min_interval=0.06):  # 0.06s ~= 16 req/s
        self.min = min_interval
        self.last = 0.0
    def wait(self):
        dt = time.time() - self.last
        if dt < self.min:
            time.sleep(self.min - dt)
        self.last = time.time()


def _parse_retry_after(value):
    """Riot usually returns Retry-After as a seconds-int; HTTP also allows a
    date format. Default to 10s on anything unparseable."""
    if not value:
        return 10
    try:
        return int(value)
    except (ValueError, TypeError):
        return 10


def http_get(url, api_key, rl, retries=4):
    rl.wait()
    req = urllib.request.Request(url, headers={
        "X-Riot-Token": api_key, "User-Agent": UA, "Accept": "application/json",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = _parse_retry_after(e.headers.get("Retry-After")) + 1
                print(f"    429 rate-limited, sleeping {wait}s", flush=True)
                time.sleep(wait); continue
            if e.code >= 500:
                wait = 2 ** attempt
                print(f"    {e.code} transient, sleeping {wait}s (attempt {attempt+1}/{retries})",
                      flush=True)
                time.sleep(wait); continue
            return {"_error": f"HTTP {e.code}", "_url": url}
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            wait = 2 ** attempt
            print(f"    network error ({type(e).__name__}: {e}), sleeping {wait}s "
                  f"(attempt {attempt+1}/{retries})", flush=True)
            time.sleep(wait); continue
        except Exception as e:
            return {"_error": str(e), "_url": url}
    return {"_error": "max retries", "_url": url}


def fetch_masters_plus(region, api_key, rl):
    """Yield {tier, puuid, leaguePoints, wins, losses, rank}.

    NOTE: As of late-2024 the league-v4 API returns puuid directly and has
    DROPPED summonerId/summonerName from the response (Riot privacy update).
    Filter by puuid; resolve to display Riot-ID via account-v1/by-puuid only
    if you actually need the name."""
    base = f"https://{region}.api.riotgames.com/lol/league/v4"
    for tier_path, tier in (
        ("challengerleagues", "Challenger"),
        ("grandmasterleagues", "Grandmaster"),
        ("masterleagues", "Master"),
    ):
        url = f"{base}/{tier_path}/by-queue/RANKED_SOLO_5x5"
        d = http_get(url, api_key, rl)
        if not d or d.get("_error"):
            print(f"  WARN: {tier} fetch failed: {d}", flush=True); continue
        entries = d.get("entries", [])
        print(f"  {tier}: {len(entries)} players", flush=True)
        for e in entries:
            yield {"tier": tier, **{k: e.get(k) for k in
                ("puuid", "leaguePoints", "wins", "losses", "rank")}}


def puuid_for_riot_id(routing, game_name, tag_line, api_key, rl):
    """account-v1 forward lookup: Riot ID → puuid."""
    url = (f"https://{routing}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/"
           f"{quote(game_name)}/{quote(tag_line)}")
    d = http_get(url, api_key, rl)
    if not d or d.get("_error"): return None
    return d.get("puuid")


def load_seed_riot_ids(path):
    """Load curated `gameName#tagLine` lines (lines starting with # = comment,
    blank lines skipped). Returns list of (game_name, tag_line)."""
    out = []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "#" not in line:
                print(f"  WARN: seed line lacks #tag, skipping: {line!r}")
                continue
            name, tag = line.rsplit("#", 1)
            out.append((name.strip(), tag.strip()))
    return out


def match_ids_by_puuid(routing, puuid, api_key, rl, count=50, queue=420):
    url = (f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/"
           f"{puuid}/ids?queue={queue}&count={count}")
    d = http_get(url, api_key, rl)
    return d if isinstance(d, list) else []


def match_detail(routing, match_id, api_key, rl):
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return http_get(url, api_key, rl)


def load_otp_filter(path):
    """Load a list of puuids (one per line, # for comments).

    NOTE: As of the 2024 Riot privacy update, league-v4 no longer exposes
    summonerName, so name-based OTP intersection is broken. If you have a
    list of curated player puuids (e.g. resolved from u.gg names via the
    account-by-riot-id endpoint), pass the puuid file here."""
    if not path: return None
    puuids = set()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                puuids.add(line)
    return puuids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--champion", required=True, help="Internal champion name (e.g. Garen, Belveth)")
    ap.add_argument("--role", required=True, help="TOP / JUNGLE / MIDDLE / BOTTOM / UTILITY")
    ap.add_argument("--patch", required=True, help="version prefix, e.g. 16.9")
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--otp-list", default=None,
                    help="optional file of puuids (one per line) to intersect against the leaderboard")
    ap.add_argument("--seed-riot-ids", default=None,
                    help="file of curated `gameName#tagLine` Riot IDs (one per line). When provided, "
                         "SKIPS the leaderboard pull and walks ONLY these players' match histories. "
                         "Use this for high-signal champion-mains lists from op.gg / u.gg.")
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--max-players", type=int, default=80,
                    help="cap how many Masters+ players we walk (after OTP filter)")
    ap.add_argument("--matches-per-player", type=int, default=50,
                    help="recent ranked-solo matches to pull per player")
    ap.add_argument("--duration-buffer", type=int, default=120,
                    help="seconds added to each match's actual gameDuration when emitting the "
                         "pipeline `duration` cap. Default 120s of headroom.")
    ap.add_argument("--duration-cap", type=int, default=1900,
                    help="absolute upper bound on the emitted `duration` in seconds. Default 1900 "
                         "(~32min — covers the vast majority of LoL games; later state is less "
                         "critical for ML training and shorter caps protect against runaway pass1).")
    ap.add_argument("--min-game-duration", type=int, default=600,
                    help="drop matches whose actual gameDuration is below this (seconds). Default "
                         "600s — filters out remakes (early /remake before 4min) and very short "
                         "early-surrenders that have too little training signal.")
    args = ap.parse_args()

    routing = ROUTING_BY_REGION.get(args.region.lower())
    if not routing:
        print(f"FATAL: unknown region {args.region!r}", file=sys.stderr); sys.exit(1)

    rl = RateLimiter(min_interval=0.07)

    # Two paths into the puuid list:
    #   A) curated seed Riot IDs (high-signal, skips leaderboard entirely)
    #   B) full Masters+ leaderboard, optionally puuid-OTP-filtered
    puuids = []  # list of (puuid, metadata-dict)
    if args.seed_riot_ids:
        seeds = load_seed_riot_ids(args.seed_riot_ids)
        print(f"\n[1] Resolving {len(seeds)} curated Riot IDs from {args.seed_riot_ids}...")
        for i, (name, tag) in enumerate(seeds, 1):
            puuid = puuid_for_riot_id(routing, name, tag, args.api_key, rl)
            if puuid:
                puuids.append((puuid, {"riot_id": f"{name}#{tag}", "tier": "curated",
                                       "leaguePoints": 0}))
                print(f"  [{i}/{len(seeds)}] {name}#{tag}  →  {puuid[:16]}…", flush=True)
            else:
                print(f"  [{i}/{len(seeds)}] {name}#{tag}  →  RESOLVE FAILED", flush=True)
        print(f"\n[2] {len(puuids)}/{len(seeds)} resolved to puuids")
    else:
        otp_filter = load_otp_filter(args.otp_list)
        if otp_filter:
            print(f"loaded {len(otp_filter)} OTP puuids from {args.otp_list}")

        print(f"\n[1] Fetching Masters+ leaderboard ({args.region})...")
        players = list(fetch_masters_plus(args.region, args.api_key, rl))
        print(f"  total Masters+ players: {len(players)}")

        if otp_filter:
            before = len(players)
            players = [p for p in players if p.get("puuid") in otp_filter]
            print(f"  after OTP puuid intersection: {len(players)} (was {before})")

        # rank by LP descending
        players.sort(key=lambda p: -(p.get("leaguePoints") or 0))
        if args.max_players and len(players) > args.max_players:
            print(f"  capping to top {args.max_players} by LP")
            players = players[:args.max_players]

        # league-v4 already includes puuid — no extra resolution needed.
        puuids = [(p["puuid"], p) for p in players if p.get("puuid")]
        print(f"\n[2] {len(puuids)}/{len(players)} players have puuid (rest dropped)")

    print(f"\n[3] Walking match histories — looking for {args.champion} {args.role} on {args.patch}.x...")
    seen_matches = set()
    out_matches = []
    rejects = Counter()  # patch / champion / role / short / detail_err / no_ids

    def _write(progress_pct=None):
        """Atomic snapshot — write to .tmp then os.replace. Called incrementally
        so a killed/crashed run still has a valid partial JSON on disk."""
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "name": f"{args.champion}_{args.role}_masters_{args.patch.replace('.', '_')}",
                "champion": args.champion, "role": args.role, "patch": args.patch,
                "region": args.region,
                "n_players_targeted": len(puuids),
                "_partial": progress_pct is not None and progress_pct < 100,
                "_progress_pct": progress_pct,
                "matches": out_matches,
            }, f, indent=2)
        os.replace(tmp, args.out)

    _write(progress_pct=0)
    print(f"  wrote empty placeholder to {args.out} (will be updated incrementally)", flush=True)

    for i, (puuid, p) in enumerate(puuids, 1):
        if len(out_matches) >= args.max_games: break
        label = p.get("riot_id") or f"{p.get('tier','?')[:4]} {p.get('leaguePoints',0)}LP"
        ids = match_ids_by_puuid(routing, puuid, args.api_key, rl, count=args.matches_per_player)
        if not ids:
            rejects["no_ids_returned"] += 1
            print(f"  [{i}/{len(puuids)}] {label:30s} puuid={puuid[:14]}…  NO MATCHES "
                  f"(empty match-id list — dev/network or genuinely 0)", flush=True)
            continue
        kept = 0
        for mid in ids:
            if mid in seen_matches: continue
            seen_matches.add(mid)
            d = match_detail(routing, mid, args.api_key, rl)
            if not d or d.get("_error"):
                rejects["detail_error"] += 1
                continue
            info = d.get("info", {})
            ver = info.get("gameVersion", "")
            if not ver.startswith(args.patch):
                rejects["wrong_patch"] += 1
                continue
            matched_part = False
            for part in info.get("participants", []):
                if part.get("championName") != args.champion:
                    continue
                if (part.get("teamPosition") or "").upper() != args.role.upper():
                    rejects["wrong_role"] += 1
                    continue
                matched_part = True
                team = "blue" if part.get("teamId") == 100 else "red"
                idx = info["participants"].index(part)
                slot = sum(1 for q in info["participants"]
                           if q.get("teamId") == part.get("teamId")
                           and info["participants"].index(q) < idx)
                riot_id = (
                    f"{part.get('riotIdGameName')}#{part.get('riotIdTagline')}"
                    if part.get('riotIdGameName') else None
                )
                actual_dur = info.get("gameDuration", 1800)
                if actual_dur < args.min_game_duration:
                    rejects["too_short"] += 1
                    break  # drop this match (remake / early ff)
                duration = min(args.duration_cap, actual_dur + args.duration_buffer)
                out_matches.append({
                    "match_id": mid,
                    "game_id":  mid.split("_")[-1],
                    "platform": args.region,
                    "champion": args.champion,
                    "garen_team": team,    # legacy key — pipeline.py also reads "team"
                    "garen_slot": slot,    # legacy key — pipeline.py also reads "slot"
                    "team": team,
                    "slot": slot,
                    "duration": duration,
                    "version": ver,
                    "summoner_name": riot_id,
                    "kda": f"{part['kills']}/{part['deaths']}/{part['assists']}",
                    "garen_win": part.get("win", False),
                    "game_duration": actual_dur,
                })
                kept += 1
                if len(out_matches) >= args.max_games: break
            if not matched_part and info.get("participants"):
                # match included no participant on (champion, anything-role) — not
                # interesting to count separately; "no_champion" check would be
                # redundant since we already filter by championName.
                pass

        print(f"  [{i}/{len(puuids)}] {label:30s} puuid={puuid[:14]}…  kept {kept}  "
              f"(running total {len(out_matches)})", flush=True)
        if kept:
            _write(progress_pct=int(100 * len(out_matches) / max(1, args.max_games)))

    _write(progress_pct=100)
    print(f"\n[4] Wrote {len(out_matches)} matches to {args.out}")
    if rejects:
        print(f"\n[5] Rejection summary (matches scanned but not kept):")
        for reason, n in rejects.most_common():
            print(f"    {reason:24s}  {n}")


if __name__ == "__main__":
    main()
