#!/usr/bin/env python3
"""Find and download high-elo Garen replays using Riot Games API + LCU API.

Two modes:
  find    - Query Riot API for high-elo Garen games, save match list to manifest
  download - Download .rofl files from manifest via League client LCU API (Windows only)

Usage:
    # Step 1: Find Garen matches (runs anywhere with internet)
    python scripts/download_replays.py find \
        --api-key RGAPI-xxx \
        --region na1 \
        --output data/replays/manifest.json

    # Step 2: Download replays (runs on Windows with League client open)
    python scripts/download_replays.py download \
        --manifest data/replays/manifest.json \
        --output-dir "C:\\Users\\daniz\\Documents\\League of Legends\\Replays"

    # Check manifest stats
    python scripts/download_replays.py stats --manifest data/replays/manifest.json

Environment:
    RIOT_API_KEY - Can also set API key via environment variable

Riot API key: https://developer.riotgames.com/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# Platform → regional routing for match-v5
PLATFORM_TO_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "ru": "europe", "tr1": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}

# Garen champion ID
GAREN_CHAMPION_ID = 86
GAREN_CHAMPION_NAME = "Garen"

# Rate limiting — dev key: 20 req/s, 100 req/2min
REQUEST_DELAY = 1.5  # seconds between requests (conservative for 100/2min)


def api_get(url: str, api_key: str, retries: int = 5) -> dict | list:
    """Make a GET request to the Riot API with rate limit handling."""
    for attempt in range(retries):
        try:
            req = Request(url, headers={
                "X-Riot-Token": api_key,
                "User-Agent": "ahriuwu-replay-finder/1.0",
                "Accept": "application/json",
            })
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            time.sleep(REQUEST_DELAY)
            return data
        except HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 30))
                print(f"  Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after + 1)
            elif e.code == 401:
                print(f"  HTTP 401: API key is invalid or expired.")
                print(f"  Regenerate at https://developer.riotgames.com/")
                sys.exit(1)
            elif e.code == 403:
                print(f"  HTTP 403: Forbidden. Key may lack permissions or be expired.")
                print(f"  Regenerate at https://developer.riotgames.com/")
                sys.exit(1)
            elif e.code == 404:
                return None
            else:
                print(f"  HTTP {e.code}: {e.reason} (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        except (ConnectionResetError, ConnectionError, OSError, TimeoutError) as e:
            wait = 5 * (attempt + 1)
            print(f"  Connection error: {e} — retrying in {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    return None


def get_high_elo_players(platform: str, api_key: str) -> list[dict]:
    """Get all Challenger + Grandmaster + Master players for a platform."""
    base = f"https://{platform}.api.riotgames.com"
    players = []

    for tier in ["challengerleagues", "grandmasterleagues", "masterleagues"]:
        url = f"{base}/lol/league/v4/{tier}/by-queue/RANKED_SOLO_5x5"
        print(f"  Fetching {tier}...")
        data = api_get(url, api_key)
        if data and "entries" in data:
            for entry in data["entries"]:
                entry["tier"] = data.get("tier", tier.replace("leagues", "").upper())
            players.extend(data["entries"])
            print(f"    Found {len(data['entries'])} players")

    print(f"  Total high-elo players: {len(players)}")
    return players


def get_puuid(platform: str, summoner_id: str, api_key: str) -> str | None:
    """Get PUUID from encrypted summoner ID."""
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    data = api_get(url, api_key)
    if data:
        return data.get("puuid")
    return None


def get_match_ids(region: str, puuid: str, api_key: str,
                  count: int = 100, queue: int = 420) -> list[str]:
    """Get recent ranked match IDs for a player."""
    base = f"https://{region}.api.riotgames.com"
    url = f"{base}/lol/match/v5/matches/by-puuid/{puuid}/ids?queue={queue}&count={count}"
    data = api_get(url, api_key)
    return data if data else []


def get_match_detail(region: str, match_id: str, api_key: str) -> dict | None:
    """Get match details to check which champion was played."""
    base = f"https://{region}.api.riotgames.com"
    url = f"{base}/lol/match/v5/matches/{match_id}"
    return api_get(url, api_key)


def find_garen_matches(platform: str, api_key: str, max_players: int = 50,
                       matches_per_player: int = 100) -> list[dict]:
    """Find high-elo matches where Garen was played.

    Returns list of match records with game ID, player info, etc.
    """
    region = PLATFORM_TO_REGION.get(platform)
    if not region:
        print(f"Error: unknown platform '{platform}'. Known: {list(PLATFORM_TO_REGION)}")
        sys.exit(1)

    print(f"Platform: {platform}, Region: {region}")

    # Get high-elo players
    players = get_high_elo_players(platform, api_key)

    # Sort by LP descending (highest elo first)
    players.sort(key=lambda p: p.get("leaguePoints", 0), reverse=True)

    if max_players > 0:
        players = players[:max_players]
        print(f"Checking top {len(players)} players...")

    garen_matches = []
    seen_match_ids = set()
    players_checked = 0
    players_with_garen = 0

    for i, player in enumerate(players):
        puuid = player.get("puuid")
        tier = player.get("tier", "?")
        lp = player.get("leaguePoints", 0)

        if not puuid:
            # Fallback: old API format with summonerId
            summoner_id = player.get("summonerId")
            if summoner_id:
                puuid = get_puuid(platform, summoner_id, api_key)
            if not puuid:
                print(f"\n[{i+1}/{len(players)}] ??? ({tier} {lp}LP) — no PUUID, skipping")
                continue

        print(f"\n[{i+1}/{len(players)}] {tier} {lp}LP")

        # Get recent matches
        match_ids = get_match_ids(region, puuid, api_key, count=matches_per_player)
        print(f"  Found {len(match_ids)} recent ranked matches")

        player_garen_count = 0
        for mid in match_ids:
            if mid in seen_match_ids:
                continue
            seen_match_ids.add(mid)

            match = get_match_detail(region, mid, api_key)
            if not match:
                continue

            info = match.get("info", {})
            participants = info.get("participants", [])

            # Check if any participant played Garen
            for p in participants:
                if p.get("championName") == GAREN_CHAMPION_NAME:
                    # Extract game ID from match ID (e.g., "NA1-5489605032" -> 5489605032)
                    game_id = mid.split("_")[-1] if "_" in mid else mid.split("-")[-1]
                    record = {
                        "match_id": mid,
                        "game_id": game_id,
                        "platform": platform,
                        "garen_puuid": p.get("puuid"),
                        "garen_summoner": p.get("summonerName", "?"),
                        "garen_team": "blue" if p.get("teamId") == 100 else "red",
                        "garen_win": p.get("win", False),
                        "game_duration": info.get("gameDuration", 0),
                        "game_version": info.get("gameVersion", "?"),
                        "queue_id": info.get("queueId", 0),
                        "timestamp": info.get("gameStartTimestamp", 0),
                    }
                    garen_matches.append(record)
                    player_garen_count += 1
                    print(f"    GAREN found in {mid} "
                          f"({p.get('summonerName')}, "
                          f"{'W' if p.get('win') else 'L'}, "
                          f"{info.get('gameDuration', 0)//60}min)")
                    break

        if player_garen_count > 0:
            players_with_garen += 1
        players_checked += 1

        print(f"  Garen games: {player_garen_count}")

    print(f"\n--- Summary ---")
    print(f"Players checked: {players_checked}")
    print(f"Players with Garen games: {players_with_garen}")
    print(f"Total Garen matches found: {len(garen_matches)}")

    return garen_matches


def read_lcu_lockfile() -> tuple[int, str] | None:
    """Read the League client lockfile to get port and auth token.

    Returns (port, auth_token) or None if not found.
    """
    # Standard lockfile locations
    lockfile_paths = [
        Path(r"C:\Riot Games\League of Legends\lockfile"),
        Path(os.path.expanduser("~")) / "Riot Games" / "League of Legends" / "lockfile",
    ]

    for path in lockfile_paths:
        if path.exists():
            content = path.read_text().strip()
            parts = content.split(":")
            if len(parts) >= 5:
                port = int(parts[2])
                token = parts[3]
                return (port, token)

    return None


def download_replays(manifest_path: str, output_dir: str):
    """Download replay files using the LCU API (requires League client running on Windows)."""
    import base64
    import ssl

    with open(manifest_path) as f:
        manifest = json.load(f)

    matches = manifest.get("matches", [])
    print(f"Manifest has {len(matches)} Garen matches")

    # Filter out already downloaded
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing = {f.stem for f in output_path.glob("*.rofl")}

    to_download = []
    for m in matches:
        game_id = m["game_id"]
        platform = m["platform"].upper()
        filename = f"{platform}-{game_id}"
        if filename not in existing:
            to_download.append(m)

    print(f"Already downloaded: {len(matches) - len(to_download)}")
    print(f"To download: {len(to_download)}")

    if not to_download:
        print("Nothing to download!")
        return

    # Read LCU lockfile
    lcu = read_lcu_lockfile()
    if not lcu:
        print("Error: cannot find League client lockfile.")
        print("Make sure the League client is running on this machine.")
        print("Lockfile expected at: C:\\Riot Games\\League of Legends\\lockfile")
        sys.exit(1)

    port, token = lcu
    auth = base64.b64encode(f"riot:{token}".encode()).decode()

    # Create SSL context that skips verification (LCU uses self-signed cert)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"LCU running on port {port}")

    downloaded = 0
    errors = 0

    for i, m in enumerate(to_download):
        game_id = m["game_id"]
        match_id = m["match_id"]
        print(f"\n[{i+1}/{len(to_download)}] Downloading {match_id} (game {game_id})...")

        url = f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{game_id}/download"
        body = json.dumps({"componentType": "replay"}).encode()
        req = Request(url, method="POST", data=body, headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json",
        })

        try:
            with urlopen(req, context=ctx) as resp:
                resp_body = resp.read()
                if resp_body:
                    result = json.loads(resp_body)
                    print(f"  Download initiated: {result}")
                else:
                    print(f"  Download initiated (204 No Content)")
                downloaded += 1
        except HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"  Error {e.code}: {body}")
            if e.code == 404:
                print(f"  Replay may have expired (replays expire each patch)")
            errors += 1
        except Exception as e:
            print(f"  Error: {e}")
            errors += 1

        time.sleep(2)  # Don't spam the client

    print(f"\n--- Download Summary ---")
    print(f"Initiated: {downloaded}")
    print(f"Errors: {errors}")
    print(f"Note: Downloads are async. Check {output_dir} for completed .rofl files.")


def print_stats(manifest_path: str):
    """Print stats about a manifest file."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    matches = manifest.get("matches", [])
    print(f"Manifest: {manifest_path}")
    print(f"Platform: {manifest.get('platform', '?')}")
    print(f"Created: {manifest.get('created', '?')}")
    print(f"Total matches: {len(matches)}")

    if not matches:
        return

    wins = sum(1 for m in matches if m.get("garen_win"))
    blue = sum(1 for m in matches if m.get("garen_team") == "blue")
    durations = [m.get("game_duration", 0) for m in matches]
    total_hours = sum(durations) / 3600

    print(f"Win rate: {wins}/{len(matches)} ({100*wins/len(matches):.1f}%)")
    print(f"Blue side: {blue}, Red side: {len(matches) - blue}")
    print(f"Avg duration: {sum(durations)/len(durations)/60:.1f} min")
    print(f"Total gameplay: {total_hours:.1f} hours")

    # Per-version breakdown
    versions = {}
    for m in matches:
        v = m.get("game_version", "?").rsplit(".", 1)[0]
        versions[v] = versions.get(v, 0) + 1
    print(f"Versions: {dict(sorted(versions.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Find and download high-elo Garen replays")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find command
    find_parser = subparsers.add_parser("find", help="Find Garen matches via Riot API")
    find_parser.add_argument("--api-key", default=os.environ.get("RIOT_API_KEY"),
                             help="Riot API key (or set RIOT_API_KEY env var)")
    find_parser.add_argument("--platform", default="na1",
                             help="Platform (default: na1)")
    find_parser.add_argument("--max-players", type=int, default=50,
                             help="Max players to check (default: 50)")
    find_parser.add_argument("--matches-per-player", type=int, default=100,
                             help="Recent matches to check per player (default: 100)")
    find_parser.add_argument("--output", "-o", default="data/replays/manifest.json",
                             help="Output manifest path")

    # download command
    dl_parser = subparsers.add_parser("download", help="Download replays via LCU API")
    dl_parser.add_argument("--manifest", required=True, help="Manifest file from 'find'")
    dl_parser.add_argument("--output-dir",
                           default=r"C:\Users\daniz\Documents\League of Legends\Replays",
                           help="Replay output directory")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show manifest stats")
    stats_parser.add_argument("--manifest", required=True, help="Manifest file")

    args = parser.parse_args()

    if args.command == "find":
        if not args.api_key:
            print("Error: provide --api-key or set RIOT_API_KEY env var")
            print("Get a key at: https://developer.riotgames.com/")
            sys.exit(1)

        matches = find_garen_matches(
            args.platform, args.api_key,
            max_players=args.max_players,
            matches_per_player=args.matches_per_player,
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "platform": args.platform,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_players": args.max_players,
            "matches": matches,
        }
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nSaved manifest to {out_path}")

    elif args.command == "download":
        download_replays(args.manifest, args.output_dir)

    elif args.command == "stats":
        print_stats(args.manifest)


if __name__ == "__main__":
    main()
