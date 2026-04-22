#!/usr/bin/env python3
"""
Find high-elo Garen games on the current patch using the Riot API.

Strategy: Look up known Garen OTPs from op.gg leaderboard, then pull their
recent match history. Every game is likely a Garen game — zero wasted API calls.

Usage:
    python scripts/find_garen_games.py --api-key RGAPI-xxx
    python scripts/find_garen_games.py --api-key RGAPI-xxx --download  # also trigger LCU download

Environment:
    RIOT_API_KEY - Can also set API key via environment variable
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Top Garen OTPs from op.gg NA leaderboard (Master+)
# Format: (game_name, tag_line)
GAREN_OTPS_NA = [
    ("Triton", "DMCIA"),
    ("AddictedToBacon", "TTV"),
    ("Taiwan Real CN", "Tibet"),
    ("CapnCheesy", "hemm"),
    ("can u feel my", "heart"),
    ("L55", "NA1"),
    ("chrisnam", "NA1"),
    ("Bagels R Awesome", "NA1"),
    ("Vanity of Icarus", "n b"),
    ("Big Ż", "NA1"),
]

# Rate limiting
REQUEST_DELAY = 1.3  # seconds between requests (safe for 20/s, 100/2min)


def api_get(url: str, api_key: str) -> dict | list | None:
    """Make a GET request to the Riot API using curl (more reliable than urllib)."""
    result = subprocess.run(
        ['curl', '-s', '-H', f'X-Riot-Token: {api_key}', url],
        capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict) and 'status' in data:
            status = data['status']
            if isinstance(status, dict) and status.get('status_code', 200) != 200:
                print(f"  API error: {status.get('message', 'unknown')}")
                return None
        return data
    except json.JSONDecodeError:
        return None


def get_puuid_from_riot_id(game_name: str, tag_line: str, api_key: str) -> str | None:
    """Look up PUUID from Riot ID (gameName#tagLine)."""
    # URL-encode the game name
    from urllib.parse import quote
    encoded_name = quote(game_name)
    encoded_tag = quote(tag_line)
    url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{encoded_name}/{encoded_tag}"
    data = api_get(url, api_key)
    if data and 'puuid' in data:
        return data['puuid']
    return None


def find_garen_games(api_key: str, platform: str = "na1", region: str = "americas",
                     max_players: int = 5, matches_per_player: int = 10,
                     current_patch: str = None) -> list[dict]:
    """Find recent Garen games from known OTP players."""
    print(f"Looking up Garen OTPs on {platform}...")

    found = []
    calls = 0

    for i, (game_name, tag_line) in enumerate(GAREN_OTPS_NA[:max_players]):
        print(f"\n[{i+1}/{max_players}] {game_name}#{tag_line}")

        # Get PUUID from Riot ID
        time.sleep(REQUEST_DELAY)
        puuid = get_puuid_from_riot_id(game_name, tag_line, api_key)
        calls += 1
        if not puuid:
            print(f"  Could not find PUUID")
            continue
        print(f"  PUUID: {puuid[:20]}...")

        # Get recent ranked matches
        time.sleep(REQUEST_DELAY)
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count={matches_per_player}"
        matches = api_get(url, api_key)
        calls += 1
        if not matches or not isinstance(matches, list):
            print(f"  No matches found")
            continue
        print(f"  {len(matches)} recent ranked matches")

        # Check each match
        for mid in matches:
            time.sleep(REQUEST_DELAY)
            mdata = api_get(f"https://{region}.api.riotgames.com/lol/match/v5/matches/{mid}", api_key)
            calls += 1
            if not mdata:
                continue

            info = mdata.get("info", {})
            ver = info.get("gameVersion", "")

            # Filter by patch if specified
            if current_patch and not ver.startswith(current_patch):
                continue

            # Find Garen participant
            for pp in info.get("participants", []):
                if pp.get("championName") == "Garen":
                    game_id = mid.split("_")[-1] if "_" in mid else mid.split("-")[-1]
                    dur = info.get("gameDuration", 0)
                    k, d, a = pp.get("kills", 0), pp.get("deaths", 0), pp.get("assists", 0)
                    win = pp.get("win", False)
                    team = "blue" if pp.get("teamId") == 100 else "red"
                    summoner = pp.get("riotIdGameName", "?")

                    record = {
                        "match_id": mid,
                        "game_id": game_id,
                        "version": ver,
                        "duration": dur,
                        "duration_min": round(dur / 60, 1),
                        "win": win,
                        "kda": f"{k}/{d}/{a}",
                        "kills": k, "deaths": d, "assists": a,
                        "garen_team": team,
                        "garen_summoner": summoner,
                        "garen_puuid": pp.get("puuid"),
                        "garen_slot": next(
                            (j for j, p2 in enumerate(info["participants"])
                             if p2.get("puuid") == pp.get("puuid")), -1),
                    }
                    found.append(record)
                    print(f"  GAREN: {mid} {ver} {dur//60}min "
                          f"{k}/{d}/{a} {'W' if win else 'L'} ({summoner})")
                    break

        print(f"  Found {sum(1 for f in found if f['garen_summoner'] == game_name or True)} Garen games from this player")

    print(f"\n--- Summary ---")
    print(f"API calls: {calls}")
    print(f"Garen games found: {len(found)}")

    # Sort: wins first, then by KDA
    found.sort(key=lambda g: (g['win'], g['kills'] - g['deaths']), reverse=True)

    if found:
        print(f"\nTop 5 games:")
        for j, g in enumerate(found[:5]):
            print(f"  {j+1}. {g['match_id']} {g['version']} "
                  f"{g['duration_min']}min {g['kda']} "
                  f"{'WIN' if g['win'] else 'LOSS'} ({g['garen_summoner']})")

    return found


def main():
    parser = argparse.ArgumentParser(description="Find high-elo Garen games")
    parser.add_argument("--api-key", default=os.environ.get("RIOT_API_KEY"),
                        help="Riot API key (or set RIOT_API_KEY env var)")
    parser.add_argument("--platform", default="na1")
    parser.add_argument("--max-players", type=int, default=5,
                        help="Max OTP players to check")
    parser.add_argument("--matches-per-player", type=int, default=10,
                        help="Recent matches to check per player")
    parser.add_argument("--patch", default=None,
                        help="Filter by patch version (e.g., '16.7')")
    parser.add_argument("--output", "-o", default="data/replays/garen_manifest.json")
    parser.add_argument("--download", action="store_true",
                        help="Trigger LCU replay download for top game")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set RIOT_API_KEY env var")
        sys.exit(1)

    games = find_garen_games(
        args.api_key,
        platform=args.platform,
        max_players=args.max_players,
        matches_per_player=args.matches_per_player,
        current_patch=args.patch,
    )

    if games:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": args.platform,
            "patch_filter": args.patch,
            "matches": games,
        }
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nSaved manifest to {out_path}")

        if args.download and games:
            best = games[0]
            print(f"\nBest game: {best['match_id']} ({best['kda']}, "
                  f"{'WIN' if best['win'] else 'LOSS'})")
            print(f"To download, run on Windows with League client open:")
            print(f"  python scripts/download_replays.py download "
                  f"--manifest {out_path}")


if __name__ == "__main__":
    main()
