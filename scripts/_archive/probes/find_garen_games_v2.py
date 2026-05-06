#!/usr/bin/env python3
"""
Find 300+ high-elo Garen games from top OTPs across NA/KR/EUW via Riot API.

Strategy: op.gg leaderboard gives us Garen one-tricks per region.
For each, pull their last 50 ranked games.
Filter for Garen + current patch.
110 players × ~5 Garen games = ~300-450 candidates.

Usage:
    python scripts/find_garen_games_v2.py --api-key RGAPI-xxx
    python scripts/find_garen_games_v2.py --api-key RGAPI-xxx --regions na kr euw
    python scripts/find_garen_games_v2.py --api-key RGAPI-xxx --regions kr euw --merge data/replays/garen_manifest_200.json

Environment:
    RIOT_API_KEY - Can also set API key via environment variable

NOTE: Riot dev API keys expire every 24 hours. Regenerate at:
    https://developer.riotgames.com/

NOTE: Must include User-Agent header or Cloudflare blocks with error 1010.

Region routing for Riot API:
    NA players → americas.api.riotgames.com (match routing: americas)
    KR players → asia.api.riotgames.com (match routing: asia)
    EUW players → europe.api.riotgames.com (match routing: europe)
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import quote
from pathlib import Path

# ── OTP Lists from op.gg leaderboards (Diamond 2+, April 2026) ──

# Source: https://www.op.gg/leaderboards/champions/garen
GAREN_OTPS_NA = [
    ("Triton","DMCIA"),("Darkphase","NA1"),("AddictedToBacon","TTV"),
    ("Taiwan Real CN","Tibet"),("CapnCheesy","hemm"),("can u feel my","heart"),
    ("L55","NA1"),("chrisnam","NA1"),("Bagels R Awesome","NA1"),
    ("Vanity of Icarus","n b"),("Big Ż","NA1"),("Freaky Phil","FEET"),
    ("Truenaux","NA1"),("saltycandice","5097"),("Sond","NA1"),
    ("LocalTh0t6213","NA1"),("Barak","Cool"),("mcmaster2","NA1"),
    ("deer12","NA1"),("SUPERSOAKER66","NA1"),("Gromp Rider","SEJ"),
    ("Emerphish","brick"),("thesourav","NA1"),("hpg","o7 o7"),
    ("KuzcoTheAdmiral","NA1"),("DaYeetGirl","7599"),("DJChong","NA1"),
    ("PEYZ","EZEZ"),("1 800 DEMACIA","riste"),("sounding fetish","NA2"),
    ("GEEK SMASHER","TRUMP"),("GooGoo GaGaren","987"),("BBC Vitamin","BBC"),
    ("OniiChan","99999"),("Blaym","NA1"),("Dog Goes Bonk","NA1"),
    ("Soloalexo1234","3422"),("King Plebus","123"),("Medium Bui","NA1"),
    ("Fred Jones","NA1"),("chinese soldier","ccp"),("Screelix","SpinR"),
    ("Ill Be Your Byul","NA1"),("Sterick","Garen"),("poo poo master 1","NA1"),
    ("Deeti","3191"),("Turn off system","NA1"),("Aberrant Demon","Demon"),
    ("dainright","b n"),
]

# Source: https://www.op.gg/leaderboards/champions/garen?region=kr
GAREN_OTPS_KR = [
    ("불기둥가렌맨","강한남자"),("짐승 다운","KR1"),("풍자 남친","9743"),("김성근","hdy"),
    ("W 안 찍는 가렌","GAREN"),("전격전","121"),("황가렌","KR1"),("가 렌","KR935"),
    ("참새우깡","KR1"),("gn z11","KR1"),("따분함은 끝났다","KR1"),("Gauni","KR1"),
    ("데마시아","000"),("비 디","qlel"),("책임없는쾌락","KR11"),("쭈쭈미","KR1"),
    ("가 렌","17900"),("Garen Crownguard","TOP"),("가붕이로마스터","파이팅"),
    ("원초의흑색느와르","KR1"),("소년가장탑솔러","KR4"),("버블리","Korea"),
    ("슈마허","KR4"),("담유이","KR8"),("욱네방네","동 네"),("Kuncle Duster","KSH"),
    ("NONAME","6419"),("04탑","KR1"),("박상후 가렌","first"),("disabled guy","KR1"),
]

# Source: https://www.op.gg/leaderboards/champions/garen?region=euw
GAREN_OTPS_EUW = [
    ("GothicLogic","EUW"),("Top Tier GarenXD","Garen"),("Palco Granko","split"),
    ("CHINESE PALCO","NA1"),("Yorweak","EUW"),("Druski","TOP"),("Garenphile","1234"),
    ("William Wallace","EUNE1"),("Pablo Escobars","12345"),("Elolesio","SPIN"),
    ("La Pain De Garen","garen"),("RX Pada","GAREN"),("Mohnstrudel","Tasty"),
    ("Guigui2326","EUW"),("LeGardienDuLow","EUW"),("Yshuro13","EUW"),("AvionPigeon","9044"),
    ("dyspraxicprodigy","EUW"),("xSuits","EUW"),("Mush RED","RANK1"),("No name sorry","EUW"),
    ("Coci","EUW"),("Arthas Menethil","immo"),("Marga38i","EUW"),("QuentinTalentino","QTtop"),
    ("Misthorm","9991"),("Crowno","030"),("wenRAR","EUW"),("Wywern","EUWW"),("mecarow","EUW"),
]

REGION_CONFIG = {
    "na":  {"otps": GAREN_OTPS_NA,  "routing": "americas", "platform": "na1"},
    "kr":  {"otps": GAREN_OTPS_KR,  "routing": "asia",     "platform": "kr"},
    "euw": {"otps": GAREN_OTPS_EUW, "routing": "europe",   "platform": "euw1"},
}

REQUEST_DELAY = 1.3  # safe for 20/s, 100/2min rate limits
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def api_get(url, api_key):
    req = urllib.request.Request(url, headers={
        "X-Riot-Token": api_key,
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"{e.code}"
    except Exception as e:
        return None, str(e)


def search_region(api_key, region, otps, routing, platform, target, matches_per_player, patch):
    """Search one region for Garen games. Returns list of game dicts."""
    print(f"\n{'='*60}")
    print(f"Region: {region.upper()} ({len(otps)} OTPs, routing={routing})")
    print(f"{'='*60}")

    # Step 1: Look up PUUIDs
    print(f"Looking up {len(otps)} players...")
    players = []
    for i, (name, tag) in enumerate(otps):
        url = f"https://{routing}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{quote(name)}/{quote(tag)}"
        d, err = api_get(url, api_key)
        if d and "puuid" in d:
            players.append({"name": name, "tag": tag, "puuid": d["puuid"], "region": region})
        else:
            print(f"  SKIP {name}#{tag}: {err}")
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(otps)} looked up, {len(players)} found")
        time.sleep(REQUEST_DELAY)
    print(f"Found {len(players)} player PUUIDs")

    # Step 2: Pull match IDs
    print(f"Pulling match IDs ({matches_per_player} per player)...")
    all_match_ids = {}
    for i, p in enumerate(players):
        url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{p['puuid']}/ids?queue=420&count={matches_per_player}"
        matches, err = api_get(url, api_key)
        if matches:
            for mid in matches:
                if mid not in all_match_ids:
                    all_match_ids[mid] = p["name"]
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(players)} players, {len(all_match_ids)} unique matches")
        time.sleep(REQUEST_DELAY)
    print(f"Total unique match IDs: {len(all_match_ids)}")

    # Step 3: Check each match for Garen
    print(f"Checking matches for Garen (target={target})...")
    garen_games = []
    checked = 0
    for mid, player_name in list(all_match_ids.items()):
        if len(garen_games) >= target + 50:
            break
        d, err = api_get(f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{mid}", api_key)
        checked += 1
        if not d:
            if err == "429":
                time.sleep(5); continue
            continue
        info = d.get("info", {})
        ver = info.get("gameVersion", "")
        if patch and not ver.startswith(patch):
            continue
        for p in info.get("participants", []):
            if p.get("championName") == "Garen":
                game_id = mid.split("_")[-1] if "_" in mid else mid
                dur = info.get("gameDuration", 0)
                team = "blue" if p.get("teamId") == 100 else "red"
                slot = next((j for j, p2 in enumerate(info["participants"]) if p2.get("puuid") == p.get("puuid")), -1)
                garen_games.append({
                    "match_id": mid, "game_id": game_id, "version": ver,
                    "duration": dur, "duration_min": round(dur/60, 1),
                    "win": p.get("win", False),
                    "kda": f"{p['kills']}/{p['deaths']}/{p['assists']}",
                    "kills": p["kills"], "deaths": p["deaths"], "assists": p["assists"],
                    "garen_team": team, "garen_slot": slot,
                    "garen_summoner": player_name,
                    "garen_puuid": p.get("puuid"),
                    "region": region,
                    "platform": platform,
                })
                break
        if checked % 20 == 0:
            print(f"  Checked {checked}, found {len(garen_games)} Garen games")
        time.sleep(REQUEST_DELAY)

    print(f"  {region.upper()}: {len(garen_games)} Garen games from {checked} matches checked")
    return garen_games


def main():
    parser = argparse.ArgumentParser(description="Find high-elo Garen games across regions")
    parser.add_argument("--api-key", default=os.environ.get("RIOT_API_KEY"),
                        help="Riot API key (or set RIOT_API_KEY env var)")
    parser.add_argument("--regions", nargs="+", default=["na", "kr", "euw"],
                        choices=["na", "kr", "euw"],
                        help="Regions to search (default: all three)")
    parser.add_argument("--target", type=int, default=300,
                        help="Target total Garen games across all regions")
    parser.add_argument("--patch", default=None,
                        help="Filter by patch version (e.g., '16.7')")
    parser.add_argument("--matches-per-player", type=int, default=50,
                        help="Recent matches to check per player")
    parser.add_argument("--merge", default=None,
                        help="Merge with existing manifest (skip duplicate match_ids)")
    parser.add_argument("-o", "--output", default="data/replays/garen_manifest_300.json")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set RIOT_API_KEY env var")
        sys.exit(1)

    # Load existing manifest to merge with
    existing_ids = set()
    existing_games = []
    if args.merge:
        try:
            with open(args.merge) as f:
                prev = json.load(f)
            existing_games = prev.get("matches", [])
            existing_ids = {g["match_id"] for g in existing_games}
            print(f"Loaded {len(existing_games)} existing games from {args.merge}")
            # Tag existing games with region if missing
            for g in existing_games:
                if "region" not in g:
                    g["region"] = "na"
                if "platform" not in g:
                    g["platform"] = "na1"
        except Exception as e:
            print(f"Warning: could not load {args.merge}: {e}")

    # Search each region
    all_games = list(existing_games)
    per_region_target = max(50, (args.target - len(existing_games)) // len(args.regions) + 30)

    for region in args.regions:
        cfg = REGION_CONFIG[region]
        games = search_region(
            args.api_key, region, cfg["otps"], cfg["routing"], cfg["platform"],
            per_region_target, args.matches_per_player, args.patch,
        )
        # Deduplicate against existing
        new = [g for g in games if g["match_id"] not in existing_ids]
        for g in new:
            existing_ids.add(g["match_id"])
        all_games.extend(new)
        print(f"  Added {len(new)} new games ({len(games) - len(new)} duplicates skipped)")

    # Sort: wins first, then by KDA
    all_games.sort(key=lambda g: (g.get("win", False), g.get("kills", 0) - g.get("deaths", 0)), reverse=True)

    # Region breakdown
    by_region = {}
    for g in all_games:
        r = g.get("region", "na")
        by_region[r] = by_region.get(r, 0) + 1

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regions": args.regions,
        "patch_filter": args.patch,
        "total_garen_games": len(all_games),
        "by_region": by_region,
        "matches": all_games,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total Garen games: {len(all_games)}")
    for r, c in sorted(by_region.items()):
        print(f"  {r.upper()}: {c}")
    if args.patch:
        on_patch = sum(1 for g in all_games if g.get("version", "").startswith(args.patch))
        print(f"On patch {args.patch}: {on_patch}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
