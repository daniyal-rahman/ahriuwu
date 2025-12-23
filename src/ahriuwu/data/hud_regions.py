"""Fixed HUD region coordinates for 1080p LoL footage.

Coordinates are (x, y, width, height) where (x,y) is top-left corner.
Screen is 1920x1080.
"""

# Replay HUD (domisumReplay-Garen channel)
# Based on actual frame analysis
REPLAY_HUD_1080P = {
    # Left side - blue team champion portraits/status
    "blue_team": (0, 145, 100, 500),

    # Right side - red team champion portraits/status
    "red_team": (1820, 145, 100, 500),  # 1920-100 = 1820

    # Bottom left - domisumReplay watermark (ignore for OCR)
    "watermark": (0, 958, 300, 122),  # y = 1080-122

    # Garen's ability bar (Q/W/E/R cooldowns, summoners, buffs)
    # Right above watermark
    "garen_hud": (0, 773, 310, 185),  # y = 958-185

    # Bottom center - scorecard with all 10 champs (CS, items, KDA)
    # Occasionally flashes individual gold
    "scorecard": (600, 847, 740, 233),  # y = 1080-233

    # Bottom right of scorecard - drake/objective timer
    "objective_timer": (1340, 980, 270, 100),  # x = 600+740

    # Bottom right corner - minimap
    "minimap": (1650, 810, 270, 270),  # 1920-270, 1080-270

    # Top middle - team gold, kills, neutrals, game timer
    "top_scoreboard": (385, 0, 1150, 100),  # centered: (1920-1150)/2

    # === Derived regions for OCR ===
    # Game timer - bottom of top_scoreboard, centered on screen, 70x30
    "game_clock": (925, 70, 70, 30),  # x: 960-35, y: 100-30

    # Garen's health bar text - shows "current/total" format (e.g., "1234/2500")
    # Located in bottom-left HUD area
    "garen_health_text": (70, 850, 100, 20),

    # Enemy health bars on side panels (use based on which side Garen is on)
    # If Garen is RED team -> enemies are on BLUE side (left panel)
    # If Garen is BLUE team -> enemies are on RED side (right panel)
    "enemy_health_blue_side": (30, 203, 46, 8),  # Left panel (blue team)
    "enemy_health_red_side": (1845, 203, 46, 8),  # Right panel (1920-75)

    # CS comes from scorecard - need to find Garen's row
    # Gold comes from scorecard (flashes) or top_scoreboard (team total)
}

# Live gameplay HUD (GarenChallenger channel)
# TODO: Get sample frame and map regions
LIVE_HUD_1080P = {
    "gold": None,  # TODO
    "cs": None,  # TODO
    "game_clock": None,  # TODO
    "minimap": None,  # TODO
    "player_health": None,  # TODO
}

# Map channel names to their HUD config
CHANNEL_HUD_MAP = {
    "domisumReplay-Garen": REPLAY_HUD_1080P,
    "GarenChallenger": LIVE_HUD_1080P,
}


def get_hud_regions(channel: str) -> dict:
    """Get HUD regions for a channel."""
    if channel not in CHANNEL_HUD_MAP:
        raise ValueError(f"Unknown channel: {channel}. Known: {list(CHANNEL_HUD_MAP.keys())}")
    return CHANNEL_HUD_MAP[channel]


def crop_region(frame, region: tuple[int, int, int, int]):
    """Crop a region from a frame. frame is numpy array (H, W, C)."""
    if region is None:
        return None
    x, y, w, h = region
    return frame[y:y+h, x:x+w]
