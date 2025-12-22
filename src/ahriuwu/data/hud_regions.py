"""Fixed HUD region coordinates for 1080p LoL footage.

These are hardcoded regions for known video sources. Coordinates are (x, y, width, height).
Tweak these values based on sample frames from each source.
"""

# Replay HUD (domisumReplay-Garen channel)
# TODO: Verify these coordinates with actual frames
REPLAY_HUD_1080P = {
    "gold": (880, 40, 100, 30),
    "cs": (920, 60, 60, 25),
    "game_clock": (930, 10, 60, 25),
    "minimap": (1630, 790, 280, 280),
    "player_health": (850, 880, 220, 30),
    "enemy_health": None,  # Need to detect dynamically or skip
}

# Live gameplay HUD (GarenChallenger channel)
# TODO: Verify these coordinates with actual frames
LIVE_HUD_1080P = {
    "gold": (1750, 1040, 100, 30),
    "cs": (1650, 1040, 60, 25),
    "game_clock": (1870, 10, 50, 25),
    "minimap": (1630, 790, 280, 280),
    "player_health": (850, 880, 220, 30),
    "enemy_health": None,
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
    x, y, w, h = region
    return frame[y:y+h, x:x+w]
