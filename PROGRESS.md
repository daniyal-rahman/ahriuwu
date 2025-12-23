# Progress Log

## Phase 1: MacBook Pipeline Setup (Complete)

### What's Done

**1. Project Structure**
- Initialized repo at `daniyal-rahman/ahriuwu`
- Python package structure under `src/ahriuwu/`
- Virtual environment with dependencies

**2. YouTube Downloader** (`src/ahriuwu/data/youtube_downloader.py`)
- Downloads from known channels: `@domisumReplay-Garen`, `@GarenChallenger`
- Extracts metadata: elo tier, patch version, side (blue/red) from title
- Saves video + metadata JSON

**3. Frame Extractor** (`src/ahriuwu/data/frame_extractor.py`)
- Extracts frames at 20 FPS using ffmpeg
- Simple, no fancy filtering

**4. HUD Region Mapping** (`src/ahriuwu/data/hud_regions.py`)
- Calibrated for 1080p replay footage (domisumReplay channel)
- Regions: blue_team, red_team, watermark, garen_hud, scorecard, objective_timer, minimap, top_scoreboard, game_clock
- Live gameplay HUD (GarenChallenger) still TODO

**5. OCR Reader** (`src/ahriuwu/ocr/reader.py`)
- EasyOCR wrapper for game clock (working: "03:35" → 215 seconds)
- Health bar detection via color (green pixel ratio)
- Gold/CS readers ready but not needed for Phase 1

**6. Reward Extractor** (`src/ahriuwu/rewards/reward_extractor.py`)
- Implements reward function from spec
- Gold gain × 0.01, health advantage × 5.0, death penalty -10.0
- **NOT USED YET** - world model training doesn't need rewards

**7. Dataset Classes** (`src/ahriuwu/data/dataset.py`)
- `FrameSequenceDataset`: Returns sequences of frames for world model
- `FrameWithStateDataset`: Returns frames + OCR state (for later)

### Key Decisions

1. **World model first, rewards later**: Replay HUD doesn't show individual gold reliably. Train world model on observation dynamics first, add reward signal with live gameplay data later.

2. **Skip complex game detection**: HUD is fixed for these videos. Hardcoded regions instead of ML-based detection.

3. **EasyOCR for game clock only**: Slow on CPU, but only need timer for timestamping. Skip other OCR for now.

---

## Phase 2: Windows Data Collection (Next)

### TODO
1. Set up environment on Windows (RTX 5080)
2. Download more videos from both channels
3. Extract frames at scale
4. Map live gameplay HUD regions (GarenChallenger)
5. Start world model training

### Future (Linux/Windows GPU)
- Implement DreamerV4 world model
- Train on extracted frame sequences
- Add action capture for live gameplay
- Reward function with proper gold/health from live HUD
