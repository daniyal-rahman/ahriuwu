# ahriuwu

Autonomous League of Legends agent using DreamerV4-style world model. Currently focused on Garen Top lane.

## Setup (Windows)

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (RTX 5080)
- ffmpeg installed and in PATH

### Install ffmpeg
```powershell
# Option 1: winget
winget install ffmpeg

# Option 2: Download from https://ffmpeg.org/download.html
# Extract and add bin/ to PATH
```

### Clone and Install
```powershell
git clone https://github.com/daniyal-rahman/ahriuwu.git
cd ahriuwu

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install package
pip install -e .

# For GPU support (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation
```powershell
python -c "import ahriuwu; print('OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Usage

### Download Videos
```powershell
python scripts/download_youtube.py --channel domisumReplay-Garen --limit 5
python scripts/download_youtube.py --channel GarenChallenger --limit 5
```

### Extract Frames
```powershell
python scripts/process_videos.py --frames-only
```

### Test OCR (optional)
```python
from ahriuwu.data import get_hud_regions
from ahriuwu.ocr.reader import read_game_clock
import cv2

frame = cv2.imread("data/samples/sample_frame.png")
regions = get_hud_regions("domisumReplay-Garen")
seconds = read_game_clock(frame, regions["game_clock"])
print(f"Game time: {seconds}s")
```

## Project Structure
```
ahriuwu/
├── src/ahriuwu/
│   ├── data/           # YouTube download, frame extraction, HUD regions
│   ├── ocr/            # Game state extraction (clock, gold, CS)
│   └── rewards/        # Reward computation (for later)
├── scripts/            # CLI tools
├── data/               # Downloaded videos and frames (gitignored)
└── PROGRESS.md         # Development progress log
```

## Data Sources
- `@domisumReplay-Garen` - Replay footage with spectator HUD
- `@GarenChallenger` - Live gameplay with player HUD

## Current Status
See [PROGRESS.md](PROGRESS.md) for detailed progress log.

**Phase 1 (MacBook)**: Pipeline setup - DONE
**Phase 2 (Windows)**: Data collection at scale - IN PROGRESS
