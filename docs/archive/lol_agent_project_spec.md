# League of Legends Autonomous Agent: Project Specification

## Executive Summary

This project aims to train a deep learning agent capable of playing League of Legends at Emerald rank or higher, using a world model approach inspired by DreamerV4. Unlike existing LoL bots that rely on rule-based scripting, this agent will learn decision-making from visual input using reinforcement learning inside a learned world model.

**Target**: Beat an Emerald-rank player (top ~15% of ranked ladder)
**Champion**: Garen (Top Lane)
**Scope**: Laning phase only (first 14 minutes)
**Approach**: DreamerV4-style world model with imagination training

---

## Background & Prior Work

### OpenAI Five (Dota 2)
- **Compute**: 770 PFlops/s·days, 256 GPUs, 128,000 CPU cores, ~$18M estimated
- **Architecture**: 4096-unit LSTM, 159M parameters
- **Training**: 10 months continuous, 180 years of gameplay per day
- **Key limitation**: Required direct API access to game state (20,000 numbers per observation)
- **Weights**: Never released publicly

### Existing LoL Approaches
| Project | Approach | Outcome |
|---------|----------|---------|
| TLoL | Replay scraping + imitation learning | Datasets released, no deployed agent |
| DeepLeague | Minimap CV detection | Detection only, no decision-making |
| AAAI Deep Learning Bot | PPO + LSTM (72 hrs on K80) | First blood in controlled scenario |
| Commercial Scripts | Rule-based evade/orbwalk | Mechanical assistance only, no macro |

**Key gap**: No ML-based LoL agent has demonstrated ranked play. This project would be novel work.

### DreamerV4 (September 2025)
- **Achievement**: First agent to obtain diamonds in Minecraft from pure offline data
- **Compute**: 2B parameters, single H100 GPU at 21 FPS inference
- **Key innovation**: Shortcut forcing enables 4 sampling steps vs 64 (16x speedup)
- **Data efficiency**: Only 100 hours of action-paired data needed; rest can be unlabeled video
- **Relevance**: Proves world model approach works for complex games with visual input

---

## Technical Architecture

### Overview

```
Vision Tokenizer (50M params)
        │
        ▼
Dynamics Model (150M params) ─── Shortcut Forcing (K=4 steps)
        │
        ├──► Policy Head (MLP)
        ├──► Reward Head (MLP)
        └──► Value Head (MLP)

Total: ~210M parameters
Target inference: 40-60 FPS on RTX 5080
```

### Component Details

**Vision Tokenizer**
- Input: 1080p frames, 16×16 patches
- Output: 256 spatial tokens per frame
- Architecture: Block-causal encoder-decoder with masked autoencoding
- Training: MSE + LPIPS reconstruction loss

**Dynamics Model**
- Architecture: Efficient transformer (24 layers total)
- Temporal attention: Every 4th layer only (efficiency optimization)
- Attention: Grouped Query Attention (GQA) with 4 KV heads
- Model dimension: 512
- Context length: 192 frames (9.6 seconds at 20 FPS)
- Training: Shortcut forcing objective with x-prediction (not v-prediction)

**Agent Heads**
- Policy: MLP outputting action distribution
- Reward: MLP with symexp twohot output
- Value: MLP with symexp twohot output
- Training: PMPO (Preference-based MPO) with behavioral prior KL

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frame rate | 20 FPS | Matches human reaction time (~200ms), proven in DreamerV4 |
| Prediction type | X-prediction | Prevents error accumulation in long rollouts |
| Sampling steps | K=4 | Shortcut forcing enables fast inference without quality loss |
| Temporal attention | Every 4 layers | 2x speedup with improved inductive bias |
| Camera | Locked | Simplifies observation space, matches most replay data |

---

## Champion Selection: Garen Top

### Rationale

| Factor | Garen Advantage |
|--------|-----------------|
| Skillshots | None (all abilities are point-click or self-AOE) |
| Mechanical complexity | Low (Q→AA→E→W→R combo) |
| Camera requirements | Locked camera sufficient for top lane |
| Lane variance | 1v1 with minimal jungle interference early |
| Decision-making focus | Enables testing macro decisions, not micro execution |
| Reaction time sensitivity | Low (W timing is forgiving) |

### Action Space

```
Keyboard (binary, can be simultaneous):
- Q, W, E, R: Abilities
- D, F: Summoner spells
- 1-6: Item actives
- B: Recall
- S: Stop
- A: Attack move

Mouse (discretized 11×11 grid with foveated center):
- Position: 121 categorical classes
- Left click: Binary
- Right click: Binary

Total: ~15 binary + 121 categorical ≈ 136 action dimensions
```

---

## Data Strategy

### Data Requirements (Based on DreamerV4 Findings)

| Data Type | Purpose | Amount Needed | Source |
|-----------|---------|---------------|--------|
| Unlabeled video | World model visual learning | 50-100 hours | YouTube |
| Action-paired video | Action conditioning | 50-100 hours | Own gameplay + keylogger |
| Reward labels | Reward model training | Derived from above | OCR extraction |

### Unlabeled Video Collection

**Source**: YouTube "Garen Top Gameplay" videos
**Processing pipeline**:
1. Download at 1080p via yt-dlp
2. Detect and crop game area (remove streamer overlays)
3. Extract frames at 20 FPS
4. Filter out non-gameplay (loading screens, shop, death screens, minimap-only views)
5. Store as frame sequences with metadata

**What this teaches the world model**:
- Visual appearance of champions, minions, abilities
- Lane state evolution patterns
- General "physics" of LoL (movement, projectiles, etc.)

### Action-Paired Data Collection

**Source**: Own gameplay with input recording
**Setup**:
- Screen capture: OBS at 1080p, 20 FPS, game capture mode
- Input capture: Python keylogger recording keyboard state, mouse position, clicks
- Synchronization: Millisecond timestamps on both streams

**Data format per frame**:
```
{
  "frame_id": int,
  "timestamp_ms": int,
  "keys": [str],        # Currently pressed keys
  "mouse_x": float,     # Normalized 0-1
  "mouse_y": float,     # Normalized 0-1
  "left_click": bool,
  "right_click": bool
}
```

**Amount**: 50-100 hours (~100-200 games)

**What this teaches the model**:
- Link between inputs and outcomes
- "Right-click here → champion moves there"
- "Press Q → ability activates"

### Reward Extraction

**Method**: OCR/CV from video frames

**Extracted values**:
- Player gold (OCR on gold counter, bottom-right UI)
- Player health percentage (health bar detection)
- Enemy health percentage (health bar detection)
- Game timestamp (OCR on game clock)
- CS count (OCR on CS counter)

**Why OCR over memory reading**:
- Works on YouTube videos (enables reward labels on unlabeled data)
- No ToS concerns
- DreamerV4 showed robustness to noisy observations

---

## Reward Function Design

### Philosophy

- **Gold-centric**: Gold naturally captures CS, kills, assists, tower plates
- **Health advantage**: Encourages winning trades (Garen's strength)
- **Sparse terminal reward**: End-of-laning-phase bonus for overall lead
- **No micro-rewards**: Avoid reward hacking on individual actions

### Reward Components

| Component | Formula | Scale | Rationale |
|-----------|---------|-------|-----------|
| Gold gain | `(current_gold - prev_gold) × 0.01` | ~0.2 per CS, ~3.0 per kill | Primary objective signal |
| Health advantage delta | `(my_hp% - enemy_hp%) - prev_delta × 5.0` | ±0.5 per trade | Encourages winning trades |
| Death penalty | `-10.0` on death | Fixed | Discourages inting |
| Lane phase bonus | `cs_lead × 0.1 + gold_lead × 0.001` | At 14 min | Terminal reward for winning lane |

### What We're NOT Rewarding Explicitly

- **CS count**: Captured by gold
- **Tower damage**: Captured by gold (plates)
- **Kill count**: Captured by gold
- **Vision**: Out of scope for laning phase
- **Roaming**: Out of scope for laning phase

---

## Training Pipeline

### Phase 1: World Model Pretraining

**Hardware**: Local RTX 5080
**Duration**: ~72 hours
**Data**: Unlabeled YouTube videos + action-paired own gameplay

**Steps**:
1. Train vision tokenizer on all video data (masked autoencoding)
2. Train dynamics model on tokenized video with shortcut forcing
3. Action conditioning learned from action-paired subset

**Target**: World model that accurately predicts game state evolution

### Phase 2: Agent Finetuning (Behavioral Cloning + Reward Model)

**Hardware**: Local RTX 5080
**Duration**: ~48 hours
**Data**: Action-paired gameplay with reward labels

**Steps**:
1. Add policy/reward/value heads to frozen dynamics model
2. Train policy head via behavioral cloning (multi-token prediction)
3. Train reward head to predict extracted rewards
4. Continue dynamics loss to maintain world model quality

**Target**: Policy that can CS, trade, and survive laning without inting

### Phase 3: Imagination Training (Reinforcement Learning)

**Hardware**: Cloud A100s (~$60-100)
**Duration**: ~100-200 GPU hours
**Data**: Rollouts generated inside world model (no real game interaction)

**Steps**:
1. Initialize value head, freeze policy as behavioral prior
2. Generate imagined rollouts from dataset contexts
3. Annotate with learned reward model
4. Train policy via PMPO to maximize rewards
5. Train value head via TD-learning on λ-returns

**Key insight**: This is on-policy RL but entirely inside the world model—no actual game needed.

**Target**: Policy that exceeds behavioral cloning baseline

### Phase 4: Online Fine-tuning (Optional)

**Hardware**: Local RTX 5080
**Duration**: Ongoing
**Data**: Real gameplay vs intermediate bots

**Steps**:
1. Deploy trained agent in practice tool or vs bots
2. Collect failure cases
3. Fine-tune on failures
4. Repeat

**Target**: Robust policy that handles edge cases

---

## Compute Budget

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Vision tokenizer | Local 5080 | 12 hrs | $0 |
| Dynamics pretraining | Local 5080 | 72 hrs | $0 |
| BC + Reward model | Local 5080 | 48 hrs | $0 |
| Imagination RL | Cloud A100 (Vast.ai) | 150 hrs | ~$60-90 |
| Optional online FT | Local 5080 | Ongoing | $0 |
| **Total** | | | **~$60-90** |

**Contingency**: $200 budget allows for ~2x iteration if needed

---

## Inference Pipeline (Deployment)

### Latency Budget

```
Target: 20 FPS = 50ms per frame

Screen capture:     ~2ms
Vision tokenizer:   ~5ms (optimized)
Dynamics model:     ~30ms (4 sampling steps)
Policy head:        ~1ms
Action execution:   ~2ms
────────────────────────
Total:              ~40ms ✓
```

### Optimization Strategies

1. **TensorRT**: Compile model for inference optimization
2. **FP16/INT8**: Quantization for faster inference
3. **KV cache**: Reuse for temporal context
4. **Async capture**: Pipeline screen capture with inference

### Reaction Time Handicap

To ensure the agent wins through decision-making, not superhuman reactions:
- Agent observes frame N, acts on frame N (no lookahead)
- 20 FPS gives ~50ms reaction floor (human average: 200-250ms)
- Agent will still be slightly faster than human but not unreasonably so

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Vision latency too high | Medium | High | Smaller model, quantization, skip frames |
| OCR noise corrupts rewards | Medium | Medium | Smooth rewards temporally, train with augmentation |
| World model doesn't capture game mechanics | Medium | High | More data, longer training, ablation studies |
| Imagination training doesn't improve over BC | Medium | Medium | Tune PMPO hyperparameters, longer training |
| Riot account ban | High | Low | Use PBE server, don't play ranked |
| Transfer to real game fails | Medium | Medium | Online fine-tuning phase |

---

## Success Criteria

### Minimum Viable Product
- Agent can CS (~5 CS/min)
- Agent doesn't die repeatedly (< 3 deaths in lane phase)
- Agent uses abilities appropriately
- **Estimated rank equivalent**: Silver/Gold

### Target Success
- Agent can CS well (~7 CS/min)
- Agent wins trades against equal skill
- Agent backs at appropriate times
- Agent responds to jungle ganks
- **Estimated rank equivalent**: Platinum/Emerald

### Stretch Goal
- Agent consistently beats Emerald players
- Agent demonstrates macro understanding (wave management, recall timing)
- **Estimated rank equivalent**: Emerald+

---

## Timeline (Effort, Not Calendar)

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Data collection (YouTube) | 1-2 weeks | None |
| Data collection (own gameplay) | 2-4 weeks | Keylogger setup |
| OCR/reward extraction pipeline | 1 week | Sample data |
| World model training | 1 week | All data |
| BC + reward model training | 1 week | World model |
| Imagination training | 1-2 weeks | BC model |
| Evaluation + iteration | Ongoing | Trained agent |

**Total estimated effort**: 8-12 weeks

---

## Open Questions

1. **Data augmentation**: Should we augment action-paired data with noise to improve robustness?
2. **Multi-champion**: Train world model on general top lane, then champion-specific policy heads?
3. **Opponent modeling**: Should we try to predict enemy actions explicitly?
4. **Curriculum**: Start with easier scenarios (vs bots) before full laning?

---

## References

- DreamerV4: Hafner et al., "Training Agents Inside of Scalable World Models", arXiv:2509.24527, Sep 2025
- OpenAI Five: OpenAI, "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv:1912.06680, Dec 2019
- TLoL Project: https://github.com/MiscellaneousStuff/tlol
- DeepLeague: https://github.com/farzaa/DeepLeague
- VPT: Baker et al., "Video PreTraining", NeurIPS 2022

---

*Document generated: December 2025*
*Version: 1.0*
