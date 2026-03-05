# ahriuwu: DreamerV4-Style World Model for Autonomous League of Legends Play

**Repository:** `daniyal-rahman/ahriuwu`  
**Task Domain:** League of Legends (current focus: Garen top lane)  
**Core Idea:** Learn a compact latent world model from replay frames, then train action and reward heads for agent finetuning.

## Abstract
This repository implements a DreamerV4-inspired pipeline for offline-to-imagination RL in League of Legends. The current system emphasizes (1) large-scale replay ingestion, (2) latent dynamics modeling, and (3) behavior-cloning and reward learning on top of pretrained world-model components. Recent experiments indicate strong visual tokenization performance with CNN-based tokenizers and stable Phase 2 finetuning metrics, while several high-priority training pipeline correctness items remain open.

## Highlights
- End-to-end data pipeline for replay/video download, frame extraction, OCR-based state parsing, and latent packing.
- Dynamics and tokenizer model implementations under [`src/ahriuwu/models`](src/ahriuwu/models).
- Agent finetuning pipeline with BC + reward prediction in [`scripts/train_agent_finetune.py`](scripts/train_agent_finetune.py).
- Audit and review artifacts in [`docs/audits`](docs/audits) and [`docs/CODE_REVIEW_PHASE2.md`](docs/CODE_REVIEW_PHASE2.md).

## Main Results
### Phase 2 agent finetuning (Issue #7)
- Top-1 action accuracy: **48.1%**
- Top-5 action accuracy: **86.1%**
- BC loss: **1.81**
- Reward loss: **0.15**
- Dynamics loss: **0.0005**

Reference: https://github.com/daniyal-rahman/ahriuwu/issues/7

### Tokenizer comparison (Issue #6)
| Model | Params | PSNR (dB, higher is better) | LPIPS (lower is better) |
|---|---:|---:|---:|
| CNN tokenizer | 13M | **32.64 ± 0.87** | **0.0216** |
| Transformer tokenizer | 40M | 27.45 ± 1.29 | 0.1032 |

Reference: https://github.com/daniyal-rahman/ahriuwu/issues/6

## Active Run (In Progress)
### Small tokenizer training
- **Status:** Running
- **Preset:** `small` transformer tokenizer
- **Architecture:** `embed_dim=512`, `num_encoder_layers=8`, `num_decoder_layers=8`, `num_heads=8`
- **Latent setup:** `num_latents=256`, `latent_dim=32`
- **Script:** `python scripts/train_transformer_tokenizer.py --model-size small`

## Method Overview
1. Data acquisition from replay sources and YouTube channels.
2. Frame-level preprocessing and HUD/OCR feature extraction.
3. Tokenizer training for image-latent compression.
4. Latent dynamics modeling for temporal prediction.
5. Agent finetuning with BC + reward supervision.
6. (Planned) Policy learning with imagination rollouts.

## Reproducibility
### Environment setup
```bash
git clone https://github.com/daniyal-rahman/ahriuwu.git
cd ahriuwu
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

For CUDA builds of PyTorch, install from the official wheel index matching your CUDA version.

### Data processing examples
```bash
python scripts/download_youtube.py --channel domisumReplay-Garen --limit 5
python scripts/process_videos.py --frames-only
python scripts/extract_features_v2.py
```

### Training examples
```bash
python scripts/train_dynamics.py
python scripts/train_transformer_tokenizer.py
python scripts/train_agent_finetune.py
```

### Evaluation examples
```bash
python scripts/eval_dynamics.py
python scripts/eval_transformer_tokenizer.py
python scripts/eval_trade_prediction.py
```

## Issue-Driven Roadmap (Snapshot: 2026-03-05)
### Open issues
- [#8 Phase 2 training pipeline code review findings](https://github.com/daniyal-rahman/ahriuwu/issues/8)
  - High-priority items include reward indexing correctness and missing death-penalty application path.
- [#7 Agent Finetuning Phase 2 Complete - BC + Reward Training Results](https://github.com/daniyal-rahman/ahriuwu/issues/7)
  - Next requested validation: held-out evaluation and prediction-vs-ground-truth analysis.
- [#6 Tokenizer Comparison: CNN (13M) achieves 30+ PSNR vs Transformer (40M) at 27 PSNR](https://github.com/daniyal-rahman/ahriuwu/issues/6)
  - Follow-up direction: investigate transformer objective/architecture mismatch and hybrid designs.

### Immediate priorities
1. Resolve correctness findings in Issue #8 before Phase 3 policy learning.
2. Run held-out evaluation protocol for Phase 2 checkpoints.
3. Decide tokenizer path (CNN vs transformer vs hybrid) for future dynamics-policy stack.

## Current Experiment Plans (2026-03-05)
1. Complete and evaluate the current `small` tokenizer run (`dim=512`, `8+8` layers, `8` heads) against the prior transformer baseline.
2. Validate tokenizer quality using PSNR/LPIPS plus rollout-facing qualitative checks on HUD-heavy frames.
3. Continue Tier-0 stabilized training defaults already adopted in code (WSD schedule, variable context noise, compile path, 8-bit optimizer where available).
4. Start the next high-impact dynamics experiments after tokenizer stabilization, prioritizing:
   - Hybrid temporal modeling (Mamba+attention) for longer-context efficiency.
   - MeanFlow vs shortcut forcing as a controlled inference-speed experiment.
   - Inverse-dynamics pseudo-labeling to expand action supervision coverage.

## Repository Map
- [`src/ahriuwu/data`](src/ahriuwu/data): data ingestion, replay processing, features, datasets
- [`src/ahriuwu/ocr`](src/ahriuwu/ocr): OCR and HUD state reading
- [`src/ahriuwu/models`](src/ahriuwu/models): tokenizer, dynamics, heads, losses
- [`scripts`](scripts): training/eval/data CLI entry points
- [`docs`](docs): progress notes, audits, technical analyses
- [`lab_notebook`](lab_notebook): experiment planning and logs
- [`eval_results`](eval_results): generated metrics/plots/comparison artifacts

## Citation
If you use this codebase in derivative work, cite the repository and include commit hashes used for training/evaluation.

```bibtex
@misc{ahriuwu2026,
  author = {Rahman, Daniyal},
  title = {ahriuwu: LoL Autonomous Agent Using DreamerV4-Style World Model},
  year = {2026},
  howpublished = {\url{https://github.com/daniyal-rahman/ahriuwu}}
}
```
