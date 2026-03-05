# Lab Notebook

**Welcome to the DreamerV4 World Model Lab Notebook.**

This directory contains all experimental design, research findings, checklists, and progress tracking for the project.

---

## 📋 Quick Navigation

### For First-Time Training
1. **Start here:** [`PRE_FLIGHT_CHECKLIST.md`](./PRE_FLIGHT_CHECKLIST.md) — Complete architecture, data, and training verification checklist before running any job
2. **Run tokenizer:** Use checklist + `EXPERIMENTS_AND_ROADMAP.md` Phase 0 section
3. **Run dynamics:** Follow Phase 1a after tokenizer is stable

### For Research & Experimentation
- **[`EXPERIMENTS_AND_ROADMAP.md`](./EXPERIMENTS_AND_ROADMAP.md)** — All Tier 0-3 experiments, proposed execution order, roadmap with timelines
- **[`PROGRESS_LOG.md`](./PROGRESS_LOG.md)** — Session notes, config decisions, architecture details (legacy; kept for reference)

### For Internal Reference
- **[`internal_docs/`](./internal_docs/)** — Detailed audits, config documentation, implementation notes (created as subdirectory)

---

## 📚 Document Overview

### PRE_FLIGHT_CHECKLIST.md
**Purpose:** Comprehensive verification checklist before training.

**Sections:**
- Tokenizer architecture & verification
- Dynamics architecture & verification
- Data pipeline setup
- Infrastructure & Slurm job templates
- Desktop setup

**Use when:** Before submitting any training job. Run through all checks to verify the model is wired correctly.

---

### EXPERIMENTS_AND_ROADMAP.md
**Purpose:** Research synthesis + experimental roadmap.

**Sections:**
- Phase overview (tokenizer, dynamics, agent)
- **Tier 0:** Zero-effort fixes (already implemented or 1-2 lines)
  - Variable noise augmentation (τ_ctx)
  - WSD learning rate schedule
  - torch.compile
  - 8-bit AdamW
  - Non-square resolution
- **Tier 1:** High-impact experiments (1-2 weeks each)
  - 1.A Mamba-Attention hybrid (O(n) temporal)
  - 1.B MeanFlow (1-step generation)
  - 1.C Inverse dynamics pipeline (pseudo-label YouTube)
  - 1.D Progressive temporal compression
- **Tier 2:** Medium-effort (2-4 weeks each)
  - ALR, multi-scale tokenizer, attention sinks, muP sweep
- **Tier 3:** Long-term / high-risk
  - Selective masking, learned noise, pre-training, pro distillation
- **Proposed execution roadmap** with timelines and success gates
- **Risk mitigation** table

**Use when:** Planning what experiment to run next; understanding dependencies between experiments; evaluating which ideas to pursue.

---

### PROGRESS_LOG.md
**Purpose:** Historical session notes and architectural decisions (legacy).

**Status:** Kept for reference. New sessions should update EXPERIMENTS_AND_ROADMAP.md instead.

---

### internal_docs/
**Purpose:** Detailed implementation audits and config documentation.

**Contents (to be created):**
- Tier 0 implementation audits (from `/docs/audits/`)
- Training config specifications
- Architecture wiring details

---

## 🚀 Typical Workflow

### Session 1: Tokenizer Training
1. Read `PRE_FLIGHT_CHECKLIST.md` → Tokenizer section
2. Read `EXPERIMENTS_AND_ROADMAP.md` → Phase 0 section
3. Verify all tokenizer checklist items before submitting job
4. Monitor training, log notes in PROGRESS_LOG.md

### Session 2: Dynamics Baseline
1. Read `PRE_FLIGHT_CHECKLIST.md` → Dynamics section
2. Read `EXPERIMENTS_AND_ROADMAP.md` → Phase 1a section
3. Pre-compute tokenizer latents (store on disk)
4. Submit dynamics job, track evaluation gates

### Session 3+: Tier 1 Experiments
1. Review `EXPERIMENTS_AND_ROADMAP.md` → Tier 1 section
2. Pick next experiment (A, B, C, or D) based on success gates
3. Implement & train
4. Log results, update PROGRESS_LOG.md
5. Move to next experiment or iterate

---

## 🎯 Key Metrics & Gates

### Tokenizer
- ✅ PSNR > 27 dB
- ✅ Reconstruction visually matches input
- ✅ Factored attention shapes verified (space 484×484, time 16×16)

### Dynamics Baseline
- ✅ 1-step PSNR > 15 dB (flow loss)
- ✅ 16-frame rollout > 8 dB (context works)
- ✅ **32-frame rollout > 6 dB** (gate for agent finetuning)
- ✅ 64-frame rollout produces recognizable frames (not blobs)
- ✅ VRAM 8-10 GB (not shrinking, not exploding)

### Tier 1 Experiment Success
- **1.A (Mamba):** T=128 runs, PSNR ≥ baseline
- **1.B (MeanFlow):** K=1 proxy PSNR ≥ K=4 baseline
- **1.C (IDM):** Faster convergence with 2.5K labeled data
- **1.D (ProMAG):** 30-40% faster wall-clock in early training

---

## 📝 Notes

### On Experiments
- **Tier 0 = free wins** — all already implemented or 1-2 lines. Do these first.
- **Tier 1 = highest ROI** — 1-2 weeks each, big impact. Run in order (C → D → A → B).
- **Tier 2 = conditional** — only pursue if Tier 1 shows promise.
- **Tier 3 = long-term** — only if plateauing.

### On Timelines
- **Tokenizer:** 1 week (bottleneck: training time)
- **Dynamics baseline:** 3 weeks (bottleneck: training time)
- **Tier 1 experiments:** 3-4 weeks (bottleneck: training + experimentation)
- **Agent finetuning:** 2-3 weeks (bottleneck: RL convergence)

### On Data
- 1531 YouTube videos (2.5K hours unlabeled)
- 30 hours labeled replays (with actions)
- 10.5M frames total
- Tokenizer trained on all; dynamics on all; agent on replay + pseudo-labeled YouTube

---

## 🔗 Related

- **Research synthesis source:** Previous session with multi-agent exploration (4 research agents across 4 ML lanes)
- **Tier 0 implementation audits:** `/docs/audits/tier0_*.md`
- **Experiment registry:** `/experiments/README.md` and `/experiments/registry.json`

---

**Last updated:** 2026-03-05
**Status:** Active (Tokenizer architecture factoring in progress)
