# iMotion-LLM

Official repository for the paper
[`iMotion-LLM: Instruction-Conditioned Trajectory Generation`](https://arxiv.org/abs/2406.06211).

This repository is the cleaned public migration of the original research codebase used for the paper.
The release keeps the original project website under `docs/`, preserves legacy experiment assets for traceability, and adds a supported public surface for environment setup, preprocessing, training, evaluation, and checkpoint loading.

## Overview

iMotion-LLM is a language-conditioned trajectory generation framework for autonomous driving.
The paper combines:

- instruction-conditioned trajectory generation
- direction feasibility reasoning on `InstructWaymo`
- safety-aware open-vocabulary instruction reasoning on `Open-Vocabulary InstructNuPlan`
- conditional motion baselines and an LLM-based motion model
- training, evaluation, and ablation studies for controllable motion generation

Project links:

- Paper: <https://arxiv.org/abs/2406.06211>
- PDF: <https://arxiv.org/pdf/2406.06211>
- Project page: [docs/](docs/)

## Status

Current repository status:

- migrated code from the protected `iMotion-LLM-ICLR` source is now present in this public repo
- public release configs are available under `configs/release/`
- supported helper scripts are available under `scripts/`
- setup, data, checkpoint, and running docs are available under `docs/setup/`
- legacy experiment folders are preserved for paper traceability

Supported code areas now in the repo:

- `instructions/` for direction and instruction utilities
- `gameformer/` for conditional trajectory backbones and preprocessing
- `trajgpt/` for the MiniGPT-4 based iMotion-LLM code
- `mtr/` for the Motion Transformer baseline
- `tools/eval/` for evaluation helpers preserved from the working tree

## Installation

Recommended quick start:

```bash
git clone https://github.com/Vision-CAIR/iMotion-LLM.git
cd iMotion-LLM

conda env create -f environment.yml
conda activate imotion-llm
pip install -e ./mtr
```

Full setup notes:

- [Installation](docs/setup/installation.md)
- [Data and checkpoints](docs/setup/data_and_checkpoints.md)
- [Running](docs/setup/running.md)

## Quick Start

Train GameFormer:

```bash
bash scripts/train_gameformer_waymo.sh --act --act_dec --level 1
```

Train iMotion-LLM on Waymo:

```bash
bash scripts/train_imotion_waymo.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth
```

Evaluate an existing iMotion-LLM checkpoint:

```bash
bash scripts/eval_imotion_waymo.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth \
  run.eval_dir=checkpoints/imotion_llm/waymo/checkpoint_last.pth
```

## Repository Structure

```text
iMotion-LLM/
├── configs/         # public release configs and config notes
├── data/            # expected dataset layout
├── docs/            # project website and supporting docs
├── experiments/     # experiment manifests and result notes
├── migration/       # migration audit docs and source inventory
├── scripts/         # public helper entrypoints
├── gameformer/      # migrated GameFormer / C-GameFormer code
├── instructions/    # migrated instruction logic
├── mtr/             # migrated MTR baseline code
├── tools/           # evaluation helpers
├── trajgpt/         # migrated iMotion-LLM / MiniGPT-4 adaptation
├── src/imotion_llm/ # future cleaned package surface
├── MIGRATION_CHECKLIST.md
├── TASKS.md
└── README.md
```

## Documentation

- [Migration checklist](MIGRATION_CHECKLIST.md)
- [Task list](TASKS.md)
- [Incoming source inventory](migration/SOURCE_INVENTORY.md)
- [Source audit: iMotion-LLM-ICLR](migration/SOURCE_AUDIT_IMOTION_LLM_ICLR.md)
- [Paper refactoring requirements](migration/PAPER_REFACTORING_REQUIREMENTS.md)
- [Paper experiment scope](docs/paper_scope.md)
- [Installation](docs/setup/installation.md)
- [Data and checkpoints](docs/setup/data_and_checkpoints.md)
- [Running](docs/setup/running.md)

## TODO

High-priority TODOs for the repo:

- [x] Initialize a clean public repository without touching older local copies
- [x] Preserve the existing project website under `docs/`
- [x] Extract paper requirements into a migration checklist
- [x] Intake and audit all legacy iMotion-LLM source locations
- [x] Migrate the main legacy working tree into this public repo
- [x] Remove critical hard-coded local paths from supported entrypoints
- [x] Add release configs for Waymo, nuPlan, and MTR
- [x] Add reproducible install, training, and evaluation instructions
- [ ] Prune unsupported duplicate legacy experiment files more aggressively
- [ ] Upload or link public research checkpoints
- [ ] Add paper-table specific reproduction manifests and verified outputs

## Availability Checklist

Available now:

- project website assets in `docs/`
- migrated working code from the protected source repo
- release configs in `configs/release/`
- helper scripts in `scripts/`
- setup/data/checkpoint/run documentation
- MTR, GameFormer, iMotion-LLM, and instruction code trees

Still missing:

- bundled public research checkpoints
- polished per-table reproduction manifests
- deeper cleanup of archival duplicate experiment files
- verification on fresh machines and non-local datasets

## Migration Note

An older local copy already exists on this machine and is treated as protected reference material, not an edit target:

- `/ibex/project/c2278/felembaa/projects/iMotion-LLM-Jan/run_ibex/iMotion-LLM`

Cleanup and refactoring work happen in this repository only.

## Acknowledgment

Parts of this codebase are being adapted and cleaned from prior internal research code that was built on top of
[`Vision-CAIR/MiniGPT-4`](https://github.com/Vision-CAIR/MiniGPT-4).

We acknowledge the MiniGPT-4 repository and its authors as an important upstream foundation for developing portions of the iMotion-LLM codebase.

## Citation

If you use this work, please cite:

```bibtex
@article{felemban2024imotionllm,
  title={iMotion-LLM: Instruction-Conditioned Trajectory Generation},
  author={Felemban, Abdulwahab and Hroub, Nussair and Ding, Jian and Abdelrahman, Eslam and Shen, Xiaoqian and Mohamed, Abduallah and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2406.06211},
  year={2024}
}
```
