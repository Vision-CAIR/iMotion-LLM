# iMotion-LLM

Official repository for the WACV 2026 paper
[`iMotion-LLM: Instruction-Conditioned Trajectory Generation`](https://openaccess.thecvf.com/content/WACV2026/html/Felemban_iMotion-LLM_Instruction-Conditioned_Trajectory_Generation_WACV_2026_paper.html).

This repository is the cleaned public migration of the original research codebase used for the paper. The release keeps legacy experiment assets for traceability, but the supported public surface is now centered on `scripts/`, `configs/release/`, and the setup and release docs under `docs/`.

## Overview

iMotion-LLM is a language-conditioned trajectory generation framework for autonomous driving. The paper combines:

- instruction-conditioned trajectory generation
- direction feasibility reasoning on `InstructWaymo`
- safety-aware open-vocabulary instruction reasoning on `Open-Vocabulary InstructNuPlan`
- conditional motion baselines and an LLM-based motion model
- training, evaluation, and ablation studies for controllable motion generation

Project links:

- WACV page: <https://openaccess.thecvf.com/content/WACV2026/html/Felemban_iMotion-LLM_Instruction-Conditioned_Trajectory_Generation_WACV_2026_paper.html>
- WACV PDF: <https://openaccess.thecvf.com/content/WACV2026/papers/Felemban_iMotion-LLM_Instruction-Conditioned_Trajectory_Generation_WACV_2026_paper.pdf>
- arXiv preprint: <https://arxiv.org/abs/2406.06211>
- project page: <https://vision-cair.github.io/iMotion-LLM/>

## Status

Current repository status:

- migrated code from the protected `iMotion-LLM-ICLR` source is now present in this public repo
- public release configs are available under `configs/release/`
- supported helper scripts are available under `scripts/`
- setup, data, checkpoint, and running docs are available under `docs/setup/`
- paper-table coverage and remaining gaps are tracked under `docs/release/`
- legacy experiment folders are preserved for paper traceability

Supported code areas now in the repo:

- `instructions/` for direction and instruction utilities
- `gameformer/` for conditional trajectory backbones and preprocessing
- `trajgpt/` for the migrated iMotion-LLM runtime code
- `mtr/` for the Motion Transformer baseline
- `tools/eval/` for evaluation helpers preserved from the working tree

Supported public runtime notes:

- use `scripts/` and `configs/release/` as the supported entry surface
- release configs now use the public model architecture name `imotion_llm`
- legacy `minigpt4` module names are still present internally for backward compatibility
- `ProSim-Instruct` is an external baseline from the paper and is not part of this repository

## Installation

Recommended quick start:

```bash
git clone https://github.com/Vision-CAIR/iMotion-LLM.git
cd iMotion-LLM

conda env create -f environment.yml
conda activate imotion-llm
pip install -e ./mtr
```

Full setup and release notes:

- [Installation](docs/setup/installation.md)
- [Data and checkpoints](docs/setup/data_and_checkpoints.md)
- [Running](docs/setup/running.md)
- [WACV 2026 experiment status](docs/release/WACV2026_EXPERIMENT_STATUS.md)
- [Release audit](docs/release/REPO_AUDIT.md)
- [Upload manifest](docs/release/HUGGINGFACE_UPLOAD_MANIFEST.md)
- [Paper reproducibility checklist](docs/release/PAPER_REPRODUCIBILITY_CHECKLIST.md)

## Quick Start

Preprocess Waymo:

```bash
bash scripts/preprocess_waymo_gameformer.sh \
  --load_path data/raw/waymo/validation_interactive \
  --save_path data/processed/waymo/gameformer/val
```

Train conditional GameFormer on Waymo:

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

Evaluate an existing iMotion-LLM checkpoint on `gt1`:

```bash
bash scripts/eval_imotion_waymo.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth \
  run.eval_dir=checkpoints/imotion_llm/waymo/checkpoint_last.pth \
  datasets.traj_align_valid.processor.valid.new_eval_mode=gt1
```

Evaluate conditional MTR on the `gt1` split:

```bash
bash scripts/eval_mtr_waymo.sh --ckpt checkpoints/mtr/waymo/checkpoint_epoch_15.pth
```

For the full paper flow, see [Running](docs/setup/running.md), which documents the `gt1`, `pos1`, `neg1`, `safe_with_context`, `safe_no_context`, `unsafe_with_context`, and `unsafe_no_context` evaluation modes.

## Repository Layout

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
- [WACV 2026 experiment status](docs/release/WACV2026_EXPERIMENT_STATUS.md)
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
- [ ] Upload public checkpoints, manifests, and generated NuPlan prompt bundle
- [ ] Add final table-to-checkpoint reproduction manifests
- [ ] Prune unsupported duplicate legacy experiment files more aggressively
- [ ] Run a fresh-machine validation using only public assets

## Availability Checklist

Available now:

- migrated working code from the protected source repo
- release configs in `configs/release/`
- helper scripts in `scripts/`
- setup, data, checkpoint, and run documentation
- MTR, GameFormer, iMotion-LLM, and instruction code trees

Still missing:

- public release checkpoints
- exact evaluation manifests and generated NuPlan prompt bundle
- polished per-table reproduction manifests
- broader fresh-machine validation using only public assets

## Migration Note

An older local copy may already exist on the development machine and is treated as protected reference material, not an edit target.

Cleanup and refactoring work happen in this repository only.

## Acknowledgment

Parts of this codebase are being adapted and cleaned from prior internal research code that was built on top of [`Vision-CAIR/MiniGPT-4`](https://github.com/Vision-CAIR/MiniGPT-4).

We acknowledge the MiniGPT-4 repository and its authors as an important upstream foundation for developing portions of the iMotion-LLM codebase.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{Felemban_2026_WACV,
  author    = {Felemban, Abdulwahab and Hroub, Nussair and Ding, Jian and Abdelrahman, Eslam and Shen, Xiaoqian and Mohamed, Abduallah and Elhoseiny, Mohamed},
  title     = {iMotion-LLM: Instruction-Conditioned Trajectory Generation},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month     = {March},
  year      = {2026},
  pages     = {2710-2720}
}
```
