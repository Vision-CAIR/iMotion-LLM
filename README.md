# iMotion-LLM

Official repository for the paper
[`iMotion-LLM: Instruction-Conditioned Trajectory Generation`](https://arxiv.org/abs/2406.06211).

This repository is the clean migration target for the public release of the paper code.
The existing project website under `docs/` has been preserved, and the codebase is being rebuilt here in a structured, reproducible form.

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

- `docs/` website is available
- repo structure for clean migration is available
- paper-to-code migration specs and TODO tracking are available
- executable research code is not migrated yet

What is planned for this repo:

- `InstructWaymo` preprocessing and instruction generation
- `Open-Vocabulary InstructNuPlan` generation and evaluation
- `GameFormer`, `C-GameFormer`, `MTR`, `C-MTR`, and `iMotion-LLM` components
- training, evaluation, and ablation scripts
- reproducibility documentation for the paper experiments

## Installation

The full dependency stack will be finalized after the legacy code is migrated and cleaned.
For now, use a clean Python environment for development and repo intake.

```bash
git clone https://github.com/Vision-CAIR/iMotion-LLM.git
cd iMotion-LLM

conda create -n imotion-llm python=3.10 -y
conda activate imotion-llm

pip install --upgrade pip
```

Once the code migration is complete, this section will be updated with:

- package requirements
- editable install instructions
- dataset preparation dependencies
- training and evaluation commands

## Repository Structure

```text
iMotion-LLM/
├── configs/         # experiment and runtime configs
├── data/            # dataset layout docs only
├── docs/            # project website and supporting docs
├── experiments/     # experiment manifests and result notes
├── migration/       # migration audit docs and source inventory
├── scripts/         # preprocessing, training, and evaluation entrypoints
├── src/imotion_llm/ # cleaned Python package for migrated code
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

## TODO

High-priority TODOs for the repo:

- [x] Initialize a clean public repository without touching older local copies
- [x] Preserve the existing project website under `docs/`
- [x] Extract paper requirements into a migration checklist
- [ ] Intake and audit all legacy iMotion-LLM source locations
- [ ] Migrate `InstructWaymo` preprocessing and direction-feasibility logic
- [ ] Migrate `Open-Vocabulary InstructNuPlan` generation and validation code
- [ ] Migrate conditional baselines and core `iMotion-LLM` modules
- [ ] Reconstruct training, evaluation, and ablation configs for the paper
- [ ] Add reproducible install, training, and evaluation instructions

## Availability Checklist

Available now:

- project website assets in `docs/`
- clean repo layout for migration
- tracking docs for missing and recovered components

Still missing:

- executable model code
- dataset preprocessing scripts
- evaluation scripts
- experiment configs
- pretrained checkpoint instructions

## Migration Note

An older local copy already exists on this machine and is treated as protected reference material, not an edit target:

- `/ibex/project/c2278/felembaa/projects/iMotion-LLM-Jan/run_ibex/iMotion-LLM`

Cleanup and refactoring work should happen in this repository only.

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
