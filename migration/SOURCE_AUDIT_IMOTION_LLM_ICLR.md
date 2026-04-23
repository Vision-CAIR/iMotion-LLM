# Source Audit: iMotion-LLM-ICLR

Audit target:

- source repo: `<legacy_repo_root>`
- source role: protected legacy working tree
- migration destination: this public `iMotion-LLM` repository

## Ground Rules

- Do not edit `iMotion-LLM-ICLR` in place.
- Migrate by copying only the needed components into this repo.
- Clean up naming, paths, and legacy comments only after code is copied here.
- Treat the source repo as non-canonical in its current state because it has a dirty working tree and many untracked experiment files.

## Why This Repo Is The Primary Source

This repo is the strongest migration source because:

- its top-level `README.md` describes a unified codebase for `iMotion-LLM`, `GameFormer`, `Conditional GameFormer`, `MTR`, and `Conditional MTR`
- its job scripts under `run_ibex/iMotion-LLM/` directly launch `trajgpt/train.py`
- it contains the broadest set of later-stage experiment configs, including backbone, projection, and finetuning ablations
- it includes both Waymo-direction and NuPlan open-vocabulary/safety codepaths

## Source State

Observed during audit:

- git status is dirty
- the repo is ahead of its remote
- there are many modified tracked files
- there are many untracked files, including potentially relevant experimental variants

Implication:

- file presence alone does not mean a file is stable
- migration should start from the most clearly named, central files first
- experimental copies and debug variants should be treated as references, not defaults

## High-Level Subtree Classification

| Subtree | Keep For Migration? | Assessment | Notes |
| --- | --- | --- | --- |
| `trajgpt/` | yes | core | Main iMotion-LLM LLM/training stack built on MiniGPT-4/LAVIS style structure. |
| `gameformer/` | yes | core | Core baseline, preprocessing, and evaluation logic for conditional motion generation. |
| `instructions/` | yes | core | Direction/speed/acceleration labeling logic and instruction utilities. |
| `mtr/` | yes | core | MTR baseline and related training/eval entrypoints. |
| `run_ibex/` | selective | important | Useful for reconstructing launch commands and config provenance; should be cleaned heavily if migrated. |
| `GameFormer-Planner/` | maybe later | supplementary | Likely relevant for planner/closed-loop experiments, not the first migration block. |
| `NuPlan-Download-CLI/` | maybe later | utility | External dataset download helper, not core paper code. |
| `dcgm/` | no | artifacts | GPU stats directories and job outputs. |
| `wandb/` | no | artifacts | Experiment logs only. |
| `nohup/` | no | artifacts | Log files only. |
| `examples/` | maybe later | release asset | Useful for repo examples after core migration. |
| `results/`, `gt1results/`, `repo_figures/` | maybe later | release asset | Results and figures, not core code. |
| `transformers/` | no | vendored dependency | Should not be copied into the public repo unless there is a hard local patch that proves necessary. |
| `longvu/`, `mtr02/` | probably no | side branches | Out of scope unless later proven necessary. |

## Core Files By Paper Area

### 1. InstructWaymo And Direction Logic

Primary candidates:

- `instructions/direction_instructions.py`
- `instructions/speed_instructions.py`
- `instructions/acceleration_instructions.py`
- `instructions/extract_instructions.py`
- `instructions/generate_instructions_json.py`
- `trajgpt/waymo_preprocess/process_ooi.py`
- `trajgpt/waymo_preprocess/process_ooi_v02.py`
- `gameformer/data_preprocess_v09.py`

Why these matter:

- `instructions/direction_instructions.py` contains the direction classifier, including the `5`-class setup used by the paper (`stationary`, `move straight`, `turn right`, `turn left`, `take left U-turn`) and the threshold-style logic for stationary/straight/u-turn classification.
- `instructions/speed_instructions.py` and `instructions/acceleration_instructions.py` appear to implement the motion description categories used in textual captions.
- `gameformer/data_preprocess_v09.py` appears to contain the heavier Waymo-side preprocessing and feasible/infeasible direction logic, including reachable-distance rules and map-lane search.
- `trajgpt/waymo_preprocess/` looks like the lower-level scene preprocessing pipeline that prepares Waymo data into the project format.

Audit notes:

- `gameformer/data_preprocess_v09.py` is likely a critical migration target, but it is large and should be split during migration.
- There are multiple earlier/later preprocess variants in the broader source history; prefer `v09` as the current main candidate, then compare against older variants only if behavior is unclear.

### 2. Open-Vocabulary InstructNuPlan

Primary candidates:

- `gameformer/nuplan_preprocess/data_preprocess_complex.py`
- `gameformer/nuplan_preprocess/scenario_situation_gpt_info.json`
- `gameformer/nuplan_preprocess/prompts_templates/*.txt`
- `trajgpt/minigpt4/datasets/datasets/complex_instruction_dataset_helper.py`
- `trajgpt/minigpt4/datasets/datasets/traj_dataset.py`

Why these matter:

- `data_preprocess_complex.py` references `scenario_situation_gpt_info.json`, writes prompt templates, and includes the 14 target scenario types described in the paper.
- `scenario_situation_gpt_info.json` encodes safe/unsafe behavior buckets per scenario type.
- `prompts_templates/` contains GPT prompt templates for safe/unsafe instruction generation with and without context.
- `complex_instruction_dataset_helper.py` and `traj_dataset.py` appear to load the generated NuPlan instruction data and expose the `nuplan_complex` codepath used during training and evaluation.

Audit notes:

- There are multiple near-duplicate variants such as `data_preprocess_complex_debug.py`, `data_preprocess_complex__.py`, and `data_preprocess_complex_direction_based.py`. These should not be migrated blindly.
- The clean target should likely keep one primary implementation plus clearly named optional variants if they correspond to distinct experiments.

### 3. iMotion-LLM Core Model

Primary candidates:

- `trajgpt/minigpt4/models/mini_gpt4.py`
- `trajgpt/minigpt4/models/gameformer_enc.py`
- `trajgpt/minigpt4/models/mini_gpt4_mtr_dev.py`
- `trajgpt/minigpt4/imotion_sft_trainer.py`
- `trajgpt/minigpt4/datasets/builders/image_text_pair_builder.py`

Why these matter:

- `mini_gpt4.py` is the main adapted MiniGPT-4-style model and includes LoRA loading, `gameformer_enc`, adapter options, and motion/instruction behavior switches.
- `mini_gpt4_mtr_dev.py` is likely the MTR-integrated variant and should be compared against the GameFormer-integrated path.
- `image_text_pair_builder.py` and dataset wiring under `minigpt4/datasets/` connect configs to the trajectory datasets.
- `imotion_sft_trainer.py` likely contains the Hugging Face / PEFT training wrapper used for the iMotion-LLM fine-tuning path.

Audit notes:

- There are many `mini_gpt4 copy*.py` and date-stamped variants. These should be treated as experiment history, not first-choice migration files.
- The public repo should converge on one clean `imotion_llm` model implementation, not preserve the copy-sprawl.

### 4. Conditional GameFormer Baselines

Primary candidates:

- `gameformer/model/GameFormer.py`
- `gameformer/model/modules.py`
- `gameformer/interaction_prediction/train.py`
- `gameformer/interaction_prediction/eval.py`

Secondary candidates:

- `gameformer/model/C_GameFormer_Forecasting.py`
- `gameformer/model/GameFormer_Forecasting.py`
- `gameformer/interaction_prediction/train_nuplan.py`
- `gameformer/interaction_prediction/train_Forecasting.py`

Why these matter:

- `GameFormer.py` and `modules.py` are the central backbone implementation.
- `interaction_prediction/train.py` and `eval.py` look like the baseline training/eval entrypoints and include direction-conditioned metrics logic.
- Some conditional or forecasting-specific files currently show up as untracked experimental files; they may still be important but require careful review before migration.

Audit notes:

- The baseline path is likely split between tracked files and newer untracked variants.
- Start with the tracked baseline, then selectively recover newer conditional/forecasting files if needed for paper parity.

### 5. MTR Baselines

Primary candidates:

- `mtr/tools/train.py`
- `mtr/tools/test.py`
- `mtr/tools/cfgs/waymo/*.yaml`
- `mtr/mtr/models/context_encoder/mtr_encoder.py`
- `mtr/mtr/models/motion_decoder/mtr_decoder.py`

Why these matter:

- They appear to contain the core MTR baseline and its Waymo configuration surface.
- There are modified config files suggesting conditional-action or instruction-aware variants were added locally.

Audit notes:

- MTR should be migrated as a minimal baseline slice, not as a full raw vendored repo if the public release only needs a focused subset.

### 6. Launch, Eval, And Reproduction Scripts

Primary candidates:

- `run_ibex/iMotion-LLM/*.sh`
- `run_ibex/preprocess/*.sh`
- `run_ibex/train_gameformer/*.sh`
- `trajgpt/train_configs_*/*.yaml`
- `trajgpt/2_iccv_rebuttal_may_2025/*.yaml`
- `trajgpt/3_wacv_rebuttal_15_sep_2025/*.yaml`

Why these matter:

- They encode the experimental surface for paper reproduction and later ablations.
- Config directories clearly contain projection-depth, finetuning-strategy, and LLM-backbone comparisons.

Audit notes:

- These scripts are useful for recovering intent and hyperparameters, but should not be copied as-is.
- Hard-coded cluster paths and output directories are widespread and will need cleanup during migration.

## Likely Mapping To Paper Deliverables

| Paper Area | Best Current Source |
| --- | --- |
| InstructWaymo label extraction | `instructions/` |
| Waymo scene preprocessing | `trajgpt/waymo_preprocess/` |
| Waymo augmentation and feasible/infeasible direction logic | `gameformer/data_preprocess_v09.py` |
| Open-Vocabulary InstructNuPlan prompt generation | `gameformer/nuplan_preprocess/` |
| NuPlan instruction dataset loading | `trajgpt/minigpt4/datasets/datasets/` |
| iMotion-LLM model | `trajgpt/minigpt4/models/mini_gpt4.py` |
| HF/LoRA training loop | `trajgpt/train.py` and `trajgpt/minigpt4/imotion_sft_trainer.py` |
| GameFormer baseline | `gameformer/model/` and `gameformer/interaction_prediction/` |
| MTR baseline | `mtr/` plus MTR-aware `trajgpt` model variant |
| Main paper ablation configs | `trajgpt/train_configs_mar_2025/`, `trajgpt/2_iccv_rebuttal_may_2025/`, `trajgpt/train_iccv_feb_2025/` |

## High-Risk Legacy Issues To Fix During Migration

- hard-coded local paths to private local and cluster directories
- copied model files like `mini_gpt4 copy.py`, `mini_gpt4 copy 2.py`, etc.
- duplicated preprocess variants with unclear ownership
- mixed tracked and untracked experimental files
- logs, notebooks, and artifact directories living beside source code
- inherited project naming from MiniGPT-4 and other parent repos
- very large monolithic preprocessing files that should be split into smaller modules

## Files And Directories To Exclude From First Migration

- `wandb/`
- `dcgm/`
- `nohup/`
- `testing_log/`
- most notebooks unless needed to recover logic
- vendored `transformers/`
- raw result dumps and temporary logs
- scratch copies and debug duplicates unless they contain unique required code

## Recommended Migration Order

1. Migrate `instructions/` into a clean dataset-labeling package.
2. Migrate `trajgpt/waymo_preprocess/` and the relevant Waymo augmentation pieces from `gameformer/data_preprocess_v09.py`.
3. Migrate `gameformer/nuplan_preprocess/` for Open-Vocabulary InstructNuPlan generation.
4. Migrate `trajgpt/minigpt4/datasets/` loaders that connect the generated data to training.
5. Migrate `trajgpt/minigpt4/models/mini_gpt4.py` and its closest dependencies.
6. Migrate `trajgpt/train.py`, `trajgpt/eval.py`, and the minimal trainer/config plumbing.
7. Migrate baseline slices from `gameformer/` and `mtr/`.
8. Reconstruct only the paper-relevant configs from the many config directories.

## Immediate Next Slice Recommendation

The best first code migration slice is:

- `instructions/`
- `trajgpt/waymo_preprocess/`
- the Waymo-specific portions of `gameformer/data_preprocess_v09.py`

Reason:

- these pieces define the paper’s data semantics
- they are easier to isolate than the full LLM stack
- they reduce ambiguity before model/config migration starts

## Secondary Sources To Use Only If Needed

- `<legacy_snapshot_repo>`
  - use to compare missing files, recover examples, or inspect later packed experiment state
- `<legacy_old_repo>`
  - use only if a needed file is absent from `iMotion-LLM-ICLR`

This document is the working source-audit baseline for upcoming code migration.
