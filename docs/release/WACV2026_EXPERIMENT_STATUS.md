# WACV 2026 Experiment Status

This note maps the final WACV 2026 paper to the public code release.

Canonical paper links:

- WACV Open Access page: <https://openaccess.thecvf.com/content/WACV2026/html/Felemban_iMotion-LLM_Instruction-Conditioned_Trajectory_Generation_WACV_2026_paper.html>
- WACV PDF: <https://openaccess.thecvf.com/content/WACV2026/papers/Felemban_iMotion-LLM_Instruction-Conditioned_Trajectory_Generation_WACV_2026_paper.pdf>
- arXiv preprint: <https://arxiv.org/abs/2406.06211>

## Table 2: InstructWaymo Main Comparison

Rows in paper:

- `GameFormer`
- `MTR`
- `C-GameFormer`
- `C-MTR`
- `ProSim-Instruct`
- `iMotion-LLM`

Public code status:

- `GameFormer`: present
  - train: `scripts/train_gameformer_waymo.sh`
  - eval: `scripts/eval_gameformer_waymo.sh`
  - unconditional row: run without `--act --act_dec`
- `C-GameFormer`: present
  - train: `scripts/train_gameformer_waymo.sh --act --act_dec --level 1`
  - eval: `scripts/eval_gameformer_waymo.sh --act --act_dec --level 1`
- `MTR`: present
  - train: `scripts/train_mtr_waymo_base.sh`
  - eval: `scripts/eval_mtr_waymo_base.sh`
- `C-MTR`: present
  - train: `scripts/train_mtr_waymo.sh`
  - eval: `scripts/eval_mtr_waymo.sh`
- `iMotion-LLM`: present
  - train: `scripts/train_imotion_waymo.sh`
  - eval: `scripts/eval_imotion_waymo.sh`
- `ProSim-Instruct`: not included in this repo

Important evaluation modes:

- `gt1`: ground-truth instruction
- `pos1`: feasible alternative instruction
- `neg1`: infeasible instruction

Paper-relevant code pointers:

- direction-conditioned dataset and class-balanced logic:
  - `trajgpt/minigpt4/datasets/datasets/traj_dataset.py`
- Waymo eval manifest builders:
  - `trajgpt/prepare_eval_meta_data.py`
  - `trajgpt/prepare_eval_meta_data_5cls.py`

Required release artifacts for strict reproduction:

- Waymo checkpoints for all reported rows
- `meta_gt1.json`
- `meta_pos1.json`
- `meta_neg1.json`
- GF↔MTR mapping JSONs used by MTR evaluation

## Table 3: Open-Vocabulary InstructNuPlan

Paper claims:

- safe/unsafe reasoning
- with-context and without-context evaluation
- safe-instruction trajectory-following evaluation

Public code status:

- `iMotion-LLM`: present
  - train: `scripts/train_imotion_nuplan.sh`
  - eval: `scripts/eval_imotion_nuplan.sh`
- conditional GameFormer baseline: legacy code path exists and now has public wrappers
  - train: `scripts/train_gameformer_nuplan.sh`
  - eval: `scripts/eval_gameformer_nuplan.sh`

Important evaluation modes:

- `safe_with_context`
- `safe_no_context`
- `unsafe_with_context`
- `unsafe_no_context`

Paper-relevant code pointers:

- prompt generation and metadata construction:
  - `gameformer/nuplan_preprocess/`
- OpenAI-dependent prompt generation step:
  - `gameformer/nuplan_preprocess/call_chatgpt.py`
- NuPlan-capable dataset path:
  - `trajgpt/minigpt4/datasets/datasets/traj_dataset.py`

Release blocker:

- raw-data-only reproduction is not enough for the exact released benchmark because part of the instruction generation uses OpenAI-based prompt synthesis
- the generated prompt and metadata bundle should therefore be released as a public artifact

## Table 4: LLM Backbone Comparison

Legacy configs are present and preserved.

Representative config paths:

- `trajgpt/train_configs_mar_2025/ablation_llms/llama_2_7b_contrastive.yaml`
- `trajgpt/train_configs_mar_2025/ablation_llms/mistral_7b_instruct.yaml`
- `trajgpt/train_configs_mar_2025/ablation_llms/vicuna_7b.yaml`
- `trajgpt/train_configs_mar_2025/ablation_llms/qwen2_7b_instruct.yaml`
- `trajgpt/train_configs_mar_2025/ablation_llms/llama_3_2_1b_instruct.yaml`
- `trajgpt/train_configs_mar_2025/ablation_llms/llama_2_13b.yaml`

Status:

- configs are present
- public wrapper surface is still centered on the main release setup rather than every ablation config

## Table 5: Class-Balanced Sampling

Code coverage:

- weighted sampling logic lives in `trajgpt/minigpt4/datasets/datasets/traj_dataset.py`
- release-trace config example: `trajgpt/train_configs_mar_2025/train_imotion_e2e_cb.yaml`
- no-balanced counterpart examples exist in preserved ablation folders, e.g. `trajgpt/2_iccv_rebuttal_may_2025/mistral_proj_both/09_mistral_7b_noCB.yaml`

Status:

- code and configs exist
- this remains a legacy-config-driven experiment rather than a first-class release config

## Table 6: Projection-Layer Ablation

Representative config paths:

- `trajgpt/2_iccv_rebuttal_may_2025/03_llama_7b_linearIn_linearOut.yaml`
- `trajgpt/2_iccv_rebuttal_may_2025/04_llama_7b_2mlpIn_2mlpOut.yaml`
- `trajgpt/2_iccv_rebuttal_may_2025/05_llama_7b_4mlpIn_4mlpOut.yaml`
- `trajgpt/2_iccv_rebuttal_may_2025/06_llama_7b_2mlpIn_2mlpOut_inShared.yaml`
- `trajgpt/2_iccv_rebuttal_may_2025/mistral_proj_both/03_mistral_7b_linearIn_linearOut.yaml`
- `trajgpt/2_iccv_rebuttal_may_2025/mistral_proj_both/05_mistral_7b_4mlpIn_4mlpOut.yaml`

Status:

- preserved and traceable
- not yet promoted to cleaned release configs

## Table 7: Fine-Tuning Strategy

Representative config paths:

- fully fine-tuned: `trajgpt/train_configs_mar_2025/train_imotion_e2e.yaml`
- decoder-only: `trajgpt/train_configs_mar_2025/train_imotion_e2e_decOnly.yaml`
- freeze backbone / encoder variants:
  - `trajgpt/2_iccv_rebuttal_may_2025/07_llama_7b_freezeGf.yaml`
  - `trajgpt/2_iccv_rebuttal_may_2025/08_llama_7b_freezeDec.yaml`
  - `trajgpt/2_iccv_rebuttal_may_2025/mistral_proj_both/07_mistral_7b_freezeGf.yaml`
  - `trajgpt/2_iccv_rebuttal_may_2025/mistral_proj_both/08_mistral_7b_freezeEnc.yaml`

Status:

- preserved in legacy config form

## Supplementary-Only Experiment Areas

Additional preserved experiment areas include:

- multi-agent settings:
  - `trajgpt/3_wacv_rebuttal_15_sep_2025/`
- cross-dataset and nuanced nuPlan evaluation variants:
  - `trajgpt/3_wacv_rebuttal_15_sep_2025/eval_complex/`

## What Users Must Download

Users should download these directly from their official sources:

- raw Waymo Open Motion Dataset files
- raw nuPlan files and maps
- gated base LLM weights such as `meta-llama/Llama-2-7b-hf`

## What We Should Release Publicly

For a strong public reproduction package, we should upload:

- released research checkpoints
- Waymo evaluation manifests: `meta_gt1.json`, `meta_pos1.json`, `meta_neg1.json`
- GF↔MTR mapping JSONs
- generated Open-Vocabulary InstructNuPlan prompt / metadata bundle
- exact config files used for the reported checkpoints

## Bottom Line

Most paper experiments are represented in the recovered codebase.

What is still missing for a strict public reproduction story is not the bulk of the code, but the release bundle:

- checkpoints
- manifests
- generated NuPlan prompt artifacts
- exact table-to-config mapping for the final reported runs
