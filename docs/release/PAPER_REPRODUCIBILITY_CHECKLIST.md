# Paper Reproducibility Checklist

This checklist maps the paper claims to the assets and configs that a public release still needs.

Paper:

- `iMotion-LLM: Instruction-Conditioned Trajectory Generation`
- arXiv: `2406.06211`

## 1. Canonical Public Entry Surface

Use these as the intended public release entrypoints:

- `scripts/preprocess_waymo_gameformer.sh`
- `scripts/preprocess_nuplan_complex.sh`
- `scripts/train_gameformer_waymo.sh`
- `scripts/train_imotion_waymo.sh`
- `scripts/eval_imotion_waymo.sh`
- `scripts/train_imotion_nuplan.sh`
- `scripts/eval_imotion_nuplan.sh`
- `scripts/train_mtr_waymo.sh`
- `scripts/eval_mtr_waymo.sh`

Canonical release configs:

- `configs/release/imotion_waymo_train.yaml`
- `configs/release/imotion_waymo_eval.yaml`
- `configs/release/imotion_nuplan_train.yaml`
- `configs/release/imotion_nuplan_eval.yaml`
- `configs/release/mtr_waymo_act.yaml`

## 2. Main Paper Results And Required Assets

### InstructWaymo Main Comparison

Expected rows to support:

- `GameFormer`
- `C-GameFormer`
- `MTR`
- `C-MTR`
- `iMotion-LLM`

Required for a strong public reproduction package:

- released checkpoint for unconditional GameFormer
- released checkpoint for conditional GameFormer
- released checkpoint for unconditional MTR
- released checkpoint for conditional MTR
- released checkpoint for iMotion-LLM on Waymo
- exact evaluation split manifest for the `1,500` balanced examples
- exact command/config used for each row

Current status:

- `C-GameFormer` training surface exists
- `MTR / C-MTR` training surface exists
- `iMotion-LLM` train/eval surface exists
- canonical public configs for unconditional GameFormer and unconditional MTR rows are still incomplete as first-class release configs

### Open-Vocabulary InstructNuPlan

Expected capabilities to support:

- safe vs unsafe instruction reasoning
- with-context and without-context evaluation
- safe-instruction IFR

Required for a strong public reproduction package:

- released conditional GameFormer NuPlan checkpoint
- released iMotion-LLM NuPlan checkpoint
- exact evaluation split manifest for the `2,078` examples
- exact category definitions for the four test buckets
- exact prompt-generation or metadata bundle used to create the released dataset

Current status:

- NuPlan training surface exists
- NuPlan eval release config now exists
- dataset-generation logic exists in migrated form
- public split manifests and released checkpoints are still missing

## 3. Checkpoints That Should Exist Publicly

Minimum recommended checkpoint list:

- `checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth`
- `checkpoints/imotion_llm/waymo/checkpoint_last.pth`
- `checkpoints/mtr/waymo/checkpoint_epoch_15.pth`
- `checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth`
- `checkpoints/imotion_llm/nuplan/checkpoint_last.pth`

Recommended additions for paper-table completeness:

- unconditional GameFormer Waymo checkpoint
- unconditional MTR Waymo checkpoint
- ablation checkpoints for the major released paper ablations if you want users to rerun those tables directly

## 4. Metadata And Small Files That Should Exist Publicly

- `trajgpt/KBinsDiscretizer_76.pkl`
- `data/processed/waymo/mtr/output/cluster_64_center_dict.pkl`
- Waymo evaluation split manifest
- NuPlan evaluation split manifest
- GF↔MTR mapping JSONs if required by released evaluation code
- any prompt-template metadata bundle required for released NuPlan evaluation

## 5. Config Coverage

Already present as canonical release configs:

- Waymo iMotion-LLM train
- Waymo iMotion-LLM eval
- NuPlan iMotion-LLM train
- NuPlan iMotion-LLM eval
- Waymo MTR act train/eval config

Still worth adding later for stronger paper reproducibility:

- explicit unconditional GameFormer Waymo release config
- explicit conditional GameFormer Waymo eval config
- explicit unconditional MTR Waymo release config
- explicit per-table ablation manifests linking:
  - checkpoint
  - config
  - command
  - expected metrics

## 6. What Blocks “Fully Reproducible” Today

The repo is not yet fully reproducible against the paper in a strict public-release sense because it still lacks:

- all released research checkpoints
- exact public split manifests for the reported test sets
- canonical baseline configs for every table row
- canonical ablation manifests for the supplementary comparisons
- fresh-machine validation with a clean environment and public-only assets

## 7. Recommended Next Release Steps

1. Upload the five minimum recommended checkpoints.
2. Upload the two split manifests and the small supporting artifacts.
3. Add missing canonical baseline configs for unconditional GameFormer and unconditional MTR.
4. Create per-table reproduction manifests under `experiments/` or `docs/release/`.
5. Run one clean-machine reproduction pass using only the public repo plus uploaded artifacts.
