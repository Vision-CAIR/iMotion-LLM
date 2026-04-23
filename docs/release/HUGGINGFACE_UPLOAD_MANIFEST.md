# Hugging Face Upload Manifest

This file lists the artifacts that should be uploaded outside the Git repo for a practical public release.

## What Should Not Be Uploaded Here

Do not mirror these into this repository:

- raw Waymo Open Motion Dataset files
- raw nuPlan dataset files
- gated base LLM weights such as Meta Llama checkpoints

These should remain external because of licensing, scale, or gated-access constraints.

## Priority 1: Checkpoints To Upload

These are the most important artifacts for immediate public usability.

### Main iMotion-LLM Checkpoints

- `checkpoints/imotion_llm/waymo/checkpoint_last.pth`
  Purpose: main InstructWaymo iMotion-LLM release checkpoint
- `checkpoints/imotion_llm/nuplan/checkpoint_last.pth`
  Purpose: Open-Vocabulary InstructNuPlan release checkpoint

### Backbone Checkpoints Required By iMotion-LLM

- `checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth`
  Purpose: conditional GameFormer backbone used by the Waymo release flow
- `checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth`
  Purpose: conditional GameFormer backbone used by the NuPlan release flow

### Baseline Checkpoints For Paper Reproduction

- `checkpoints/mtr/waymo/checkpoint_epoch_15.pth`
  Purpose: released MTR / C-MTR baseline checkpoint
- optional unconditional GameFormer Waymo checkpoint
  Purpose: paper-table reproduction for unconditional baseline rows
- optional unconditional MTR Waymo checkpoint
  Purpose: paper-table reproduction for unconditional baseline rows

## Priority 2: Small Supporting Artifacts To Upload

These are lightweight but important for users who want reproducible behavior.

- `trajgpt/KBinsDiscretizer_76.pkl`
  Purpose: discretization artifact used by parts of the migrated trajectory pipeline
- `data/processed/waymo/mtr/output/cluster_64_center_dict.pkl`
  Purpose: MTR intention-point file referenced by the public runtime
- Waymo evaluation split manifest for the `1,500` balanced examples
  Suggested format: JSON or CSV with scenario/object identifiers
- NuPlan test split manifest for the `2,078` evaluation examples
  Suggested format: JSON or CSV with scenario identifiers and category labels
- any GF↔MTR mapping JSONs needed by released evaluation code
  Suggested format: a small `metadata/` bundle rather than embedding them in large processed datasets

## Priority 3: Config Bundle To Upload Alongside Checkpoints

These config files should be uploaded or duplicated in the model card release assets.

- `configs/release/imotion_waymo_train.yaml`
- `configs/release/imotion_waymo_eval.yaml`
- `configs/release/imotion_nuplan_train.yaml`
- `configs/release/imotion_nuplan_eval.yaml`
- `configs/release/mtr_waymo_act.yaml`

Also include the public helper scripts or link the exact Git commit that contains them:

- `scripts/train_imotion_waymo.sh`
- `scripts/eval_imotion_waymo.sh`
- `scripts/train_imotion_nuplan.sh`
- `scripts/eval_imotion_nuplan.sh`
- `scripts/train_gameformer_waymo.sh`
- `scripts/train_mtr_waymo.sh`
- `scripts/eval_mtr_waymo.sh`

## Recommended Hugging Face Layout

One practical layout is:

```text
imotion-llm-release/
├── README.md
├── configs/
├── metadata/
│   ├── instructwaymo_eval_split.json
│   ├── nuplan_eval_split.json
│   └── mtr_mapping/
├── checkpoints/
│   ├── imotion_llm/
│   ├── gameformer/
│   └── mtr/
└── artifacts/
    ├── KBinsDiscretizer_76.pkl
    └── cluster_64_center_dict.pkl
```

## Still To Decide Before Upload

- whether processed prompt/caption datasets can be redistributed under Waymo / nuPlan terms
- whether to release only final checkpoints or also intermediate ablation checkpoints
- whether to store all checkpoints in one HF repo or split them into:
  - `Vision-CAIR/iMotion-LLM-checkpoints`
  - `Vision-CAIR/iMotion-LLM-metadata`

## Minimum Viable Public Release

If time is limited, upload at least:

- main Waymo iMotion-LLM checkpoint
- Waymo conditional GameFormer checkpoint
- NuPlan iMotion-LLM checkpoint
- NuPlan conditional GameFormer checkpoint
- MTR checkpoint
- `KBinsDiscretizer_76.pkl`
- `cluster_64_center_dict.pkl`
- the `configs/release/` files
- the exact evaluation split manifests
