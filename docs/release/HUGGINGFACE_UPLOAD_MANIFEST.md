# Hugging Face Upload Manifest

## Status: datasets published, checkpoints outstanding

**Published (2026-10-13)** — under the `KAUST` org:

| Repo | Contents |
|---|---|
| [`KAUST/InstructNuPlan`](https://huggingface.co/datasets/KAUST/InstructNuPlan) | Open-Vocabulary InstructNuPlan: LLM-generated instructions with reasoning and safety labels over 14 nuPlan scenario types; prompts and scenario metadata for provenance |
| [`KAUST/InstructWaymo`](https://huggingface.co/datasets/KAUST/InstructWaymo) | InstructWaymo instruction/reasoning/decision records; the exact evaluation manifests (`meta_gt1.json`, `meta_pos1.json`, `meta_neg1.json`, `eval_meta.json`); GameFormer-to-MTR index mappings |

Fetch with `tools/instruct_data/download_instruct_data.py`.

Both are stored as gzipped JSONL shards rather than the original millions of
per-scene files, so they load in one pass; `_source` on every row preserves the
original path and `--expand` rebuilds the original tree (raw-text files come
back byte-identical).

**Deliberately not uploaded** — reproducible from code in this repo:

- `npz` scenario tensor caches: ~912 GiB (nuPlan) + ~432 GiB (Waymo)
- rendered scenario figures (~22 GiB of `.png`)

**Still outstanding** — the checkpoints listed under Priority 1 below.

---


This file lists the artifacts that should be uploaded outside the Git repo for a practical public release.

## What Users Should Download From Official Sources

Users should obtain these directly from the original providers:

- raw Waymo Open Motion Dataset files
- raw nuPlan dataset files
- raw nuPlan map files
- gated base LLM weights such as Meta Llama checkpoints

These should stay external because of licensing, scale, or gated-access constraints.

## What We Should Upload Ourselves

The release artifacts below are the pieces that make the repo practical and reproducible.

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
- `checkpoints/gameformer/waymo/gf_l1/epochs_29.pth`
  Purpose: unconditional GameFormer baseline row for Table 2
- `checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth`
  Purpose: conditional GameFormer backbone used by the NuPlan release flow
- `checkpoints/gameformer/nuplan/gf_l1/epochs_29.pth`
  Purpose: unconditional GameFormer NuPlan baseline if released

### Baseline Checkpoints For Paper Reproduction

- `checkpoints/mtr/waymo/checkpoint_epoch_15.pth`
  Purpose: released conditional MTR baseline checkpoint
- `checkpoints/mtr/waymo_base/checkpoint_epoch_15.pth`
  Purpose: unconditional MTR baseline row for Table 2

## Priority 2: Small Supporting Artifacts To Upload

These are lightweight but important for users who want reproducible behavior.

- `trajgpt/KBinsDiscretizer_76.pkl`
  Purpose: discretization artifact used by parts of the migrated trajectory pipeline
- `data/processed/waymo/mtr/output/cluster_64_center_dict.pkl`
  Purpose: MTR intention-point file referenced by the public runtime
- Waymo evaluation manifests for the `1,500` balanced examples
  Suggested files:
  - `meta_gt1.json`
  - `meta_pos1.json`
  - `meta_neg1.json`
- NuPlan test split manifest for the `2,078` evaluation examples
  Suggested format: JSON or CSV with scenario identifiers and category labels
- any GF↔MTR mapping JSONs needed by released evaluation code
  Suggested format: a small `metadata/` bundle rather than embedding them in large processed datasets
- generated Open-Vocabulary InstructNuPlan prompt and metadata bundle
  Purpose: deterministic public reuse of the benchmark without requiring OpenAI-based regeneration

## Priority 3: Config Bundle To Upload Alongside Checkpoints

These config files should be uploaded or duplicated in the model card release assets.

- `configs/release/imotion_waymo_train.yaml`
- `configs/release/imotion_waymo_eval.yaml`
- `configs/release/imotion_nuplan_train.yaml`
- `configs/release/imotion_nuplan_eval.yaml`
- `configs/release/mtr_waymo.yaml`
- `configs/release/mtr_waymo_act.yaml`

Also include the public helper scripts or link the exact Git commit that contains them:

- `scripts/train_gameformer_waymo.sh`
- `scripts/eval_gameformer_waymo.sh`
- `scripts/train_imotion_waymo.sh`
- `scripts/eval_imotion_waymo.sh`
- `scripts/train_gameformer_nuplan.sh`
- `scripts/eval_gameformer_nuplan.sh`
- `scripts/train_imotion_nuplan.sh`
- `scripts/eval_imotion_nuplan.sh`
- `scripts/train_mtr_waymo_base.sh`
- `scripts/eval_mtr_waymo_base.sh`
- `scripts/train_mtr_waymo.sh`
- `scripts/eval_mtr_waymo.sh`

## Recommended Hugging Face Layout

```text
imotion-llm-release/
├── README.md
├── configs/
├── metadata/
│   ├── instructwaymo_eval/
│   │   ├── meta_gt1.json
│   │   ├── meta_pos1.json
│   │   └── meta_neg1.json
│   ├── nuplan_eval_split.json
│   ├── nuplan_prompt_bundle/
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

- whether the generated NuPlan prompt bundle is released as standalone metadata or as part of a larger processed-data package
- whether to release only final checkpoints or also intermediate ablation checkpoints
- whether to store all checkpoints in one HF repo or split them into:
  - `Vision-CAIR/iMotion-LLM-checkpoints`
  - `Vision-CAIR/iMotion-LLM-metadata`

## Minimum Viable Public Release

If time is limited, upload at least:

- main Waymo iMotion-LLM checkpoint
- Waymo conditional GameFormer checkpoint
- Waymo unconditional GameFormer checkpoint
- NuPlan iMotion-LLM checkpoint
- NuPlan conditional GameFormer checkpoint
- MTR checkpoint
- unconditional MTR checkpoint
- `KBinsDiscretizer_76.pkl`
- `cluster_64_center_dict.pkl`
- the `configs/release/` files
- the exact Waymo evaluation manifests
- the GF↔MTR mapping JSONs
- the NuPlan evaluation metadata and generated prompt bundle
