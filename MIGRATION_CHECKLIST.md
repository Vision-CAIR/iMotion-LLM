# Migration Checklist

Status legend:

- `available`: present in this repo in usable form
- `scaffolded`: structure/docs exist, code not migrated yet
- `missing`: not yet found or not yet migrated

## Current Inventory

| Area | Paper Deliverable | Status | Notes |
| --- | --- | --- | --- |
| Project website | Public project page under `docs/` | available | Already existed in the remote repo and was preserved. |
| Repository bootstrap | README, task tracker, migration inventory, paper scope notes | available | Added during repo initialization. |
| Clean package layout | `src/`, `configs/`, `scripts/`, `experiments/`, `data/` | available | Public release structure now exists and is partially wired. |
| InstructWaymo preprocessing | WOMD augmentation and instruction generation | available | Migrated under `gameformer/` and `instructions/`; cleanup is still ongoing. |
| Direction labels | GT / feasible / infeasible direction logic | available | Preserved in `instructions/direction_instructions.py` and related helpers. |
| Conditional baseline | Conditional GameFormer or equivalent baseline code | available | Migrated under `gameformer/`. |
| iMotion-LLM core model | LLM projection, scene mapper, instruction mapper, decoder integration | available | Migrated under `trajgpt/`. |
| LLM fine-tuning | LoRA setup and training pipeline | available | Public release configs and scripts added. |
| Open-Vocabulary InstructNuPlan pipeline | Instruction generation, curation, or validation code | available | Migrated, with path cleanup on the supported release path. |
| InstructWaymo evaluation | IFR, feasibility detection, minADE/minFDE | available | Evaluation helpers are present, but not all paper tables are re-manifested yet. |
| InstructNuPlan evaluation | Safety detection and safe-instruction IFR | available | Code is present; public checkpoint packaging remains TODO. |
| Ablations | Balanced sampling, LLM backbones, mapper depth, fine-tuning strategy | available | Legacy YAMLs are preserved; release curation is still ongoing. |
| Qualitative tools | Visualization or export helpers for examples/figures | available | Preserved across `tools/`, `gameformer/`, and legacy experiment folders. |
| Pretrained checkpoints | Weights or load instructions | scaffolded | Load instructions are documented, but public research checkpoints are not bundled. |
| Repro instructions | End-to-end commands for data prep, training, eval | available | Added under `docs/setup/` and `scripts/`. |

## Availability Summary

Available now:

- Project website assets in `docs/`
- Migration planning docs
- Migrated code copied from the protected source repo
- Release configs, helper scripts, and environment files
- Setup, checkpoint, and runtime documentation

Still TODO:

- Public research checkpoint hosting
- Verified paper-table manifests
- More aggressive cleanup of archival duplicate legacy files
