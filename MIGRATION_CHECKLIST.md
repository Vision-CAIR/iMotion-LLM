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
| Clean package layout | `src/`, `configs/`, `scripts/`, `experiments/`, `data/` | scaffolded | Ready for migration targets. |
| InstructWaymo preprocessing | WOMD augmentation and instruction generation | missing | Expected from legacy source code. |
| Direction labels | GT / feasible / infeasible direction logic | missing | Needs source recovery and validation. |
| Conditional baseline | Conditional GameFormer or equivalent baseline code | missing | Needed for paper parity and iMotion-LLM initialization. |
| iMotion-LLM core model | LLM projection, scene mapper, instruction mapper, decoder integration | missing | Main migration target. |
| LLM fine-tuning | LoRA setup and training pipeline | missing | Must remove local paths and old project assumptions. |
| Open-Vocabulary InstructNuPlan pipeline | Instruction generation, curation, or validation code | missing | May be split across scripts/notebooks/utilities. |
| InstructWaymo evaluation | IFR, feasibility detection, minADE/minFDE | missing | Needed to reproduce Table 2 and related analysis. |
| InstructNuPlan evaluation | Safety detection and safe-instruction IFR | missing | Needed to reproduce Table 3 and Table 4. |
| Ablations | Balanced sampling, LLM backbones, mapper depth, fine-tuning strategy | missing | Needed for Tables 4-7. |
| Qualitative tools | Visualization or export helpers for examples/figures | missing | Optional if unavailable, but should be tracked. |
| Pretrained checkpoints | Weights or load instructions | missing | Track separately if hosted externally. |
| Repro instructions | End-to-end commands for data prep, training, eval | missing | To be added after code migration. |

## Availability Summary

Available now:

- Project website assets in `docs/`
- Migration planning docs
- Clean destination layout for code intake

Still TODO once source code is provided:

- All executable research code
- Experiment configs
- Evaluation scripts
- Reproduction instructions
- Checkpoints and release packaging
