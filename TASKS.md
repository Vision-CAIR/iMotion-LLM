# iMotion-LLM Tasks

This file tracks the repo bootstrap, migration, cleanup, and reproducibility work.

## Phase 0: Repository Bootstrap

- [x] Create a fresh local clone in a new top-level `iMotion-LLM` directory.
- [x] Preserve the existing website content under `docs/`.
- [x] Review the paper and map the experiment surface to repo tasks.
- [x] Add migration planning documents and protected-source notes.
- [x] Create clean target directories for incoming code migration.

## Phase 1: Source Intake And Audit

- [x] Collect all legacy source directories/files that contain iMotion-LLM-related code.
- [x] Record each source in `migration/SOURCE_INVENTORY.md`.
- [x] Identify which files are direct iMotion-LLM code versus inherited MiniGPT-4/GameFormer legacy.
- [x] Mark files that must remain untouched in their original locations.
- [x] Map legacy filenames/functions to cleaned destination modules in this repo.

## Phase 2: Dataset And Preprocessing Migration

- [x] Migrate `InstructWaymo` data preparation code.
- [x] Migrate direction bucketing and GT/F/IF instruction generation logic.
- [x] Migrate `Open-Vocabulary InstructNuPlan` generation/validation pipeline.
- [x] Document external dataset prerequisites and expected directory layout.
- [x] Remove hard-coded local paths from the supported public entrypoints.

## Phase 3: Model And Training Migration

- [x] Migrate the conditional trajectory baseline code.
- [x] Migrate iMotion-LLM projection, scene-mapper, and instruction-mapper modules.
- [x] Migrate LoRA/LLM fine-tuning code and model-loading utilities.
- [ ] Clean outdated naming inherited from older parent projects where safe.
- [ ] Separate reusable library code from experiment entrypoints.

## Phase 4: Evaluation And Reproduction

- [x] Migrate evaluation code for `minADE`, `minFDE`, and instruction-following recall.
- [x] Migrate feasibility and safety detection evaluation.
- [x] Reconstruct baseline public configs/scripts for training and evaluation.
- [ ] Reconstruct configs for each specific paper table and ablation.
- [x] Add qualitative visualization/export scripts if available.

## Phase 5: Cleanup And Release Readiness

- [ ] Remove dead experimental branches, stale comments, and obsolete utilities from migrated copies.
- [x] Add minimal setup instructions and runnable examples.
- [ ] Verify naming consistency across modules, configs, and docs.
- [x] Add a clear "available now" vs "still missing" section to the main README.
- [ ] Prepare the repo for commit/push once migration work is complete.
