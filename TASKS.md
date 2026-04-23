# iMotion-LLM Tasks

This file tracks the repo bootstrap, migration, cleanup, and reproducibility work.

## Phase 0: Repository Bootstrap

- [x] Create a fresh local clone in a new top-level `iMotion-LLM` directory.
- [x] Preserve the existing website content under `docs/`.
- [x] Review the paper and map the experiment surface to repo tasks.
- [x] Add migration planning documents and protected-source notes.
- [x] Create clean target directories for incoming code migration.

## Phase 1: Source Intake And Audit

- [ ] Collect all legacy source directories/files that contain iMotion-LLM-related code.
- [ ] Record each source in `migration/SOURCE_INVENTORY.md`.
- [ ] Identify which files are direct iMotion-LLM code versus inherited MiniGPT-4/GameFormer legacy.
- [ ] Mark files that must remain untouched in their original locations.
- [ ] Map legacy filenames/functions to cleaned destination modules in this repo.

## Phase 2: Dataset And Preprocessing Migration

- [ ] Migrate `InstructWaymo` data preparation code.
- [ ] Migrate direction bucketing and GT/F/IF instruction generation logic.
- [ ] Migrate `Open-Vocabulary InstructNuPlan` generation/validation pipeline.
- [ ] Document external dataset prerequisites and expected directory layout.
- [ ] Remove hard-coded local paths and replace them with config/CLI inputs.

## Phase 3: Model And Training Migration

- [ ] Migrate the conditional trajectory baseline code.
- [ ] Migrate iMotion-LLM projection, scene-mapper, and instruction-mapper modules.
- [ ] Migrate LoRA/LLM fine-tuning code and model-loading utilities.
- [ ] Clean outdated naming inherited from older parent projects where safe.
- [ ] Separate reusable library code from experiment entrypoints.

## Phase 4: Evaluation And Reproduction

- [ ] Migrate evaluation code for `minADE`, `minFDE`, and instruction-following recall.
- [ ] Migrate feasibility and safety detection evaluation.
- [ ] Reconstruct configs/scripts needed for the main paper tables.
- [ ] Reconstruct ablation configs for LLM backbone, projection depth, and fine-tuning strategy.
- [ ] Add qualitative visualization/export scripts if available.

## Phase 5: Cleanup And Release Readiness

- [ ] Remove dead experimental branches, stale comments, and obsolete utilities from migrated copies.
- [ ] Add minimal setup instructions and runnable examples.
- [ ] Verify naming consistency across modules, configs, and docs.
- [ ] Add a clear "available now" vs "still missing" section to the main README.
- [ ] Prepare the repo for commit/push once migration work is complete.
