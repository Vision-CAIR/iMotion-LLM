# Repo Audit

Date: `2026-04-23`

This audit reflects the current public release state of the repository after migration cleanup.

## Current Assessment

The repository is partially refactored and usable through the supported public surface, but it is not fully normalized internally.

Supported and reviewed:

- `scripts/*.sh`
- `configs/release/*.yaml`
- `trajgpt/train.py`
- `trajgpt/eval.py`
- `gameformer/interaction_prediction/train.py`
- `gameformer/data_preprocess_v09.py`
- `gameformer/nuplan_preprocess/data_preprocess_complex.py`
- `mtr/tools/train.py`
- `mtr/tools/test.py`

Legacy and not fully normalized:

- many archival files under `trajgpt/`, `gameformer/`, and other preserved experiment folders
- duplicate model variants and historical configs kept for traceability
- inherited package/module naming such as `minigpt4` retained internally for compatibility

## Checks Performed

- Python compile checks passed on the supported runtime entrypoints and patched helpers.
- Release YAML files under `configs/release/` parse successfully.
- Shell syntax checks passed for the public helper scripts.
- Private path and token scans were cleaned before this audit.

## Public Runtime Naming

The supported public release now exposes:

- model architecture name: `imotion_llm`
- public model aliases:
  - `IMotionLLMModel`
  - `IMotionLLMMTRModel`

Backward compatibility is still preserved internally through:

- `MiniGPT4`
- `gameformer_gpt`
- the inherited `minigpt4` package layout

## What Seems Runnable Now

From a repository-structure and reference-consistency perspective, these are the intended public workflows:

- Waymo preprocessing for GameFormer / iMotion-LLM
- NuPlan preprocessing for Open-Vocabulary InstructNuPlan
- GameFormer training through `gameformer/interaction_prediction/train.py`
- iMotion-LLM train/eval through `trajgpt/train.py` and `trajgpt/eval.py`
- MTR train/eval through `mtr/tools/train.py` and `mtr/tools/test.py`

## Remaining Risks

- The preserved legacy tree is much larger than the supported public surface. Not every archived script or config was validated end-to-end.
- Some runtime dependencies are heavy and environment-sensitive, so compile success does not guarantee full training success on a fresh machine.
- Full paper reproducibility still depends on uploading checkpoints, exact split manifests, and canonical experiment manifests. See [PAPER_REPRODUCIBILITY_CHECKLIST.md](PAPER_REPRODUCIBILITY_CHECKLIST.md).

## Recommendation

For public use, treat only the following as stable:

- `scripts/`
- `configs/release/`
- `docs/setup/`
- the specific Python entrypoints called by those scripts

Treat the rest of the migrated tree as:

- useful reference material
- source for future cleanup
- not yet guaranteed as a polished public API
