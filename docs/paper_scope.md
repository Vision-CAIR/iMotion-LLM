# Paper Scope

This note maps the final paper to the code and experiment surface that this repository is expected to deliver.

Paper:

- `iMotion-LLM: Instruction-Conditioned Trajectory Generation`
- WACV 2026 Open Access page: <https://openaccess.thecvf.com/content/WACV2026/html/Felemban_iMotion-LLM_Instruction-Conditioned_Trajectory_Generation_WACV_2026_paper.html>
- WACV 2026 PDF: <https://openaccess.thecvf.com/content/WACV2026/papers/Felemban_iMotion-LLM_Instruction-Conditioned_Trajectory_Generation_WACV_2026_paper.pdf>
- arXiv preprint: <https://arxiv.org/abs/2406.06211>

## Core Contributions To Support In Code

The paper describes a trajectory-generation system that combines:

- a scene encoder and multimodal trajectory decoder baseline
- an LLM with LoRA fine-tuning
- a projection from scene features into the LLM token space
- a scene mapper back into the motion model space
- an instruction mapper that conditions the motion queries
- text outputs for feasibility, safety reasoning, and justification

## Datasets Mentioned In The Paper

### 1. InstructWaymo

Primary use:

- direction-conditioned controllable trajectory generation
- feasible vs infeasible direction reasoning

Paper setup highlights:

- built from Waymo Open Motion Dataset preprocessing
- approximately `1.3M` usable vehicle-centric conditional samples
- evaluation set of `1,500` examples balanced across GT / feasible / infeasible instructions

Expected code surface:

- WOMD preprocessing glue
- direction bucketing
- GT / feasible / infeasible instruction assignment
- conditional training and evaluation

### 2. Open-Vocabulary InstructNuPlan

Primary use:

- safety-aware natural-language instruction understanding
- safe/unsafe classification with optional context

Paper setup highlights:

- `115,097` training examples
- `2,078` test examples
- test categories: safe without context, safe with context, unsafe without context, unsafe with context

Expected code surface:

- instruction generation / curation pipeline
- context handling
- safety evaluation
- safe-instruction trajectory-following evaluation

## Main Reported Results To Reproduce

### InstructWaymo

Key metrics:

- `GT-IFR`
- `F-IFR`
- `minADE`
- `minFDE`
- feasibility detection accuracy

Main comparison table in the paper includes:

- `GameFormer`
- `MTR`
- `C-GameFormer`
- `C-MTR`
- `ProSim-Instruct`
- `iMotion-LLM`

Headline result for `iMotion-LLM`:

- `GT-IFR = 87.30`
- `F-IFR = 52.24`
- `minADE/minFDE = 0.67 / 1.25`

### Open-Vocabulary InstructNuPlan

Key metrics:

- safety detection accuracy
- safe-instruction IFR

Headline results reported:

- safe without context: `96.61%` accuracy, `68.96` IFR
- safe with context: `98.39%` accuracy, `70.22` IFR
- unsafe without context: `95.00%` accuracy
- unsafe with context: `96.88%` accuracy

## Ablations Mentioned In The Paper

The migrated code should ideally cover:

- class-balanced sampling
- LLM backbone comparison
- projection layer depth / configuration
- backbone fine-tuning strategy

Selected reference numbers from the paper:

- best reported LLM backbone in the main setup: `LLaMA-7B`
- LoRA settings in supplementary: rank `32`, alpha `16`, dropout `0.05`, batch size `16`
- training strategy with the best overall balance: fine-tune trajectory decoder only

## Current Public Coverage

The cleaned public release now includes:

- iMotion-LLM training and evaluation entrypoints for Waymo and nuPlan
- GameFormer / C-GameFormer training on Waymo and public evaluation wrappers for Waymo and nuPlan
- MTR / C-MTR public train and eval wrappers on Waymo
- legacy ablation configs kept under `trajgpt/` for paper traceability

## Known Remaining Gaps

- `ProSim-Instruct` is not part of this repository
- strict public reproducibility still requires uploaded checkpoints and manifests
- Open-Vocabulary InstructNuPlan generation is not fully self-contained from raw data because part of the prompt pipeline uses OpenAI-based generation; the generated prompt bundle should therefore be released

See [release/WACV2026_EXPERIMENT_STATUS.md](release/WACV2026_EXPERIMENT_STATUS.md) for the table-by-table status and concrete file paths.
