# Paper Refactoring Requirements

This file is the implementation-facing extraction of the paper.
Its purpose is to define what code, configs, data logic, and evaluation behavior must exist in the cleaned repository after migration.

Paper reference:

- `iMotion-LLM: Instruction-Conditioned Trajectory Generation`
- arXiv: <https://arxiv.org/abs/2406.06211>
- Main paper plus supplementary were used to prepare this checklist.

## How To Use This File

When auditing legacy code, every candidate file should be checked against one or more requirements below:

- `required`: needed for the paper’s core claims or main tables
- `important`: strongly expected for reproducibility and clean release
- `optional`: supplementary or nice-to-have if source exists
- `missing`: not yet found in migrated code

The migration is complete only when all `required` items are either:

- implemented in this repo, or
- explicitly marked unavailable with a reason

## 1. Datasets And Data Generation Requirements

### 1.1 InstructWaymo

Status target: `required`

We need code that does all of the following:

- augment WOMD preprocessing into instruction-conditioned samples
- preprocess approximately `400K` Waymo driving scenarios in a GameFormer-like format
- support scenes with up to `32` neighbors plus the focal agent (`33` total agents)
- treat each agent as a possible focal agent, producing the `4.2M` sample construction described in the paper
- filter down to the approximately `1.3M` usable vehicle-centric samples with valid trajectories and detected instructions
- restrict trajectory training cases to focal agents that are vehicles

We need the following direction categories and associated logic:

- `Stationary`
- `Straight`
- `Right`
- `Left`
- `Left U-turn`

Expected direction-class counts mentioned in the paper:

- `Stationary`: `16,748`
- `Straight`: `863,910`
- `Right`: `184,286`
- `Left`: `221,762`
- `Left U-turn`: `4,575`

We need code for direction extraction and heuristics:

- ground-truth direction extraction from future trajectory
- feasible direction derivation
- infeasible direction derivation as the complement of feasible directions
- candidate destination lanes relative to current ego position and heading
- reachable range based on current speed plus a max speed increase of `15 km/h` within `8 seconds`
- range capped by the road speed limit and by a maximum of `60 meters`
- stationary feasibility rule that allows stopping within `8 seconds` when current speed is within `65 km/h`
- configurable heuristics rather than fixed hard-coded constants

We need code for additional motion description attributes:

- `5` speed categories
- `9` acceleration/deceleration categories
- caption generation that includes:
  - final target direction
  - two intermediate wayfinding steps
  - indicative speed
  - indicative acceleration

We need instruction/caption generation behavior:

- input instruction should specify the target final direction
- output caption should describe how the instruction is executed
- captions should be suitable for LLM autoregressive generation

We need evaluation split behavior:

- an evaluation set of `1,500` examples balanced across `GT`, `F`, and `IF`
- test data filtering/verification logic if it exists in legacy code

Important supplementary details to preserve if found:

- speed-category thresholds: `20, 40, 90, 120, >120 km/h`
- acceleration/deceleration thresholds over `8s`: `6, 25, 46, 65, >65 km/h`
- direction extraction thresholds:
  - `vstationary = 2 m/s`
  - `dstationary = 5 m`
  - `theta_straight = 30 degrees`
  - `dv = 5 m`
  - `du = 5 m`

### 1.2 Open-Vocabulary InstructNuPlan

Status target: `required`

We need code that does all of the following:

- build or load the open-vocabulary instruction dataset on top of NuPlan
- select the `14` scenario types described in the paper
- sample an equal number of scenarios per type
- produce `7,119` training scenarios and `124` testing scenarios
- expand those into `115,097` training instruction-caption pairs and `2,078` testing pairs
- use `1.1s` of history to predict `8s` into the future

Scenario types that should appear in code, configs, or metadata:

- accelerating at crosswalk
- traversing crosswalk
- waiting for pedestrian to cross
- following lane
- following lane with lead
- following lane with slow lead
- stopping with lead
- starting protected turn cross
- starting protected turn non-cross
- starting non-protected turn cross
- starting non-protected turn non-cross
- behind bike
- behind long vehicle
- traversing intersection

We need safety-grounding logic:

- scenario-type metadata defining safe and unsafe behaviors
- up to `10` safe and `10` unsafe behaviors per scenario
- behavior bucket matching based on ego future motion
- behavior types including:
  - not moving
  - stopping
  - waiting then moving
  - slowing down
  - speeding up
  - slowing down then speeding up
  - speeding up then slowing down
  - maintaining speed

We need instruction generation behavior:

- GPT-4o mini based generation or compatible replacement wrapper if the original script used it
- safe and unsafe instruction-caption pair generation
- two instruction modes:
  - with context
  - without context
- captions that retain context even when the instruction is context-free

We need test/eval split categories:

- safe without context
- safe with context
- unsafe without context
- unsafe with context

Important supplementary details to preserve if found:

- metadata or templates used to reduce GPT hallucinations
- the lightweight verification web tool for reviewing generated instruction-reasoning pairs
- quality-validation pipeline on a `10%` random sample using human/GPT scoring

## 2. Model And Architecture Requirements

### 2.1 Backbone Families

Status target: `required`

The repo should end up with clear support for these model families where paper results depend on them:

- `GameFormer`
- `MTR`
- `C-GameFormer`
- `C-MTR`
- `iMotion-LLM`

The following structure must be recognizable in migrated code:

- `Scene Encoder`
- `Multimodal Trajectory Decoder`
- motion queries
- keys/values from scene tokens
- Gaussian Mixture Model trajectory output

Important pseudocode-level details from the supplementary:

- observed steps: `tobs = 11`
- predicted steps: `tpred = 80`
- selected loss steps: `tselect = [29, 49, 79]`
- target agents in the default setup: `Npred = 2`
- multimodal outputs: `M` futures
- predicted output parameterization: `Pred in R^(M x Npred x tpred x 4)` for `(mu_x, mu_y, sigma_x, sigma_y)`

### 2.2 Conditional Baselines

Status target: `required`

We need code for the conditional adaptation used in `C-GameFormer` and `C-MTR`:

- a learnable instruction query `Q'`
- `Q'` fused into motion generation queries by element-wise addition
- categorical instruction embedding for non-LLM conditional baselines
- training/evaluation entrypoints that accept direction labels as conditions

The migration should preserve the distinction between:

- unconditional baseline behavior
- categorical conditional baseline behavior
- language-conditioned iMotion-LLM behavior

### 2.3 iMotion-LLM Core Modules

Status target: `required`

We need identifiable code for these four blocks from the paper:

- `LLM Projection`
- `LLM`
- `Scene Mapper`
- `Instruction Mapper`

Specific behaviors that should exist in code:

- linear projection from `dscene` to `dLLM`
- LLM input formed by concatenating instruction text embeddings with projected scene embeddings
- LLM output that includes:
  - an ego token
  - an instruction token
  - text/caption tokens
- scene mapper MLP from `dLLM` back to `dscene`
- replacement of the original ego token with the instruction-grounded ego token
- instruction mapper from LLM instruction token back into motion-query space
- fusion of mapped instruction query into the decoder queries

We also need output-text behavior:

- caption generation
- feasibility decision token equivalent to `[Accept]` / `[Reject]`
- safety justification text

The inference/training path should preserve the paper’s special-token behavior if present:

- generation token for instruction query, written in the supplementary as `[I]`
- scene-related generated tokens such as `[S1] ... [SN]`
- masked generation path during inference after the first trajectory token is detected

### 2.4 Fine-Tuning Strategy

Status target: `required`

We need code or configs that show:

- iMotion-LLM starts from a pretrained `C-GameFormer`
- InstructWaymo fine-tuning uses mixed supervision:
  - `70%` GT instructions with trajectories
  - `30%` IF instructions without trajectories
- text loss is always backpropagated
- trajectory loss is applied only when GT trajectory exists

We also need backbone-tuning options reflected in code or configs:

- frozen backbone
- trajectory decoder only
- scene encoder only
- fully finetuned

## 3. Training Configuration Requirements

### 3.1 Baseline Training

Status target: `important`

We need training scripts/configs that reflect the paper’s reported baseline setup:

- `GameFormer` and `C-GameFormer` trained from scratch on `4 x V100` for `30 epochs`
- `MTR` and `C-MTR` trained from scratch on `4 x V100` for `15 epochs`

### 3.2 iMotion-LLM Fine-Tuning

Status target: `required`

We need code/configs that preserve these reported settings:

- starting checkpoint: pretrained `C-GameFormer`
- InstructWaymo fine-tuning: `0.25` epochs, `6,726` iterations, `3 x A100`
- Open-Vocabulary InstructNuPlan fine-tuning: `1` epoch, `7,194` iterations
- 4-bit `bfloat16` fine-tuning
- optimizer: `AdamW`
- learning rate: `1e-4`
- max gradient norm: `10`
- cosine annealing scheduler
- warm-up: `0.1` epochs
- LoRA dropout: `0.05`
- batch size: `16`
- LoRA rank: `32`
- LoRA alpha: `16`
- higher learning rate for projection and mapping modules

### 3.3 LLM Backbones

Status target: `important`

We need evidence in code/configs that these backbones were supported or evaluated:

- `LLaMA-7B`
- `Mistral-7B`
- `LLaMA-1B`
- `Vicuna-7B`

The default main-paper setup should resolve to `LLaMA-7B`.

## 4. Evaluation Requirements

### 4.1 Core Trajectory Metrics

Status target: `required`

We need implementations or wrappers for:

- `minADE`
- `minFDE`

These must work for:

- unconditional baselines
- conditional baselines
- iMotion-LLM

### 4.2 Instruction Following Recall (IFR)

Status target: `required`

We need the exact paper-defined behavior as closely as possible:

- direction extraction from generated trajectories using the same direction-extraction logic used for ground truth
- evaluate `M = 6` generated futures per example
- count how many generated trajectories satisfy the instructed direction
- average accuracy separately per direction
- macro-average across directions for the final IFR

We need support for:

- `GT-IFR`
- `F-IFR`
- safe-instruction IFR on Open-Vocabulary InstructNuPlan

### 4.3 Feasibility And Safety Evaluation

Status target: `required`

We need feasibility/safety evaluation outputs for:

- `GT-Acc`
- `F-Acc`
- `IF-Acc`
- safe accuracy without context
- safe accuracy with context
- unsafe accuracy without context
- unsafe accuracy with context
- average safety accuracy

We also need evaluation behavior that matches the paper:

- no IFR reported for unsafe instructions because no GT trajectory exists

### 4.4 Text And Reasoning Evaluation

Status target: `optional`

If present in legacy code, preserve:

- justification quality evaluation tooling
- scoring rubric for:
  - safety and risk awareness
  - consistency with instruction
  - clarity and driving-principle justification
- web-based annotation/review helpers

Reported supplementary scores worth preserving if such code exists:

- overall average: `7.81`
- safety and risk awareness: `7.91`
- consistency with instruction: `8.11`
- clarity and driving-principle justification: `7.41`

### 4.5 Efficiency Analysis

Status target: `optional`

If the code exists, preserve latency/memory benchmarking for:

- iMotion-LLM `7B` with text: `2100 ms`, `7697 MB`
- iMotion-LLM `7B` no text: `250 ms`, `7010 MB`
- iMotion-LLM `1B` with text: `1200 ms`, `2890 MB`
- iMotion-LLM `1B` no text: `130 ms`, `2600 MB`
- `GameFormer`: `40 ms`, `139 MB`
- `C-GameFormer`: `37 ms`, `139 MB`
- `ProSim-Instruct`: `324 ms`

## 5. Main Paper Tables We Need To Be Able To Reproduce

### 5.1 Table 2: InstructWaymo Results

Status target: `required`

We need code/configs/results flow covering:

| Model | GT-IFR | F-IFR | minADE/minFDE | Feasibility Detection |
| --- | ---: | ---: | --- | --- |
| GameFormer | 66.89 | 15.21 | 0.78 / 1.64 | No |
| MTR | 53.46 | 16.64 | 0.74 / 1.62 | No |
| C-GameFormer | 83.10 | 48.70 | 0.65 / 1.20 | No |
| C-MTR | 64.29 | 52.22 | 0.67 / 1.39 | No |
| ProSim-Instruct | 86.32 | 24.91 | 3.52 / 3.9 | No |
| iMotion-LLM | 87.30 | 52.24 | 0.67 / 1.25 | Yes |

### 5.2 Table 3: Open-Vocabulary InstructNuPlan Results

Status target: `required`

We need evaluation code that can produce:

| Input Type | Context | Accuracy | IFR |
| --- | --- | ---: | ---: |
| Safe | No | 96.61 | 68.96 |
| Safe | Yes | 98.39 | 70.22 |
| Unsafe | No | 95.00 | N/A |
| Unsafe | Yes | 96.88 | N/A |

### 5.3 Table 4: LLM Backbone Comparison

Status target: `important`

We need enough code/config support to compare these backbones:

| LLM | minADE/FDE | GT-IFR | GT-Acc | F-IFR | F-Acc | IF-Acc | Safe IFR | Safe+Context IFR | Avg Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaMA-7B | 0.67 / 1.25 | 87.30 | 97.33 | 52.24 | 62.27 | 92.73 | 68.96 | 70.22 | 96.25 |
| Mistral-7B | 0.67 / 1.25 | 86.61 | 97.93 | 47.23 | 64.87 | 95.00 | 63.69 | 63.95 | 97.25 |
| LLaMA-1B | 0.69 / 1.33 | 79.31 | 97.20 | 35.36 | 63.93 | 91.13 | 62.05 | 63.20 | 96.50 |
| Vicuna-7B | 0.70 / 1.35 | 80.81 | 94.47 | 36.56 | 39.80 | 86.13 | 62.26 | 62.75 | 96.25 |

### 5.4 Table 5: Class-Balanced Sampling

Status target: `important`

We need code/config switches for class-balanced vs unbalanced training on InstructWaymo:

| Class | GT-IFR Without | F-IFR Without | GT-IFR With | F-IFR With |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 27.61 | 0.83 | 51.00 | 16.89 |
| Straight | 92.11 | 26.61 | 98.83 | 70.39 |
| Left | 80.78 | 19.56 | 95.94 | 68.11 |
| Right | 82.28 | 19.83 | 97.61 | 68.56 |
| Left U-turn | 94.00 | 32.33 | 93.11 | 37.28 |

### 5.5 Table 6: Projection-Layer Ablation

Status target: `important`

We need mapper/projection architecture variants:

| In-Projection | Out-Projection | minADE/FDE | GT-IFR | GT-Acc | F-IFR | F-Acc | IF-Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linear | Linear | 0.78 / 1.58 | 80.78 | 97.40 | 36.83 | 65.87 | 92.13 |
| Linear | MLP 2L | 0.67 / 1.25 | 87.30 | 97.33 | 52.24 | 62.27 | 92.73 |
| MLP 2L | MLP 2L | 0.82 / 1.77 | 66.49 | 97.13 | 21.42 | 63.20 | 92.13 |
| MLP 4L | MLP 4L | 0.76 / 1.57 | 68.73 | 97.13 | 16.49 | 63.93 | 91.07 |

### 5.6 Table 7: Backbone Fine-Tuning Strategy

Status target: `important`

We need config switches for:

| Strategy | minADE/FDE | GT-IFR | GT-Acc | F-IFR | F-Acc | IF-Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 0.78 / 1.64 | 68.46 | 97.00 | 27.19 | 63.00 | 92.33 |
| Traj Decoder Only | 0.67 / 1.25 | 87.30 | 97.33 | 52.24 | 62.27 | 92.73 |
| Scene Encoder Only | 0.70 / 1.33 | 82.32 | 97.33 | 39.04 | 64.53 | 90.33 |
| Fully Finetuned | 0.76 / 1.57 | 74.53 | 97.20 | 41.70 | 66.07 | 91.60 |

## 6. Supplementary Items Worth Preserving If The Code Exists

### 6.1 Cross-Dataset Generalization

Status target: `optional`

Keep code/configs if we find them for:

- InstructWaymo-only fine-tuning
- Open-Vocabulary InstructNuPlan-only fine-tuning
- Direction InstructNuPlan fine-tuning
- two-stage Direction InstructNuPlan -> Open-Vocabulary InstructNuPlan fine-tuning

Reported Table S4 values:

| Stage 1 | Stage 2 | minADE/FDE | IFR |
| --- | --- | --- | ---: |
| Direction InstructWaymo | None | 4.12 / 7.82 | 52.92 |
| Open-Vocabulary InstructNuPlan | None | 6.33 / 9.19 | 70.22 |
| Direction InstructNuPlan | None | 1.22 / 2.79 | 61.91 |
| Direction InstructNuPlan | Open-Vocabulary InstructNuPlan | 1.03 / 2.08 | 67.16 |

### 6.2 Closed-Loop Small-Scale Evaluation

Status target: `optional`

Preserve if present:

- NuPlan planning simulator integration
- non-reactive-agent evaluation setup
- online directional instruction generation over a `30m` horizon
- qualitative export scripts for closed-loop examples

### 6.3 Multi-Agent Joint Prediction

Status target: `optional`

Preserve if present, but keep clearly marked as supplementary/limited:

- joint prediction using two interactive agents
- conditioning query per target agent for `C-GameFormer`
- extra special tokens / condition queries for multi-agent iMotion-LLM

Reported Table S6 values:

| Model | minADE/FDE Ego | minADE/FDE Both | GT-IFR | F-IFR |
| --- | --- | --- | ---: | ---: |
| GameFormer | 0.84 / 1.79 | 1.08 / 2.36 | 74.97 | 10.20 |
| C-GameFormer | 0.72 / 1.33 | 0.94 / 1.91 | 96.70 | 52.20 |
| iMotion-LLM | 0.88 / 1.83 | 1.33 / 2.72 | 75.54 | 13.43 |

## 7. Legacy-Code Search Guide

These are the kinds of functions, classes, constants, or comments we should expect to find in legacy sources, even if names are stale or inherited from MiniGPT-4/GameFormer:

- Waymo preprocessing utilities
- direction bucket extraction
- feasible/infeasible direction heuristics
- lane-destination search
- instruction templating
- caption templating
- speed/acceleration discretization
- conditional query embeddings
- scene encoder wrappers
- motion decoder wrappers
- LoRA setup
- tokenizer special tokens
- accept/reject text labels
- projection mapper modules
- scene mapper / instruction mapper modules
- trajectory decoder fine-tuning controls
- NuPlan scenario metadata
- GPT-driven instruction generation scripts
- evaluation code for IFR / GT-Acc / F-Acc / IF-Acc
- scripts producing Table 2 through Table 7

Likely legacy-source warning signs that are still relevant and should be migrated carefully:

- names referring to `MiniGPT-4`
- names referring to earlier VLM codepaths
- hard-coded local dataset roots
- one-off notebook exports
- experimental comments around ablations
- duplicated utility files with slightly different heuristics

## 8. What Must Be Explicitly Tracked As Missing If We Cannot Find It

If any of the following cannot be recovered from source, they must stay on the repo TODO list:

- InstructWaymo augmentation scripts
- Open-Vocabulary InstructNuPlan generation scripts
- exact IFR implementation
- special-token generation path for iMotion-LLM inference
- mixed GT/IF training logic
- class-balanced sampling implementation
- projection-depth ablation configs
- backbone fine-tuning ablation configs
- backbone comparison configs
- verification-tool code for instruction-reasoning review
- closed-loop evaluation scripts
- multi-agent supplementary scripts

## 9. Migration Completion Definition

For this repo, “paper-aligned refactoring complete” means:

- all main-paper required items are present in cleaned form
- old local paths are removed
- parent-project naming is cleaned where safe
- the codebase separates reusable modules from one-off experiments
- missing paper components are documented explicitly instead of being silently absent

Until then, this file remains the ground-truth migration checklist.
