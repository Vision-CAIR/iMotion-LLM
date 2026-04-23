# Running

## 1. Preprocess Waymo For GameFormer / iMotion-LLM

```bash
bash scripts/preprocess_waymo_gameformer.sh \
  --load_path data/raw/waymo/validation_interactive \
  --save_path data/processed/waymo/gameformer/val
```

For training data, point `--load_path` and `--save_path` to the training split instead.

## 2. Train And Evaluate GameFormer / C-GameFormer On Waymo

Train unconditional `GameFormer`:

```bash
bash scripts/train_gameformer_waymo.sh \
  --level 1 \
  --save_path outputs/gameformer/waymo \
  --name gf_l1_waymo_release
```

Train conditional `C-GameFormer`:

```bash
bash scripts/train_gameformer_waymo.sh \
  --act \
  --act_dec \
  --level 1 \
  --save_path outputs/gameformer/waymo \
  --name cgf_l1_waymo_release
```

Evaluate on Waymo `gt1`:

```bash
IMOTION_GF_CKPT=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth \
bash scripts/eval_gameformer_waymo.sh --act --act_dec --level 1
```

Switch evaluation mode with:

- `IMOTION_GF_EVAL_MODE=gt1`
- `IMOTION_GF_EVAL_MODE=pos1`
- `IMOTION_GF_EVAL_MODE=neg1`

`gt1`, `pos1`, and `neg1` correspond to ground-truth, feasible-alternative, and infeasible instructions.

## 3. Train And Evaluate MTR / C-MTR On Waymo

Train unconditional `MTR`:

```bash
bash scripts/train_mtr_waymo_base.sh
```

Train conditional `C-MTR`:

```bash
bash scripts/train_mtr_waymo.sh
```

Evaluate unconditional `MTR`:

```bash
bash scripts/eval_mtr_waymo_base.sh \
  --ckpt checkpoints/mtr/waymo_base/checkpoint_epoch_15.pth
```

Evaluate conditional `C-MTR`:

```bash
bash scripts/eval_mtr_waymo.sh \
  --ckpt checkpoints/mtr/waymo/checkpoint_epoch_15.pth
```

Switch evaluation mode with:

- `IMOTION_MTR_EVAL_MODE=gt1`
- `IMOTION_MTR_EVAL_MODE=pos1`
- `IMOTION_MTR_EVAL_MODE=neg1`

## 4. Train And Evaluate iMotion-LLM On Waymo

Train iMotion-LLM:

```bash
bash scripts/train_imotion_waymo.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth \
  run.output_dir=outputs/imotion_waymo/my_run
```

Evaluate an existing checkpoint on `gt1`:

```bash
bash scripts/eval_imotion_waymo.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth \
  run.eval_dir=checkpoints/imotion_llm/waymo/checkpoint_last.pth \
  datasets.traj_align_valid.processor.valid.new_eval_mode=gt1
```

Run the same checkpoint on the other Table 2 modes by changing:

- `datasets.traj_align_valid.processor.valid.new_eval_mode=pos1`
- `datasets.traj_align_valid.processor.valid.new_eval_mode=neg1`

## 5. Preprocess nuPlan For Open-Vocabulary InstructNuPlan

```bash
bash scripts/preprocess_nuplan_complex.sh \
  --data_path data/raw/nuplan/cache/train_combined \
  --map_path data/raw/nuplan/maps \
  --save_path data/processed/nuplan/gpt_prompt_14types
```

Important note:

- if you want exact paper-style public reproduction, prefer the released generated prompt bundle over regenerating prompts yourself
- raw-data regeneration is possible, but parts of the prompt pipeline use OpenAI-based generation and require `OPENAI_API_KEY`

## 6. Train And Evaluate GameFormer / C-GameFormer On nuPlan

Train a nuPlan GameFormer baseline:

```bash
bash scripts/train_gameformer_nuplan.sh \
  --act \
  --act_dec \
  --level 1
```

Evaluate on `safe_with_context`:

```bash
IMOTION_GF_NUPLAN_CKPT=checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth \
bash scripts/eval_gameformer_nuplan.sh --act --act_dec --level 1
```

Switch evaluation mode with:

- `IMOTION_GF_NUPLAN_EVAL_MODE=safe_with_context`
- `IMOTION_GF_NUPLAN_EVAL_MODE=safe_no_context`
- `IMOTION_GF_NUPLAN_EVAL_MODE=unsafe_with_context`
- `IMOTION_GF_NUPLAN_EVAL_MODE=unsafe_no_context`

## 7. Train And Evaluate iMotion-LLM On nuPlan

Train iMotion-LLM on nuPlan:

```bash
bash scripts/train_imotion_nuplan.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth \
  run.output_dir=outputs/imotion_nuplan/my_run
```

Evaluate an existing checkpoint on `safe_with_context`:

```bash
bash scripts/eval_imotion_nuplan.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth \
  run.eval_dir=checkpoints/imotion_llm/nuplan/checkpoint_last.pth \
  datasets.traj_align_valid.processor.valid.new_eval_mode=safe_with_context
```

Run the same checkpoint on the other Table 3 categories by changing:

- `datasets.traj_align_valid.processor.valid.new_eval_mode=safe_no_context`
- `datasets.traj_align_valid.processor.valid.new_eval_mode=unsafe_with_context`
- `datasets.traj_align_valid.processor.valid.new_eval_mode=unsafe_no_context`

## 8. Loading Existing Checkpoints

- place checkpoints under `checkpoints/` using the layout described in `data_and_checkpoints.md`, or
- override the paths directly at runtime with `--options` and env vars

Common overrides:

- `run.eval_dir=...`
- `model.gf_encoder_path=...`
- `IMOTION_GF_CKPT=...`
- `IMOTION_MTR_EVAL_MODE=...`
- `IMOTION_LLM_MTR_EVAL_META_DIR=...`
- `IMOTION_LLM_MTR_EVAL_MAPPING_DIR=...`

## Notes

- run commands from the repo root unless the command explicitly changes directory for you
- the release configs are the supported public defaults; the many legacy YAMLs under `trajgpt/` are preserved mainly for experiment archaeology
- if a path in the default config does not match your local layout, override it with `--options key=value`
- for multi-GPU training, prefer `torchrun` on the underlying Python entrypoints
