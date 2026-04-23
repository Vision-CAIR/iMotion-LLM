# Running

## 1. Preprocess Waymo For GameFormer / iMotion-LLM

```bash
bash scripts/preprocess_waymo_gameformer.sh \
  --load_path data/raw/waymo/validation_interactive \
  --save_path data/processed/waymo/gameformer/val
```

For training data, point `--load_path` and `--save_path` to the training split instead.

## 2. Preprocess nuPlan For Open-Vocabulary InstructNuPlan

```bash
bash scripts/preprocess_nuplan_complex.sh \
  --data_path data/raw/nuplan/cache/train_combined \
  --map_path data/raw/nuplan/maps \
  --save_path data/processed/nuplan/gpt_prompt_14types
```

## 3. Train GameFormer

```bash
bash scripts/train_gameformer_waymo.sh \
  --act \
  --act_dec \
  --level 1 \
  --save_path outputs/gameformer/waymo \
  --name cgf_l1_waymo_release
```

For multi-GPU training, call the Python entrypoint with `torchrun` directly:

```bash
torchrun --nproc-per-node 4 gameformer/interaction_prediction/train.py \
  --act --act_dec --level 1 \
  --train_set data/processed/waymo/gameformer/train \
  --valid_set data/processed/waymo/gameformer/val \
  --save_path outputs/gameformer/waymo \
  --name cgf_l1_waymo_release
```

## 4. Train iMotion-LLM On Waymo

```bash
bash scripts/train_imotion_waymo.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth \
  run.output_dir=outputs/imotion_waymo/my_run
```

## 5. Evaluate An Existing iMotion-LLM Checkpoint

```bash
bash scripts/eval_imotion_waymo.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth \
  run.eval_dir=checkpoints/imotion_llm/waymo/checkpoint_last.pth
```

If your checkpoint is stored elsewhere, just override `run.eval_dir`.

## 6. Fine-Tune On nuPlan

```bash
bash scripts/train_imotion_nuplan.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth \
  run.output_dir=outputs/imotion_nuplan/my_run
```

## 7. Evaluate An Existing iMotion-LLM NuPlan Checkpoint

```bash
bash scripts/eval_imotion_nuplan.sh \
  --options \
  model.llama_model=meta-llama/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth \
  run.eval_dir=checkpoints/imotion_llm/nuplan/checkpoint_last.pth
```

## 8. Train / Evaluate MTR

```bash
bash scripts/train_mtr_waymo.sh
```

```bash
bash scripts/eval_mtr_waymo.sh --ckpt checkpoints/mtr/waymo/checkpoint_epoch_15.pth
```

## Notes

- Run commands from the repo root unless the command explicitly changes directory for you.
- The release configs are the supported public defaults; the many legacy YAMLs under `trajgpt/` are preserved mainly for experiment archaeology.
- If a path in the default config does not match your local layout, override it with `--options key=value`.
