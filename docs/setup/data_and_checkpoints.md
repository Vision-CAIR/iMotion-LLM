# Data And Checkpoints

## Expected Layout

The release configs assume this directory layout under the repo root:

```text
data/
├── raw/
│   ├── waymo/
│   │   ├── training_interactive/
│   │   └── validation_interactive/
│   └── nuplan/
│       ├── cache/
│       │   ├── train_combined/
│       │   └── test/
│       └── maps/
└── processed/
    ├── waymo/
    │   ├── gameformer/
    │   │   ├── train/
    │   │   ├── train_templateLLM/
    │   │   ├── val/
    │   │   └── val_templateLLM/
    │   └── mtr/
    │       ├── processed_scenarios_training/
    │       ├── processed_scenarios_validation/
    │       └── output/cluster_64_center_dict.pkl
    └── nuplan/
        ├── gpt_prompt_14types/
        └── test_gpt_prompt_14types/

checkpoints/
├── gameformer/
│   ├── waymo/cgf_l1/epochs_29.pth
│   └── nuplan/cgf_l1/epochs_29.pth
├── imotion_llm/
│   └── waymo/checkpoint_last.pth
└── mtr/
    └── waymo/checkpoint_epoch_15.pth
```

## Official Dataset Sources

- Waymo Open Dataset download portal: <https://waymo.com/open/download>
- Waymo Open Motion Dataset overview: <https://waymo.com/intl/it/open/data/motion/>
- nuPlan dataset setup docs: <https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html>

Tips:

- Download the official raw data first, then run the preprocessing scripts in this repo to create the `data/processed/...` structure expected by the release configs.
- The Waymo and nuPlan datasets have their own licenses and account requirements; this repo does not mirror them.
- The old internal `docs/nuplan_download_links_legacy.txt` file is preserved for provenance, but the official links above should be treated as the source of truth.

## Base LLM Weights

The release configs currently default to Meta Llama 2 7B:

- Hugging Face model page: <https://huggingface.co/meta-llama/Llama-2-7b-hf>

You can either:

- use the Hugging Face model id directly in the config, or
- download it locally and override `model.llama_model=/absolute/path/to/Llama-2-7b-hf`

## Project Checkpoints

This public repo currently ships code and load instructions, but not bundled research checkpoints.

If you already have local checkpoints:

- place them under the `checkpoints/` tree shown above, or
- pass explicit paths at runtime with `--options` or env vars

Useful overrides:

- `model.gf_encoder_path=...`
- `run.eval_dir=...`
- `model.mtr_ckpt_path=...`
- `IMOTION_LLM_MTR_CKPT=...`
- `IMOTION_LLM_MTR_DATA_ROOT=...`
