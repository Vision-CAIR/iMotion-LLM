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
    │   │   ├── val_eval_meta/
    │   │   │   ├── meta_gt1.json
    │   │   │   ├── meta_pos1.json
    │   │   │   └── meta_neg1.json
    │   │   ├── val_templateLLM/
    │   │   └── gf_mtr_mapping/gf_templatellm_maps/
    │   └── mtr/
    │       ├── processed_scenarios_training/
    │       ├── processed_scenarios_validation/
    │       └── output/cluster_64_center_dict.pkl
    └── nuplan/
        ├── gpt_prompt_14types/
        └── test_gpt_prompt_14types/

checkpoints/
├── gameformer/
│   ├── waymo/
│   │   ├── gf_l1/epochs_29.pth
│   │   └── cgf_l1/epochs_29.pth
│   └── nuplan/
│       ├── gf_l1/epochs_29.pth
│       └── cgf_l1/epochs_29.pth
├── imotion_llm/
│   ├── waymo/checkpoint_last.pth
│   └── nuplan/checkpoint_last.pth
└── mtr/
    ├── waymo/checkpoint_epoch_15.pth
    └── waymo_base/checkpoint_epoch_15.pth
```

## Users Download These From Official Sources

- Waymo Open Dataset download portal: <https://waymo.com/open/download>
- Waymo Open Motion Dataset overview: <https://waymo.com/intl/it/open/data/motion/>
- nuPlan dataset setup docs: <https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html>
- Meta Llama 2 7B model page: <https://huggingface.co/meta-llama/Llama-2-7b-hf>

Tips:

- download the official raw data first, then run the preprocessing scripts in this repo to create the `data/processed/...` structure expected by the release configs
- the Waymo and nuPlan datasets have their own licenses and account requirements; this repo does not mirror them
- the old internal `docs/nuplan_download_links_legacy.txt` file is preserved for provenance, but the official links above should be treated as the source of truth

## We Should Provide These As Release Artifacts

To make the repo practical for public users, upload these separately from the Git repo:

- research checkpoints
- Waymo evaluation manifests:
  - `meta_gt1.json`
  - `meta_pos1.json`
  - `meta_neg1.json`
- GF↔MTR mapping JSONs used by MTR evaluation
- generated Open-Vocabulary InstructNuPlan prompt / metadata bundle
- small supporting artifacts such as `KBinsDiscretizer_76.pkl` and `cluster_64_center_dict.pkl`

See [../release/HUGGINGFACE_UPLOAD_MANIFEST.md](../release/HUGGINGFACE_UPLOAD_MANIFEST.md) for the full upload list.

## Base LLM Weights

The release configs currently default to Meta Llama 2 7B.

You can either:

- use the Hugging Face model id directly in the config, or
- download it locally and override `model.llama_model=/absolute/path/to/Llama-2-7b-hf`

Example local download:

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir checkpoints/base_models/Llama-2-7b-hf
```

Then run with:

```bash
bash scripts/train_imotion_waymo.sh \
  --options \
  model.llama_model=checkpoints/base_models/Llama-2-7b-hf \
  model.gf_encoder_path=checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth
```

## Project Checkpoints

This public repo currently ships code and load instructions, but not bundled research checkpoints.

If you already have local checkpoints:

- place them under the `checkpoints/` tree shown above, or
- pass explicit paths at runtime with `--options` or env vars

Suggested placement if you are copying checkpoints from an older internal machine:

```bash
mkdir -p checkpoints/gameformer/waymo/gf_l1
mkdir -p checkpoints/gameformer/waymo/cgf_l1
mkdir -p checkpoints/gameformer/nuplan/gf_l1
mkdir -p checkpoints/gameformer/nuplan/cgf_l1
mkdir -p checkpoints/imotion_llm/waymo
mkdir -p checkpoints/imotion_llm/nuplan
mkdir -p checkpoints/mtr/waymo
mkdir -p checkpoints/mtr/waymo_base

cp /path/to/old/gameformer_waymo_epoch_29.pth checkpoints/gameformer/waymo/gf_l1/epochs_29.pth
cp /path/to/old/cgameformer_waymo_epoch_29.pth checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth
cp /path/to/old/gameformer_nuplan_epoch_29.pth checkpoints/gameformer/nuplan/gf_l1/epochs_29.pth
cp /path/to/old/cgameformer_nuplan_epoch_29.pth checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth
cp /path/to/old/imotion_waymo_checkpoint.pth checkpoints/imotion_llm/waymo/checkpoint_last.pth
cp /path/to/old/imotion_nuplan_checkpoint.pth checkpoints/imotion_llm/nuplan/checkpoint_last.pth
cp /path/to/old/mtr_checkpoint_epoch_15.pth checkpoints/mtr/waymo/checkpoint_epoch_15.pth
cp /path/to/old/mtr_base_checkpoint_epoch_15.pth checkpoints/mtr/waymo_base/checkpoint_epoch_15.pth
```

Project research checkpoints are not bundled in this public repo yet. If no old checkpoints are available locally, use the training commands in [running.md](running.md) to produce fresh ones.

## NuPlan Prompt Generation Caveat

Open-Vocabulary InstructNuPlan is not a pure raw-data-only pipeline.

Parts of the prompt generation flow under `gameformer/nuplan_preprocess/` use OpenAI-based generation and require `OPENAI_API_KEY` if you regenerate the prompts yourself. For a strict public reproduction package, the generated prompt and metadata bundle should be downloaded from the project release instead of being regenerated from scratch.

Useful overrides:

- `model.gf_encoder_path=...`
- `run.eval_dir=...`
- `model.mtr_ckpt_path=...`
- `IMOTION_LLM_MTR_CKPT=...`
- `IMOTION_LLM_MTR_DATA_ROOT=...`
- `IMOTION_LLM_MTR_EVAL_META_DIR=...`
- `IMOTION_LLM_MTR_EVAL_MAPPING_DIR=...`
