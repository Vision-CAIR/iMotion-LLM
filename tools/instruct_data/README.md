# Instruction Data Tools

Tools for obtaining and regenerating the two instruction datasets used in the
WACV 2026 paper.

## Published datasets

| Dataset | Hub | Contents |
|---|---|---|
| Open-Vocabulary InstructNuPlan | [`KAUST/InstructNuPlan`](https://huggingface.co/datasets/KAUST/InstructNuPlan) | LLM-generated safe/unsafe instructions with reasoning, over 14 nuPlan scenario types |
| InstructWaymo | [`KAUST/InstructWaymo`](https://huggingface.co/datasets/KAUST/InstructWaymo) | Instruction/reasoning/decision records over Waymo, plus the exact evaluation manifests |

Both contain **annotations only** — no Waymo or nuPlan sensor data, maps or
logs. Records are keyed to scene tokens. You must obtain the raw datasets
yourself from their official sources (see
[`docs/setup/data_and_checkpoints.md`](../../docs/setup/data_and_checkpoints.md)).

## Download

```bash
pip install huggingface_hub

# fetch the gzipped JSONL shards
python tools/instruct_data/download_instruct_data.py --dest data/processed

# also rebuild the original per-scene file tree the configs expect
python tools/instruct_data/download_instruct_data.py --dest data/processed --expand
```

`--expand` is only needed for configs that read per-scene files. Every row
stores its original relative path in `_source`, so expansion restores the
original tree; raw-text files come back byte-identical.

## What is published vs. regenerated

We publish only what code cannot reproduce:

- **Published** — the LLM-generated instruction text, and the evaluation
  manifests that define the exact paper splits (`meta_gt1.json`,
  `meta_pos1.json`, `meta_neg1.json`, `eval_meta.json`). Regenerating the
  nuPlan instructions would mean re-prompting an LLM: non-deterministic, so the
  paper's numbers would not be reproducible, and it costs API credits.
- **Not published** — the `npz` scenario tensor caches (~1.3 TiB across both
  datasets) and rendered figures. These are deterministic output of the
  preprocessing scripts in this repo, so they are regenerated locally rather
  than mirrored.

## Regenerating the nuPlan prompt bundle

`build_nuplan_prompts_v01/v02/v03.py` and `prepare_gpt_json.py` are the original
research scripts that construct the prompts sent to the LLM. They are included
for provenance and to let others re-run the pipeline on new scenarios.

They take paths from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `IMOTION_SCENE_JSON_DIR` | `data/processed/waymo/gameformer/train_agentJsons` | per-scene agent/map JSON from preprocessing |
| `IMOTION_VAL_SPLIT_DIR` | `data/processed/waymo/gameformer/val_splits/split_1` | validation split to build prompts for |

```bash
IMOTION_SCENE_JSON_DIR=data/processed/waymo/gameformer/train_agentJsons \
  python tools/instruct_data/build_nuplan_prompts_v03.py
```

`v03` is the version used for the released bundle; `v01` and `v02` are earlier
variants kept for traceability. These scripts build prompts only — sending them
to an LLM and collecting responses was a separate step, which is why the
responses are published rather than regenerated.
