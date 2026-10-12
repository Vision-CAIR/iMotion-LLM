#!/usr/bin/env python3
"""Fetch the published iMotion-LLM instruction annotations and restore the
on-disk layout that the release configs expect.

The datasets on the Hub are stored as gzipped JSONL shards rather than the
millions of tiny per-scene files the research pipeline produced, because that
loads in one pass instead of millions of HTTP requests. Every row keeps its
original relative path in `_source`, so this script can rebuild the original
tree byte-for-byte when a config wants real files.

Usage
-----
    # just download the shards (no expansion)
    python tools/instruct_data/download_instruct_data.py --dest data/processed

    # also expand shards back into per-scene files
    python tools/instruct_data/download_instruct_data.py --dest data/processed --expand

    # only one dataset
    python tools/instruct_data/download_instruct_data.py --only nuplan --dest data/processed

Requires: pip install huggingface_hub
"""
import argparse
import gzip
import json
import os
import sys

NUPLAN_REPO = "KAUST/InstructNuPlan"
WAYMO_REPO = "KAUST/InstructWaymo"

# shard prefix -> directory it expands into, relative to --dest
EXPAND_MAP = {
    "gpt_responses": "nuplan/gpt_prompt_14types",
    "template_instructions": "nuplan/gpt_prompt_14types",
    "meta_prompts": "nuplan/gpt_prompt_14types",
    "gpt_positive_prompts": "nuplan/gpt_prompt_14types",
    "gpt_negative_prompts": "nuplan/gpt_prompt_14types",
    "train_instructions": "waymo/gameformer/train_templateLLM",
    "val_instructions": "waymo/gameformer/val_templateLLM",
    "train_synth_instructions": "waymo/gameformer/train_synth_templateLLM",
    "gf_mtr_mapping": "waymo/gameformer/gf_mtr_mapping",
}


def download(repo_id, dest):
    from huggingface_hub import snapshot_download
    print(f"[download] {repo_id}")
    path = snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        local_dir=os.path.join(dest, "_hub", repo_id.split("/")[-1]),
    )
    print(f"           -> {path}")
    return path


def expand_shards(snapshot_dir, dest):
    """Rebuild per-scene files from the shards, using each row's _source."""
    written = 0
    for name in sorted(os.listdir(snapshot_dir)):
        if not name.endswith(".jsonl.gz"):
            continue
        prefix = name.split("-")[0]
        target_root = EXPAND_MAP.get(prefix)
        if target_root is None:
            print(f"  [skip] no expand mapping for {name}")
            continue
        out_root = os.path.join(dest, target_root)
        grouped = {}
        with gzip.open(os.path.join(snapshot_dir, name), "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                src = row.get("_source")
                if not src:
                    continue
                grouped.setdefault(src, []).append(row)
        for src, rows in grouped.items():
            out_path = os.path.join(out_root, src)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as out:
                if len(rows) == 1 and "text" in rows[0]:
                    out.write(rows[0]["text"])
                elif len(rows) == 1 and "content" in rows[0]:
                    json.dump(rows[0]["content"], out, indent=4)
                else:
                    # file originally held one JSON object per line
                    for r in sorted(rows, key=lambda x: x.get("_line", 0)):
                        payload = {k: v for k, v in r.items()
                                   if k not in ("_scene_id", "_source", "_line")}
                        out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
        print(f"  [expand] {name} -> {out_root} ({len(grouped)} files)")
    print(f"[expand] wrote {written} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data/processed",
                    help="root to place data under (default: data/processed)")
    ap.add_argument("--only", choices=["nuplan", "waymo"], default=None)
    ap.add_argument("--expand", action="store_true",
                    help="also rebuild the original per-scene files from shards")
    a = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    os.makedirs(a.dest, exist_ok=True)
    todo = []
    if a.only in (None, "nuplan"):
        todo.append(NUPLAN_REPO)
    if a.only in (None, "waymo"):
        todo.append(WAYMO_REPO)

    for repo in todo:
        snap = download(repo, a.dest)
        if a.expand:
            expand_shards(snap, a.dest)

    print("\nDone. Note: these are annotations only — you still need the raw")
    print("Waymo Open Motion Dataset and nuPlan data from their official sources.")


if __name__ == "__main__":
    main()
