import os
import sys
from pathlib import Path
from typing import Union


def repo_root_from(anchor_file: str, levels_up: int = 1) -> Path:
    root = Path(anchor_file).resolve()
    for _ in range(levels_up):
        root = root.parent
    return root


def ensure_repo_root_on_path(anchor_file: str, levels_up: int = 1) -> Path:
    root = repo_root_from(anchor_file, levels_up=levels_up)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def env_or_default(env_name: str, default: Union[Path, str]) -> Path:
    return Path(os.path.expanduser(os.environ.get(env_name, str(default)))).resolve()


def first_existing_path(*candidates: Union[Path, str]) -> Path:
    normalized = [Path(candidate).expanduser() for candidate in candidates if candidate]
    for candidate in normalized:
        if candidate.exists():
            return candidate.resolve()
    if normalized:
        return normalized[0].resolve()
    raise ValueError("No path candidates were provided.")
