"""Template / prompt builders, under the module name the pipeline imports.

The Waymo and nuPlan preprocessing scripts in ``gameformer/`` import this module
by its historical name::

    from chatgpt_instruct_v02 import *

and call ``generate_template_json`` to write the ``*_templateLLM/*.txt``
instruction files next to each preprocessed ``.npz``. The implementation lives
in ``tools/instruct_data/build_nuplan_prompts_v02.py``, which is also runnable
on its own for nuPlan prompt construction; this module re-exports it so the
published preprocessing entrypoints resolve without renaming the import in the
31 scripts that use it, and without a second copy of the templates.

Kept as a re-export rather than a copy so the Waymo template text and the
nuPlan prompt text can never drift apart.
"""

import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path

_IMPL = _Path(__file__).resolve().parent / "tools" / "instruct_data" / "build_nuplan_prompts_v02.py"

if not _IMPL.is_file():
    raise ImportError(
        f"prompt builder implementation not found at {_IMPL}; "
        "the repository checkout is incomplete"
    )

_spec = _importlib_util.spec_from_file_location("_imotion_prompt_builders_v02", _IMPL)
_module = _importlib_util.module_from_spec(_spec)
_sys.modules["_imotion_prompt_builders_v02"] = _module
_spec.loader.exec_module(_module)

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("_")})

__all__ = [k for k in vars(_module) if not k.startswith("_")]
