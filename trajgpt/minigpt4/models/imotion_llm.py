"""Project-native iMotion-LLM model names.

The underlying implementation still lives in the migrated legacy modules.
This wrapper provides cleaner public names for supported release entrypoints.
"""

from minigpt4.models.mini_gpt4 import IMotionLLMModel, MiniGPT4
from minigpt4.models.mini_gpt4_mtr_dev import IMotionLLMMTRModel

__all__ = [
    "IMotionLLMModel",
    "IMotionLLMMTRModel",
    "MiniGPT4",
]
