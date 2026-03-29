"""
Multi-Model Router — Task-Specialized Model Swapping

Instead of one model for everything, uses the fastest model that's
perfect at each task category:

  code      → SmolLM2 135M  (159 tok/s, 3/3 raw)
  math      → Qwen2.5 0.5B  (45 tok/s, 3/3 raw)
  json      → SmolLM2 360M  (65 tok/s, 2/2 raw)
  reasoning → Llama 3.2 1B   (28 tok/s, needs experts)
  general   → SmolLM2 135M  (159 tok/s, 2/2 raw)
  instruct  → SmolLM2 135M  (159 tok/s, 1/1 raw)

Models are pre-loaded and hot-swapped based on the router's
module selection. This gives maximum speed per task while
maintaining quality.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.base_model import BaseModel
from engine.config import BaseModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelSlot:
    """A loaded model in the pool."""
    name: str
    path: str
    chat_format: str
    context_length: int
    model: Optional[BaseModel] = None
    categories: list[str] = None  # Which task categories this model handles

    def __post_init__(self):
        if self.categories is None:
            self.categories = []


# Benchmark-derived routing table: category → best model
DEFAULT_ROUTING = {
    # Speed-optimized: smallest model with perfect score per category
    "speed": {
        "code": "fast",       # SmolLM2 135M — 159 tok/s
        "math": "math",       # Qwen2.5 0.5B — 45 tok/s, 3/3 raw
        "json_format": "fast", # SmolLM2 135M — works for JSON too
        "general": "fast",    # SmolLM2 135M — 159 tok/s
        "default": "balanced", # Llama 3.2 1B fallback
    },
    # Quality-optimized: best quality per category
    "quality": {
        "code": "balanced",
        "math": "math",
        "json_format": "balanced",
        "general": "balanced",
        "default": "balanced",
    },
    # Balanced: one model for everything (current default behavior)
    "single": {
        "default": "balanced",
    },
}


class MultiModelRouter:
    """
    Manages a pool of models and routes requests to the optimal one.

    Modes:
      - "single":  One model for everything (default, backward-compatible)
      - "speed":   Fastest model per task category
      - "quality": Best quality model per category
    """

    def __init__(self, mode: str = "single", gpu_layers: int = 99, threads: int = 8):
        self.mode = mode
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.slots: dict[str, ModelSlot] = {}
        self._active_slot: Optional[str] = None
        self._routing = DEFAULT_ROUTING.get(mode, DEFAULT_ROUTING["single"])

    def register_model(self, slot_name: str, path: str, chat_format: str,
                       context_length: int = 2048, categories: list[str] = None):
        """Register a model in the pool (doesn't load yet)."""
        self.slots[slot_name] = ModelSlot(
            name=slot_name,
            path=path,
            chat_format=chat_format,
            context_length=context_length,
            categories=categories or [],
        )
        logger.info(f"Registered model slot: {slot_name} -> {Path(path).name}")

    def load_all(self):
        """Pre-load all registered models."""
        for name, slot in self.slots.items():
            if not Path(slot.path).exists():
                logger.warning(f"Model not found: {slot.path}")
                continue

            config = BaseModelConfig(
                path=slot.path,
                context_length=slot.context_length,
                gpu_layers=self.gpu_layers,
                threads=self.threads,
            )
            slot.model = BaseModel(config)
            slot.model.load()
            logger.info(f"Loaded model: {name} ({Path(slot.path).name})")

    def unload_all(self):
        """Unload all models."""
        for slot in self.slots.values():
            if slot.model and slot.model.is_loaded:
                slot.model.unload()

    def select(self, module_name: Optional[str] = None) -> tuple[BaseModel, str]:
        """
        Select the best model for a task.

        Args:
            module_name: The router's selected module (e.g., "code", "math")

        Returns:
            (model, chat_format) tuple
        """
        # Look up which slot to use
        slot_name = self._routing.get(module_name, self._routing.get("default", "balanced"))
        slot = self.slots.get(slot_name)

        if slot is None or slot.model is None or not slot.model.is_loaded:
            # Fallback to any loaded model
            for s in self.slots.values():
                if s.model and s.model.is_loaded:
                    slot = s
                    break

        if slot is None:
            raise RuntimeError("No models loaded")

        if slot_name != self._active_slot:
            logger.debug(f"Switched to model: {slot.name} (for {module_name})")
            self._active_slot = slot_name

        return slot.model, slot.chat_format

    def status(self) -> dict:
        """Get router status."""
        return {
            "mode": self.mode,
            "slots": {
                name: {
                    "path": Path(slot.path).name,
                    "loaded": slot.model is not None and slot.model.is_loaded,
                    "chat_format": slot.chat_format,
                    "categories": slot.categories,
                }
                for name, slot in self.slots.items()
            },
            "routing": self._routing,
            "active": self._active_slot,
        }


def create_speed_router(gpu_layers: int = 99, threads: int = 8) -> MultiModelRouter:
    """Create a speed-optimized multi-model router with default model assignments."""
    router = MultiModelRouter(mode="speed", gpu_layers=gpu_layers, threads=threads)

    router.register_model(
        "fast", "models/SmolLM2-135M-Instruct-Q4_K_M.gguf",
        chat_format="chatml", context_length=2048,
        categories=["code", "general", "instruct", "json_format"],
    )
    router.register_model(
        "math", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        chat_format="chatml", context_length=2048,
        categories=["math"],
    )
    router.register_model(
        "balanced", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        chat_format="llama3", context_length=4096,
        categories=["reasoning", "default"],
    )

    return router


def create_quality_router(gpu_layers: int = 99, threads: int = 8) -> MultiModelRouter:
    """Create a quality-optimized multi-model router."""
    router = MultiModelRouter(mode="quality", gpu_layers=gpu_layers, threads=threads)

    router.register_model(
        "math", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        chat_format="chatml", context_length=2048,
        categories=["math"],
    )
    router.register_model(
        "balanced", "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        chat_format="llama3", context_length=4096,
        categories=["code", "json_format", "reasoning", "general", "instruct", "default"],
    )

    return router
