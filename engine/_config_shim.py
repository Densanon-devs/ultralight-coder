"""
Lightweight Config shim for ulcagent standalone mode.

Replaces densanon.core.config.Config when densanon-core is not installed.
Only implements what ulcagent needs: YAML loading + base_model attribute access.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class BaseModelConfig:
    path: str = ""
    context_length: int = 4096
    gpu_layers: int = 99
    threads: int = 8
    temperature: float = 0.2
    max_tokens: int = 512
    batch_size: int = 512


@dataclass
class StandaloneConfig:
    """Minimal config object — drop-in for densanon.core.config.Config."""
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    config_path: Optional[Path] = None

    def setup_logging(self):
        pass  # ulcagent handles its own logging


def load_config(config_path: str) -> StandaloneConfig:
    """Load a YAML config file and return a StandaloneConfig."""
    p = Path(config_path)
    cfg = StandaloneConfig(config_path=p)

    if not p.exists():
        logger.warning(f"Config not found: {config_path}")
        return cfg

    if yaml is None:
        # Fallback: parse just the base_model section with regex
        import re
        text = p.read_text(encoding="utf-8", errors="replace")
        def _get(key, default, cast=str):
            m = re.search(rf"^\s*{key}:\s*(.+)$", text, re.MULTILINE)
            if m:
                try:
                    return cast(m.group(1).strip().strip("'\""))
                except (ValueError, TypeError):
                    pass
            return default
        cfg.base_model = BaseModelConfig(
            path=_get("path", ""),
            context_length=_get("context_length", 4096, int),
            gpu_layers=_get("gpu_layers", 99, int),
            threads=_get("threads", 8, int),
            temperature=_get("temperature", 0.2, float),
            max_tokens=_get("max_tokens", 512, int),
            batch_size=_get("batch_size", 512, int),
        )
    else:
        with open(p, "r") as f:
            raw = yaml.safe_load(f) or {}
        bm = raw.get("base_model", {})
        cfg.base_model = BaseModelConfig(
            path=bm.get("path", ""),
            context_length=bm.get("context_length", 4096),
            gpu_layers=bm.get("gpu_layers", 99),
            threads=bm.get("threads", 8),
            temperature=bm.get("temperature", 0.2),
            max_tokens=bm.get("max_tokens", 512),
            batch_size=bm.get("batch_size", 512),
        )

    # Resolve relative model paths against the config file's directory
    model_path = cfg.base_model.path
    if model_path and not Path(model_path).is_absolute():
        resolved = (p.parent / model_path).resolve()
        cfg.base_model.path = str(resolved)

    return cfg
