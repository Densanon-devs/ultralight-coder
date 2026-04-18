"""
Native Speculative Decoding — llama-cpp-python draft-model integration.

Sibling to engine/speculative.py, which implements application-level tricks
(parallel generation, n-gram cache, early exit). This module taps llama-cpp's
built-in speculative decoding support via the `draft_model` kwarg on Llama.

### What actually works in llama-cpp-python (verified 2026-04-12, v0.3.9)

Only ONE mode is actually supported by llama-cpp-python's high-level API:

    prompt_lookup:
        Uses LlamaPromptLookupDecoding — no second model required. Scans
        repeated n-grams in the current context and predicts multiple tokens
        at once. Works with any base model. Real speedup on code generation
        whenever the prompt contains repetition the target is likely to
        reproduce (variable names reused in tests, boilerplate, signatures).
        Low risk, bit-identical output, ships today.

### What does NOT work (and why)

A second-Llama draft-model path (e.g., Qwen Coder 0.5B drafting 14B) is
BROKEN from this module in llama-cpp-python 0.3.9. The `draft_model` kwarg
expects an object implementing the `LlamaDraftModel` protocol, and the
library ships exactly one implementation: `LlamaPromptLookupDecoding`. A
raw `Llama` instance does NOT satisfy the protocol.

A custom adapter wrapping a second `Llama` via reset/eval/sample is
theoretically possible but would need KV-cache state tracking and rollback
to be faster than baseline. Without state tracking, every call to the
adapter triggers a full forward pass over the entire context on the draft
model, which kills any speedup. We do not ship this path.

The community-reported 2.5× speedups (ggml-org/llama.cpp discussion #10466)
come from the llama.cpp C++ CLI tool `llama-speculative-simple`, not from
llama-cpp-python's Python API. A subprocess-based benchmark that shells out
to the CLI is a legitimate future direction — see experiment_backlog.md.

### Why draft_model_path config still exists

Kept as a forward-compatible config knob. When/if llama-cpp-python adds a
native second-Llama draft adapter (or we build a subprocess path), the
config key is already wired up. Attempting to enable it today raises a
clear NotImplementedError instead of crashing at generation time.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class NativeSpeculativeConfig:
    """Configuration for native (llama-cpp-python) speculative decoding."""

    enabled: bool = False
    mode: str = "prompt_lookup"  # "prompt_lookup" or "draft_model"

    # prompt_lookup mode
    num_pred_tokens: int = 10
    max_ngram_size: int = 2

    # draft_model mode
    draft_model_path: str = ""
    draft_gpu_layers: int = 99
    draft_context_length: int = 4096


def _probe_prompt_lookup() -> Optional[Any]:
    """Return the LlamaPromptLookupDecoding class if importable, else None."""
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
        return LlamaPromptLookupDecoding
    except ImportError:
        logger.debug("LlamaPromptLookupDecoding not available in this llama-cpp-python build")
        return None


def _probe_llama() -> Optional[Any]:
    """Return the Llama class if importable, else None."""
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        logger.debug("llama-cpp-python not installed")
        return None


def build_draft_model(cfg: NativeSpeculativeConfig) -> Optional[Any]:
    """
    Construct a draft model object suitable for passing as Llama(draft_model=...).

    Returns None if speculative decoding is disabled, unavailable, or
    misconfigured. Callers should treat None as "run without spec decoding".

    Does NOT raise on failure — this is a perf feature, not a correctness one,
    and we always want the base pipeline to keep working.
    """
    if not cfg.enabled:
        return None

    if cfg.mode == "prompt_lookup":
        return _build_prompt_lookup(cfg)

    if cfg.mode == "draft_model":
        return _build_draft_llama(cfg)

    logger.warning(f"Unknown speculative mode '{cfg.mode}' — disabling")
    return None


def _build_prompt_lookup(cfg: NativeSpeculativeConfig) -> Optional[Any]:
    Cls = _probe_prompt_lookup()
    if Cls is None:
        logger.warning(
            "prompt_lookup speculative decoding requested but "
            "llama_cpp.llama_speculative.LlamaPromptLookupDecoding is unavailable. "
            "Upgrade llama-cpp-python or disable speculative.enabled."
        )
        return None

    try:
        draft = Cls(
            num_pred_tokens=cfg.num_pred_tokens,
            max_ngram_size=cfg.max_ngram_size,
        )
        logger.info(
            f"Native speculative decoding: prompt_lookup "
            f"(num_pred_tokens={cfg.num_pred_tokens}, max_ngram_size={cfg.max_ngram_size})"
        )
        return draft
    except TypeError:
        try:
            draft = Cls(num_pred_tokens=cfg.num_pred_tokens)
            logger.info(
                f"Native speculative decoding: prompt_lookup "
                f"(num_pred_tokens={cfg.num_pred_tokens}, max_ngram_size=<default>)"
            )
            return draft
        except Exception as e:
            logger.warning(f"Failed to build LlamaPromptLookupDecoding: {e}")
            return None
    except Exception as e:
        logger.warning(f"Failed to build LlamaPromptLookupDecoding: {e}")
        return None


def _build_draft_llama(cfg: NativeSpeculativeConfig) -> Optional[Any]:
    """
    Second-Llama draft-model path is intentionally not implemented.

    See the module docstring for the full rationale. TL;DR:
      - llama-cpp-python 0.3.9's `draft_model` kwarg only accepts
        LlamaDraftModel subclasses, and the only one shipped is
        LlamaPromptLookupDecoding.
      - A raw Llama is not a LlamaDraftModel.
      - A custom adapter using reset/eval/sample would do a full-context
        forward pass on every call, costing more than it saves.
      - The real 2.5x speedups reported on llama.cpp #10466 come from the
        `llama-speculative-simple` C++ CLI binary, not from Python.

    If you want the draft-model speedup today, that's a subprocess-based
    effort that belongs in its own module (e.g., `benchmark_phase13_cli.py`
    shelling out to `llama-speculative-simple --model-draft X --model Y`).
    """
    logger.warning(
        "speculative.mode=draft_model is not supported by the native "
        "llama-cpp-python path. Falling back to no speculative decoding. "
        "Use mode=prompt_lookup instead, or see experiment_backlog.md for "
        "the subprocess-based CLI approach."
    )
    return None


def describe(cfg: NativeSpeculativeConfig) -> str:
    """Short human-readable description for startup logging."""
    if not cfg.enabled:
        return "native speculative: disabled"
    if cfg.mode == "prompt_lookup":
        return f"native speculative: prompt_lookup (n={cfg.num_pred_tokens})"
    if cfg.mode == "draft_model":
        name = Path(cfg.draft_model_path).name if cfg.draft_model_path else "<unset>"
        return f"native speculative: draft_model ({name})"
    return f"native speculative: unknown mode '{cfg.mode}'"
