"""
Base Model — The "Tiny Brain"

Loads and manages the core GGUF model via llama-cpp-python.
Supports dynamic LoRA adapter injection, generation with
configurable parameters, and graceful resource management.

This is the workhorse: it handles raw token generation.
Everything else (routing, memory, fusion) builds on top.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BaseModel:
    """
    Wraps llama-cpp-python to provide:
    - Model loading with quantization/GPU layer control
    - LoRA adapter hot-loading and unloading
    - Text generation with streaming support
    - Token counting for prompt budget management
    """

    def __init__(self, config):
        """
        Args:
            config: BaseModelConfig dataclass from engine.config
        """
        self.config = config
        self.model = None
        self._loaded_lora: Optional[str] = None

    def load(self):
        """Load the base GGUF model into memory."""
        model_path = Path(self.config.path)

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Download a GGUF model and update config.yaml.\n"
                f"Recommended starters:\n"
                f"  - TinyLlama 1.1B: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF\n"
                f"  - Qwen2 1.5B: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF\n"
                f"  - Phi-2 2.7B: https://huggingface.co/TheBloke/phi-2-GGUF"
            )

        logger.info(f"Loading model: {model_path}")
        logger.info(f"  Context: {self.config.context_length} tokens")
        logger.info(f"  GPU layers: {self.config.gpu_layers}")
        logger.info(f"  Threads: {self.config.threads}")

        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=self.config.threads,
                n_batch=self.config.batch_size,
                verbose=logger.isEnabledFor(logging.DEBUG),
            )
            logger.info("Model loaded successfully")

        except ImportError:
            logger.error(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_lora(self, lora_path: str):
        """
        Load a LoRA adapter on top of the base model.

        Args:
            lora_path: Path to the LoRA adapter file (.gguf)
        """
        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load() first.")

        lora_file = Path(lora_path)
        if not lora_file.exists():
            logger.warning(f"LoRA file not found: {lora_path}")
            return False

        try:
            # llama-cpp-python supports LoRA via model recreation
            # or via the lora_path parameter. For hot-swapping,
            # we use the adapter loading API.
            from llama_cpp import Llama

            logger.info(f"Loading LoRA adapter: {lora_file.name}")

            # For v1, we reload the model with the LoRA applied.
            # Future: use llama_lora_adapter_set for true hot-swap.
            self.model = Llama(
                model_path=self.config.path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=self.config.threads,
                n_batch=self.config.batch_size,
                lora_path=str(lora_file),
                verbose=logger.isEnabledFor(logging.DEBUG),
            )
            self._loaded_lora = str(lora_file)
            logger.info(f"LoRA adapter loaded: {lora_file.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False

    def unload_lora(self):
        """Remove the current LoRA adapter, reverting to base model."""
        if self._loaded_lora is None:
            return

        logger.info("Unloading LoRA adapter, reverting to base model")
        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.config.path,
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.gpu_layers,
                n_threads=self.config.threads,
                n_batch=self.config.batch_size,
                verbose=logger.isEnabledFor(logging.DEBUG),
            )
            self._loaded_lora = None
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        grammar=None,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The full assembled prompt
            max_tokens: Override max generation length
            temperature: Override sampling temperature
            stop: Stop sequences
            stream: If True, returns a generator that yields tokens
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k sampling (0 = disabled)
            repeat_penalty: Repetition penalty (1.0 = none, >1.0 = penalize)
            grammar: GBNF grammar object for constrained generation

        Returns:
            Generated text string (or generator if stream=True)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tok = max_tokens or self.config.max_tokens
        temp = temperature or self.config.temperature
        stop_seqs = stop or ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]

        logger.debug(f"Generating (max_tokens={max_tok}, temp={temp}, stream={stream})")

        if stream:
            return self._generate_stream(prompt, max_tok, temp, stop_seqs)

        # Build kwargs for llama-cpp
        gen_kwargs = {
            "max_tokens": max_tok,
            "temperature": temp,
            "stop": stop_seqs,
            "echo": False,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if repeat_penalty is not None:
            gen_kwargs["repeat_penalty"] = repeat_penalty
        if grammar is not None:
            gen_kwargs["grammar"] = grammar

        try:
            output = self.model(prompt, **gen_kwargs)

            response = output["choices"][0]["text"].strip()
            usage = output.get("usage", {})
            logger.debug(
                f"Generated {usage.get('completion_tokens', '?')} tokens "
                f"(prompt: {usage.get('prompt_tokens', '?')})"
            )
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Generation Error: {e}]"

    def _generate_stream(self, prompt: str, max_tokens: int,
                         temperature: float, stop: list[str]):
        """
        Phase 5: Streaming generation — yields tokens one at a time.
        Returns a generator that yields (token_text, is_final) tuples.
        """
        try:
            stream = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                echo=False,
                stream=True,
            )

            for chunk in stream:
                token = chunk["choices"][0]["text"]
                finish = chunk["choices"][0].get("finish_reason")
                yield token, finish is not None

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"[Generation Error: {e}]", True

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a string.
        Useful for prompt budget management.
        """
        if self.model is None:
            # Rough estimate: ~4 chars per token
            return len(text) // 4

        try:
            tokens = self.model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception:
            return len(text) // 4

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def active_lora(self) -> Optional[str]:
        return self._loaded_lora

    def unload(self):
        """Free the model from memory."""
        if self.model is not None:
            logger.info("Unloading base model from memory")
            del self.model
            self.model = None
            self._loaded_lora = None

    def __repr__(self):
        status = "loaded" if self.is_loaded else "not loaded"
        lora = f", lora={Path(self._loaded_lora).name}" if self._loaded_lora else ""
        return f"BaseModel({status}{lora})"
