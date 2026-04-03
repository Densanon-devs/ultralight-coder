#!/usr/bin/env python3
"""
Ultralight Code Assistant — CLI Entry Point

A local coding assistant powered by tiny code-specialized models.
Branched from Plug-in Intelligence Engine.

Usage:
    python main.py                    # Interactive mode
    python main.py --config my.yaml   # Custom config
    python main.py --dry-run          # Test routing without model
    python main.py --list-modules     # Show available modules
    python main.py --explain "prompt" # Explain routing decision
    python main.py --no-pipeline      # Disable async pipeline
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import Config
from engine.base_model import BaseModel
from engine.router import Router
from engine.module_manager import ModuleManager
from engine.memory import MemorySystem
from engine.fusion import FusionLayer
from engine.pipeline import Pipeline
from engine.kv_cache import KVCacheManager
from engine.micro_adapters import MicroAdapterEngine
from engine.tools import ToolRegistry
from engine.augmentors import AugmentorRouter
from engine.speculative import SpeculativeEngine
from engine.project_context import ProjectIndex

logger = logging.getLogger("UCA")


class UltralightCodeAssistant:
    """
    Main orchestrator. Wires all components together
    and runs the inference loop for code assistance.
    """

    def __init__(self, config_path: str = "config.yaml", dry_run: bool = False, use_pipeline: bool = True):
        self.config = Config(config_path)
        self._apply_tuned_config()
        self.config.setup_logging()
        self.dry_run = dry_run

        logger.info(f"=== {self.config.system.name} v{self.config.system.version} ===")

        # Core components
        self.base_model = BaseModel(self.config.base_model)
        self.router = Router(self.config.router)
        self.modules = ModuleManager(self.config.modules)
        self.memory = MemorySystem(self.config.memory)
        self.fusion = FusionLayer(
            self.config.fusion,
            token_counter=self.base_model.count_tokens,
        )

        # Pipeline
        pipeline_enabled = use_pipeline and self.config.pipeline.enabled
        self.pipeline = Pipeline(
            parallel_workers=self.config.pipeline.parallel_workers,
            enable_queue=self.config.pipeline.enable_generation_queue and not dry_run,
        )

        # KV Cache
        self.kv_cache = KVCacheManager(
            context_length=self.config.base_model.context_length,
            max_generation_tokens=self.config.base_model.max_tokens,
        )
        self.kv_cache.set_token_counter(self.base_model.count_tokens)

        # Micro-adapters
        self.micro_adapters = None
        if self.config.micro_adapters.enabled:
            self.micro_adapters = MicroAdapterEngine(
                storage_dir=self.config.micro_adapters.storage_dir,
                embedding_model=self.config.micro_adapters.embedding_model,
                min_cluster_size=self.config.micro_adapters.min_cluster_size,
                max_adapters=self.config.micro_adapters.max_adapters,
                regenerate_interval=self.config.micro_adapters.regenerate_interval,
            )

        # Tools
        self.tools = ToolRegistry()

        # Augmentor system (code-focused, YAML examples + auto mode)
        self.augmentor_router = AugmentorRouter(yaml_dir="data/augmentor_examples")
        self._augmentors_enabled = self._should_enable_augmentors()

        # Response cache
        self.speculative = SpeculativeEngine(storage_dir="data/cache")

        # Project context indexer
        self.project_index = ProjectIndex(self.config.project_context)

        # Performance tracking
        self._perf_history: list[dict] = []

    def _should_enable_augmentors(self) -> bool:
        """Enable augmentors for all supported models.

        Phase 6 proved auto mode works for all model sizes:
        - Sub-1.5GB: rerank1 (single example injection)
        - 1.5GB+: rerank (two example injection)
        """
        model_path = Path(self.config.base_model.path)
        if not model_path.exists():
            return True
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Augmentor system: ON (model={size_mb:.0f}MB, 230 YAML examples)")
        return True

    def _apply_tuned_config(self):
        """Load and apply tuned_config.json if it exists."""
        tuned_path = Path(self.config.config_path).parent / "tuned_config.json"
        if not tuned_path.exists():
            return

        try:
            import json
            with open(tuned_path) as f:
                tuned = json.load(f)

            opt = tuned.get("optimal", {})
            applied = []

            if "threads" in opt:
                self.config.base_model.threads = opt["threads"]
                applied.append(f"threads={opt['threads']}")
            if "temperature" in opt:
                self.config.base_model.temperature = opt["temperature"]
                applied.append(f"temperature={opt['temperature']}")
            if "batch_size" in opt:
                self.config.base_model.batch_size = opt["batch_size"]
                applied.append(f"batch_size={opt['batch_size']}")
            if "gpu_layers" in opt:
                self.config.base_model.gpu_layers = opt["gpu_layers"]
                applied.append(f"gpu_layers={opt['gpu_layers']}")
            if "chat_format" in opt:
                self.config.fusion.chat_format = opt["chat_format"]
                applied.append(f"chat_format={opt['chat_format']}")

            if applied:
                logger.info(f"Applied tuned config: {', '.join(applied)}")

        except Exception as e:
            logger.warning(f"Failed to load tuned config: {e}")

    def initialize(self):
        """Load model, discover modules, start pipeline."""
        available = self.modules.discover()
        logger.info(f"Available modules: {available}")

        if self.router.classifier.needs_training:
            logger.info("Training classifier...")
            stats = self.router.train_classifier()
            logger.info(f"Classifier training result: {stats}")

        if not self.dry_run:
            try:
                self.base_model.load()
            except FileNotFoundError as e:
                print(f"\n{e}\n")
                print("Download a model first: python download_model.py")
                print("Or use --dry-run to test routing.\n")
                sys.exit(1)
            except ImportError:
                print(
                    "\nllama-cpp-python not installed.\n"
                    "Install with: pip install llama-cpp-python\n"
                    "Or use --dry-run to test without a model.\n"
                )
                sys.exit(1)

            def generate_fn(prompt, max_tokens=None, temperature=None, **kwargs):
                return self.base_model.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature, **kwargs,
                )
            self.pipeline.set_generate_fn(generate_fn)

            def stream_fn(prompt, max_tokens=None, temperature=None):
                return self.base_model.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature, stream=True,
                )
            self.pipeline.set_stream_fn(stream_fn)
        else:
            logger.info("DRY RUN mode — no model loaded")

        # Initialize augmentor embeddings + auto mode
        try:
            from engine.embedder import get_embedder
            embedder = get_embedder()
            if embedder:
                self.augmentor_router.init_embeddings(embedder)
                self.project_index.init_embedder(embedder)
        except Exception as e:
            logger.debug(f"Augmentor embeddings not initialized: {e}")

        if self._augmentors_enabled:
            model_path = Path(self.config.base_model.path)
            model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 1000
            self.augmentor_router.use_auto_augmentors(model_size_mb)

        self.pipeline.start()
        logger.info("Ready")

    def process(self, user_input: str) -> str:
        """Full pipeline: route -> augmentor/fusion -> generate -> post-process."""
        result = self._handle_commands(user_input)
        if result is not None:
            return result

        perf = {"start": time.monotonic()}

        # Check response cache
        cached = self.speculative.try_cache(user_input)
        if cached is not None:
            perf["cache_hit"] = True
            perf["total"] = time.monotonic() - perf["start"]
            self.memory.add_interaction("user", user_input)
            self.memory.add_interaction("assistant", cached)
            self._perf_history.append(perf)
            return cached

        # Route
        routing = self.router.route(
            user_prompt=user_input,
            conversation_history=self.memory.short_term.get_turns(),
            available_modules=self.modules.available_modules,
        )
        perf["route"] = time.monotonic() - perf["start"]

        # Augmentor system
        if not self.dry_run and self._augmentors_enabled and routing.selected_modules:
            module_hint = routing.selected_modules[0]

            # Build extra context for augmentor: conversation history + project context
            extra_parts = []
            conv_history = self.memory.short_term.get_context()
            if conv_history:
                extra_parts.append(conv_history)
            if self.project_index.is_indexed:
                project_ctx = self.project_index.get_context(user_input, top_k=3)
                if project_ctx:
                    extra_parts.append(project_ctx)
            extra_ctx = "\n\n".join(extra_parts)

            augmentor_result = self.augmentor_router.process(
                query=user_input,
                model=self.base_model,
                chat_format=self.config.fusion.chat_format,
                module_hint=module_hint,
                extra_context=extra_ctx,
                gen_kwargs={"max_tokens": self.config.base_model.max_tokens,
                            "temperature": self.config.base_model.temperature},
            )
            if augmentor_result is not None:
                perf["augmentor"] = augmentor_result.augmentor_name
                perf["augmentor_attempts"] = augmentor_result.attempts
                response = augmentor_result.response
                perf["generation"] = time.monotonic() - perf["start"] - perf["route"]
                self._post_process(user_input, response, routing, perf)
                return response

        # Micro-adapter selection
        active_adapter = None
        adapter_applied = None
        if self.micro_adapters:
            active_adapter = self.micro_adapters.select_adapter(user_input)
            if active_adapter:
                adapter_applied = self.micro_adapters.apply(
                    active_adapter,
                    base_temperature=self.config.base_model.temperature,
                    base_max_tokens=self.config.base_model.max_tokens,
                )
                perf["adapter"] = active_adapter.name

        # Parallel I/O
        def _load_modules_and_lora():
            active = self.modules.get_multiple(routing.selected_modules)
            lora_modules = [m for m in active if m.manifest.lora_path]
            if lora_modules:
                top_lora = max(lora_modules, key=lambda m: m.manifest.priority)
                if self.base_model.active_lora != top_lora.manifest.lora_path:
                    self.base_model.load_lora(top_lora.manifest.lora_path)
            elif self.base_model.active_lora:
                self.base_model.unload_lora()
            return active

        def _recall_memory():
            return self.memory.recall(user_input)

        def _predictive_preload():
            if self.config.modules.predictive_preload:
                predicted = self.modules.predict_next_modules(
                    routing.selected_modules,
                    top_k=self.config.modules.preload_top_k,
                )
                if predicted:
                    self.modules.preload(predicted)
                    return predicted
            return []

        io_result = self.pipeline.run_parallel_io(
            module_loader=_load_modules_and_lora,
            memory_retriever=_recall_memory,
            preloader=_predictive_preload,
            knowledge_loader=lambda: None,
        )
        active_modules = io_result.active_modules
        memory_context = io_result.memory_context
        perf["parallel_io"] = time.monotonic() - perf["start"] - perf.get("route", 0)

        # Fusion
        if adapter_applied and adapter_applied.get("context_prompt"):
            memory_context["adapter"] = adapter_applied["context_prompt"]

        tool_prompt = self.tools.get_tool_prompt()
        if tool_prompt:
            memory_context["tools"] = tool_prompt

        # Project context — retrieve relevant code from indexed project
        if self.project_index.is_indexed:
            project_ctx = self.project_index.get_context(user_input)
            if project_ctx:
                memory_context["project"] = project_ctx

        prompt = self.fusion.assemble(
            user_input=user_input,
            active_modules=active_modules,
            memory_context=memory_context,
            blend_weights=routing.blend_weights,
        )

        # Generate
        gen_start = time.monotonic()
        if self.dry_run:
            response = self._dry_run_response(routing, active_modules, prompt, perf)
        else:
            gen_kwargs = {}
            if adapter_applied:
                gen_kwargs["temperature"] = adapter_applied["temperature"]
                gen_kwargs["max_tokens"] = adapter_applied["max_tokens"]
            response = self.pipeline.generate(prompt, **gen_kwargs)

        perf["generation"] = time.monotonic() - gen_start
        self._post_process(user_input, response, routing, perf)
        return response

    def process_stream(self, user_input: str):
        """Streaming version of process(). Yields tokens."""
        result = self._handle_commands(user_input)
        if result is not None:
            yield result
            return

        if self.dry_run:
            yield self.process(user_input)
            return

        routing = self.router.route(
            user_prompt=user_input,
            conversation_history=self.memory.short_term.get_turns(),
            available_modules=self.modules.available_modules,
        )

        def _load_modules_and_lora():
            active = self.modules.get_multiple(routing.selected_modules)
            lora_modules = [m for m in active if m.manifest.lora_path]
            if lora_modules:
                top_lora = max(lora_modules, key=lambda m: m.manifest.priority)
                if self.base_model.active_lora != top_lora.manifest.lora_path:
                    self.base_model.load_lora(top_lora.manifest.lora_path)
            elif self.base_model.active_lora:
                self.base_model.unload_lora()
            return active

        io_result = self.pipeline.run_parallel_io(
            module_loader=_load_modules_and_lora,
            memory_retriever=lambda: self.memory.recall(user_input),
            preloader=lambda: None,
            knowledge_loader=lambda: None,
        )

        prompt = self.fusion.assemble(
            user_input=user_input,
            active_modules=io_result.active_modules,
            memory_context=io_result.memory_context,
            blend_weights=routing.blend_weights,
        )

        full_response = []
        for token, is_final in self.pipeline.generate_stream(prompt):
            full_response.append(token)
            yield token

        response = "".join(full_response).strip()
        self.memory.add_interaction("user", user_input)
        self.memory.add_interaction("assistant", response)
        if routing.selected_modules:
            self.router.record_interaction(user_input, routing.selected_modules)
            self.modules.record_usage(routing.selected_modules)

    def _post_process(self, user_input: str, response: str, routing, perf: dict):
        """Post-processing: memory, classifier training, cache."""
        self.memory.add_interaction("user", user_input)
        self.memory.add_interaction("assistant", response)

        module = routing.selected_modules[0] if routing.selected_modules else None
        self.speculative.record(
            user_input, response, module=module, quality_verified=True,
        )

        if routing.selected_modules:
            self.router.record_interaction(user_input, routing.selected_modules)
            self.modules.record_usage(routing.selected_modules)

        if self.micro_adapters and routing.selected_modules:
            self.micro_adapters.record_interaction(
                prompt=user_input, response=response, modules=routing.selected_modules,
            )

        self.modules.cleanup_stale()

        perf["total"] = time.monotonic() - perf["start"]
        self._perf_history.append(perf)
        if len(self._perf_history) > 50:
            self._perf_history = self._perf_history[-50:]

    def _handle_commands(self, user_input: str) -> str | None:
        """Handle /commands."""
        cmd = user_input.strip().lower()

        if cmd == "/modules":
            modules = self.modules.list_all()
            if not modules:
                return "No modules found."
            lines = ["Available modules:"]
            for m in modules:
                cached = f" [{m['tier']}]" if m["cached"] else ""
                lines.append(f"  {m['name']}{cached} — {m['description']}")
            return "\n".join(lines)

        if cmd == "/memory":
            status = self.memory.status()
            return (
                f"Memory: {status['short_term_turns']} turns, "
                f"{status['long_term_count']} long-term entries"
            )

        if cmd.startswith("/remember "):
            fact = user_input[10:].strip()
            self.memory.remember(fact, source="user", importance=0.8)
            return f"Stored: {fact}"

        if cmd.startswith("/explain "):
            query = user_input[9:].strip()
            return self.router.explain(query)

        if cmd == "/status":
            return (
                f"Model: {'loaded' if self.base_model.is_loaded else 'not loaded'}\n"
                f"Augmentors: {'ON' if self._augmentors_enabled else 'OFF'}\n"
                f"Modules: {len(self.modules.available_modules)} available\n"
                f"Memory: {self.memory.status()['short_term_turns']} turns"
            )

        if cmd == "/perf":
            if not self._perf_history:
                return "No performance data yet."
            last = self._perf_history[-1]
            lines = ["Last turn:"]
            for k, v in last.items():
                if k == "start":
                    continue
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.3f}s")
                else:
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        if cmd in ("/help", "/h"):
            return (
                "Commands:\n"
                "  /modules    — List available modules\n"
                "  /memory     — Memory status\n"
                "  /remember X — Store a fact\n"
                "  /explain X  — Explain routing for a prompt\n"
                "  /status     — System status\n"
                "  /perf       — Last turn performance\n"
                "  /help       — This help\n"
                "  /quit       — Exit"
            )

        if cmd in ("/quit", "/exit", "/q"):
            return "__QUIT__"

        return None

    def _dry_run_response(self, routing, active_modules, prompt, perf):
        """Generate a mock response for dry-run mode."""
        lines = [
            "[DRY RUN]",
            f"Routed to: {routing.selected_modules}",
            f"Mode: {routing.routing_mode}",
            f"Modules loaded: {[m.manifest.name for m in active_modules]}",
            f"Prompt length: ~{len(prompt)//4} tokens",
        ]
        return "\n".join(lines)

    def shutdown(self):
        """Clean shutdown."""
        self.pipeline.shutdown()
        self.base_model.unload()
        logger.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Ultralight Code Assistant"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Test without model")
    parser.add_argument("--no-pipeline", action="store_true", help="Disable async pipeline")
    parser.add_argument("--list-modules", action="store_true", help="List available modules")
    parser.add_argument("--explain", type=str, help="Explain routing for a prompt")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    args = parser.parse_args()

    engine = UltralightCodeAssistant(
        config_path=args.config,
        dry_run=args.dry_run,
        use_pipeline=not args.no_pipeline,
    )
    engine.initialize()

    if args.list_modules:
        print(engine.process("/modules"))
        return

    if args.explain:
        print(engine.router.explain(args.explain))
        return

    if args.train:
        stats = engine.router.train_classifier()
        print(f"Training result: {stats}")
        return

    # Interactive REPL
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    print(f"\n  Ultralight Code Assistant v{engine.config.system.version}")
    print(f"  Model: {Path(engine.config.base_model.path).name}")
    print(f"  Type /help for commands, /quit to exit\n")

    while True:
        try:
            user_input = input("you> ").strip()
            if not user_input:
                continue

            response = engine.process(user_input)
            if response == "__QUIT__":
                break

            if use_rich and ("```" in response or "**" in response or "#" in response):
                console.print(Markdown(response))
            else:
                print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except EOFError:
            break

    engine.shutdown()


if __name__ == "__main__":
    main()
