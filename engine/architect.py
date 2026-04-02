"""
Multi-Agent Architect — Task Decomposition and Assembly

The big model (3B) thinks. The small models (0.5B) build. The big model stitches.

Pipeline:
    1. ARCHITECT (3B): Decomposes complex prompt into independent subtasks
    2. WORKERS (0.5B x N): Build each piece in parallel with augmentor injection
    3. ASSEMBLER (3B): Reviews pieces, resolves interfaces, produces final code

Design principles:
    - Subtasks must be independently buildable (no circular deps)
    - Each subtask specifies: function/class name, inputs, outputs, description
    - Workers use the full augmentor system (failure routing + similarity)
    - Assembler gets all worker outputs + the original plan for context
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("UCA.architect")


@dataclass
class Subtask:
    """A single piece of work for a worker agent."""
    id: int
    name: str           # function/class name to implement
    description: str    # what it should do
    inputs: str         # input types/parameters
    outputs: str        # return type/value
    dependencies: list[int] = field(default_factory=list)  # subtask IDs this depends on


@dataclass
class SubtaskResult:
    """Result from a worker agent."""
    subtask: Subtask
    code: str
    elapsed: float
    success: bool
    error: str = ""


@dataclass
class ArchitectPlan:
    """The architect's decomposition of a complex task."""
    original_prompt: str
    overview: str
    subtasks: list[Subtask]
    plan_time: float


@dataclass
class AssemblyResult:
    """The final assembled output."""
    code: str
    plan: ArchitectPlan
    worker_results: list[SubtaskResult]
    total_time: float
    plan_time: float
    build_time: float
    assemble_time: float


# ── Prompt Templates ──

ARCHITECT_SYSTEM = """You are a software architect. Your job is to decompose a coding task into independent subtasks.

RULES:
- Break the task into 2-5 small, independent pieces
- Each piece should be one function or one class
- Specify exact function/class names, parameters, and return types
- Pieces should have minimal dependencies on each other
- Simple tasks (1 function) should NOT be decomposed — return just 1 subtask

OUTPUT FORMAT (strict JSON, no markdown):
{
  "overview": "Brief description of the architecture",
  "subtasks": [
    {
      "id": 1,
      "name": "function_or_class_name",
      "description": "What this piece does",
      "inputs": "parameter types",
      "outputs": "return type",
      "dependencies": []
    }
  ]
}"""

ASSEMBLER_SYSTEM = """You are a code integrator. You receive independently-built code pieces and must combine them into one working Python module.

RULES:
- Combine all pieces into a single ```python block
- Deduplicate imports (keep all unique imports at the top)
- Resolve any naming conflicts
- Add a brief main/demo section at the bottom showing usage
- Do NOT rewrite the pieces — just stitch and integrate
- If pieces reference each other, ensure the order is correct (dependencies first)
- Fix any obvious interface mismatches (wrong parameter names, missing returns)"""


class Architect:
    """Decomposes complex tasks using the large model."""

    def __init__(self, model, chat_format: str = "chatml",
                 max_tokens: int = 1024, temperature: float = 0.3):
        self.model = model
        self.chat_format = chat_format
        self.max_tokens = max_tokens
        self.temperature = temperature

    def decompose(self, prompt: str) -> ArchitectPlan:
        """Break a complex prompt into independent subtasks."""
        full_prompt = self._build_prompt(ARCHITECT_SYSTEM, prompt)

        start = time.monotonic()
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]
        output = self.model(
            full_prompt, max_tokens=self.max_tokens,
            temperature=self.temperature, stop=stop, echo=False,
        )
        elapsed = time.monotonic() - start

        text = output["choices"][0]["text"].strip()
        plan = self._parse_plan(text, prompt)
        plan.plan_time = elapsed

        logger.info(
            f"Architect: {len(plan.subtasks)} subtasks in {elapsed:.1f}s "
            f"[{', '.join(s.name for s in plan.subtasks)}]"
        )
        return plan

    def _parse_plan(self, text: str, original_prompt: str) -> ArchitectPlan:
        """Parse the architect's JSON output into a plan."""
        # Extract JSON from response (may have surrounding text)
        json_str = text
        if "{" in text:
            start = text.index("{")
            # Find matching closing brace
            depth = 0
            end = start
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = text[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: treat the whole thing as a single subtask
            logger.warning("Architect output not valid JSON, falling back to single task")
            return ArchitectPlan(
                original_prompt=original_prompt,
                overview="Single task (decomposition failed)",
                subtasks=[Subtask(
                    id=1, name="solution",
                    description=original_prompt,
                    inputs="", outputs="",
                )],
                plan_time=0,
            )

        subtasks = []
        for s in data.get("subtasks", []):
            subtasks.append(Subtask(
                id=s.get("id", len(subtasks) + 1),
                name=s.get("name", f"task_{len(subtasks) + 1}"),
                description=s.get("description", ""),
                inputs=s.get("inputs", ""),
                outputs=s.get("outputs", ""),
                dependencies=s.get("dependencies", []),
            ))

        if not subtasks:
            subtasks = [Subtask(
                id=1, name="solution",
                description=original_prompt,
                inputs="", outputs="",
            )]

        return ArchitectPlan(
            original_prompt=original_prompt,
            overview=data.get("overview", ""),
            subtasks=subtasks,
            plan_time=0,
        )

    def _build_prompt(self, system: str, user: str) -> str:
        if self.chat_format == "chatml":
            return (f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        return f"{system}\n\nUser: {user}\nAssistant:\n"


class Worker:
    """Builds individual subtasks using the small model + augmentors."""

    def __init__(self, model, augmentor_router, chat_format: str = "chatml",
                 max_tokens: int = 512, temperature: float = 0.2):
        self.model = model
        self.augmentor_router = augmentor_router
        self.chat_format = chat_format
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build(self, subtask: Subtask) -> SubtaskResult:
        """Generate code for a single subtask using the augmentor system."""
        # Build a clear, specific prompt from the subtask spec
        prompt = self._subtask_to_prompt(subtask)

        start = time.monotonic()
        try:
            if self.augmentor_router:
                # Use the full PIE augmentor pipeline
                from engine.augmentors import AugmentorResult
                shim = _ModelShim(self.model, self.max_tokens, self.temperature)
                result = self.augmentor_router.process(
                    query=prompt, model=shim, chat_format=self.chat_format,
                    module_hint="code_gen",
                    gen_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
                )
                if result:
                    code = self._extract_code(result.response)
                else:
                    code = self._generate_direct(prompt)
            else:
                code = self._generate_direct(prompt)

            elapsed = time.monotonic() - start
            logger.info(f"Worker [{subtask.name}]: {len(code)} chars in {elapsed:.1f}s")

            return SubtaskResult(
                subtask=subtask, code=code,
                elapsed=elapsed, success=True,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error(f"Worker [{subtask.name}] failed: {e}")
            return SubtaskResult(
                subtask=subtask, code="",
                elapsed=elapsed, success=False, error=str(e),
            )

    def _subtask_to_prompt(self, subtask: Subtask) -> str:
        """Convert a subtask spec into a natural coding prompt."""
        parts = [f"Write a Python {subtask.name}"]
        if subtask.description:
            parts.append(f"that {subtask.description}")
        if subtask.inputs:
            parts.append(f"Inputs: {subtask.inputs}")
        if subtask.outputs:
            parts.append(f"Returns: {subtask.outputs}")
        return ". ".join(parts)

    def _generate_direct(self, prompt: str) -> str:
        """Generate without augmentors (fallback)."""
        system = ("You are a Python coding assistant. Write clean, correct, complete Python code "
                  "in ```python blocks. Include all imports.")
        full = self._build_prompt(system, prompt)
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]
        output = self.model(
            full, max_tokens=self.max_tokens,
            temperature=self.temperature, stop=stop, echo=False,
        )
        return self._extract_code(output["choices"][0]["text"].strip())

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def _build_prompt(self, system: str, user: str) -> str:
        if self.chat_format == "chatml":
            return (f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        return f"{system}\n\nUser: {user}\nAssistant:\n"


class Assembler:
    """Stitches worker outputs into a final working module."""

    def __init__(self, model, chat_format: str = "chatml",
                 max_tokens: int = 2048, temperature: float = 0.2):
        self.model = model
        self.chat_format = chat_format
        self.max_tokens = max_tokens
        self.temperature = temperature

    def assemble(self, plan: ArchitectPlan, results: list[SubtaskResult]) -> str:
        """Combine worker outputs into final code."""
        # Build the assembly prompt
        pieces = []
        for r in results:
            if r.success and r.code:
                pieces.append(f"# === {r.subtask.name}: {r.subtask.description} ===\n{r.code}")

        if len(pieces) <= 1:
            # Single piece — no assembly needed
            return pieces[0] if pieces else ""

        assembly_input = (
            f"Original task: {plan.original_prompt}\n\n"
            f"Architecture: {plan.overview}\n\n"
            f"Code pieces to combine:\n\n"
            + "\n\n".join(pieces)
        )

        full_prompt = self._build_prompt(ASSEMBLER_SYSTEM, assembly_input)

        start = time.monotonic()
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]
        output = self.model(
            full_prompt, max_tokens=self.max_tokens,
            temperature=self.temperature, stop=stop, echo=False,
        )
        elapsed = time.monotonic() - start

        text = output["choices"][0]["text"].strip()
        code = self._extract_code(text)

        logger.info(f"Assembler: {len(code)} chars in {elapsed:.1f}s")
        return code

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def _build_prompt(self, system: str, user: str) -> str:
        if self.chat_format == "chatml":
            return (f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        return f"{system}\n\nUser: {user}\nAssistant:\n"


class MultiAgentOrchestrator:
    """
    Top-level orchestrator: Architect (3B) -> Workers (0.5B) -> Assembler (3B)

    Loads both models simultaneously on GPU. The 3B plans and assembles,
    the 0.5B builds with augmentor-injected examples.
    """

    def __init__(self,
                 architect_model_path: str = "models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
                 worker_model_path: str = "models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
                 max_workers: int = 1,
                 gpu_layers: int = 99,
                 context_length: int = 4096):
        self.architect_model_path = architect_model_path
        self.worker_model_path = worker_model_path
        self.max_workers = max_workers
        self.gpu_layers = gpu_layers
        self.context_length = context_length

        self.architect_llm = None
        self.worker_llm = None
        self.architect = None
        self.worker = None
        self.assembler = None
        self.augmentor_router = None

    def initialize(self):
        """Load both models and set up augmentors."""
        from llama_cpp import Llama
        from benchmark_exec import detect_chat_format

        # Load architect model (3B)
        logger.info(f"Loading architect model: {self.architect_model_path}")
        arch_format = detect_chat_format(self.architect_model_path)
        self.architect_llm = Llama(
            model_path=self.architect_model_path,
            n_ctx=self.context_length,
            n_gpu_layers=self.gpu_layers,
            n_threads=8, n_batch=512, verbose=False,
        )
        self.architect = Architect(self.architect_llm, chat_format=arch_format)
        self.assembler = Assembler(self.architect_llm, chat_format=arch_format)
        logger.info("Architect model loaded")

        # Load worker model (0.5B)
        logger.info(f"Loading worker model: {self.worker_model_path}")
        worker_format = detect_chat_format(self.worker_model_path)
        self.worker_llm = Llama(
            model_path=self.worker_model_path,
            n_ctx=self.context_length,
            n_gpu_layers=self.gpu_layers,
            n_threads=8, n_batch=512, verbose=False,
        )

        # Set up augmentors for workers
        from engine.augmentors import AugmentorRouter
        from engine.embedder import get_embedder
        self.augmentor_router = AugmentorRouter(yaml_dir="data/augmentor_examples")
        embedder = get_embedder()
        self.augmentor_router.init_embeddings(embedder)

        worker_size_mb = Path(self.worker_model_path).stat().st_size / (1024 * 1024)
        self.augmentor_router.use_auto_augmentors(worker_size_mb)

        self.worker = Worker(
            self.worker_llm, self.augmentor_router,
            chat_format=worker_format,
        )
        logger.info(f"Worker model loaded (auto mode, {worker_size_mb:.0f}MB)")

    def process(self, prompt: str) -> AssemblyResult:
        """Full pipeline: decompose -> build in parallel -> assemble."""
        total_start = time.monotonic()

        # Step 1: Architect decomposes
        print(f"\n  [ARCHITECT] Decomposing task...")
        plan = self.architect.decompose(prompt)
        print(f"  [ARCHITECT] {len(plan.subtasks)} subtasks in {plan.plan_time:.1f}s:")
        for s in plan.subtasks:
            deps = f" (needs: {s.dependencies})" if s.dependencies else ""
            print(f"    {s.id}. {s.name}: {s.description[:60]}{deps}")

        # Step 2: Workers build pieces (respecting dependencies)
        print(f"\n  [WORKERS] Building {len(plan.subtasks)} pieces...")
        build_start = time.monotonic()
        results = self._execute_subtasks(plan.subtasks)
        build_time = time.monotonic() - build_start

        for r in results:
            status = "OK" if r.success else "FAIL"
            lines = len(r.code.strip().split("\n")) if r.code else 0
            print(f"    [{status}] {r.subtask.name}: {lines} lines in {r.elapsed:.1f}s")

        # Step 3: Assembler stitches
        print(f"\n  [ASSEMBLER] Combining pieces...")
        assemble_start = time.monotonic()
        if len(results) <= 1 and results[0].success:
            # Single piece — no assembly needed
            final_code = results[0].code
        else:
            final_code = self.assembler.assemble(plan, results)
        assemble_time = time.monotonic() - assemble_start

        total_time = time.monotonic() - total_start
        lines = len(final_code.strip().split("\n")) if final_code else 0
        print(f"  [ASSEMBLER] Done: {lines} lines in {assemble_time:.1f}s")
        print(f"\n  Total: {total_time:.1f}s (plan={plan.plan_time:.1f}s build={build_time:.1f}s assemble={assemble_time:.1f}s)")

        return AssemblyResult(
            code=final_code,
            plan=plan,
            worker_results=results,
            total_time=total_time,
            plan_time=plan.plan_time,
            build_time=build_time,
            assemble_time=assemble_time,
        )

    def _execute_subtasks(self, subtasks: list[Subtask]) -> list[SubtaskResult]:
        """Execute subtasks respecting dependency order.

        Groups tasks into waves: wave 1 = no deps, wave 2 = depends on wave 1, etc.
        Tasks within a wave can run in parallel (if max_workers > 1).
        """
        completed: dict[int, SubtaskResult] = {}
        remaining = list(subtasks)

        while remaining:
            # Find tasks whose dependencies are all completed
            ready = [s for s in remaining
                     if all(d in completed for d in s.dependencies)]
            if not ready:
                # Circular dependency or missing dep — force execute remaining
                logger.warning("Dependency deadlock, forcing remaining tasks")
                ready = remaining

            remaining = [s for s in remaining if s not in ready]

            # Execute ready tasks (parallel if multiple workers)
            if self.max_workers > 1 and len(ready) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {pool.submit(self.worker.build, s): s for s in ready}
                    for future in as_completed(futures):
                        result = future.result()
                        completed[result.subtask.id] = result
            else:
                for s in ready:
                    result = self.worker.build(s)
                    completed[result.subtask.id] = result

        # Return in original order
        return [completed[s.id] for s in subtasks if s.id in completed]


class _ModelShim:
    """Wraps llama_cpp.Llama to look like the augmentor's expected model interface."""

    def __init__(self, model, max_tokens: int, temperature: float):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.total_time = 0.0

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]

        start = time.monotonic()
        output = self._model(
            prompt, max_tokens=max_tokens,
            temperature=temperature, stop=stop, echo=False,
        )
        self.total_time += time.monotonic() - start
        return output["choices"][0]["text"].strip()

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate
