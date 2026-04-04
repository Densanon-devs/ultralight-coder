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

GLUE_SYSTEM = """You are a code integrator. Given component classes with their exact methods, write ONLY a short main() that wires them together.

RULES:
- Output ONLY the main() code in a ```python block
- Do NOT redefine any classes or functions
- Do NOT include imports
- Use ONLY the exact constructor signatures and method names shown
- Pass all required constructor parameters when instantiating
- Keep it under 30 lines"""


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
                 max_tokens: int = 768, temperature: float = 0.2):
        self.model = model
        self.augmentor_router = augmentor_router
        self.chat_format = chat_format
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build(self, subtask: Subtask, siblings: list[Subtask] | None = None) -> SubtaskResult:
        """Generate code for a single subtask using the augmentor system."""
        # Build a clear, specific prompt from the subtask spec + sibling interfaces
        prompt = self._subtask_to_prompt(subtask, siblings)

        start = time.monotonic()
        try:
            if self.augmentor_router:
                # Use the full PIE augmentor pipeline
                # Route to component augmentor when building with sibling context
                from engine.augmentors import AugmentorResult
                hint = "component" if siblings else "code_gen"
                shim = _ModelShim(self.model, self.max_tokens, self.temperature)
                result = self.augmentor_router.process(
                    query=prompt, model=shim, chat_format=self.chat_format,
                    module_hint=hint,
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

    def _subtask_to_prompt(self, subtask: Subtask, siblings: list[Subtask] | None = None) -> str:
        """Convert a subtask spec into a natural coding prompt with interface context."""
        parts = [f"Write a Python {subtask.name}"]
        if subtask.description:
            parts.append(f"that {subtask.description}")
        if subtask.inputs:
            parts.append(f"Inputs: {subtask.inputs}")
        if subtask.outputs:
            parts.append(f"Returns: {subtask.outputs}")

        prompt = ". ".join(parts)

        # Inject sibling interface specs so this piece can reference others
        if siblings:
            others = [s for s in siblings if s.id != subtask.id]
            if others:
                iface_lines = []
                for s in others:
                    sig = s.name
                    if s.inputs:
                        sig += f"({s.inputs})"
                    if s.outputs:
                        sig += f" -> {s.outputs}"
                    iface_lines.append(f"  - {sig}: {s.description}")
                prompt += (
                    "\n\nOther components in this project (use these interfaces, "
                    "do NOT redefine them):\n" + "\n".join(iface_lines)
                )

        return prompt

    def _generate_direct(self, prompt: str) -> str:
        """Generate without augmentors (fallback)."""
        if "Other components in this project" in prompt:
            system = (
                "You are a Python component builder. Write ONLY the ONE class or function requested. "
                "Do NOT define other components. Do NOT add methods that belong to other components. "
                "Each component has its own job — only implement methods for YOUR component. "
                "Accept dependencies as constructor parameters. "
                "Use forward references (quotes) for type hints to sibling classes. "
                "Include only the imports YOUR component needs. "
                "Output in a ```python block."
            )
        else:
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
    """Fast deterministic assembly: concat + dedup imports + short LLM glue.

    Instead of asking the 3B to rewrite all code (~100s, error-prone),
    this concatenates worker outputs deterministically and only uses the
    LLM for a short glue/main section (~10-15s).
    """

    def __init__(self, model=None, chat_format: str = "chatml",
                 max_tokens: int = 512, temperature: float = 0.2):
        self.model = model
        self.chat_format = chat_format
        self.max_tokens = max_tokens
        self.temperature = temperature

    def assemble(self, plan: ArchitectPlan, results: list[SubtaskResult]) -> str:
        """Fast assembly: deterministic concat + optional LLM glue."""
        pieces = [(r.subtask, r.code) for r in results if r.success and r.code]
        if not pieces:
            return ""
        if len(pieces) == 1:
            return pieces[0][1]

        # Step 1: Extract only the assigned component from each worker output,
        # strip if-main blocks, dedup methods, collect imports
        all_imports: list[str] = []
        code_blocks: list[tuple[Subtask, str]] = []
        for subtask, code in pieces:
            code = self._strip_if_main(code)
            code = self._dedup_methods(code)
            imports, body = self._split_imports(code)
            all_imports.extend(imports)
            # Extract only the class/function this worker was assigned
            extracted = self._extract_named(body, subtask.name)
            if extracted:
                code_blocks.append((subtask, extracted))
            elif body.strip():
                code_blocks.append((subtask, body.strip()))

        unique_imports = self._dedup_imports(all_imports)

        # Step 2: Deduplicate and prune
        # - Remove class/function defs that belong to other workers
        # - Remove methods that semantically belong to sibling components
        all_subtasks = [s for s, _ in code_blocks]
        assigned_names = {s.name for s in all_subtasks}
        final_blocks: list[tuple[Subtask, str]] = []
        for subtask, body in code_blocks:
            cleaned = self._remove_foreign_defs(body, subtask.name, assigned_names)
            cleaned = self._prune_foreign_methods(cleaned, subtask, all_subtasks)
            if cleaned.strip():
                final_blocks.append((subtask, cleaned.strip()))

        # Step 3: Concatenate bodies in dependency order (already ordered)
        # Skip sections that are only comments or whitespace
        sections = []
        for subtask, body in final_blocks:
            # Check if body has any actual code (not just comments)
            has_code = any(line.strip() and not line.strip().startswith("#")
                          for line in body.split("\n"))
            if has_code:
                sections.append(f"# -- {subtask.name} --\n{body}")

        # Prepend future annotations to make all type hints lazy —
        # prevents NameError for forward references (Iterator, Plugin, etc.)
        combined = "from __future__ import annotations\n\n"
        if unique_imports:
            combined += "\n".join(unique_imports) + "\n\n\n"
        combined += "\n\n\n".join(sections)

        # Step 4: Short LLM pass for glue/main section only
        if self.model:
            glue = self._generate_glue(plan, [s.name for s, _ in final_blocks],
                                        assembled_code=combined)
            if glue:
                combined += f"\n\n\n# -- main --\n{glue}"

        logger.info(f"FastAssembler: {len(combined)} chars, "
                     f"{len(unique_imports)} imports, {len(final_blocks)} sections")
        return combined

    def _dedup_methods(self, code: str) -> str:
        """Remove duplicate method definitions within a class."""
        import re
        lines = code.split("\n")
        result = []
        seen_methods = set()
        current_class = None
        skip_method = False
        skip_indent = 0

        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip()) if line.strip() else 999

            # Track class context
            if re.match(r'^class\s+\w+', stripped):
                current_class = stripped.split('(')[0].split(':')[0]
                seen_methods = set()
                skip_method = False

            # Track method definitions
            m = re.match(r'^def\s+(\w+)\(', stripped)
            if m and current_class and indent > 0:
                method_name = m.group(1)
                key = f"{current_class}.{method_name}"
                if key in seen_methods:
                    skip_method = True
                    skip_indent = indent
                    continue
                seen_methods.add(key)
                skip_method = False

            if skip_method:
                if indent > skip_indent or not stripped:
                    continue
                else:
                    skip_method = False

            result.append(line)
        return "\n".join(result)

    def _strip_if_main(self, code: str) -> str:
        """Remove if __name__ == '__main__' blocks from worker output."""
        lines = code.split("\n")
        result = []
        skip = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if __name__") and "__main__" in stripped:
                skip = True
                continue
            if skip:
                # Keep skipping indented lines under the if-main block
                if line and not line[0].isspace() and stripped:
                    skip = False
                    result.append(line)
                continue
            result.append(line)
        return "\n".join(result)

    def _extract_named(self, body: str, name: str) -> str:
        """Extract only the class or function definition matching `name`."""
        import re
        lines = body.split("\n")
        # Find the start of the matching class/def
        pattern = re.compile(rf'^(class\s+{re.escape(name)}\b|def\s+{re.escape(name)}\b)')
        start_idx = None
        for i, line in enumerate(lines):
            if pattern.match(line.strip()):
                start_idx = i
                break
        if start_idx is None:
            return ""
        # Find the end: next top-level class/def/comment-header or EOF
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped and not lines[i][0].isspace():
                # Top-level line — check if it's a new def/class
                if stripped.startswith("class ") or stripped.startswith("def "):
                    end_idx = i
                    break
                # Or an "Example usage" / standalone expression
                if stripped.startswith("#") and any(kw in stripped.lower()
                        for kw in ["example", "usage", "test"]):
                    end_idx = i
                    break
        return "\n".join(lines[start_idx:end_idx]).rstrip()

    def _prune_foreign_methods(self, body: str, subtask: Subtask,
                                siblings: list[Subtask]) -> str:
        """Remove methods from a class that semantically belong to sibling components.

        Uses the Architect's subtask descriptions to determine ownership:
        - Extract keyword stems from each subtask's description and name
        - If a method name matches a sibling's keywords but NOT the owner's, strip it
        - Always keep __init__, __repr__, __str__, and private methods
        """
        import re
        if not siblings:
            return body

        def extract_keywords(text: str) -> set[str]:
            """Extract lowercase word stems from description text."""
            words = re.findall(r'[a-z]+', text.lower())
            # Also split camelCase/PascalCase names
            for w in list(words):
                parts = re.findall(r'[a-z]+', re.sub(r'([A-Z])', r' \1', w).lower())
                words.extend(parts)
            # Stem: keep both full word and 5-char prefix for fuzzy matching
            stems = set()
            for w in words:
                if len(w) > 2:
                    stems.add(w)
                    if len(w) > 5:
                        stems.add(w[:5])
            return stems

        own_keywords = extract_keywords(f"{subtask.name} {subtask.description}")
        sibling_keywords: dict[str, set[str]] = {}
        for s in siblings:
            if s.id != subtask.id:
                sibling_keywords[s.name] = extract_keywords(f"{s.name} {s.description}")

        lines = body.split("\n")
        result = []
        skip_method = False
        skip_indent = 0

        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip()) if stripped else 999

            if skip_method:
                if indent > skip_indent or not stripped:
                    continue
                else:
                    skip_method = False

            # Check method definitions inside a class (indented def)
            m = re.match(r'^def\s+(\w+)\(', stripped)
            if m and indent > 0:
                method_name = m.group(1)
                # Always keep special methods and private methods
                if method_name.startswith('_'):
                    result.append(line)
                    continue
                # Check if this method name matches sibling keywords (with stems)
                raw_words = re.findall(r'[a-z]+', method_name.lower())
                method_stems = set()
                for w in raw_words:
                    if len(w) > 2:
                        method_stems.add(w)
                        if len(w) > 5:
                            method_stems.add(w[:5])
                matches_sibling = any(
                    method_stems & sib_kw for sib_kw in sibling_keywords.values()
                )
                matches_own = bool(method_stems & own_keywords)
                if matches_sibling and not matches_own:
                    skip_method = True
                    skip_indent = indent
                    continue

            result.append(line)
        return "\n".join(result)

    def _remove_foreign_defs(self, body: str, owner_name: str,
                              all_names: set[str]) -> str:
        """Remove class/def definitions that belong to other workers."""
        import re
        lines = body.split("\n")
        result = []
        skip_until_dedent = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if skip_until_dedent:
                if line and not line[0].isspace() and stripped:
                    skip_until_dedent = False
                else:
                    continue
            # Check if this line defines a class/function owned by another worker
            for name in all_names:
                if name == owner_name:
                    continue
                pat = re.compile(rf'^(class\s+{re.escape(name)}\b|def\s+{re.escape(name)}\b)')
                if pat.match(stripped):
                    skip_until_dedent = True
                    break
            if not skip_until_dedent:
                result.append(line)
        return "\n".join(result)

    def _split_imports(self, code: str) -> tuple[list[str], str]:
        """Split code into import lines and body."""
        imports = []
        body_lines = []
        in_body = False
        for line in code.split("\n"):
            stripped = line.strip()
            if not in_body and (stripped.startswith("import ") or
                                stripped.startswith("from ")):
                imports.append(stripped)
            else:
                if stripped and not stripped.startswith("#"):
                    in_body = True
                body_lines.append(line)
        # Strip leading blank lines from body
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        return imports, "\n".join(body_lines)

    def _dedup_imports(self, imports: list[str]) -> list[str]:
        """Deduplicate, validate, and sort imports. stdlib first, then third-party."""
        import sys
        stdlib_names = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()

        seen = set()
        stdlib = []
        thirdparty = []
        for imp in imports:
            if imp in seen:
                continue
            seen.add(imp)
            # Extract module name for sorting
            parts = imp.split()
            if len(parts) < 2:
                continue
            mod = parts[1] if parts[0] == "import" else parts[1]
            mod = mod.split(".")[0]
            if not mod:
                continue
            # Validate: check if module actually exists
            if mod not in stdlib_names:
                try:
                    __import__(mod)
                except (ImportError, ValueError):
                    logger.debug(f"Stripping unresolvable import: {imp}")
                    continue
            if mod in stdlib_names:
                stdlib.append(imp)
            else:
                thirdparty.append(imp)

        result = sorted(stdlib)
        if thirdparty:
            if result:
                result.append("")  # blank line between stdlib and third-party
            result.extend(sorted(thirdparty))
        return result

    def _extract_signatures(self, code: str) -> str:
        """Extract class/function signatures from assembled code for glue context."""
        import re
        sigs = []
        lines = code.split("\n")
        current_class = None
        for line in lines:
            stripped = line.strip()
            # Class definition
            m = re.match(r'^class\s+(\w+)', stripped)
            if m:
                current_class = m.group(1)
                sigs.append(f"\n{stripped}")
                continue
            # Method definition inside a class
            if current_class and re.match(r'^def\s+', stripped):
                # Extract just the signature, skip private/dunder except __init__
                m2 = re.match(r'^def\s+(\w+)\(([^)]*)\)', stripped)
                if m2:
                    name = m2.group(1)
                    if name.startswith('_') and name != '__init__':
                        continue
                    sigs.append(f"    {stripped.split(':')[0]}:")
                continue
            # Top-level function
            if not current_class and re.match(r'^def\s+', stripped):
                sigs.append(stripped.split(':')[0] + ":")
        return "\n".join(sigs)

    def _generate_glue(self, plan: ArchitectPlan, component_names: list[str],
                        assembled_code: str = "") -> str:
        """Short LLM pass to generate ONLY the main/demo wiring section."""
        # Extract actual signatures so the 3B knows real method names
        sig_block = self._extract_signatures(assembled_code) if assembled_code else ""
        user_msg = f"Task: {plan.original_prompt}\n"
        if sig_block:
            user_msg += f"Components and their methods:\n{sig_block}\n\n"
        else:
            user_msg += f"Components defined above: {', '.join(component_names)}\n"
        user_msg += "Write a main() that wires these together and demonstrates usage. Use ONLY the methods shown above."
        full_prompt = self._build_prompt(GLUE_SYSTEM, user_msg)

        start = time.monotonic()
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]
        output = self.model(
            full_prompt, max_tokens=self.max_tokens,
            temperature=self.temperature, stop=stop, echo=False,
        )
        elapsed = time.monotonic() - start

        text = output["choices"][0]["text"].strip()
        code = self._extract_code(text)
        logger.info(f"Glue generation: {len(code)} chars in {elapsed:.1f}s")
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

        lines = len(final_code.strip().split("\n")) if final_code else 0
        print(f"  [ASSEMBLER] Done: {lines} lines in {assemble_time:.1f}s")

        # Step 4: Validate — try to compile, fix if broken
        final_code = self._validate_and_fix(final_code, plan)

        total_time = time.monotonic() - total_start
        lines = len(final_code.strip().split("\n")) if final_code else 0
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

    def _fix_empty_bodies(self, code: str) -> str:
        """Deterministic fix: add 'pass' to empty def/class bodies.

        Common 0.5B issue: generates `def __init__(self):\\n    # comment`
        with no actual body, causing SyntaxError.
        """
        import re
        lines = code.split("\n")
        result = []
        for i, line in enumerate(lines):
            result.append(line)
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                if stripped.endswith(":"):
                    # Check if next non-blank, non-comment line is at same or lower indent
                    indent = len(line) - len(line.lstrip())
                    needs_pass = True
                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_stripped = lines[j].strip()
                        if not next_stripped or next_stripped.startswith("#"):
                            continue
                        next_indent = len(lines[j]) - len(lines[j].lstrip())
                        if next_indent > indent:
                            needs_pass = False
                        break
                    if needs_pass:
                        result.append(" " * (indent + 4) + "pass")
        return "\n".join(result)

    def _validate_and_fix(self, code: str, plan: ArchitectPlan,
                          max_attempts: int = 2) -> str:
        """Try to compile the assembled code; if it fails, do a targeted fix pass."""
        # First: deterministic fixes for common 0.5B issues
        code = self._fix_empty_bodies(code)

        for attempt in range(max_attempts):
            try:
                compile(code, "<assembled>", "exec")
                if attempt == 0:
                    print(f"  [VALIDATE] Syntax OK")
                else:
                    print(f"  [VALIDATE] Fixed after {attempt} attempt(s)")
                return code
            except SyntaxError as e:
                print(f"  [VALIDATE] Syntax error: {e.msg} (line {e.lineno})")
                if not self.architect_llm:
                    break
                code = self._fix_pass(code, e)

        # Final check — return whatever we have
        try:
            compile(code, "<assembled>", "exec")
        except SyntaxError:
            print(f"  [VALIDATE] Could not fix syntax errors after {max_attempts} attempts")
        return code

    def _fix_pass(self, code: str, error: SyntaxError) -> str:
        """Targeted fix: extract a window around the error, fix it, splice back."""
        lines = code.split("\n")
        err_line = (error.lineno or 1) - 1  # 0-indexed
        # Window: 5 lines before and after the error
        win_start = max(0, err_line - 5)
        win_end = min(len(lines), err_line + 6)
        snippet = "\n".join(lines[win_start:win_end])

        fix_system = (
            "Fix ONLY the syntax error in this Python code snippet. "
            "Output ONLY the fixed snippet in a ```python block. "
            "Do NOT add, remove, or change anything else."
        )
        fix_user = (
            f"Error: {error.msg} on line {error.lineno}\n\n"
            f"```python\n{snippet}\n```"
        )

        chat_format = getattr(self.architect, 'chat_format', 'chatml')
        if chat_format == "chatml":
            prompt = (f"<|im_start|>system\n{fix_system}<|im_end|>\n"
                      f"<|im_start|>user\n{fix_user}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
        else:
            prompt = f"{fix_system}\n\nUser: {fix_user}\nAssistant:\n"

        start = time.monotonic()
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]
        output = self.architect_llm(
            prompt, max_tokens=512,
            temperature=0.1, stop=stop, echo=False,
        )
        elapsed = time.monotonic() - start
        text = output["choices"][0]["text"].strip()

        # Extract fixed snippet
        if "```python" in text:
            fixed_snippet = text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            fixed_snippet = text.split("```")[1].split("```")[0].strip()
        else:
            fixed_snippet = text.strip()

        if not fixed_snippet:
            logger.info(f"Fix pass: empty response in {elapsed:.1f}s")
            print(f"  [VALIDATE] Fix pass: empty ({elapsed:.1f}s)")
            return code

        # Splice fixed snippet back into the full code
        fixed_lines = fixed_snippet.split("\n")
        result_lines = lines[:win_start] + fixed_lines + lines[win_end:]

        logger.info(f"Fix pass: {elapsed:.1f}s (lines {win_start+1}-{win_end})")
        print(f"  [VALIDATE] Fix pass: {elapsed:.1f}s (lines {win_start+1}-{win_end})")
        return "\n".join(result_lines)

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
                    futures = {pool.submit(self.worker.build, s, subtasks): s for s in ready}
                    for future in as_completed(futures):
                        result = future.result()
                        completed[result.subtask.id] = result
            else:
                for s in ready:
                    result = self.worker.build(s, subtasks)
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
