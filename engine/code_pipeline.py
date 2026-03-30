"""
Code Quality Pipeline — Generate → Execute → Self-Repair

Instead of few-shot augmentors (which hurt execution accuracy),
this pipeline uses actual code execution as the quality gate:

1. Generate code from prompt
2. Extract Python code from response
3. Run quick sanity tests (if provided) or syntax check
4. If fails → feed error back to model → retry (max 2x)
5. Return best result

Also supports multi-model mode:
- Model A generates the code
- If it fails, Model B (a debugger) gets the error + code and fixes it
- Allows pairing a fast generator with a cheap debugger

This replaces the augmentor system for code tasks.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class CodeResult:
    """Result from the code pipeline."""
    code: str
    response: str  # Full model response
    passed: bool
    error: str = ""
    attempts: int = 1
    models_used: list[str] = field(default_factory=list)
    total_time: float = 0.0
    test_results: Optional[dict] = None


def extract_code(response: str) -> str:
    """Extract Python code from a model response."""
    # Try markdown code blocks first
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        return "\n".join(blocks)

    # Try to find function/class definitions
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            in_code = True
        if in_code:
            if stripped and not stripped.startswith(("#", " ", "\t", "def ", "class ",
                "import ", "from ", "return", "if ", "else", "elif", "for ", "while ",
                "try:", "except", "finally", "with ", "raise", "yield", "pass",
                "break", "continue", "assert", "@", ")", "]", "}", "'''", '"""')):
                if not any(c in stripped for c in ["=", "(", "[", "{"]):
                    break
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)
    return response


def check_syntax(code: str) -> tuple[bool, str]:
    """Check if code has valid Python syntax."""
    try:
        compile(code, "<generated>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"


def execute_code(code: str, test_code: str = "") -> tuple[bool, str, dict]:
    """
    Execute code and optionally run test assertions.
    Returns (passed, error_message, details).
    """
    namespace = {"__builtins__": __builtins__}

    # Execute the generated code
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Execution error: {type(e).__name__}: {e}", {"stage": "exec"}

    if not test_code.strip():
        return True, "", {"stage": "exec_only"}

    # Run test assertions
    tests = [t.strip() for t in test_code.strip().split("\n")
             if t.strip() and not t.strip().startswith("#")]

    passed = 0
    total = len(tests)
    errors = []

    for test in tests:
        try:
            exec(test, namespace)
            passed += 1
        except AssertionError as e:
            errors.append(f"FAIL: {test}")
        except Exception as e:
            errors.append(f"ERROR: {test} -> {type(e).__name__}: {e}")

    all_passed = passed == total
    error_msg = "; ".join(errors[:3]) if errors else ""

    return all_passed, error_msg, {
        "stage": "tests",
        "passed": passed,
        "total": total,
        "errors": errors,
    }


def build_generate_prompt(task: str, chat_format: str) -> str:
    """Build a prompt for code generation."""
    system = (
        "Write a Python function as requested. Return ONLY the code in a ```python block. "
        "No explanation needed. The function must be complete and runnable."
    )
    return _wrap_chat(system, task, chat_format)


def build_repair_prompt(task: str, code: str, error: str, chat_format: str) -> str:
    """Build a prompt for code repair given an error."""
    system = (
        "You are a code debugger. Fix the broken code. Return ONLY the corrected code "
        "in a ```python block. No explanation needed."
    )
    user = (
        f"Original task: {task}\n\n"
        f"Code that failed:\n```python\n{code}\n```\n\n"
        f"Error: {error}\n\n"
        f"Fix the code."
    )
    return _wrap_chat(system, user, chat_format)


def _wrap_chat(system: str, user: str, fmt: str) -> str:
    if fmt == "chatml":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    elif fmt == "llama3":
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif fmt == "phi3":
        return (f"<|system|>\n{system}<|end|>\n"
                f"<|user|>\n{user}<|end|>\n"
                f"<|assistant|>\n")
    elif fmt == "alpaca":
        return (f"### System:\n{system}\n\n"
                f"### Instruction:\n{user}\n\n"
                f"### Response:\n")
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


class CodePipeline:
    """
    Generate → Execute → Self-Repair pipeline.

    Modes:
    - single: One model generates and repairs its own code
    - multi: Model A generates, Model B debugs failures
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def run_single(
        self,
        task: str,
        model,
        chat_format: str,
        test_code: str = "",
        gen_kwargs: dict = None,
    ) -> CodeResult:
        """Single-model pipeline: generate, test, self-repair."""
        kwargs = gen_kwargs or {}
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]
        start_time = time.monotonic()
        models_used = [getattr(model, '_model_name', 'unknown')]

        # Step 1: Generate
        prompt = build_generate_prompt(task, chat_format)
        output = model(prompt, stop=stop, echo=False, **kwargs)
        response = output["choices"][0]["text"].strip()

        code = extract_code(response)
        attempts = 1

        # Step 2: Check syntax
        syntax_ok, syntax_err = check_syntax(code)
        if not syntax_ok:
            # Retry with syntax error feedback
            for retry in range(self.max_retries):
                repair_prompt = build_repair_prompt(task, code, syntax_err, chat_format)
                output = model(repair_prompt, stop=stop, echo=False, **kwargs)
                response = output["choices"][0]["text"].strip()
                code = extract_code(response)
                attempts += 1
                syntax_ok, syntax_err = check_syntax(code)
                if syntax_ok:
                    break

            if not syntax_ok:
                return CodeResult(
                    code=code, response=response, passed=False,
                    error=syntax_err, attempts=attempts,
                    models_used=models_used,
                    total_time=time.monotonic() - start_time,
                )

        # Step 3: Execute and test
        passed, error, details = execute_code(code, test_code)

        if passed:
            return CodeResult(
                code=code, response=response, passed=True,
                attempts=attempts, models_used=models_used,
                total_time=time.monotonic() - start_time,
                test_results=details,
            )

        # Step 4: Retry with execution error feedback
        for retry in range(self.max_retries):
            repair_prompt = build_repair_prompt(task, code, error, chat_format)
            output = model(repair_prompt, stop=stop, echo=False, **kwargs)
            response = output["choices"][0]["text"].strip()
            code = extract_code(response)
            attempts += 1

            syntax_ok, syntax_err = check_syntax(code)
            if not syntax_ok:
                error = syntax_err
                continue

            passed, error, details = execute_code(code, test_code)
            if passed:
                break

        return CodeResult(
            code=code, response=response, passed=passed,
            error=error if not passed else "",
            attempts=attempts, models_used=models_used,
            total_time=time.monotonic() - start_time,
            test_results=details,
        )

    def run_multi(
        self,
        task: str,
        generator_model,
        generator_format: str,
        debugger_model,
        debugger_format: str,
        test_code: str = "",
        gen_kwargs: dict = None,
        debug_kwargs: dict = None,
    ) -> CodeResult:
        """
        Multi-model pipeline:
        - generator_model writes the initial code
        - debugger_model fixes failures
        """
        gkw = gen_kwargs or {}
        dkw = debug_kwargs or {}
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]
        start_time = time.monotonic()
        gen_name = getattr(generator_model, '_model_name', 'generator')
        dbg_name = getattr(debugger_model, '_model_name', 'debugger')
        models_used = [gen_name]

        # Step 1: Generate with Model A
        prompt = build_generate_prompt(task, generator_format)
        output = generator_model(prompt, stop=stop, echo=False, **gkw)
        response = output["choices"][0]["text"].strip()
        code = extract_code(response)
        attempts = 1

        # Step 2: Check + Execute
        syntax_ok, syntax_err = check_syntax(code)
        if syntax_ok:
            passed, error, details = execute_code(code, test_code)
        else:
            passed = False
            error = syntax_err
            details = {"stage": "syntax"}

        if passed:
            return CodeResult(
                code=code, response=response, passed=True,
                attempts=attempts, models_used=models_used,
                total_time=time.monotonic() - start_time,
                test_results=details,
            )

        # Step 3: Hand off to Model B (debugger) for repair
        models_used.append(dbg_name)
        for retry in range(self.max_retries):
            repair_prompt = build_repair_prompt(task, code, error, debugger_format)
            output = debugger_model(repair_prompt, stop=stop, echo=False, **dkw)
            response = output["choices"][0]["text"].strip()
            code = extract_code(response)
            attempts += 1

            syntax_ok, syntax_err = check_syntax(code)
            if not syntax_ok:
                error = syntax_err
                continue

            passed, error, details = execute_code(code, test_code)
            if passed:
                break

        return CodeResult(
            code=code, response=response, passed=passed,
            error=error if not passed else "",
            attempts=attempts, models_used=models_used,
            total_time=time.monotonic() - start_time,
            test_results=details,
        )
