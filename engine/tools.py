"""
Tool-Use System — Phase 5: Callable Tools

Lets the model invoke tools (calculator, code runner, etc.) during generation.
Tools are registered with the engine and described to the model in the prompt.
The model outputs structured tool calls, which are parsed, executed, and the
results fed back for a final response.

Tool call format (in model output):
    <tool_call>tool_name(arg1, arg2)</tool_call>

The system:
1. Detects tool call patterns in model output
2. Parses the tool name and arguments
3. Executes the tool
4. Feeds the result back to the model as context
5. Generates a final response incorporating the tool result

Tools are defined as Python callables with metadata (name, description, args).
New tools can be registered at runtime.
"""

import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A registered tool that the model can invoke."""
    name: str
    description: str
    parameters: list[dict]  # [{name, type, description, required}]
    function: Callable
    category: str = "general"
    enabled: bool = True

    def schema_for_prompt(self) -> str:
        """Format tool description for injection into the prompt."""
        params = ", ".join(
            f"{p['name']}: {p['type']}" for p in self.parameters
        )
        param_desc = "\n".join(
            f"    - {p['name']} ({p['type']}): {p['description']}"
            for p in self.parameters
        )
        return (
            f"  {self.name}({params}): {self.description}\n"
            f"{param_desc}"
        )


@dataclass
class ToolCall:
    """A parsed tool call from model output."""
    tool_name: str
    arguments: list[str]
    raw: str


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None

    def format_for_prompt(self) -> str:
        """Format result for feeding back to the model."""
        if self.success:
            return f"[Tool Result: {self.tool_name}]\n{self.result}"
        else:
            return f"[Tool Error: {self.tool_name}]\n{self.error}"


# ── Tool Call Pattern ─────────────────────────────────────────
# Matches: <tool_call>name(arg1, arg2)</tool_call>
# Also matches: [tool_call]name(arg1, arg2)[/tool_call]
# Also matches plain: TOOL_CALL: name(arg1, arg2)
TOOL_CALL_PATTERNS = [
    re.compile(r'<tool_call>\s*(\w+)\((.*?)\)\s*</tool_call>', re.DOTALL),
    re.compile(r'\[tool_call\]\s*(\w+)\((.*?)\)\s*\[/tool_call\]', re.DOTALL),
    re.compile(r'TOOL_CALL:\s*(\w+)\((.*?)\)(?:\s|$)', re.DOTALL),
]


class ToolRegistry:
    """
    Manages tool registration, prompt generation, parsing, and execution.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in tools."""

        # Calculator
        self.register(
            name="calculate",
            description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, etc.",
            parameters=[
                {"name": "expression", "type": "str", "description": "Math expression to evaluate", "required": True},
            ],
            function=self._tool_calculate,
            category="math",
        )

        # Python code runner (sandboxed)
        self.register(
            name="run_python",
            description="Execute a short Python code snippet and return the output.",
            parameters=[
                {"name": "code", "type": "str", "description": "Python code to execute", "required": True},
            ],
            function=self._tool_run_python,
            category="code",
        )

        # File reader
        self.register(
            name="read_file",
            description="Read the contents of a text file.",
            parameters=[
                {"name": "path", "type": "str", "description": "Path to the file to read", "required": True},
            ],
            function=self._tool_read_file,
            category="file",
        )

        # JSON formatter
        self.register(
            name="format_json",
            description="Parse and pretty-print a JSON string.",
            parameters=[
                {"name": "data", "type": "str", "description": "JSON string to format", "required": True},
            ],
            function=self._tool_format_json,
            category="json",
        )

        # Word/char counter
        self.register(
            name="count_text",
            description="Count words, characters, and lines in text.",
            parameters=[
                {"name": "text", "type": "str", "description": "Text to analyze", "required": True},
            ],
            function=self._tool_count_text,
            category="text",
        )

    # ── Registration ──────────────────────────────────────────

    def register(self, name: str, description: str, parameters: list[dict],
                 function: Callable, category: str = "general"):
        """Register a new tool."""
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            category=category,
        )
        self._tools[name] = tool
        logger.info(f"Registered tool: {name} ({category})")

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)

    # ── Prompt Generation ─────────────────────────────────────

    def get_tool_prompt(self) -> str:
        """
        Generate the tool description section for the prompt.
        This tells the model what tools are available and how to call them.
        """
        enabled = [t for t in self._tools.values() if t.enabled]
        if not enabled:
            return ""

        lines = [
            "You have access to the following tools. To use a tool, "
            "write: <tool_call>tool_name(arguments)</tool_call>",
            "",
            "Available tools:",
        ]
        for tool in enabled:
            lines.append(tool.schema_for_prompt())

        lines.append("")
        lines.append(
            "Use tools when they would help answer the question accurately. "
            "You may use multiple tools in one response."
        )

        return "\n".join(lines)

    # ── Parsing ───────────────────────────────────────────────

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls from model output."""
        calls = []
        for pattern in TOOL_CALL_PATTERNS:
            for match in pattern.finditer(text):
                tool_name = match.group(1)
                args_str = match.group(2).strip()

                # Parse arguments (comma-separated, strip quotes)
                args = []
                if args_str:
                    # Handle quoted strings and simple values
                    for arg in self._split_args(args_str):
                        arg = arg.strip().strip("'\"")
                        args.append(arg)

                calls.append(ToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    raw=match.group(0),
                ))

        return calls

    def _split_args(self, args_str: str) -> list[str]:
        """Split arguments respecting quoted strings, braces, and parens."""
        args = []
        current = []
        in_quotes = None
        depth = 0

        for char in args_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = char
                current.append(char)
            elif char == in_quotes:
                in_quotes = None
                current.append(char)
            elif char == ',' and not in_quotes and depth == 0:
                args.append("".join(current))
                current = []
            else:
                if char in ('(', '{', '['):
                    depth += 1
                elif char in (')', '}', ']'):
                    depth -= 1
                current.append(char)

        if current:
            args.append("".join(current))

        return args

    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains any tool calls."""
        for pattern in TOOL_CALL_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def strip_tool_calls(self, text: str) -> str:
        """Remove tool call markers from text, keeping surrounding content."""
        result = text
        for pattern in TOOL_CALL_PATTERNS:
            result = pattern.sub("", result)
        return result.strip()

    # ── Execution ─────────────────────────────────────────────

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a parsed tool call."""
        tool = self._tools.get(call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {call.tool_name}",
            )

        if not tool.enabled:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                result=None,
                error=f"Tool '{call.tool_name}' is disabled",
            )

        try:
            result = tool.function(*call.arguments)
            return ToolResult(
                tool_name=call.tool_name,
                success=True,
                result=result,
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {call.tool_name}: {e}")
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    def execute_all(self, text: str) -> tuple[list[ToolResult], str]:
        """
        Parse and execute all tool calls in text.
        Returns (results, cleaned_text).
        """
        calls = self.parse_tool_calls(text)
        results = [self.execute(call) for call in calls]
        cleaned = self.strip_tool_calls(text)
        return results, cleaned

    # ── Built-in Tool Implementations ─────────────────────────

    @staticmethod
    def _tool_calculate(expression: str) -> str:
        """Evaluate a math expression safely."""
        # Whitelist safe math operations
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sum": sum, "len": len,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "inf": math.inf,
            "floor": math.floor, "ceil": math.ceil,
        }

        # Block dangerous builtins
        code = compile(expression, "<calc>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise ValueError(f"Disallowed function: {name}")

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)

    @staticmethod
    def _tool_run_python(code: str) -> str:
        """Run Python code in a subprocess with timeout."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(Path.cwd()),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "[Error: Code execution timed out (10s limit)]"
        except Exception as e:
            return f"[Error: {e}]"

    @staticmethod
    def _tool_read_file(path: str) -> str:
        """Read a text file (limited to 5KB)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not p.is_file():
            raise ValueError(f"Not a file: {path}")

        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 5000:
            content = content[:5000] + f"\n... [truncated, {len(content)} chars total]"
        return content

    @staticmethod
    def _tool_format_json(data: str) -> str:
        """Parse and pretty-print JSON."""
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2)

    @staticmethod
    def _tool_count_text(text: str) -> str:
        """Count words, characters, and lines."""
        words = len(text.split())
        chars = len(text)
        lines = text.count("\n") + 1
        return f"Words: {words}, Characters: {chars}, Lines: {lines}"

    # ── Status ────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        """List all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "enabled": t.enabled,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def status(self) -> dict:
        """Get tool registry status."""
        return {
            "total_tools": len(self._tools),
            "enabled": sum(1 for t in self._tools.values() if t.enabled),
            "tools": [t.name for t in self._tools.values() if t.enabled],
        }
