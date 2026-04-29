#!/usr/bin/env python
"""
ulcagent — Adaptive local coding & system agent.

Usage:
    cd into any project directory, then:
        ulcagent                # interactive REPL (auto-detects profile)
        ulcagent "fix the bug"  # one-shot
        ulcagent --warm         # keep model loaded between goals

Profiles:
    code    — Qwen 2.5 Coder 14B (precise code edits, tests, refactoring)
    general — Qwen 2.5 14B Instruct (exploration, system tasks, Q&A)

Auto-detects which profile to use from your goal. Override with /code or /general.
Zero servers, zero network, 100% local.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path

# Bootstrap paths
_SELF = Path(__file__).resolve().parent
sys.path.insert(0, str(_SELF))
_CORE = _SELF.parent / "densanon-core"
if _CORE.exists() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

# UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress noisy logging
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
for _n in ("engine", "densanon", "llama_cpp"):
    logging.getLogger(_n).setLevel(logging.WARNING)


# ── Profiles ─────────────────────────────────────────────────────

PROFILES = {
    "code": {
        "config": str(_SELF / "config_agent14b.yaml"),
        "label": "Qwen Coder 14B",
        "hint": (
            "You are a precise coding agent. Execute the task using tools, "
            "then give a concise final answer (2-3 sentences max). "
            "Do not repeat what the tools already showed."
        ),
    },
    "general": {
        "config": str(_SELF / "config_agent14b_general.yaml"),
        "label": "Qwen Instruct 14B",
        "hint": (
            "You are a helpful local assistant with full access to the user's "
            "files and system. Use tools to answer questions, find information, "
            "and perform tasks. Be conversational but concise."
        ),
    },
}

# Keywords that signal each profile
_CODE_PATTERNS = re.compile(
    r"\b(fix|bug|error|refactor|add function|add method|add class|implement|"
    r"write test|run test|pytest|unittest|import|syntax|compile|build|"
    r"edit_file|write_file|def |class |return |raise |except |"
    r"\.py\b|\.js\b|\.ts\b|\.go\b|\.rs\b|endpoint|api|handler|middleware|"
    r"rename.*function|add.*flag|add.*parameter|type hint|docstring|decorator|"
    r"dataclass|argparse|fastapi|flask|django)\b",
    re.IGNORECASE,
)

_GENERAL_PATTERNS = re.compile(
    r"\b(what is|what are|what files|what project|tell me|describe|explain|"
    r"summarize|overview|how does|why does|list.*files|find.*files|search for|"
    r"show me|disk|space|process|running|memory|system|installed|clean up|"
    r"delete|remove|move|copy|rename files|organize|"
    r"read.*and tell|read.*and describe|read.*and summarize)\b",
    re.IGNORECASE,
)


def _detect_profile(goal: str) -> str:
    code_score = len(_CODE_PATTERNS.findall(goal))
    general_score = len(_GENERAL_PATTERNS.findall(goal))
    # Code wins ties — it's the more precise tool
    return "code" if code_score >= general_score else "general"


# ── Colors ───────────────────────────────────────────────────────

_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        _USE_COLOR = True
    except Exception:
        pass


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _dim(t): return _c("2", t)
def _cyan(t): return _c("36", t)
def _green(t): return _c("32", t)
def _red(t): return _c("31", t)
def _yellow(t): return _c("33", t)
def _bold(t): return _c("1", t)
def _magenta(t): return _c("35", t)


# ── Spinner ──────────────────────────────────────────────────────

class _Spinner:
    """Animated progress spinner. No-op when stdout isn't a TTY — without
    this guard, headless runs (one-shot piped to a log file) accumulate
    kilobytes of `[2m| thinking...[0m` ANSI escapes per second. Surfaced
    in the 2026-04-26 handheld walkthrough."""
    _FRAMES = ["|", "/", "-", "\\"]
    def __init__(self):
        self._active = False
        self._thread = None
        self._stop = threading.Event()
        self._enabled = sys.stdout.isatty()
    def start(self):
        if not self._enabled:
            return
        self._stop.clear()
        self._active = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    def stop(self):
        if not self._enabled:
            return
        if not self._active:
            return
        self._stop.set()
        self._active = False
        if self._thread:
            self._thread.join(timeout=1)
        print("\r" + " " * 30 + "\r", end="", flush=True)
    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            print(f"\r  {_dim(frame + ' thinking...')}", end="", flush=True)
            i += 1
            self._stop.wait(0.15)

_spinner = _Spinner()


# ── Model manager ────────────────────────────────────────────────

class ModelManager:
    """Manages loading/unloading models by profile. Only one loaded at a time."""

    def __init__(self):
        self._bm = None
        self._config = None
        self._profile = None
        self._override_path = None

    @property
    def profile(self):
        return self._profile

    @property
    def loaded(self):
        return self._bm is not None and self._bm.model is not None

    def ensure_profile(self, profile: str, quiet: bool = False):
        """Load the requested profile's model. Swaps if different."""
        # Check for /model override
        override = PROFILES[profile].get("_override_path")
        if override and self._override_path == override and self.loaded:
            return
        if not override and self._profile == profile and self.loaded:
            return

        if self.loaded:
            if not quiet:
                label = PROFILES[profile]["label"]
                print(f"  {_dim(f'Swapping to {label}...')}", end=" ", flush=True)
            self._bm.unload()
            self._bm = None
            self._config = None
        else:
            if not quiet:
                label = PROFILES[profile]["label"]
                print(f"  {_dim(f'Loading {label}...')}", end=" ", flush=True)

        try:
            from densanon.core.config import Config
            self._config = Config(PROFILES[profile]["config"])
        except ImportError:
            from engine._config_shim import load_config
            self._config = load_config(PROFILES[profile]["config"])
        from engine.base_model import BaseModel
        # Apply model path override if /model was used
        if override:
            self._config.base_model.path = override
            self._override_path = override
        else:
            self._override_path = None
        self._bm = BaseModel(self._config.base_model)
        t0 = time.monotonic()
        self._bm.load()
        self._profile = profile
        if not quiet:
            print(f"{_green('ready')} {_dim(f'({time.monotonic() - t0:.1f}s)')}")

    def unload(self):
        if self._bm is not None:
            self._bm.unload()
            self._bm = None
        self._profile = None

    @property
    def bm(self):
        return self._bm

    @property
    def config(self):
        return self._config


# ── Agent builder ────────────────────────────────────────────────


def _parse_mcp_arg(argv: list[str]) -> list[str]:
    """Pull `--mcp <comma-separated-names>` out of argv.

    Supports both `--mcp foo,bar` (two args) and `--mcp=foo,bar` (one arg)
    forms. Returns an empty list when --mcp is absent (the default).

    Built-in shortcuts the value can use are listed in
    `engine.mcp_adapter._BUILTIN_SERVERS`. The `register_mcp_tools`
    call validates them and raises a clear ValueError on unknowns.
    """
    for i, a in enumerate(argv):
        if a == "--mcp" and i + 1 < len(argv):
            return [s.strip() for s in argv[i + 1].split(",") if s.strip()]
        if a.startswith("--mcp="):
            return [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
    return []


def _build_agent(mgr: ModelManager, workspace: Path):
    from engine.agent import Agent, AgentEvent
    from engine.agent_builtins import build_default_registry
    from engine.agent_memory import AgentMemory

    profile = mgr.profile
    memory = AgentMemory(workspace=workspace)
    extended = "--extended" in sys.argv
    lsp = "--lsp" in sys.argv
    # `--mcp <name1,name2>` opt-in MCP-server mounting. Scaffolded today,
    # raises NotImplementedError if used (see engine/mcp_adapter.py).
    # Default = no MCP, identical behavior to before this scaffold landed.
    mcp_servers = _parse_mcp_arg(sys.argv)
    registry = build_default_registry(
        workspace, memory=memory,
        ask_user_fn=_ask_user,
        extended_tools=extended,
        lsp_tools=lsp,
        mcp_servers=mcp_servers,
    )

    # Add system tools for the general profile
    if profile == "general":
        _register_system_tools(registry)

    # Load plugins
    plugin_count = _load_plugins(registry)

    # Load model-specific prompt profile. mgr.bm.config is a BaseModelConfig
    # (the per-model section, not the top-level Config), so .path is the
    # GGUF path directly. Earlier code wrote `.base_model.path` which crashed
    # on AttributeError when the agent path was first exercised in one-shot.
    model_profile = _load_model_profile(getattr(mgr.bm.config, "path", "") if hasattr(mgr.bm, 'config') else "")

    # Auto-approve risky tools when explicitly requested (--yes) or when
    # stdin isn't a TTY (one-shot pipes, automation, headless runs). The
    # walkthrough's Goal 1 hit this: every run_bash silently denied
    # because the input() prompt got EOFError → defaulted to deny.
    auto_yes = "--yes" in sys.argv or not sys.stdin.isatty()

    def _confirm_risky(call):
        if auto_yes:
            args_s = ', '.join(f'{k}={str(v)[:60]}' for k, v in call.arguments.items())
            print(f"  {_dim('[auto-approved risky]')} {call.name}({args_s})")
            return True
        _spinner.stop()
        args_s = ', '.join(f'{k}={str(v)[:60]}' for k, v in call.arguments.items())
        print(f"\n  {_yellow('[risky]')} {call.name}({args_s})")
        try:
            answer = input(f"  {_yellow('Approve? [y/N]')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        return answer in ("y", "yes")

    def _on_event(e: AgentEvent):
        if e.type == "iteration":
            _spinner.stop()
            print(f"\n  {_dim(f'[{e.iteration}]')}", end="", flush=True)
            _spinner.start()
        elif e.type == "tool_call":
            _spinner.stop()
            args = ", ".join(f"{k}={str(v)[:50]}" for k, v in e.payload.arguments.items())
            print(f"\n    {_cyan('->')} {_cyan(e.payload.name)}({args[:80]})")
        elif e.type == "tool_result":
            _spinner.stop()
            r = e.payload
            if r.success:
                preview = str(r.content)[:120].replace("\n", " ")
                print(f"       {_dim(preview)}")
            else:
                print(f"       {_red('err:')} {r.error[:120]}")
        elif e.type == "pre_finish_retry":
            _spinner.stop()
            p = e.payload or {}
            print(f"\n  {_yellow('[retry]')} {p.get('feedback', '')[:100]}")
        elif e.type == "compacted":
            _spinner.stop()
            p = e.payload or {}
            before = p.get('total_before', 0)
            after = p.get('total_after', 0)
            print(f"\n  {_dim(f'[compacted] {before} -> {after} chars')}")
        elif e.type == "final":
            _spinner.stop()

    bm_cfg = mgr.config.base_model
    cfg_temp = getattr(bm_cfg, "temperature", None)
    cfg_max = getattr(bm_cfg, "max_tokens", None)

    workspace_hint = (
        f"Workspace: {workspace}\n"
        f"{PROFILES[profile]['hint']}"
    )

    agent = Agent(
        model=mgr.bm,
        registry=registry,
        system_prompt_extra=workspace_hint,
        workspace_root=workspace,
        memory=memory,
        auto_verify_python=True,
        max_iterations=20,
        max_wall_time=600.0,
        max_tokens_per_turn=int(cfg_max) if cfg_max else 1024,
        temperature=cfg_temp if cfg_temp is not None else 0.1,
        confirm_risky=_confirm_risky,
        on_event=_on_event,
    )
    # Stash the build inputs on the agent so we can spawn a parallel
    # ArchitectAgent against the same registry/model/memory if needed.
    agent._ulcagent_ctx = {
        "model": mgr.bm,
        "registry": registry,
        "system_prompt_extra": workspace_hint,
        "workspace_root": workspace,
        "memory": memory,
        "max_tokens_per_turn": int(cfg_max) if cfg_max else 1024,
        "temperature": cfg_temp if cfg_temp is not None else 0.1,
        "confirm_risky": _confirm_risky,
        "on_event": _on_event,
    }
    return agent


# ── Architect mode (B6) ──────────────────────────────────────────

# Heuristic patterns that signal the goal is multi-file scaffolding —
# the failure mode the flat Agent flaps on (build_todo_cli @ 14B).
# Auto-architect kicks in when one of these matches and the user hasn't
# explicitly disabled it with --no-architect.
_ARCH_TRIGGER_PHRASES = (
    "build a", "build the", "scaffold", "create a project", "create a new",
    "implement a", "set up a", "set up the",
    "multi-file", "multiple files",
    "todo cli", "todo app", "rest api", "fastapi app", "flask app",
    "with these files", "with the following files",
)

# A list of >=3 distinct filename-like tokens in one sentence is a strong
# scaffold signal (e.g. "todo.py, storage.py, cli.py, tests/test_todo.py").
_FILE_TOKEN_RE = re.compile(
    r"\b[\w/\\.-]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cs|rb|sh|html|yaml|yml|json|toml|md)\b",
    re.IGNORECASE,
)

_ARCH_NEGATION = re.compile(r"\b(?:no|not|don't|dont|skip|avoid|don't use)\s+architect\b", re.IGNORECASE)

# Goals containing these patterns are FIX/DEBUG/REFACTOR shapes — they
# require cross-file consistency, which architect's per-step isolation
# explicitly destroys. Goal 1.5 of the 2026-04-26 handheld walkthrough
# proved this catastrophically: architect rewrote storage.py to a single
# import line and duplicated cli.py handlers because each sub-agent had
# no shared context. Suppress architect even with 3+ file tokens when
# any of these phrases match.
_ARCH_SUPPRESS_PHRASES = (
    "fix the bug", "fix the bugs", "fix three bugs", "fix two bugs",
    "fix this", "fix that", "fix it", "fix:", "fix —", "fix -",
    "broken", "is failing", "are failing", "doesn't work", "does not work",
    "rename ", "refactor ", "debug ", "repair ",
    "remove the duplicate", "remove duplicate",
    "shadows", "shadow the builtin",
    "cross-file", "cross-module",
    "inconsistent", "out of sync", "out-of-sync",
)


def _should_use_architect(goal: str, force_on: bool = False, force_off: bool = False) -> bool:
    """Decide whether to route this goal through ArchitectAgent.

    Force flags from the CLI (`--architect` / `--no-architect`) win. Otherwise
    return True iff the goal looks like multi-file SCAFFOLDING (not fix/debug):
      - matches one of the scaffold-trigger phrases, OR
      - mentions 3+ distinct filename-like tokens AND no fix/debug verbs.

    Goals with fix/debug/rename/refactor language ALWAYS suppress architect,
    regardless of file-token count (per the 2026-04-26 walkthrough finding).
    """
    if force_off:
        return False
    if force_on:
        return True
    if _ARCH_NEGATION.search(goal):
        return False

    g = goal.lower()

    # Suppress architect for fix/debug/refactor goals — these need cross-file
    # consistency which per-step isolation destroys.
    if any(p in g for p in _ARCH_SUPPRESS_PHRASES):
        return False

    if any(p in g for p in _ARCH_TRIGGER_PHRASES):
        return True

    tokens = {m.group(0).lower() for m in _FILE_TOKEN_RE.finditer(goal)}
    if len(tokens) >= 3:
        return True

    return False


def _build_architect_agent(flat_agent):
    """Spawn an ArchitectAgent that shares model + registry + memory + workspace
    with the existing flat Agent. Cheap — no new model load.
    """
    from engine.architect_agent import ArchitectAgent
    ctx = getattr(flat_agent, "_ulcagent_ctx", None)
    if ctx is None:
        raise RuntimeError("flat agent has no _ulcagent_ctx; rebuild via _build_agent first")
    return ArchitectAgent(
        model=ctx["model"],
        registry=ctx["registry"],
        system_prompt_extra=ctx["system_prompt_extra"],
        workspace_root=ctx["workspace_root"],
        memory=ctx["memory"],
        auto_verify_python=True,
        max_iterations_per_step=6,
        max_wall_time=600.0,
        max_tokens_per_turn=ctx["max_tokens_per_turn"],
        temperature=ctx["temperature"],
        confirm_risky=ctx["confirm_risky"],
        on_event=ctx["on_event"],
    )


def _build_handheld_driver(flat_agent):
    """Spawn a HandheldDriver that shares model + registry + memory +
    workspace with the existing flat Agent. Different from architect:
    each step's sub-agent gets prior-step DELIVERABLE summaries injected
    into its system prompt — true cross-step context, not just isolated
    workspace sharing. Use for projects too large for the flat agent."""
    from engine.handheld_driver import HandheldDriver
    ctx = getattr(flat_agent, "_ulcagent_ctx", None)
    if ctx is None:
        raise RuntimeError("flat agent has no _ulcagent_ctx; rebuild via _build_agent first")
    return HandheldDriver(
        model=ctx["model"],
        registry=ctx["registry"],
        system_prompt_extra=ctx["system_prompt_extra"],
        workspace_root=ctx["workspace_root"],
        memory=ctx["memory"],
        auto_verify_python=True,
        max_iterations_per_step=8,
        max_wall_time=1800.0,  # 30 min — large projects need it
        max_tokens_per_turn=ctx["max_tokens_per_turn"],
        temperature=ctx["temperature"],
        confirm_risky=ctx["confirm_risky"],
        on_event=ctx["on_event"],
    )


def _register_system_tools(registry):
    """Add system-awareness tools for the general profile."""
    from engine.agent_tools import ToolSchema
    import subprocess

    def _disk_usage(path="."):
        import shutil
        total, used, free = shutil.disk_usage(Path(path).resolve())
        gb = lambda b: f"{b / (1024**3):.1f} GB"
        return f"Disk: {gb(total)} total, {gb(used)} used, {gb(free)} free ({used/total*100:.0f}%)"

    def _processes(filter_name=""):
        result = subprocess.run(
            ["tasklist"] if os.name == "nt" else ["ps", "aux"],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.splitlines()
        if filter_name:
            lines = [l for l in lines if filter_name.lower() in l.lower()]
        if len(lines) > 30:
            lines = lines[:30] + [f"... ({len(lines) - 30} more)"]
        return "\n".join(lines) if lines else f"No processes matching '{filter_name}'"

    def _env_var(name=""):
        if name:
            val = os.environ.get(name)
            return f"{name}={val}" if val else f"{name} is not set"
        # List interesting ones
        show = ["PATH", "PYTHON", "HOME", "USERPROFILE", "CUDA_PATH", "NODE_PATH", "GOPATH"]
        lines = []
        for k in sorted(os.environ):
            if any(s in k.upper() for s in show) or k in show:
                lines.append(f"{k}={os.environ[k][:100]}")
        return "\n".join(lines[:20]) if lines else "No matching env vars"

    def _recent_files(path=".", count=20):
        root = Path(path).resolve()
        files = []
        for p in root.rglob("*"):
            if p.is_file() and ".git" not in p.parts:
                files.append((p.stat().st_mtime, p))
        files.sort(reverse=True)
        lines = []
        for mtime, p in files[:int(count)]:
            rel = p.relative_to(root)
            t = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            lines.append(f"  {t}  {rel}")
        return "\n".join(lines) if lines else "No files found"

    for name, desc, params, fn in [
        ("disk_usage", "Show disk space usage for a path.",
         {"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
         lambda path=".": _disk_usage(path)),
        ("processes", "List running processes. Optional filter by name.",
         {"type": "object", "properties": {"filter_name": {"type": "string", "default": ""}}},
         lambda filter_name="": _processes(filter_name)),
        ("env_var", "Get environment variable(s). Empty name lists interesting vars.",
         {"type": "object", "properties": {"name": {"type": "string", "default": ""}}},
         lambda name="": _env_var(name)),
        ("recent_files", "List most recently modified files in a directory.",
         {"type": "object", "properties": {
             "path": {"type": "string", "default": "."},
             "count": {"type": "integer", "default": 20},
         }},
         lambda path=".", count=20: _recent_files(path, count)),
    ]:
        registry.register(ToolSchema(
            name=name, description=desc, parameters=params,
            function=fn, category="system",
        ))


# ── Interactive helpers ──────────────────────────────────────────

def _ask_user(question: str) -> str:
    _spinner.stop()
    print(f"\n  {_yellow('[agent asks]')} {question}")
    try:
        answer = input(f"  {_bold('>')} ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = "(no response)"
    return answer or "(no response)"


# ── Model management ─────────────────────────────────────────────

_MODELS_DIRS = [
    _SELF / "models",                       # built-in: ultralight-coder/models/
]
# User-configured extra model directories (one path per line)
_MODELS_PATHS_FILE = Path.home() / ".ulcagent_model_paths"
if _MODELS_PATHS_FILE.exists():
    for _line in _MODELS_PATHS_FILE.read_text(encoding="utf-8").splitlines():
        _p = Path(_line.strip())
        if _p.is_dir() and _p not in _MODELS_DIRS:
            _MODELS_DIRS.append(_p)

_DEFAULT_FILE = Path.home() / ".ulcagent_default_model"


def _scan_models() -> list[dict]:
    """Scan all model directories for GGUF files."""
    models = []
    for models_dir in _MODELS_DIRS:
        if not models_dir.exists():
            continue
        for p in sorted(models_dir.glob("*.gguf")):
            # Skip split parts (only show first part)
            name = p.stem
            if "-00002-" in name or "-00003-" in name:
                continue
            size_gb = p.stat().st_size / (1024**3)
            # Check for split files
            parts = list(models_dir.glob(f"{name.split('-00001')[0]}*.gguf")) if "-00001-" in name else [p]
            total_gb = sum(pp.stat().st_size for pp in parts) / (1024**3)
            models.append({
                "name": name,
                "path": str(p),
                "size_gb": total_gb,
                "parts": len(parts),
            })
    return models


def _get_default_model() -> str:
    """Get the default model path from config or saved preference."""
    if _DEFAULT_FILE.exists():
        saved = _DEFAULT_FILE.read_text(encoding="utf-8").strip()
        if saved and Path(saved).exists():
            return saved
    # Fall back to config
    return PROFILES["code"]["config"]


def _list_models():
    """Show available GGUF models."""
    models = _scan_models()
    if not models:
        print(f"  {_red('No GGUF files found in')} {_MODELS_DIR}")
        return
    # Check current default
    default_path = ""
    if _DEFAULT_FILE.exists():
        default_path = _DEFAULT_FILE.read_text(encoding="utf-8").strip()

    print(f"\n  {_bold('Available models')} ({_MODELS_DIR}):\n")
    for m in models:
        parts_note = f" ({m['parts']} parts)" if m['parts'] > 1 else ""
        is_default = " *default*" if m['path'] == default_path else ""
        label = _green(m['name']) if is_default else m['name']
        size = f"{m['size_gb']:.1f} GB"
        print(f"    {label}  {_dim(size)}{_dim(parts_note)}{_green(is_default)}")
    print(f"\n  {_dim('Use /model <name> to switch, /default <name> to set permanent default')}")


def _switch_model(name: str, mgr):
    """Switch model by profile. Syntax: /model code <name>, /model general <name>, or /model <name> (both)."""
    parts = name.split(None, 1)

    # Detect if first word is a profile name
    target_profiles = ["code", "general"]
    if len(parts) == 2 and parts[0].lower() in ("code", "general"):
        target_profiles = [parts[0].lower()]
        search = parts[1]
    elif len(parts) == 2 and parts[0].lower() == "both":
        target_profiles = ["code", "general"]
        search = parts[1]
    else:
        search = name

    models = _scan_models()
    matches = [m for m in models if search.lower() in m['name'].lower()]
    if not matches:
        print(f"  {_red('No model matching')} '{search}'")
        print(f"  {_dim('Available:')} {', '.join(m['name'] for m in models)}")
        return
    if len(matches) > 1:
        print(f"  {_yellow('Multiple matches:')} {', '.join(m['name'] for m in matches)}")
        print(f"  {_dim('Be more specific')}")
        return
    model = matches[0]
    for p in target_profiles:
        PROFILES[p]["_override_path"] = model["path"]
    if mgr.loaded:
        mgr.unload()
    profiles_str = " + ".join(target_profiles)
    print(f"  {_green('Set')} {_cyan(profiles_str)} -> {model['name']} ({model['size_gb']:.1f} GB)")
    # Show current mapping
    for p in ("code", "general"):
        override = PROFILES[p].get("_override_path")
        if override:
            short = Path(override).stem
            label = _cyan("code") if p == "code" else _magenta("general")
            print(f"    {label}: {short}")
        else:
            default_cfg = PROFILES[p]["config"]
            label = _cyan("code") if p == "code" else _magenta("general")
            print(f"    {label}: {_dim('(default from config)')}")


def _manage_model_paths(args: str):
    """Handle /modelpath command: add, remove, or list model directories."""
    parts = args.strip().split(None, 1)
    action = parts[0].lower() if parts else "list"

    if action == "list" or not parts:
        print(f"\n  {_bold('Model search directories:')}")
        for d in _MODELS_DIRS:
            exists = d.exists()
            count = len(list(d.glob("*.gguf"))) if exists else 0
            status = f"{count} models" if exists else "not found"
            print(f"    {d}  {_dim(f'({status})')}")
        print(f"\n  {_dim(f'Config: {_MODELS_PATHS_FILE}')}")
        return

    if action == "add" and len(parts) > 1:
        new_dir = Path(parts[1].strip()).resolve()
        if not new_dir.is_dir():
            print(f"  {_red('Not a directory:')} {new_dir}")
            return
        if new_dir in _MODELS_DIRS:
            print(f"  {_dim('Already in search paths:')} {new_dir}")
            return
        _MODELS_DIRS.append(new_dir)
        # Persist to config file
        existing = []
        if _MODELS_PATHS_FILE.exists():
            existing = [l.strip() for l in _MODELS_PATHS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        existing.append(str(new_dir))
        _MODELS_PATHS_FILE.write_text("\n".join(existing) + "\n", encoding="utf-8")
        count = len(list(new_dir.glob("*.gguf")))
        print(f"  {_green('Added:')} {new_dir} ({count} models found)")
        return

    if action == "remove" and len(parts) > 1:
        target = Path(parts[1].strip()).resolve()
        if target == _SELF / "models":
            print(f"  {_red('Cannot remove the built-in models directory')}")
            return
        if target in _MODELS_DIRS:
            _MODELS_DIRS.remove(target)
        # Update config file
        if _MODELS_PATHS_FILE.exists():
            lines = [l.strip() for l in _MODELS_PATHS_FILE.read_text(encoding="utf-8").splitlines()
                     if l.strip() and Path(l.strip()).resolve() != target]
            _MODELS_PATHS_FILE.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        print(f"  {_green('Removed:')} {target}")
        return

    print(f"  {_dim('Usage: /modelpath add <dir> | /modelpath remove <dir> | /modelpath list')}")


def _set_default_model(name: str):
    """Save a model as the default for future sessions."""
    models = _scan_models()
    matches = [m for m in models if name.lower() in m['name'].lower()]
    if not matches:
        print(f"  {_red('No model matching')} '{name}'")
        return
    if len(matches) > 1:
        print(f"  {_yellow('Multiple matches:')} {', '.join(m['name'] for m in matches)}")
        return
    model = matches[0]
    _DEFAULT_FILE.write_text(model["path"], encoding="utf-8")
    print(f"  {_green('Default set:')} {model['name']}")
    print(f"  {_dim(f'Saved to {_DEFAULT_FILE}')}")


# ── Slash commands ───────────────────────────────────────────────

_context_files: dict[str, str] = {}  # path -> content, injected into system prompt


def _inject_project_index(agent, workspace: Path):
    """Scan workspace and add a file tree to the agent's system prompt."""
    entries = []
    try:
        for p in sorted(workspace.rglob("*")):
            if any(skip in p.parts for skip in (
                ".git", "__pycache__", "node_modules", ".venv", "dist",
                "build", ".egg-info", ".tox",
            )):
                continue
            if p.is_file():
                rel = str(p.relative_to(workspace)).replace("\\", "/")
                size = p.stat().st_size
                if size > 1024 * 1024:
                    label = f"{size / (1024*1024):.1f}MB"
                elif size > 1024:
                    label = f"{size // 1024}KB"
                else:
                    label = f"{size}B"
                entries.append(f"  {rel} ({label})")
                if len(entries) >= 60:
                    entries.append(f"  ... and more files")
                    break
    except OSError:
        return
    if entries:
        tree = "\n".join(entries)
        agent.system_prompt_extra += (
            f"\n\nProject files ({len(entries)} indexed):\n{tree}"
        )


def _load_context(goal: str, workspace: Path):
    """Handle /context command: load files into working memory."""
    parts = goal.split()[1:]  # skip "/context"
    if not parts or parts[0].lower() == "clear":
        _context_files.clear()
        print(f"  {_dim('Context cleared.')}")
        return
    if parts[0].lower() == "list":
        if not _context_files:
            print(f"  {_dim('No files in context.')}")
        else:
            for path in _context_files:
                lines = _context_files[path].count("\n")
                print(f"  {_cyan(path)} ({lines} lines)")
        return
    for fname in parts:
        p = (workspace / fname).resolve()
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                _context_files[fname] = content
                lines = content.count("\n")
                print(f"  {_green('+')} {fname} ({lines} lines)")
            except OSError as e:
                print(f"  {_red('err:')} {fname}: {e}")
        else:
            print(f"  {_red('not found:')} {fname}")


def _show_diff(workspace: Path):
    """Show git diff for the workspace."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, timeout=10, cwd=str(workspace),
        )
        if result.returncode != 0:
            print(f"  {_red('not a git repo or git error')}")
            return
        stat = result.stdout.strip()
        if not stat:
            print(f"  {_dim('No changes.')}")
            return
        print(f"\n{stat}")
        # Also show the full diff (truncated)
        full = subprocess.run(
            ["git", "diff"],
            capture_output=True, text=True, timeout=10, cwd=str(workspace),
        )
        lines = full.stdout.splitlines()
        for line in lines[:80]:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"  {_green(line)}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"  {_red(line)}")
            elif line.startswith("@@"):
                print(f"  {_cyan(line)}")
            else:
                print(f"  {line}")
        if len(lines) > 80:
            print(f"  {_dim(f'... ({len(lines) - 80} more lines)')}")
    except Exception as e:
        print(f"  {_red('error:')} {e}")


def _do_commit(workspace: Path, mgr, warm: bool):
    """Auto-commit with a generated message."""
    import subprocess
    # Check for changes
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=10, cwd=str(workspace),
        )
        if not status.stdout.strip():
            print(f"  {_dim('Nothing to commit.')}")
            return
        print(f"\n  {_bold('Changes to commit:')}")
        for line in status.stdout.strip().splitlines():
            print(f"    {line}")

        # Generate commit message from diff
        diff = subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True, text=True, timeout=10, cwd=str(workspace),
        )
        if not diff.stdout.strip():
            # Nothing staged — stage everything first
            subprocess.run(["git", "add", "-A"], cwd=str(workspace), timeout=10)
            diff = subprocess.run(
                ["git", "diff", "--staged"],
                capture_output=True, text=True, timeout=10, cwd=str(workspace),
            )

        # Simple message from changed files
        files = [l.split()[-1] for l in status.stdout.strip().splitlines()]
        if len(files) <= 3:
            msg = f"Update {', '.join(files)}"
        else:
            msg = f"Update {len(files)} files"

        print(f"\n  {_bold('Commit message:')} {msg}")
        try:
            answer = input(f"  {_yellow('Commit? [y/N/edit]')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer == "edit":
            try:
                msg = input(f"  {_bold('Message:')} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"  {_dim('Cancelled.')}")
                return
        elif answer not in ("y", "yes"):
            print(f"  {_dim('Cancelled.')}")
            return

        result = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True, text=True, timeout=30, cwd=str(workspace),
        )
        if result.returncode == 0:
            print(f"  {_green('Committed:')} {msg}")
        else:
            print(f"  {_red('Failed:')} {result.stderr.strip()[:200]}")
    except Exception as e:
        print(f"  {_red('error:')} {e}")


# ── Startup awareness ────────────────────────────────────────────

def _startup_greeting(workspace: Path):
    """Read cross-session memory + git status for a contextual greeting."""
    parts = []

    # Project name from directory
    project = workspace.name

    # Cross-session memory notes
    try:
        from engine.agent_memory import AgentMemory
        mem = AgentMemory(workspace=workspace)
        notes = mem.load()
        if notes:
            # Extract last 2 bullet points
            bullets = [l.strip() for l in notes.strip().splitlines() if l.strip().startswith("-")]
            if bullets:
                last = bullets[-1][2:].strip()  # remove "- " prefix
                parts.append(f"last note: {_dim(last[:80])}")
    except Exception:
        pass

    # Git status
    try:
        import subprocess
        status = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=5, cwd=str(workspace),
        )
        if status.returncode == 0:
            changes = len([l for l in status.stdout.strip().splitlines() if l.strip()])
            if changes > 0:
                parts.append(f"{_yellow(str(changes))} uncommitted change{'s' if changes != 1 else ''}")
            else:
                parts.append(f"{_green('clean')} working tree")

            # Last commit
            log = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                capture_output=True, text=True, timeout=5, cwd=str(workspace),
            )
            if log.returncode == 0 and log.stdout.strip():
                parts.append(f"last commit: {_dim(log.stdout.strip()[:60])}")
    except Exception:
        pass

    if parts:
        info = " | ".join(parts)
        print(f"  {_bold(project)}: {info}")
    else:
        print(f"  {_bold(project)}")


# ── Undo system ──────────────────────────────────────────────────

_undo_snapshot: dict[str, bytes] = {}  # relative_path -> content bytes


def _snapshot_workspace(workspace: Path):
    """Capture all file contents before a goal runs."""
    _undo_snapshot.clear()
    try:
        for p in workspace.rglob("*"):
            if not p.is_file():
                continue
            if any(skip in p.parts for skip in (
                ".git", "__pycache__", "node_modules", ".venv",
            )):
                continue
            rel = str(p.relative_to(workspace))
            try:
                _undo_snapshot[rel] = p.read_bytes()
            except OSError:
                pass
    except OSError:
        pass


def _do_undo(workspace: Path):
    """Restore workspace to the pre-goal snapshot."""
    if not _undo_snapshot:
        print(f"  {_dim('Nothing to undo.')}")
        return
    restored = 0
    for rel, content in _undo_snapshot.items():
        p = workspace / rel
        try:
            current = p.read_bytes() if p.exists() else None
            if current != content:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
                restored += 1
        except OSError:
            pass
    # Remove files that didn't exist in snapshot
    for p in workspace.rglob("*"):
        if not p.is_file():
            continue
        if any(skip in p.parts for skip in (".git", "__pycache__", "node_modules", ".venv")):
            continue
        rel = str(p.relative_to(workspace))
        if rel not in _undo_snapshot:
            try:
                p.unlink()
                restored += 1
            except OSError:
                pass
    print(f"  {_green(f'Undone:')} {restored} file{'s' if restored != 1 else ''} restored to pre-goal state.")
    _undo_snapshot.clear()


# ── Post-task suggestions ────────────────────────────────────────

def _suggest_next(result) -> str:
    """Detect what happened and suggest a natural follow-up."""
    if result is None:
        return ""
    calls = [c.name for c in result.tool_calls] if result.tool_calls else []
    wrote = any(n in calls for n in ("write_file", "edit_file", "insert_at_line"))
    ran_tests = "run_tests" in calls
    read_only = all(n in ("read_file", "list_dir", "glob", "grep", "read_function",
                          "find_definition", "find_usages") for n in calls) and calls

    if wrote and not ran_tests:
        return "Run the tests?"
    if ran_tests and not result.passed if hasattr(result, 'passed') else False:
        return "Want me to fix the failures?"
    if read_only:
        return "Want me to make changes?"
    return ""


def _maybe_auto_flag(result, goal: str) -> None:
    """Run the failure flagger silently after every goal. Writes YAML
    augmentor entries to data/augmentor_examples/_auto_generated/ when
    known failure patterns are detected; no-op on clean runs.

    Print is one line — visible only when records are written, so clean
    runs don't add noise.

    Disabled via --no-auto-flag CLI flag.
    """
    if result is None:
        return
    try:
        from engine.failure_flagger import flag, summarize
        from engine.yaml_augmentor_builder import write_all
        from engine.recovery_detector import write_all_recoveries
    except Exception:
        return  # missing module — silently skip rather than break the REPL
    failure_paths: list = []
    recovery_paths: list = []
    try:
        records = flag(result, goal)
        if records:
            failure_paths = write_all(records, goal, _SELF)
    except Exception:
        records = []
    try:
        recovery_paths = write_all_recoveries(result, goal, _SELF)
    except Exception:
        pass
    if not failure_paths and not recovery_paths:
        return
    parts = []
    if records:
        counts = summarize(records)
        parts.append(", ".join(f"{k}×{v}" for k, v in counts.items()))
    if recovery_paths:
        parts.append(f"recoveries×{len(recovery_paths)}")
    summary = " | ".join(parts)
    total = len(failure_paths) + len(recovery_paths)
    print(f"  {_dim('[auto-flagged:')} {_yellow(summary)} {_dim(f'-> {total} YAML(s) under data/auto_generated_review/]')}")


# ── Project rules (.ulcagent) ────────────────────────────────────

def _load_project_rules(workspace: Path) -> str:
    """Load .ulcagent project rules file if present."""
    for name in (".ulcagent", ".ulcagent.md"):
        p = workspace / name
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    return content
            except OSError:
                pass
    return ""


def _load_aliases(workspace: Path) -> dict:
    """Parse [aliases] section from .ulcagent file."""
    aliases = {
        # Built-in aliases
        "/test": "Run pytest on this project. Report any failures with file and line number. Be concise.",
        "/lint": "Run the linter (flake8 or pylint or eslint, whichever is configured) on this project. Report issues. Be concise.",
        "/format": "Run the code formatter (black or prettier, whichever is configured) on all source files. Report what changed. Be concise.",
    }
    for name in (".ulcagent", ".ulcagent.md"):
        p = workspace / name
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        in_aliases = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower() == "[aliases]":
                in_aliases = True
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                in_aliases = False
                continue
            if in_aliases and "=" in stripped:
                key, val = stripped.split("=", 1)
                key = key.strip()
                if not key.startswith("/"):
                    key = "/" + key
                aliases[key] = val.strip()
    return aliases


# ── Session export ───────────────────────────────────────────────

_session_log: list[dict] = []  # {"role": "user"|"agent", "content": str, "stats": str}


def _export_session(workspace: Path, args: str):
    """Export session conversation to markdown."""
    if not _session_log:
        print(f"  {_dim('Nothing to export.')}")
        return
    fname = args.strip() if args.strip() else f"session_{time.strftime('%Y%m%d_%H%M%S')}.md"
    if not fname.endswith(".md"):
        fname += ".md"
    p = workspace / fname
    lines = [f"# ulcagent session — {time.strftime('%Y-%m-%d %H:%M')}\n"]
    lines.append(f"workspace: {workspace}\n\n---\n")
    for entry in _session_log:
        if entry["role"] == "user":
            lines.append(f"\n## >>> {entry['content']}\n")
        else:
            lines.append(f"\n{entry['content']}\n")
            if entry.get("stats"):
                lines.append(f"\n*{entry['stats']}*\n")
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {_green('Exported:')} {p} ({len(_session_log)} entries)")


# ── Clipboard ────────────────────────────────────────────────────

_last_answer: str = ""


def _update_last_answer(answer: str):
    global _last_answer
    _last_answer = answer


def _clipboard_paste() -> str:
    """Read clipboard contents."""
    import subprocess
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
        else:
            result = subprocess.run(
                ["pbpaste"] if sys.platform == "darwin" else ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=5,
            )
        content = result.stdout.strip()
        if content:
            lines = content.count("\n") + 1
            print(f"  {_green('Pasted:')} {lines} lines from clipboard")
            return content
        print(f"  {_dim('Clipboard is empty.')}")
    except Exception as e:
        print(f"  {_red('Clipboard error:')} {e}")
    return ""


def _clipboard_copy(text: str):
    """Copy text to clipboard."""
    import subprocess
    try:
        if os.name == "nt":
            proc = subprocess.Popen(
                ["clip"], stdin=subprocess.PIPE, text=True,
            )
            proc.communicate(input=text, timeout=5)
        elif sys.platform == "darwin":
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE, text=True)
            proc.communicate(input=text, timeout=5)
        else:
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, text=True,
            )
            proc.communicate(input=text, timeout=5)
        lines = text.count("\n") + 1
        print(f"  {_green('Copied:')} {lines} lines to clipboard")
    except Exception as e:
        print(f"  {_red('Clipboard error:')} {e}")


# ── Code review ──────────────────────────────────────────────────

def _do_review(workspace: Path):
    """Return a goal string that asks the agent to review the current diff."""
    import subprocess
    try:
        diff = subprocess.run(
            ["git", "diff"], capture_output=True, text=True, timeout=15, cwd=str(workspace),
        )
        staged = subprocess.run(
            ["git", "diff", "--staged"], capture_output=True, text=True, timeout=15, cwd=str(workspace),
        )
        combined = (diff.stdout + staged.stdout).strip()
        if not combined:
            print(f"  {_dim('No changes to review.')}")
            return None
        lines = len(combined.splitlines())
        if lines > 200:
            combined = "\n".join(combined.splitlines()[:200]) + f"\n... ({lines - 200} more lines)"
        return (
            f"Review this git diff for bugs, security issues, and code quality. "
            f"Be concise — list issues with severity (critical/high/medium/low).\n\n"
            f"```diff\n{combined}\n```"
        )
    except Exception as e:
        print(f"  {_red('Error:')} {e}")
        return None


def _do_review_deep(workspace: Path, base: str = "HEAD"):
    """Return a goal string for a structured deep review against a base ref.

    Different posture from `/review`: the agent is told to read the changed
    files in full (not just diff context), then output a structured review
    with explicit categories. Uses read-only tools — the agent should not
    modify the working tree.
    """
    import subprocess
    try:
        # Find changed files vs base (handles both committed-on-branch and uncommitted)
        names_cmd = subprocess.run(
            ["git", "diff", "--name-only", base], capture_output=True, text=True, timeout=15, cwd=str(workspace),
        )
        unstaged = subprocess.run(
            ["git", "diff", "--name-only"], capture_output=True, text=True, timeout=15, cwd=str(workspace),
        )
        staged = subprocess.run(
            ["git", "diff", "--name-only", "--staged"], capture_output=True, text=True, timeout=15, cwd=str(workspace),
        )
        files = sorted({
            ln.strip()
            for src in (names_cmd.stdout, unstaged.stdout, staged.stdout)
            for ln in src.splitlines()
            if ln.strip()
        })
        if not files:
            print(f"  {_dim('No changed files to review.')}")
            return None

        diff_cmd = subprocess.run(
            ["git", "diff", base], capture_output=True, text=True, timeout=30, cwd=str(workspace),
        )
        diff_text = diff_cmd.stdout.strip()
        if not diff_text:
            # Fall back to working-tree diff if base has no diff (e.g. base == HEAD)
            diff_text = (subprocess.run(
                ["git", "diff"], capture_output=True, text=True, timeout=15, cwd=str(workspace),
            ).stdout + subprocess.run(
                ["git", "diff", "--staged"], capture_output=True, text=True, timeout=15, cwd=str(workspace),
            ).stdout).strip()

        diff_lines = diff_text.splitlines()
        if len(diff_lines) > 400:
            diff_text = "\n".join(diff_lines[:400]) + f"\n... ({len(diff_lines) - 400} more lines truncated; read the files directly with read_file)"

        file_list = "\n".join(f"  - {f}" for f in files[:30])
        if len(files) > 30:
            file_list += f"\n  ... +{len(files) - 30} more"

        return (
            f"DEEP CODE REVIEW (read-only — do NOT modify any files).\n\n"
            f"Base: {base}. Changed files ({len(files)}):\n{file_list}\n\n"
            f"WORKFLOW:\n"
            f"  1. Read each changed file in full with read_file (not just the diff).\n"
            f"  2. For each file, look for: bugs, security issues, missing error handling,\n"
            f"     missing tests, style/idiom violations, performance regressions,\n"
            f"     documentation gaps.\n"
            f"  3. Cross-check: do the changes break callers in unchanged files?\n"
            f"     Use grep to find callers. Do NOT edit anything.\n\n"
            f"OUTPUT — produce a structured review with these sections (skip empty ones):\n"
            f"  ## Critical    — bugs, data loss, security holes, crashes\n"
            f"  ## High        — likely bugs, broken contracts, missing validation\n"
            f"  ## Medium      — code smell, missing tests, weak error handling\n"
            f"  ## Low         — style nits, doc gaps, minor refactor opportunities\n"
            f"  ## Questions   — anything ambiguous you'd ask the author\n"
            f"Each item: one line, format `file:line — description`. No prose intro.\n\n"
            f"```diff\n{diff_text}\n```"
        )
    except Exception as e:
        print(f"  {_red('Error:')} {e}")
        return None


# ── Snippets ─────────────────────────────────────────────────────

_SNIPPETS_DIR = Path.home() / ".ulcagent_snippets"


def _manage_snippets(args: str):
    """Handle /snippet commands."""
    parts = args.strip().split(None, 1)
    if not parts or parts[0] == "list":
        if not _SNIPPETS_DIR.exists():
            print(f"  {_dim('No snippets saved.')}")
            return
        files = sorted(_SNIPPETS_DIR.glob("*.txt"))
        if not files:
            print(f"  {_dim('No snippets saved.')}")
            return
        print(f"\n  {_bold('Saved snippets:')}")
        for f in files:
            lines = f.read_text(errors="replace").count("\n") + 1
            print(f"    {_cyan(f.stem)}  {_dim(f'({lines} lines)')}")
        return

    if parts[0] == "save" and len(parts) > 1:
        name = parts[1].strip()
        if not _last_answer:
            print(f"  {_dim('Nothing to save — run a goal first.')}")
            return
        _SNIPPETS_DIR.mkdir(exist_ok=True)
        (_SNIPPETS_DIR / f"{name}.txt").write_text(_last_answer, encoding="utf-8")
        print(f"  {_green('Saved:')} {name} ({_last_answer.count(chr(10)) + 1} lines)")
        return

    if parts[0] == "delete" and len(parts) > 1:
        name = parts[1].strip()
        p = _SNIPPETS_DIR / f"{name}.txt"
        if p.exists():
            p.unlink()
            print(f"  {_green('Deleted:')} {name}")
        else:
            print(f"  {_red('Not found:')} {name}")
        return

    # Load a snippet as context
    name = parts[0]
    p = _SNIPPETS_DIR / f"{name}.txt"
    if p.exists():
        content = p.read_text(errors="replace")
        print(f"  {_green('Loaded snippet:')} {name} ({content.count(chr(10)) + 1} lines)")
        return content
    print(f"  {_red('Snippet not found:')} {name}")
    print(f"  {_dim('Use /snippet list to see available snippets')}")
    return None


# ── Session stats ────────────────────────────────────────────────

_stats = {"goals": 0, "tool_calls": 0, "iterations": 0, "wall_time": 0.0, "ctx_peak": 0.0}


def _show_stats():
    """Display session statistics."""
    print(f"\n  {_bold('Session stats:')}")
    print(f"    Goals completed:  {_stats['goals']}")
    print(f"    Total iterations: {_stats['iterations']}")
    print(f"    Total tool calls: {_stats['tool_calls']}")
    print(f"    Wall time:        {_stats['wall_time']:.0f}s ({_stats['wall_time']/60:.1f} min)")
    print(f"    Peak context:     {_stats['ctx_peak']:.0f}%")


# ── Autofix loop ─────────────────────────────────────────────────

def _autofix_goal(max_rounds: int = 5) -> str:
    """Generate a goal string for the test-fix loop."""
    return (
        f"Run pytest (or the project's test suite). If any tests fail, read the "
        f"failure output, identify the bug, fix it, and re-run the tests. "
        f"Repeat until all tests pass or you've tried {max_rounds} fix attempts. "
        f"Be concise — just fix and verify."
    )


def _tdd_goal(user_goal: str, max_rounds: int = 5) -> str:
    """Wrap a user goal into a test-driven development loop.

    The agent writes pytest tests for the goal first, then iterates on
    implementation until the tests pass. Designed to convert one-shot
    multi-file emissions (where 14B flaps) into incremental loops where
    each turn has a clear pass/fail signal.
    """
    user_goal = (user_goal or "").strip()
    if not user_goal:
        return ""
    return (
        "Follow a strict test-driven workflow for this task:\n"
        f"  GOAL: {user_goal}\n\n"
        "STEP 1. Write pytest tests for the goal in tests/ (or the project's existing\n"
        "        test directory). Cover the happy path and at least one edge case.\n"
        "        Use write_file. Do NOT write the implementation yet.\n"
        "STEP 2. Run the tests with run_tests — they should fail (red phase).\n"
        "STEP 3. Write the minimum implementation to make the tests pass. Use\n"
        "        write_file or edit_file. Keep it small.\n"
        "STEP 4. Run the tests again. If any fail, read the failure, fix one bug,\n"
        f"        re-run. Loop up to {max_rounds} times.\n"
        "STEP 5. When all tests pass, give a 1-2 sentence summary. Do not refactor\n"
        "        beyond what was needed.\n\n"
        "Stop and report if step 1 or 3 produces no testable behavior."
    )


def _expand_slash_command(goal: str) -> str:
    """Pre-expand goal-rewriting slash commands so they work in one-shot
    mode the same way they do in the REPL dispatcher.

    Supports: /proof [path], /tdd <goal>, /autofix [N]. Other slash
    commands (/diff, /commit, /context, /clear, ...) are interactive
    and don't make sense as one-shot args — they pass through unchanged.

    /proof PRE-EXECUTES pytest on the user's behalf and embeds the
    failure traceback into the goal. Required because empirically the
    14B cannot drive open-ended diagnose-and-fix loops — it bails at
    iter 1-2 with "I cannot complete..." or "Let's read the file..."
    narration. Embedding the concrete failure converts "discover the
    bug" (which the model can't reliably do) into "fix this specific
    traceback" (which it can).
    """
    if not goal:
        return goal
    g = goal.strip()
    g_lower = g.lower()
    if g_lower.startswith("/proof"):
        target = g[len("/proof"):].strip()
        return _proof_goal(target, run_pytest_preflight=True)
    if g_lower.startswith("/tdd"):
        user_goal = g[4:].strip()
        if user_goal:
            return _tdd_goal(user_goal)
    if g_lower.startswith("/autofix"):
        parts = g.split()
        rounds = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
        return _autofix_goal(rounds)
    return goal


def _run_pytest_preflight(target: str = "") -> tuple[bool, str]:
    """Run pytest in the current workspace and capture the result.
    Returns (passed, output). Used by /proof to embed the actual
    traceback in the goal so the 14B has a concrete fix-this task
    instead of open-ended discovery (which it can't reliably drive).

    `target` is an optional pytest argument (file path or test node).
    If empty, pytest discovers tests in the current directory.

    Times out at 60s — pytest on a small project should finish in <2s,
    so 60s is a generous safety net for a misconfigured project.
    """
    import subprocess
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--no-header"]
    if target:
        cmd.append(target)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            cwd=str(Path.cwd()),
        )
    except FileNotFoundError:
        return False, "[pytest preflight: python or pytest not on PATH]"
    except subprocess.TimeoutExpired:
        return False, "[pytest preflight: timed out after 60s]"
    output = (proc.stdout or "") + (proc.stderr or "")
    # exit 0 = all passed; 5 = no tests collected; anything else = failure
    passed = proc.returncode == 0
    return passed, output


def _proof_goal(target: str = "", run_pytest_preflight: bool = False) -> str:
    """Self-proof goal: pre-execute pytest, embed any failure traceback
    into a concrete fix-this goal.

    KNOWN LIMITATION (2026-04-26): With Qwen 2.5 Coder 14B, the /proof
    goal shape consistently triggers a stock "I can't execute commands
    directly on your system" / "outside the workspace" refusal at iter 1
    regardless of:
      - whether tests actually fail (preflight success path also bails)
      - target path quoted vs unquoted
      - terse vs elaborate goal text
      - workspace (pomodoro/ AND ulcagent's own repo both bailed)
    The 14B appears to recognize the slash-command-derived goal shape
    as a "system action" pattern and refuse defensively. The same model
    in the SAME workspace handles human-written fix goals fine
    (self-proof v2: 3/3 tests fixed in 100s) — the shape, not the
    workspace or content, is the trigger.

    The pre-flight pytest invocation + traceback embedding remains
    sound infrastructure. A different model with looser refusal
    behavior (or augmentor-driven counter-bias against this stock
    refusal) should be able to use /proof end-to-end. Shipping the
    plumbing now so it's ready when that model is available.

    Set run_pytest_preflight=False for unit tests that don't want to
    actually invoke pytest at expansion time (e.g. test_proof_command.py).
    """
    target = (target or "").strip()

    if not run_pytest_preflight:
        # Test-friendly path: returns the no-preflight goal text without
        # invoking pytest. Used by unit tests that just want to verify
        # the goal shape.
        if target:
            return (
                f"Run the tests in {target} via run_tests. If any fail, "
                f"read the full traceback, find the bug (it could be in "
                f"source code OR test setup — check mocks, fixtures, "
                f"imports), fix it with a targeted edit_file (use "
                f"write_file only for a genuine full-file rewrite). Re-run "
                f"until tests pass. Do not give up — the bug is fixable."
            )
        return (
            "Run all tests in this project via run_tests. If any fail, "
            "read the full traceback, find the bug (it could be in source "
            "code OR test setup — check mocks, fixtures, imports), fix it "
            "with a targeted edit_file (use write_file only for a genuine "
            "full-file rewrite). Re-run until tests pass. Do not give up "
            "— the bug is fixable."
        )

    # Real path — pre-execute pytest and embed result.
    passed, output = _run_pytest_preflight(target)
    if passed:
        # Trivial success path: tell the agent "everything is green,
        # nothing to do" so it gives a one-line confirmation and exits.
        return (
            "Pytest already passed for this project (verified at goal "
            "expansion time). No edits needed. Confirm with a single "
            "sentence and stop."
        )

    # Trim the captured output so the goal stays under ~3k chars
    # (keeps the model's prompt budget intact). Most useful info is the
    # last 2KB which contains the traceback + summary line.
    output = output.strip()
    if len(output) > 2400:
        head = output[:400]
        tail = output[-2000:]
        output = head + "\n... [middle truncated] ...\n" + tail

    target_clause = f"in {target}" if target else "in this project"
    return (
        f"Pytest is currently FAILING {target_clause}. Captured output:\n\n"
        f"```\n{output}\n```\n\n"
        f"Diagnose the failure(s) above and fix them. The bug may be in "
        f"source code OR test setup — check mocks, fixtures, imports. "
        f"Use targeted edit_file with a SHORT unique anchor for line "
        f"changes; use write_file only for a genuine full-file rewrite "
        f"(NEVER use edit_file with empty old_string + multi-construct "
        f"new_string — that pattern is rejected). After your fix, call "
        f"run_tests again to confirm the failures are gone. Do NOT bail "
        f"early — the bug shown above IS fixable from the traceback."
    )


# ── Watch mode ───────────────────────────────────────────────────

def _watch_loop(workspace: Path, action: str, mgr, warm: bool, build_agent_fn):
    """Poll for file changes and run an action on each change."""
    import time as _time
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".yaml", ".yml", ".json"}
    action_goals = {
        "test": "Run pytest. Report results concisely.",
        "lint": "Run the linter on changed files. Report issues concisely.",
        "check": "Check all .py files for syntax errors. Report any found.",
    }
    goal = action_goals.get(action, action)

    # Snapshot mtimes
    def _snapshot():
        times = {}
        for p in workspace.rglob("*"):
            if p.is_file() and p.suffix in extensions:
                if any(skip in p.parts for skip in (".git", "__pycache__", "node_modules", ".venv")):
                    continue
                try:
                    times[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        return times

    prev = _snapshot()
    print(f"  {_dim(f'Watching {len(prev)} files for changes...')}")
    print(f"  {_dim(f'Action: {goal[:80]}')}")
    print(f"  {_dim('Press Ctrl+C to stop.')}\n")

    try:
        while True:
            _time.sleep(2)
            curr = _snapshot()
            changed = [f for f in curr if curr[f] != prev.get(f, 0)]
            new_files = [f for f in curr if f not in prev]
            changed.extend(new_files)
            if changed:
                names = [Path(f).name for f in changed[:5]]
                more = f" +{len(changed)-5}" if len(changed) > 5 else ""
                print(f"\n  {_yellow('Changed:')} {', '.join(names)}{more}")
                # Load model and run
                if not warm:
                    mgr.ensure_profile("code")
                else:
                    mgr.ensure_profile("code", quiet=True)
                agent = build_agent_fn(mgr, workspace)
                _run_one(agent, goal)
                if not warm:
                    mgr.unload()
                prev = _snapshot()
            else:
                prev = curr
    except KeyboardInterrupt:
        print(f"\n  {_dim('Watch stopped.')}")


# ── Daemon mode ──────────────────────────────────────────────────

def _detect_test_command(workspace: Path) -> str | None:
    """Return the most likely test command for the workspace, or None."""
    if (workspace / "pytest.ini").exists() or (workspace / "pyproject.toml").exists():
        # Best-effort — assume pytest if there's any python project file
        return "pytest"
    if any((workspace / f).exists() for f in ("tests", "test")):
        return "pytest"
    if (workspace / "package.json").exists():
        return "npm test"
    if (workspace / "Cargo.toml").exists():
        return "cargo test"
    if (workspace / "go.mod").exists():
        return "go test ./..."
    return None


_DAEMON_GOAL_TEMPLATE = (
    "Run the project's tests using run_tests. "
    "If they pass, respond with EXACTLY one line: PASS\n"
    "If they fail, respond with at most 5 lines: the first line is `FAIL`, "
    "and the next lines list each failing test and the file:line where it broke. "
    "No prose, no fix attempt — this is a status check, not a repair."
)


def _daemon_loop(workspace: Path, mgr, build_agent_fn):
    """Long-lived background daemon. Watches the workspace for file changes
    and runs the project test suite on each batch of changes. Stays quiet on
    pass; surfaces a one-line FAIL summary on failure.

    Privacy invariant preserved: stdin/stdout only, no sockets, no HTTP.
    """
    import time as _time
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".yaml", ".yml", ".json", ".toml"}

    cmd = _detect_test_command(workspace)
    if cmd is None:
        print(f"  {_yellow('No test framework detected.')} "
              f"{_dim('Daemon will still watch for changes but cannot self-verify.')}")
    else:
        print(f"  {_dim(f'detected test command: {cmd}')}")

    def _snapshot():
        times = {}
        for p in workspace.rglob("*"):
            if not p.is_file() or p.suffix not in extensions:
                continue
            if any(skip in p.parts for skip in (".git", "__pycache__", "node_modules", ".venv", "dist", "build")):
                continue
            try:
                times[str(p)] = p.stat().st_mtime
            except OSError:
                pass
        return times

    prev = _snapshot()
    print(f"  {_dim(f'watching {len(prev)} files. Ctrl+C to stop.')}\n")

    # Warm-load once — the daemon stays running and re-uses the model
    mgr.ensure_profile("code", quiet=True)
    agent = build_agent_fn(mgr, workspace)

    debounce_until = 0.0
    try:
        while True:
            _time.sleep(2)
            now = _time.time()
            curr = _snapshot()
            changed = [f for f in curr if curr[f] != prev.get(f, 0)]
            new_files = [f for f in curr if f not in prev]
            changed.extend(new_files)
            if not changed:
                prev = curr
                continue

            # Debounce: wait 3s of quiet before firing, to coalesce save bursts
            prev = curr
            debounce_until = now + 3.0
            while _time.time() < debounce_until:
                _time.sleep(0.5)
                curr2 = _snapshot()
                more = [f for f in curr2 if curr2[f] != prev.get(f, 0)]
                more.extend([f for f in curr2 if f not in prev])
                if more:
                    prev = curr2
                    debounce_until = _time.time() + 3.0

            ts = _time.strftime("%H:%M:%S")
            stamp = _dim(f"[{ts}]")
            names = sorted({Path(f).name for f in changed})[:5]
            tail = f" +{len(changed) - 5}" if len(changed) > 5 else ""
            print(f"{stamp} change: {', '.join(names)}{tail}")

            if cmd is None:
                continue

            try:
                result = agent.run(_DAEMON_GOAL_TEMPLATE, continue_session=False)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"{stamp} {_red('[error]')} {exc}")
                continue

            answer = (result.final_answer or "").strip()
            first = answer.splitlines()[0].strip().upper() if answer else ""
            if first.startswith("PASS"):
                # Quiet on green — single character so the user knows we ran
                print(f"{stamp} {_green('PASS')}")
            elif first.startswith("FAIL"):
                print(f"{stamp} {_red('FAIL')}")
                for line in answer.splitlines()[1:6]:
                    if line.strip():
                        print(f"        {line.rstrip()}")
            else:
                # Unknown shape — print first line so the user sees something
                print(f"{stamp} {_yellow('?')} {answer.splitlines()[0][:120] if answer else '(no output)'}")
    except KeyboardInterrupt:
        print(f"\n  {_dim('daemon stopped.')}")


# ── Batch mode ───────────────────────────────────────────────────

def _run_batch(filepath: str, workspace: Path, mgr, warm: bool, build_agent_fn):
    """Run goals from a file, one per line."""
    p = Path(filepath.strip())
    if not p.is_file():
        print(f"  {_red('File not found:')} {p}")
        return
    goals = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
    if not goals:
        print(f"  {_dim('No goals in file.')}")
        return
    print(f"  {_bold(f'Batch: {len(goals)} goals from {p.name}')}\n")
    passed = 0
    for i, goal in enumerate(goals, 1):
        print(f"  {_bold(f'[{i}/{len(goals)}]')} {goal[:80]}")
        if not warm:
            profile = _detect_profile(goal)
            mgr.ensure_profile(profile)
        else:
            mgr.ensure_profile(_detect_profile(goal), quiet=True)
        agent = build_agent_fn(mgr, workspace)
        result = _run_one(agent, goal)
        if result and result.final_answer:
            passed += 1
        if not warm:
            mgr.unload()
    print(f"\n  {_bold(f'Batch complete: {passed}/{len(goals)} goals')}")


# ── Plugin system ────────────────────────────────────────────────

_PLUGINS_DIR = _SELF / "plugins"


def _load_plugins(registry):
    """Scan plugins/ for .py files and call their register() function."""
    if not _PLUGINS_DIR.exists():
        return 0
    count = 0
    for p in sorted(_PLUGINS_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"plugin_{p.stem}", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "register"):
                mod.register(registry)
                count += 1
        except Exception as e:
            print(f"  {_yellow(f'Plugin {p.name} failed:')} {e}")
    return count


# ── Model prompt profiles ────────────────────────────────────────

_PROFILES_DIR = _SELF / "profiles"


def _load_model_profile(model_path: str) -> dict:
    """Load a per-model prompt profile if one exists.
    Returns {"system_prompt": str, "temperature": float, ...} or empty dict."""
    if not _PROFILES_DIR.exists():
        return {}
    model_name = Path(model_path).stem.lower()
    for p in _PROFILES_DIR.glob("*.yaml"):
        try:
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            patterns = data.get("match", [])
            if any(pat.lower() in model_name for pat in patterns):
                return data
        except Exception:
            continue
    # Also try .txt files (plain system prompt)
    for p in _PROFILES_DIR.glob("*.txt"):
        if p.stem.lower() in model_name:
            return {"system_prompt": p.read_text(encoding="utf-8").strip()}
    return {}


# ── Doc generation ───────────────────────────────────────────────

_DOC_GOALS = {
    "readme": (
        "Read all files in this project. Generate a comprehensive README.md that includes: "
        "project name, what it does, installation steps, usage examples, file structure, "
        "and key dependencies. Write the README using write_file."
    ),
    "api": (
        "Read all source files. Find every public function, class, and API endpoint. "
        "Generate API documentation in markdown format listing each with its signature, "
        "parameters, return value, and a one-line description. Write to API_DOCS.md."
    ),
    "arch": (
        "Read the project structure and key source files. Write an ARCHITECTURE.md that "
        "describes: high-level design, main components and how they interact, data flow, "
        "key design decisions, and file-by-file purpose summary."
    ),
}


_HELP_TEXT = """
  {bold}Three ways to use ulcagent:{end}
    Terminal:   {cyan}ulcagent{end}
    Browser:    {cyan}python web_agent.py{end}  (localhost:8899)
    VS Code:    right-click → Ask/Fix/Explain with ulcagent

  {bold}ulcagent commands (30):{end}
    {cyan}?{end}  / {cyan}help{end}              Show this help
    {cyan}/code{end} <goal>          Force the Coder model for this goal
    {cyan}/general{end} <goal>       Force the General model for this goal
    {cyan}/context{end} f1 f2 ...    Load files into working memory for cross-file reasoning
    {cyan}/context clear{end}        Clear loaded files
    {cyan}/context list{end}         Show loaded files
    {cyan}/diff{end}                 Show uncommitted changes (git diff)
    {cyan}/commit{end}               Stage + commit with auto-generated message
    {cyan}/undo{end}                 Revert all changes from the last goal
    {cyan}/clear{end}                Reset session memory (start fresh conversation)
    {cyan}/models{end}              List available GGUF models in models/
    {cyan}/model code <name>{end}   Set the code profile's model
    {cyan}/model general <name>{end} Set the general profile's model
    {cyan}/default <name>{end}      Set default model for both profiles
    {cyan}/modelpath{end}           List model search directories
    {cyan}/modelpath add <dir>{end} Add a directory to scan for GGUFs
    {cyan}/modelpath remove <dir>{end} Remove a search directory
    {cyan}/review{end}              Review uncommitted changes for bugs + security
    {cyan}/review-deep{end} [base]   Structured deep review (reads full files, severity buckets)
    {cyan}/export{end} [file]       Save session conversation to markdown
    {cyan}/paste{end}               Send clipboard contents as context
    {cyan}/copy{end}                Copy last answer to clipboard
    {cyan}/snippet save <n>{end}    Save last answer as a named snippet
    {cyan}/snippet list{end}        Show saved code snippets
    {cyan}/snippet delete <n>{end}  Delete a saved snippet
    {cyan}/snippet <name>{end}      Load a snippet as context
    {cyan}/stats{end}               Show session statistics
    {cyan}/test{end}                Run project tests (auto-detects framework)
    {cyan}/lint{end}                Run the project linter
    {cyan}/format{end}              Run the project code formatter
    {cyan}/autofix{end} [N]         Run tests, fix failures, re-run — loop up to N times (default 5)
    {cyan}/tdd{end} <goal>           Test-driven loop: write tests first, then implement until they pass
    {cyan}/proof{end} [path]         Self-proof: run tests, diagnose any failures (source OR test bugs), fix, re-run
    {cyan}/distill{end}              Capture the last run as a reusable trajectory artifact
    {cyan}/flag-errors{end}          Analyze last run for known failure patterns + auto-synthesize YAML augmentor fixes
    {cyan}/promote{end} [cat|--all|--list]  Validate + promote auto-generated YAMLs from review queue into retrieval index
    {cyan}/library{end}                     Status snapshot: trajectories, review queue, promoted YAMLs, total augmentor count
    {cyan}/watch{end} [action]      Watch for file changes, auto-run: test, lint, or custom goal
    {cyan}/batch{end} <file>        Run goals from a text file (one per line)
    {cyan}/docs{end} readme|api|arch  Auto-generate project documentation
    {cyan}/plugins{end}             List loaded plugins from plugins/ directory
    {cyan}/learn{end}               Capture a correction for future sessions
    {cyan}/learn list{end}          List stored corrections
    {cyan}/learn clear{end}         Clear all corrections
    {cyan}/learn delete{end} N      Delete correction at index N
    {cyan}cd <path>{end}             Switch workspace directory
    {cyan}exit{end} / {cyan}quit{end}            Exit the agent
    {cyan}Ctrl+C{end}               Cancel a running task

  {bold}Session memory:{end}
    Goals build on each other. Ask "read server.py", then "now fix the
    endpoint you just read" — the agent remembers the conversation.

    The context meter {cyan}[ctx: 45%]{end} shows how full the window is.
    When it passes 70% you'll get a warning — use {cyan}/clear{end} to reset.

    {cyan}/clear{end} only resets the conversation. Your project notes (things
    the agent learned via the "remember" tool) persist across sessions
    and are never cleared. Switching profiles also resets the conversation.

  {bold}Profiles (auto-detected from your goal):{end}
    {cyan}code{end}      Qwen Coder 14B — precise code edits, tests, refactoring
    {cyan}general{end}   Qwen Instruct 14B — exploration, system tasks, Q&A

    Auto-detects which profile fits. Override with /code or /general prefix.

  {bold}Project rules (.ulcagent file):{end}
    Drop a .ulcagent file in any project root with instructions:
      "Use tabs not spaces"
      "Tests go in tests/ directory"
      "Always type-hint function signatures"
      [aliases]
      /deploy = Run the deploy script via run_bash
      /check = Run mypy type checking and report errors
    Rules auto-load on startup. Custom aliases appear in the command list.

  {bold}How to think about this tool:{end}
    ulcagent is a surgical tool, not a chatbot. Give it the WHERE and
    the WHAT — it figures out the HOW. You don't need to know the exact
    fix, but you do need to point it in the right direction.

  {bold}Good prompts:{end}
    >>> Read the dashboard component and fix the title spacing — it overlaps the content
    >>> Find all .py files that import UserService and list them
    >>> Add input validation to the POST handler in server.py
    >>> The login test is failing — read it, find the bug, fix it, run the tests
    >>> What's using the most disk space in this project?
    >>> Show me the most recently changed files

  {bold}Weak prompts (rephrase these):{end}
    >>> "Something is broken"          -> name the file or symptom
    >>> "Make it better"               -> say what to improve
    >>> "Fix everything"               -> one task at a time

  {bold}What it handles well:{end}
    - Single-file edits: bugs, features, refactors, style fixes
    - File search: find definitions, usages, imports across a project
    - Multi-file changes: rename, add imports, update references
    - Test workflows: run tests, read failures, fix, re-run
    - Shell commands: build, lint, deploy (asks permission first)
    - System info: disk usage, processes, environment, recent files

  {bold}Where it struggles (use a cloud AI instead):{end}
    - Files over ~200 lines (break into smaller reads)
    - Deep cross-file reasoning across many components
    - Vague exploration without a concrete goal
    - Large file creation (100+ lines): the model can't write big HTML/config
      files in one shot. Scaffold the file yourself (or use a template), then
      ask ulcagent for targeted edits.

  {bold}Scaffolding workflow:{end}
    The model works best on TARGETED EDITS to existing files, not creating
    large files from scratch. For new projects:
    1. Create the file structure yourself (or copy a template)
    2. Use ulcagent to fill in logic, fix bugs, add features
    3. Use /diff and /commit to review and save changes
    Example: create an empty Flask app.py with routes, then:
    >>> Add input validation to the /users POST handler
    >>> Add rate limiting middleware
    >>> Write tests for the auth endpoints

  {bold}Startup flags:{end}
    --warm          Keep model loaded between goals (instant, ~10GB VRAM)
    --extended      Enable 21 advanced tools (git, checkpoint, etc.)
    --lsp           Enable code-intelligence tools (goto_def, refs, diagnostics, completions; needs jedi)
    --daemon        Headless watcher: runs project tests on every file save, quiet on PASS
    --architect     Force plan-then-execute on every goal (auto-triggered for multi-file scaffolding)
    --no-architect  Disable architect auto-trigger (always run flat agent)
    --handheld      Plan + per-step driver with prior-state injection. For LARGE projects (8+ files). Slower than architect but each step sees what prior steps actually wrote, eliminating cross-file class duplication.
    --no-auto-flag  Disable auto-flag-on-fail (silent failure-pattern detector that writes YAML augmentors)
    --yes           Auto-approve all risky tool calls (run_bash etc) — no interactive prompt. Auto-enabled when stdin is not a TTY.
"""


def _print_help():
    text = _HELP_TEXT
    if _USE_COLOR:
        text = text.replace("{bold}", "\033[1m").replace("{end}", "\033[0m")
        text = text.replace("{cyan}", "\033[36m")
    else:
        for tag in ("{bold}", "{end}", "{cyan}"):
            text = text.replace(tag, "")
    print(text)


def _run_one(agent, goal: str, continue_session: bool = False):
    if goal in ("?", "help"):
        _print_help()
        return None
    if agent is None:
        return None

    _spinner.start()
    t0 = time.monotonic()
    try:
        try:
            result = agent.run(goal, continue_session=continue_session)
        except TypeError:
            # ArchitectAgent.run takes goal only — no continue_session
            result = agent.run(goal)
    except KeyboardInterrupt:
        _spinner.stop()
        print("\n[interrupted]")
        return None
    except Exception as exc:
        _spinner.stop()
        print(f"\n{_red('[error]')} {exc}")
        return None
    _spinner.stop()
    wall = time.monotonic() - t0
    answer = (result.final_answer or "").strip()
    print()
    if answer:
        print(answer)
    elif result.iterations == 1 and len(result.tool_calls) == 0:
        print(_dim(f"(no response -- stop: {result.stop_reason})"))
        for turn in reversed(result.transcript):
            if turn.get("role") == "assistant":
                raw = turn.get("content", "").strip()
                if raw:
                    print(f"{_dim('raw:')} {raw[:500]}")
                    break
    else:
        print(_dim("(done)"))
    # Context usage meter
    transcript_chars = sum(len(t.get("content", "")) for t in result.transcript)
    budget = 52000  # matches Agent default context_char_budget
    pct = min(transcript_chars / budget * 100, 100)
    if pct > 70:
        ctx_color = _red
    elif pct > 40:
        ctx_color = _yellow
    else:
        ctx_color = _dim
    print(f"  {_dim(f'[{result.iterations} iter, {len(result.tool_calls)} calls, {wall:.1f}s')} | {ctx_color(f'ctx: {pct:.0f}%')}{_dim(']')}")

    # Warn when context is getting full
    if pct > 70:
        print(f"  {_yellow('Session memory getting full.')} Use {_cyan('/clear')} to start fresh (your project notes are saved).")

    return result


# ── Main ─────────────────────────────────────────────────────────

def main():
    # Readline history
    try:
        import readline
        histfile = Path.home() / ".ulcagent_history"
        try:
            readline.read_history_file(str(histfile))
        except FileNotFoundError:
            pass
        import atexit
        atexit.register(readline.write_history_file, str(histfile))
        readline.set_history_length(500)
    except ImportError:
        pass

    workspace = Path.cwd().resolve()
    warm = "--warm" in sys.argv
    daemon = "--daemon" in sys.argv

    print(f"{_bold('ulcagent')} {_dim('- adaptive local agent')}")
    print(f"  {_dim('workspace:')} {workspace}")
    _startup_greeting(workspace)
    if warm:
        print(f"  {_dim('mode: --warm')}")
    if daemon:
        print(f"  {_dim('mode: --daemon (quiet on success, surfaces failures only)')}")
    print()

    mgr = ModelManager()

    # Daemon mode — long-lived, watches workspace, runs tests on save, quiet
    # except on failure. Reuses the model in --warm style by default.
    if daemon:
        _daemon_loop(workspace, mgr, _build_agent)
        mgr.unload()
        return

    # Architect mode flags (override the auto-trigger heuristic)
    arch_force_on = "--architect" in sys.argv
    arch_force_off = "--no-architect" in sys.argv
    auto_flag_off = "--no-auto-flag" in sys.argv
    handheld_force_on = "--handheld" in sys.argv

    # One-shot mode
    goal_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if goal_args:
        goal = " ".join(goal_args)
        # Slash-command pre-expansion (parity with REPL dispatcher).
        # /proof, /tdd, /autofix etc. should work as one-shot args too.
        goal = _expand_slash_command(goal)
        profile = _detect_profile(goal)
        mgr.ensure_profile(profile)
        agent = _build_agent(mgr, workspace)
        if handheld_force_on:
            print(f"  {_dim('[handheld driver — plan + per-step with prior-state injection]')}")
            agent = _build_handheld_driver(agent)
        elif _should_use_architect(goal, arch_force_on, arch_force_off):
            print(f"  {_dim('[architect mode — plan-then-execute]')}")
            agent = _build_architect_agent(agent)
        result = _run_one(agent, goal)
        # Same auto-flag hook the REPL loop uses — the one-shot path was
        # silently skipping it, surfaced by the 2026-04-26 re-walkthrough.
        if result is not None and not auto_flag_off:
            _maybe_auto_flag(result, goal)
        mgr.unload()
        return

    # Interactive REPL with session memory
    hint = "Type a goal and press Enter. '?' for help. Ctrl+C to cancel. 'exit' to quit."
    print(f"  {_dim(hint)}\n")

    agent = None
    session_active = False  # True after first goal — enables continue_session
    last_profile = None
    last_result = None
    last_goal_for_distill = ""

    while True:
        try:
            prompt_tag = ""
            if last_profile and warm:
                tag = _cyan("code") if last_profile == "code" else _magenta("general")
                prompt_tag = f"[{tag}] "
            goal = input(f"{prompt_tag}{_bold('>>>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if not goal:
            continue
        if goal.lower() in ("exit", "quit", "q"):
            break
        if goal.lower() in ("?", "help"):
            _print_help()
            continue

        # /clear — reset session memory
        if goal.lower() in ("/clear", "clear"):
            session_active = False
            agent = None
            print(f"  {_dim('Session cleared.')}")
            continue

        # cd command — also clears session
        if goal.lower().startswith("cd "):
            new_path = Path(goal[3:].strip()).resolve()
            if new_path.is_dir():
                workspace = new_path
                os.chdir(str(workspace))
                session_active = False
                agent = None
                print(f"  {_dim('workspace:')} {workspace}")
            else:
                print(f"  {_red('not a directory:')} {new_path}")
            continue

        # /undo command
        if goal.lower() in ("/undo", "undo"):
            _do_undo(workspace)
            continue

        # /models — list available GGUFs
        if goal.lower() in ("/models", "/model"):
            _list_models()
            continue

        # /model <name> — switch to a specific GGUF
        if goal.lower().startswith("/model "):
            name = goal[7:].strip()
            _switch_model(name, mgr)
            agent = None
            session_active = False
            continue

        # /default <name> — set default model for future sessions
        if goal.lower().startswith("/default "):
            name = goal[9:].strip()
            _set_default_model(name)
            continue

        # /modelpath — manage model search directories
        if goal.lower().startswith("/modelpath"):
            args = goal[10:].strip()
            _manage_model_paths(args)
            continue

        # /diff command
        if goal.lower() in ("/diff", "diff"):
            _show_diff(workspace)
            continue

        # /commit command
        if goal.lower().startswith("/commit"):
            _do_commit(workspace, mgr, warm)
            continue

        # /context command
        if goal.lower().startswith("/context"):
            _load_context(goal, workspace)
            continue

        # /export — save session to markdown
        if goal.lower().startswith("/export"):
            _export_session(workspace, goal[7:].strip())
            continue

        # /paste — clipboard to context
        if goal.lower() == "/paste":
            clip = _clipboard_paste()
            if clip:
                _context_files["(clipboard)"] = clip
            continue

        # /copy — last answer to clipboard
        if goal.lower() == "/copy":
            if _last_answer:
                _clipboard_copy(_last_answer)
            else:
                print(f"  {_dim('No answer to copy yet.')}")
            continue

        # /review — code review current diff
        if goal.lower() in ("/review",):
            review_goal = _do_review(workspace)
            if review_goal:
                goal = review_goal  # fall through to normal goal processing
            else:
                continue

        # /review-deep — structured deep review against a base ref (default HEAD)
        if goal.lower().startswith("/review-deep"):
            base = goal[len("/review-deep"):].strip() or "HEAD"
            review_goal = _do_review_deep(workspace, base)
            if review_goal:
                goal = review_goal
            else:
                continue

        # /snippet — manage snippets
        if goal.lower().startswith("/snippet"):
            result = _manage_snippets(goal[8:].strip())
            if isinstance(result, str):
                _context_files["(snippet)"] = result
            continue

        # /stats — session statistics
        if goal.lower() in ("/stats",):
            _show_stats()
            continue

        # /autofix — test-fix loop
        if goal.lower().startswith("/autofix"):
            parts = goal.split()
            rounds = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            goal = _autofix_goal(rounds)
            # fall through to normal goal processing

        # /tdd — test-driven development loop
        if goal.lower().startswith("/tdd"):
            user_goal = goal[4:].strip()
            if not user_goal:
                print(f"  {_dim('Usage: /tdd <goal> — write tests first, then implement until they pass')}")
                continue
            goal = _tdd_goal(user_goal)
            # fall through to normal goal processing

        # /proof — self-diagnose-and-fix failing tests
        if goal.lower().startswith("/proof"):
            target = goal[len("/proof"):].strip()
            goal = _proof_goal(target)
            # fall through to normal goal processing

        # /distill — capture the LAST run as a trajectory artifact
        if goal.lower() in ("/distill",):
            if not last_result:
                print(f"  {_dim('No prior run to distill. Run a goal first.')}")
                continue
            try:
                from engine.trajectory_distiller import distill, save
                traj = distill(last_result, last_goal_for_distill)
                p = save(traj, _SELF)
                rel = p.relative_to(_SELF) if p.is_relative_to(_SELF) else p
                bucket = "examples" if traj.success else "review"
                meta = f"[intent={traj.intent} lang={traj.language} iters={traj.iterations} files={len(traj.files_touched)}]"
                print(f"  {_green('Distilled')} {bucket}: {rel}  {_dim(meta)}")
            except Exception as exc:
                print(f"  {_red('Distill failed:')} {exc}")
            continue

        # /library — single-pane status report on every learning artifact
        if goal.lower() in ("/library", "/lib"):
            try:
                from engine.library_status import collect, render_text
                s = collect(_SELF)
                print(f"  {_bold('Library status:')}")
                for line in render_text(s).splitlines():
                    print(f"  {line}")
            except Exception as exc:
                print(f"  {_red('Library status failed:')} {exc}")
            continue

        # /promote — copy validated auto-generated YAMLs from the review
        # queue into data/augmentor_examples/agentic/<category>/ so the
        # retrieval system picks them up on next ulcagent boot.
        if goal.lower().startswith("/promote"):
            args = goal[len("/promote"):].strip()
            try:
                from engine.augmentor_promoter import (
                    promote_all, promote_category, list_review_queue, summarize,
                )
            except Exception as exc:
                print(f"  {_red('Promoter not importable:')} {exc}")
                continue
            queue = list_review_queue(_SELF)
            if not queue:
                print(f"  {_dim('No YAMLs in data/auto_generated_review/.')}")
                continue
            if not args or args == "--all":
                results = promote_all(_SELF)
            elif args.startswith("--list"):
                print(f"  {_bold('Review queue:')}")
                for cat, files in queue.items():
                    print(f"    {cat}: {len(files)} file(s)")
                continue
            else:
                results = promote_category(args, _SELF)
            counts = summarize(results)
            print(f"  {_green('Promoted:')} {counts['promoted']}, "
                  f"{_yellow('skipped:')} {counts['skipped']}, "
                  f"{_red('rejected:')} {counts['rejected']}")
            for r in results:
                if r.promoted:
                    rel = r.target.relative_to(_SELF) if r.target and r.target.is_relative_to(_SELF) else r.target
                    print(f"    {_dim('OK   ')} {r.source.name} -> {rel}")
                elif "validation failed" in r.reason:
                    print(f"    {_red('REJECT')} {r.source.name}: {r.reason}")
                else:
                    print(f"    {_dim('SKIP ')} {r.source.name}: {r.reason}")
            continue

        # /flag-errors — analyze the LAST run for known failure patterns and
        # synthesize YAML augmentor entries that demonstrate the correct fix.
        if goal.lower() in ("/flag-errors", "/flag"):
            if not last_result:
                print(f"  {_dim('No prior run to analyze. Run a goal first.')}")
                continue
            try:
                from engine.failure_flagger import flag, summarize
                from engine.yaml_augmentor_builder import write_all
                records = flag(last_result, last_goal_for_distill)
                if not records:
                    print(f"  {_green('No known failure patterns detected.')}")
                    continue
                summary = summarize(records)
                summary_str = ", ".join(f"{k}×{v}" for k, v in summary.items())
                print(f"  {_yellow('Detected:')} {summary_str}")
                paths = write_all(records, last_goal_for_distill, _SELF)
                if paths:
                    print(f"  {_green('Wrote')} {len(paths)} augmentor YAML file(s) to data/auto_generated_review/:")
                    for p in paths:
                        rel = p.relative_to(_SELF) if p.is_relative_to(_SELF) else p
                        print(f"    {_dim('->')} {rel}")
                    print(f"  {_dim('Review and move to data/augmentor_examples/<domain>/ to enable retrieval.')}")
                else:
                    print(f"  {_dim('No YAML templates available for the detected categories.')}")
            except Exception as exc:
                print(f"  {_red('Flag failed:')} {exc}")
            continue

        # /watch — file watcher
        if goal.lower().startswith("/watch"):
            action = goal[6:].strip() or "test"
            _watch_loop(workspace, action, mgr, warm, _build_agent)
            continue

        # /batch — run goals from file
        if goal.lower().startswith("/batch"):
            filepath = goal[6:].strip()
            if filepath:
                _run_batch(filepath, workspace, mgr, warm, _build_agent)
            else:
                print(f"  {_dim('Usage: /batch goals.txt')}")
            continue

        # /docs — generate documentation
        if goal.lower().startswith("/docs"):
            doc_type = goal[5:].strip().lower() or "readme"
            if doc_type in _DOC_GOALS:
                goal = _DOC_GOALS[doc_type]
                # fall through to normal goal processing
            else:
                print(f"  {_dim('Usage: /docs readme | /docs api | /docs arch')}")
                continue

        # /plugins — list loaded plugins
        if goal.lower() in ("/plugins",):
            if _PLUGINS_DIR.exists():
                plugins = [p.stem for p in _PLUGINS_DIR.glob("*.py") if not p.name.startswith("_")]
                if plugins:
                    print(f"  {_bold('Plugins:')} {', '.join(plugins)}")
                else:
                    print(f"  {_dim('No plugins in plugins/')}")
            else:
                print(f"  {_dim('No plugins/ directory. Create it and add .py files with a register(registry) function.')}")
            continue

        # Check aliases (built-in + project-specific)
        aliases = _load_aliases(workspace)
        if goal.lower() in aliases:
            goal = aliases[goal.lower()]

        # Profile override: /code or /general prefix
        forced_profile = None
        if goal.startswith("/code "):
            forced_profile = "code"
            goal = goal[6:].strip()
        elif goal.startswith("/general "):
            forced_profile = "general"
            goal = goal[9:].strip()

        # Auto-detect or use forced profile
        profile = forced_profile or _detect_profile(goal)
        profile_label = _cyan("code") if profile == "code" else _magenta("general")

        # If profile changed, reset session (different model = different conversation)
        if profile != last_profile:
            session_active = False
            agent = None

        # Load/swap model if needed
        if not warm:
            print(f"  {profile_label} ", end="", flush=True)
            mgr.ensure_profile(profile)
        else:
            mgr.ensure_profile(profile)

        last_profile = profile

        # Build agent on first goal or after clear/profile-swap
        if agent is None:
            agent = _build_agent(mgr, workspace)
            # Inject project index into first run
            _inject_project_index(agent, workspace)
            # Inject project rules from .ulcagent file
            rules = _load_project_rules(workspace)
            if rules:
                agent.system_prompt_extra += f"\n\nProject rules (.ulcagent):\n{rules}"
            # Inject any loaded /context files
            if _context_files:
                ctx_block = "\n\nLoaded context files:\n"
                for fname, content in _context_files.items():
                    ctx_block += f"\n--- {fname} ---\n{content[:3000]}\n"
                agent.system_prompt_extra += ctx_block

        # Snapshot workspace for /undo
        _snapshot_workspace(workspace)

        # Log user goal
        _session_log.append({"role": "user", "content": goal})

        # Mode routing (per-goal): handheld > architect > flat
        run_target = agent
        if handheld_force_on:
            print(f"  {_dim('[handheld driver — plan + per-step with prior-state injection]')}")
            try:
                run_target = _build_handheld_driver(agent)
            except Exception as exc:
                print(f"  {_yellow('handheld setup failed; falling back to flat:')} {exc}")
        elif _should_use_architect(goal, arch_force_on, arch_force_off):
            print(f"  {_dim('[architect mode — plan-then-execute]')}")
            try:
                run_target = _build_architect_agent(agent)
            except Exception as exc:
                print(f"  {_yellow('architect setup failed; falling back to flat:')} {exc}")

        # Run with session memory (architect agents ignore continue_session
        # — each step is a fresh sub-agent, which is the whole point)
        if run_target is agent:
            last_result = _run_one(agent, goal, continue_session=session_active)
        else:
            last_result = _run_one(run_target, goal, continue_session=False)
        last_goal_for_distill = goal  # preserved across iterations for /distill
        session_active = True

        # Track stats + log answer
        if last_result:
            _stats["goals"] += 1
            _stats["iterations"] += last_result.iterations
            _stats["tool_calls"] += len(last_result.tool_calls)
            _stats["wall_time"] += last_result.wall_time
            transcript_chars = sum(len(t.get("content", "")) for t in last_result.transcript)
            pct = transcript_chars / 52000 * 100
            if pct > _stats["ctx_peak"]:
                _stats["ctx_peak"] = pct
            answer = (last_result.final_answer or "").strip()
            _update_last_answer(answer)
            stats_str = f"{last_result.iterations} iter, {len(last_result.tool_calls)} calls, {last_result.wall_time:.1f}s"
            _session_log.append({"role": "agent", "content": answer, "stats": stats_str})

            # Auto-flag-on-fail: silently scan the run for known failure
            # patterns and write YAML augmentor entries to
            # data/auto_generated_review/ (OUTSIDE the retrieval scan path).
            # No-op on clean runs. Disable via --no-auto-flag CLI flag.
            if not auto_flag_off:
                _maybe_auto_flag(last_result, goal)

        # Post-task suggestion
        suggestion = _suggest_next(last_result)
        if suggestion:
            print(f"  {_dim(f'Suggestion: {suggestion}')} {_dim('(press Enter to accept, or type something else)')}")

        if not warm:
            mgr.unload()

    mgr.unload()
    print("Goodbye.")


if __name__ == "__main__":
    main()
