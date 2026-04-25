# ulcagent User Guide

A step-by-step guide from first launch to daily driver.

## First Launch

Open PowerShell, go to any project, type `ulcagent`:

```
PS> cd D:\LLCWork\myproject
PS> ulcagent
ulcagent - adaptive local agent
  workspace: D:\LLCWork\myproject
  myproject: clean working tree | last commit: a3f1b2c Fix auth
  Type a goal and press Enter. '?' for help. Ctrl+C to cancel. 'exit' to quit.

>>>
```

That's it. You're in. The `>>>` prompt is waiting for you.

## Your First 5 Minutes

Type these one at a time. Read each answer before moving on.

```
>>> ?
```
Shows help. Skim it — you'll come back to it later.

```
>>> List the files in this project
```
The agent reads your directory and tells you what's there.

```
>>> Read the main entry point and summarize it
```
Now you see your project through the agent's eyes.

```
>>> Find any TODO comments in the code
```
Quick discovery — find what needs work.

```
>>> /diff
```
Check if anything changed (nothing should have — we only read files).

**Congratulations.** You just used ulcagent. Everything else builds on this.

## Making Changes

Once you're comfortable reading, start editing:

```
>>> Fix the typo on line 12 of utils.py
```

```
>>> Add a --verbose flag to cli.py that sets logging to DEBUG
```

```
>>> Read test_auth.py, find why it's failing, and fix it
```

After any change:
```
>>> /diff
```
See exactly what changed. If it looks right:
```
>>> /commit
```
If it looks wrong:
```
>>> /undo
```

## The Golden Rule

**Be specific. Name the file. Describe the change.**

| Works well | Doesn't work well |
|---|---|
| "Read server.py and add rate limiting to the /login endpoint" | "Make the server better" |
| "Fix the import error in utils/db.py" | "Something is broken" |
| "Add a pytest test for the calculate_tax function" | "Add tests" |
| "Rename the function 'process' to 'handle_request' in all .py files" | "Refactor the code" |

You don't need to know the exact fix — just point the agent at the right file and describe what you want. It figures out the HOW.

## Session Memory

Goals build on each other within a session:

```
>>> Read server.py
  (reads the file, shows you the content)

>>> What framework is it using?
  (remembers the file from the last goal — doesn't need to re-read)

>>> Add rate limiting to the main endpoint
  (knows which file and which endpoint from context)
```

The context meter `[ctx: 45%]` shows how full the window is. When it gets above 70%, you'll see a warning. Type `/clear` to reset — your project notes (cross-session memory) are never lost.

## Daily Workflows

### Fix a bug
```
>>> Read the error traceback and fix it:
    File "auth.py", line 42, in login
    TypeError: 'NoneType' has no attribute 'email'
```

### Add a feature
```
>>> Read app.py. Add a /health endpoint that returns {"status": "ok", "uptime": seconds_since_start}
>>> Run the tests to make sure nothing broke
>>> /diff
>>> /commit
```

### Code review
```
>>> /review
```
Reviews your uncommitted changes for bugs and security issues.

### Test-fix loop
```
>>> /autofix
```
Runs tests, reads failures, fixes them, re-runs — loops until green.

### Explore a new codebase
```
>>> List all files and describe the project structure
>>> Read the main entry point and explain how it works
>>> Find all API endpoints and list them with their HTTP methods
```

### Multi-file context
```
>>> /context server.py routes.py models.py
>>> How does the auth flow work across these files?
```

### Security audit
```
>>> Run python D:\LLCWork\security-toolkit\runner.py secrets . and summarize
>>> Run python D:\LLCWork\security-toolkit\runner.py headers https://localhost:8000
```

## Commands Reference (Quick)

### Essential (use these daily)
| Command | What it does |
|---|---|
| `?` | Show help |
| `/diff` | Show what changed |
| `/commit` | Save changes to git |
| `/undo` | Revert last change |
| `/clear` | Reset conversation (project notes persist) |

### File & context
| Command | What it does |
|---|---|
| `/context file1 file2` | Load files for cross-file reasoning |
| `/context clear` | Clear loaded files |
| `/review` | Code review current changes |
| `/paste` | Send clipboard as context |
| `/copy` | Copy last answer to clipboard |

### Testing & automation
| Command | What it does |
|---|---|
| `/test` | Run pytest |
| `/lint` | Run linter |
| `/format` | Run formatter |
| `/autofix` | Test → fix → re-test loop |
| `/watch test` | Auto-run tests on file save |
| `/batch goals.txt` | Run goals from a file |

### Models & profiles
| Command | What it does |
|---|---|
| `/models` | List available GGUF models |
| `/model code coder-14` | Set code profile's model |
| `/model general phi-3.5` | Set general profile's model |
| `/modelpath add D:\Models` | Add a GGUF search directory |

### Documentation & output
| Command | What it does |
|---|---|
| `/docs readme` | Generate a README |
| `/docs api` | Generate API docs |
| `/export` | Save session to markdown |
| `/snippet save name` | Save last answer as template |
| `/snippet name` | Load a saved template |
| `/stats` | Show session statistics |

### Learning
| Command | What it does |
|---|---|
| `/learn` | Teach it a correction ("don't do X, do Y") |
| `/learn list` | See stored corrections |

## Three Ways to Use It

### 1. Terminal (default)
```
PS> ulcagent
>>> your goal here
```

### 2. Browser
```
PS> python D:\LLCWork\ultralight-coder\web_agent.py
```
Opens a chat UI at http://localhost:8899. Same features, visual interface.

### 3. VS Code
Install the extension from `vscode-ulcagent/`. Then:
- Highlight code → right-click → **Ask ulcagent** / **Fix** / **Explain**
- Or: Ctrl+Shift+P → "ulcagent"

## Project Rules (.ulcagent file)

Drop a `.ulcagent` file in any project root:

```
Use pytest for all tests.
Always add type hints to function signatures.
Prefer f-strings over .format().

[aliases]
/deploy = Run the deploy script via run_bash
/check = Run mypy and report type errors
```

The instructions are injected into every goal. Custom aliases appear as commands.

## Startup Flags

```
PS> ulcagent --warm         # keep model in memory (instant, uses ~10GB VRAM)
PS> ulcagent --extended     # enable 21 tools (git, rename, checkpoint — for guided work)
```

Default mode unloads the model between goals (4s reload, 0 VRAM at prompt). Use `--warm` when you're in a focused coding session.

## Plugins

Drop a `.py` file in `plugins/` with a `register(registry)` function:

```python
# plugins/my_tool.py
def register(registry):
    from engine.agent_tools import ToolSchema
    registry.register(ToolSchema(
        name="my_tool",
        description="Does something useful",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}},
        function=lambda input: f"Result: {input}",
        category="plugin",
    ))
```

The tool is available immediately on next launch.

## Known Limits

| Limit | Workaround |
|---|---|
| Can't create files over ~60 lines in one shot | Scaffold the file yourself, then ask ulcagent for targeted edits |
| Files over ~200 lines use a lot of context | Use `read_function` (--extended) or read with offset/limit |
| 16k token context window | Watch the `[ctx: %]` meter, `/clear` when high |
| ~1-4 min per task | The model thinks at ~20 tok/s — be patient or add "be brief" |
| Only Qwen models work reliably | Hermes tool-call format is Qwen-native; other models need adapters |

## Troubleshooting

**"No response" or empty answer:**
The model may have emitted a malformed tool call. Try rephrasing with more specific file names.

**Model not found:**
Run `python setup_ulcagent.py` to download a model, or place a GGUF in `models/`.

**Slow first goal:**
The model loads in ~4s. Subsequent goals in `--warm` mode are instant.

**Context meter at 100%:**
Type `/clear` to reset. Your project notes (cross-session memory) are preserved.

**ulcagent command not found:**
Restart PowerShell after setup, or run directly: `python D:\LLCWork\ultralight-coder\ulcagent.py`

## Security Toolkit

Separate from ulcagent — standalone tools for security testing:

```
PS> sectool ?                                    # see all 17 tools
PS> sectool ports mysite.com                     # port scan
PS> sectool headers https://mysite.com           # header audit
PS> sectool secrets D:\LLCWork\myproject         # secret scan
PS> sectool report --target https://mysite.com   # full HTML report
```

See `D:\LLCWork\security-toolkit\README.md` for the full guide.
