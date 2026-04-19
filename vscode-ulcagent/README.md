# ulcagent - VS Code Extension

Local AI coding agent, powered by ultralight-coder. 100% local, zero cloud.

## Install

**Option A -- Dev mode (recommended):**
1. Open this folder in VS Code.
2. Press **F5** to launch an Extension Development Host.

**Option B -- Manual install:**
Copy the `vscode-ulcagent` folder into your extensions directory:
- Windows: `%USERPROFILE%\.vscode\extensions\ulcagent`
- macOS/Linux: `~/.vscode/extensions/ulcagent`

Restart VS Code after copying.

## Configure

Open **Settings** (Ctrl+,) and set:

| Setting | Required | Description |
|---------|----------|-------------|
| `ulcagent.agentPath` | Yes | Absolute path to `ulcagent.py` (e.g. `D:\LLCWork\ultralight-coder\ulcagent.py`) |
| `ulcagent.pythonPath` | No | Path to your Python interpreter (default: `python`) |

## Usage

### Right-click context menu
1. Highlight code in any file (or leave nothing selected to use the entire file).
2. Right-click and choose:
   - **Ask ulcagent** -- type a custom goal (e.g. "Add error handling")
   - **Fix with ulcagent** -- auto-sends "Fix this code" with the selection
   - **Explain with ulcagent** -- auto-sends "Explain this code" with the selection

### Command palette
1. Press **Ctrl+Shift+P** (or Cmd+Shift+P on macOS).
2. Type `ulcagent` and pick a command.

## How It Works

The extension spawns `ulcagent.py` in one-shot mode with your goal and code as input.
Results appear in the **ulcagent** output channel. A progress notification shows while
the model is running.

## Requirements

- Python 3.10+
- ultralight-coder with a downloaded GGUF model
- A GPU is strongly recommended (CPU inference is slow on 14B models)
