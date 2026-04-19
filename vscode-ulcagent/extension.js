const vscode = require("vscode");
const { spawn } = require("child_process");

/** Strip ANSI escape codes from a string. */
function stripAnsi(str) {
  return str.replace(
    /[\u001b\u009b][[\]()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nq-uy=><~]/g,
    ""
  );
}

/** Get selected text or entire file content. */
function getCode(editor) {
  const sel = editor.selection;
  if (!sel.isEmpty) return editor.document.getText(sel);
  return editor.document.getText();
}

/** Resolve the ulcagent.py path from settings. */
function agentPath() {
  const cfg = vscode.workspace.getConfiguration("ulcagent");
  const p = cfg.get("agentPath", "");
  if (!p) {
    vscode.window.showErrorMessage(
      'ulcagent: Set "ulcagent.agentPath" in settings to the absolute path of ulcagent.py.'
    );
    return null;
  }
  return p;
}

/** Run ulcagent as a subprocess and return stdout. */
function runAgent(goal, cwd) {
  return new Promise((resolve, reject) => {
    const cfg = vscode.workspace.getConfiguration("ulcagent");
    const python = cfg.get("pythonPath", "python");
    const script = agentPath();
    if (!script) return reject(new Error("agentPath not configured"));

    const proc = spawn(python, [script, goal], {
      cwd: cwd || undefined,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
      timeout: 300000,
    });

    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d) => (stdout += d.toString()));
    proc.stderr.on("data", (d) => (stderr += d.toString()));

    proc.on("error", (err) => {
      if (err.code === "ENOENT") {
        reject(new Error(`Python not found at "${python}". Check ulcagent.pythonPath.`));
      } else {
        reject(err);
      }
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        const msg = stripAnsi(stderr || stdout).trim();
        reject(new Error(`ulcagent exited with code ${code}:\n${msg}`));
      } else {
        resolve(stripAnsi(stdout).trim());
      }
    });
  });
}

/** Show result in an output channel and reveal it. */
function showResult(channel, label, text) {
  channel.clear();
  channel.appendLine(`--- ${label} ---\n`);
  channel.appendLine(text);
  channel.show(true);
}

/** Core handler: build goal, run agent, display result. */
async function handleCommand(goalPrefix, label) {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("ulcagent: Open a file first.");
    return;
  }
  if (!agentPath()) return;

  const code = getCode(editor);
  let goal;
  if (goalPrefix === null) {
    const input = await vscode.window.showInputBox({
      prompt: "What should ulcagent do?",
      placeHolder: "e.g. Add error handling to this function",
    });
    if (!input) return;
    goal = `${input}\n\n--- context ---\n${code}`;
  } else {
    goal = `${goalPrefix}\n\n${code}`;
  }

  const cwd =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || undefined;
  const channel = vscode.window.createOutputChannel("ulcagent");

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `ulcagent: ${label}...`,
      cancellable: false,
    },
    async () => {
      try {
        const result = await runAgent(goal, cwd);
        showResult(channel, label, result);
      } catch (err) {
        vscode.window.showErrorMessage(`ulcagent error: ${err.message}`);
      }
    }
  );
}

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("ulcagent.ask", () =>
      handleCommand(null, "Ask ulcagent")
    ),
    vscode.commands.registerCommand("ulcagent.fix", () =>
      handleCommand("Fix this code:", "Fix with ulcagent")
    ),
    vscode.commands.registerCommand("ulcagent.explain", () =>
      handleCommand("Explain this code:", "Explain with ulcagent")
    )
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
