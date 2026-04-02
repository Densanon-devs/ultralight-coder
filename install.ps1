# Ultralight Code Assistant — Windows Installer
#
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/densanon-devs/ultralight-coder/master/install.ps1 | iex
#
# Or manually:
#   .\install.ps1
#   .\install.ps1 -NoModel       # Skip model download
#   .\install.ps1 -GPU           # Install GPU-accelerated llama-cpp-python

param(
    [switch]$NoModel,
    [switch]$GPU
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  Ultralight Code Assistant - Installer" -ForegroundColor White
Write-Host "  Local AI coding assistant, 12 languages, 469MB model" -ForegroundColor DarkGray
Write-Host ""

# Check Python
$py = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) { $py = $cmd; break }
    } catch {}
}

if (-not $py) {
    Write-Host "  Error: Python 3.10+ is required but not found." -ForegroundColor Red
    Write-Host "  Install from https://python.org"
    exit 1
}

$pyVersion = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pyMajor = & $py -c "import sys; print(sys.version_info.major)"
$pyMinor = & $py -c "import sys; print(sys.version_info.minor)"

if ([int]$pyMajor -lt 3 -or ([int]$pyMajor -eq 3 -and [int]$pyMinor -lt 10)) {
    Write-Host "  Error: Python 3.10+ required (you have $pyVersion)" -ForegroundColor Red
    exit 1
}
Write-Host "  Python $pyVersion OK" -ForegroundColor Green

# Clone
$installDir = "ultralight-coder"
if (Test-Path $installDir) {
    Write-Host "  Directory $installDir already exists, pulling latest..."
    Set-Location $installDir
    git pull --ff-only
} else {
    Write-Host "  Cloning repository..."
    git clone https://github.com/densanon-devs/ultralight-coder.git
    Set-Location $installDir
}

# Install deps (use pre-built wheels to avoid 20min C++ compile)
Write-Host ""
Write-Host "  Installing dependencies..."
if ($GPU) {
    Write-Host "  (Using pre-built CUDA wheels)" -ForegroundColor DarkGray
    & $py -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --quiet --force-reinstall
} else {
    Write-Host "  (Using pre-built CPU wheels)" -ForegroundColor DarkGray
    & $py -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --quiet
}
& $py -m pip install -r requirements.txt --quiet
Write-Host "  Dependencies installed" -ForegroundColor Green

# Download model
if (-not $NoModel) {
    Write-Host ""
    Write-Host "  Downloading default model (Qwen 0.5B, 469MB)..."
    & $py download_model.py --model coder-0.5b
    Write-Host "  Model ready" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  Skipping model download (-NoModel)" -ForegroundColor DarkGray
    Write-Host "  Run later: python download_model.py"
}

# Done
Write-Host ""
Write-Host "  Setup complete!" -ForegroundColor White
Write-Host ""
Write-Host "  Start the web UI:"
Write-Host "    cd $installDir"
Write-Host "    $py launch.py"
Write-Host ""
Write-Host "  Then open http://localhost:8000 in your browser." -ForegroundColor Green
Write-Host ""
