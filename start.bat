@echo off
title Ultralite Code Assistant
echo.
echo   Ultralite Code Assistant
echo   ========================
echo.

set "ROOT=%~dp0"
set "PYDIR=%ROOT%python"
set "PYTHON=%PYDIR%\python.exe"
set "MODELS=%ROOT%models"

:: Check if portable Python exists
if exist "%PYTHON%" goto :have_python

echo   Downloading portable Python 3.12 (no install needed)...
echo.
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile '%ROOT%python.zip' }"
if not exist "%ROOT%python.zip" (
    echo   Error: Failed to download Python.
    echo   Check your internet connection and try again.
    pause
    exit /b 1
)

echo   Extracting Python...
powershell -Command "Expand-Archive -Path '%ROOT%python.zip' -DestinationPath '%PYDIR%' -Force"
del "%ROOT%python.zip"

:: Enable pip in embedded Python (uncomment import site in ._pth file)
for %%f in ("%PYDIR%\python*._pth") do (
    powershell -Command "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
)

:: Install pip
echo   Installing pip...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYDIR%\get-pip.py' }"
"%PYTHON%" "%PYDIR%\get-pip.py" --quiet
del "%PYDIR%\get-pip.py"
echo   pip installed.
echo.

:have_python
echo   Python: %PYTHON%

:: Check if deps are installed (use fastapi as marker)
"%PYTHON%" -c "import fastapi" 2>nul
if %errorlevel%==0 goto :have_deps

echo.
echo   Installing dependencies (first run only, may take a few minutes)...
echo.
"%PYTHON%" -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --quiet 2>nul
"%PYTHON%" -m pip install -r "%ROOT%requirements.txt" --quiet
echo.
echo   Dependencies installed.

:have_deps

:: Check if model exists
set "HAS_MODEL=0"
if exist "%MODELS%\*.gguf" set "HAS_MODEL=1"

if "%HAS_MODEL%"=="1" goto :have_model

echo.
echo   No model found. Downloading default model (Qwen 0.5B, 469MB)...
echo   This is a one-time download.
echo.
"%PYTHON%" "%ROOT%download_model.py" --model coder-0.5b
echo.

:have_model

:: Launch
echo.
echo   Starting Ultralite Code Assistant...
echo   Close this window to stop.
echo.
"%PYTHON%" "%ROOT%desktop.py"
if %errorlevel% neq 0 (
    echo.
    echo   Desktop mode not available, starting in browser...
    echo   Open http://localhost:8000 in your browser.
    echo.
    "%PYTHON%" "%ROOT%server.py"
)

pause
