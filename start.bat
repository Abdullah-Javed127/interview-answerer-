@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo   Groq Interview Voice Agent - Start
echo ==========================================================
echo.

REM ── Find the right Python executable ──
set "PYTHON_CMD="

REM Priority 1: venv Python (created by install.bat with system Python)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    goto :run
  )
  echo [WARN] .venv Python is broken. Checking for portable Python...
)

REM Priority 2: embedded portable Python (created by install.bat auto-download)
if exist ".python\python.exe" (
  ".python\python.exe" --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=.python\python.exe"
    goto :run
  )
  echo [WARN] Portable Python is broken.
)

echo [ERROR] No working Python found.
echo Please run install.bat first.
pause
exit /b 1

:run
if not exist "config.json" (
  if exist "config.json.example" (
    copy /Y "config.json.example" "config.json" >nul
  )
)

echo Launching interview agent...
call "%PYTHON_CMD%" main.py

if errorlevel 1 (
  echo.
  echo Agent exited with an error.
)

pause
