@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ==========================================================
echo   Groq Interview Voice Agent - Installer
echo ==========================================================
echo.

REM ── Step 0: Find a working Python 3 executable ──
set "PYTHON_CMD="

REM Try: py launcher
where py >nul 2>nul
if not errorlevel 1 (
  py -3 --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :found_python
  )
)

REM Try: python3 on PATH
where python3 >nul 2>nul
if not errorlevel 1 (
  python3 --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=python3"
    goto :found_python
  )
)

REM Try: python on PATH
where python >nul 2>nul
if not errorlevel 1 (
  python --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :found_python
  )
)

REM Try: common install locations
for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  "C:\Python313\python.exe"
  "C:\Python312\python.exe"
  "C:\Python311\python.exe"
  "C:\Python310\python.exe"
  "%ProgramFiles%\Python313\python.exe"
  "%ProgramFiles%\Python312\python.exe"
  "%ProgramFiles%\Python311\python.exe"
  "%ProgramFiles%\Python310\python.exe"
) do (
  if exist %%~P (
    set "PYTHON_CMD=%%~P"
    goto :found_python
  )
)

REM Try: our own embedded Python from a previous install
if exist ".python\python.exe" (
  set "PYTHON_CMD=%~dp0.python\python.exe"
  goto :found_python
)

REM ── No Python found anywhere — download portable Python ──
echo [INFO] Python not found on this system. Downloading portable Python...
echo.

set "PY_VERSION=3.12.8"
set "PY_ZIP=python-%PY_VERSION%-embed-amd64.zip"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/%PY_ZIP%"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "EMBED_DIR=%~dp0.python"

REM Download Python embeddable zip
echo Downloading Python %PY_VERSION% portable...
powershell -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ZIP%' -UseBasicParsing } catch { Write-Error $_.Exception.Message; exit 1 }"
if errorlevel 1 (
  echo [ERROR] Failed to download Python. Check your internet connection.
  pause
  exit /b 1
)

REM Extract
echo Extracting Python...
if exist "%EMBED_DIR%" rmdir /s /q "%EMBED_DIR%"
powershell -NoProfile -Command "try { Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%EMBED_DIR%' -Force } catch { Write-Error $_.Exception.Message; exit 1 }"
if errorlevel 1 (
  echo [ERROR] Failed to extract Python.
  pause
  exit /b 1
)
del /q "%PY_ZIP%" 2>nul

REM Enable pip/site-packages: uncomment "import site" in the ._pth file
for %%F in ("%EMBED_DIR%\python*._pth") do (
  powershell -NoProfile -Command "(Get-Content '%%F') -replace '^#import site','import site' | Set-Content '%%F'"
)

REM Download and run get-pip.py
echo Installing pip into portable Python...
powershell -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile '%EMBED_DIR%\get-pip.py' -UseBasicParsing } catch { Write-Error $_.Exception.Message; exit 1 }"
if errorlevel 1 (
  echo [ERROR] Failed to download pip installer.
  pause
  exit /b 1
)
"%EMBED_DIR%\python.exe" "%EMBED_DIR%\get-pip.py" --no-warn-script-location
if errorlevel 1 (
  echo [ERROR] Failed to install pip.
  pause
  exit /b 1
)
del /q "%EMBED_DIR%\get-pip.py" 2>nul

set "PYTHON_CMD=%EMBED_DIR%\python.exe"
echo.
echo [OK] Portable Python %PY_VERSION% installed to .python\ folder.
echo.
goto :skip_venv

:found_python
echo Found Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM ── Step 1: Create venv (only when using system Python) ──
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    echo Trying portable Python fallback instead...
    goto :skip_venv_fallback
  )
) else (
  echo [1/4] Virtual environment already exists.
)

echo [2/4] Upgrading pip...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo [WARN] pip upgrade failed, continuing...
)

echo [3/4] Installing dependencies...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  echo If this is your first run, check internet/firewall and try again.
  pause
  exit /b 1
)
goto :finish_install

:skip_venv_fallback
REM If venv creation failed with system Python, we still have PYTHON_CMD
REM Fall through to direct install

:skip_venv
REM ── Direct install (for embedded Python, no venv) ──
echo [2/4] Upgrading pip...
"%PYTHON_CMD%" -m pip install --upgrade pip --no-warn-script-location
if errorlevel 1 (
  echo [WARN] pip upgrade failed, continuing...
)

echo [3/4] Installing dependencies...
"%PYTHON_CMD%" -m pip install -r requirements.txt --no-warn-script-location
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  echo If this is your first run, check internet/firewall and try again.
  pause
  exit /b 1
)

:finish_install
if not exist "config.json" (
  if exist "config.json.example" (
    echo [4/4] Creating config.json from template...
    copy /Y "config.json.example" "config.json" >nul
  ) else (
    echo [4/4] No config template found, skipping.
  )
) else (
  echo [4/4] config.json already exists (kept as-is).
)

echo.
echo ========================================
echo   Install complete!
echo ========================================
echo Next steps:
echo   1) Double-click start.bat
echo   2) Fill details in the setup window and click Start Interview
echo.
pause
