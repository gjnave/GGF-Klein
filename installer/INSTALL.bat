@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Get Going Fast - GGF Flux 3 Klein - Dream Setup
if exist "disclaimer.md" type "disclaimer.md"
if not defined GGF_UNATTENDED pause
if exist "about.nfo" type "about.nfo"
set "PYTHONUTF8=1"
set "APP=%CD%\GGF-Klein"
set "CACHE=%CD%\installer-cache"
if not exist "%CACHE%" mkdir "%CACHE%"
if not defined GGF_SOURCE_PRIMARY set "GGF_SOURCE_PRIMARY=https://codeberg.org/Cognibuild/GGF-Klein/archive/main.zip"
if not defined GGF_SOURCE_SECONDARY set "GGF_SOURCE_SECONDARY=https://github.com/gjnave/GGF-Klein/archive/refs/heads/main.zip"
call :source "%GGF_SOURCE_PRIMARY%"
if not errorlevel 1 goto source_ready
echo Codeberg unavailable. Trying GitHub...
call :source "%GGF_SOURCE_SECONDARY%"
if errorlevel 1 goto failed
:source_ready
if exist "%APP%\app.py" (
  echo Existing app detected. Close it before continuing. Data and models will be preserved.
  if not defined GGF_UNATTENDED pause
)
if not exist "%APP%" mkdir "%APP%"
if not exist "%APP%\runtime" mkdir "%APP%\runtime"
if exist "%APP%\runtime\uv.exe" goto uv_ready
curl.exe --fail --location --retry 3 --output "%CACHE%\uv-0.12.13.zip" "https://github.com/astral-sh/uv/releases/download/0.12.13/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 goto failed
certutil -hashfile "%CACHE%\uv-0.12.13.zip" SHA256 | findstr /i /c:"a86c9dc7bad9b03f388583b7187c05fe9951c2e0d392217e8fd43d97787f6ec2" >nul
if errorlevel 1 goto failed
tar.exe -xf "%CACHE%\uv-0.12.13.zip" -C "%APP%\runtime" uv.exe
if errorlevel 1 goto failed
:uv_ready
set "UV_PYTHON_INSTALL_DIR=%APP%\runtime\python"
set "UV_PYTHON_DOWNLOADS=automatic"
set "VIRTUAL_ENV="
if exist "%APP%\.venv\Scripts\python.exe" goto venv_ready
"%APP%\runtime\uv.exe" venv --managed-python --python 3.12 "%APP%\.venv"
if errorlevel 1 goto failed
:venv_ready
set "PY=%APP%\.venv\Scripts\python.exe"
rem Use the staged updater to validate every archive path and back up overwritten source.
"%PY%" "%SOURCE_ROOT%\bootstrap_source.py" "%SOURCE_ROOT%" "%APP%"
if errorlevel 1 goto failed
"%PY%" "%APP%\setup_environment.py"
if errorlevel 1 goto failed
"%PY%" "%APP%\preflight.py"
if errorlevel 1 goto failed
rem Source was already downloaded and installed above. Do not run an updater here:
rem an unrelated update failure must not prevent the required model installation.
echo.
echo Installing all four REQUIRED models. Setup is not complete until they verify.
echo Accept access at https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
echo Models download directly from the authorized repository, not from GGF.
if defined HF_TOKEN goto download
"%PY%" -c "from huggingface_hub import get_token;raise SystemExit(0 if get_token() else 1)"
if not errorlevel 1 goto download
echo Sign in with a Hugging Face read token from the account granted access.
"%APP%\.venv\Scripts\hf.exe" auth login
if errorlevel 1 goto failed
:download
"%PY%" "%APP%\model_manager.py" --directory "%APP%\models" --components image text vae upsampler
if errorlevel 1 goto failed
"%PY%" "%APP%\model_manager.py" --directory "%APP%\models" --verify-only
if errorlevel 1 goto failed
:complete
echo Setup complete. All four required models are installed and verified.
echo Double-click RUN.bat. Phone access and optional models are in Settings.
if not defined GGF_UNATTENDED pause
exit /b 0
:source
set "SOURCE_ROOT="
set "STAGE=%CACHE%\source-%RANDOM%-%RANDOM%"
mkdir "%STAGE%"
curl.exe --fail --location --retry 2 --connect-timeout 20 --output "%STAGE%\source.zip" "%~1"
if errorlevel 1 exit /b 1
rem Inspect paths before extracting untrusted archive entries.
tar.exe -tf "%STAGE%\source.zip" | findstr /r /c:"\.\./" /c:"^/" /c:"^\\" >nul
if not errorlevel 1 exit /b 1
tar.exe -xf "%STAGE%\source.zip" -C "%STAGE%"
if errorlevel 1 exit /b 1
for /d %%D in ("%STAGE%\*") do if exist "%%D\app.py" if exist "%%D\worker.py" if exist "%%D\bootstrap_source.py" if exist "%%D\release_sources.json" set "SOURCE_ROOT=%%D"
if not defined SOURCE_ROOT exit /b 1
exit /b 0
:failed
echo Setup stopped. Existing files, models, images and partial downloads are preserved.
if not defined GGF_UNATTENDED pause
exit /b 1
