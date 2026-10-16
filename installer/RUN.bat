@echo off
cd /d "%~dp0"
if exist "about.nfo" type "about.nfo"
if not exist "GGF-Klein\.venv\Scripts\python.exe" (
  echo Run INSTALL.bat first.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
"GGF-Klein\.venv\Scripts\python.exe" "GGF-Klein\app.py"
if errorlevel 1 pause
