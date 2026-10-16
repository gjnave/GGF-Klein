@echo off
cd /d "%~dp0"
if not exist "GGF-Klein\.venv\Scripts\python.exe" (
  echo Run INSTALL.bat first.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
"GGF-Klein\.venv\Scripts\python.exe" "GGF-Klein\update_app.py"
if errorlevel 1 (
  pause
  exit /b 1
)
call RUN.bat
