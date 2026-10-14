@echo off
REM Starts the web app. Double-click this file after setup_windows.bat.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
venv\Scripts\python -m streamlit run app.py
pause
