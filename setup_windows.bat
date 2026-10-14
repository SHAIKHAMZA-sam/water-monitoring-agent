@echo off
REM One-time setup for Windows. Double-click this file.
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer from python.org
  echo and tick "Add Python to PATH" during installation.
  pause
  exit /b 1
)
if not exist venv (
  echo Creating virtual environment...
  %PY% -m venv venv
)
echo Installing packages (this can take a few minutes)...
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package install failed. Check your internet connection and try again.
  pause
  exit /b 1
)
echo Creating data and training the model...
venv\Scripts\python generate_data.py
venv\Scripts\python train_model.py
echo Running self-test...
venv\Scripts\python test_agent.py
echo.
echo Setup complete. Now double-click run_windows.bat to start the app.
pause
