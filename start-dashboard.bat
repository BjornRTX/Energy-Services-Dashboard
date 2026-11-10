@echo off
rem Starts the Energy Dashboard. First start installs the packages into a local folder (.venv).
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please install Python 3.12 from python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the environment ...
    python -m venv .venv
    if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
)

echo Checking the packages ...
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Check the internet connection and that Python is 3.10 to 3.12.
    pause
    exit /b 1
)

echo.
echo Dashboard starting. Open http://127.0.0.1:8050 if the browser does not open by itself.
echo Close this window to stop it.
start "" "http://127.0.0.1:8050"
".venv\Scripts\python.exe" local-dashboard.py
pause
