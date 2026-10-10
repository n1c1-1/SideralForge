@echo off
rem Launches the astro_GPU interface with its dedicated virtual environment.
cd /d "%~dp0"
start "" ".venv\Scripts\astro-gpu-gui.exe" %*
