@echo off
rem Launches the new client (on the GPU when there is one).
cd /d "%~dp0"
"C:\Users\Nicolas\sandbox\ComfyUI\.venv\Scripts\python.exe" -m astro_gpu.client %*
if errorlevel 1 pause
