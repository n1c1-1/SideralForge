@echo off
rem Launches the new client. While the projects' runs hold both GPUs (pilot run), the client runs on the CPU only:
rem remove the next line once the runs are finished.
set CUDA_VISIBLE_DEVICES=-1
cd /d "%~dp0"
"C:\Users\Nicolas\sandbox\ComfyUI\.venv\Scripts\python.exe" -m astro_gpu.client %*
if errorlevel 1 pause
