@echo off
rem Start ComfyUI with the astro_GPU nodes and open it in the browser
cd /d C:\Users\Nicolas\sandbox\ComfyUI
start "" http://127.0.0.1:8188
.venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188
