@echo off
rem Start ComfyUI with the astro_GPU nodes and open it in the browser
cd /d C:\Users\Nicolas\sandbox\ComfyUI
start "" http://127.0.0.1:8188
rem --cuda-device all: both GPUs (on Windows ComfyUI otherwise shows only the first; projects may use another)
.venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188 --cuda-device all
