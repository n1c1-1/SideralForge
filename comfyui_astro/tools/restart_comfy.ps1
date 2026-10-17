# Restart the local ComfyUI (C:\Users\Nicolas\sandbox\ComfyUI) in the background, logs in comfy_stdout/stderr.log
$root = "C:\Users\Nicolas\sandbox\ComfyUI"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*main.py*--port*8188*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 2
Start-Process -WindowStyle Hidden -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList 'main.py','--listen','127.0.0.1','--port','8188' -WorkingDirectory $root -RedirectStandardOutput "$root\comfy_stdout.log" -RedirectStandardError "$root\comfy_stderr.log"
