@echo off
setlocal
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0Ultralytics"
set "PYTHON_EXE="

if exist "%TEMP%\sticker_yolo_venv\Scripts\pythonw.exe" (
    set "PYTHON_EXE=%TEMP%\sticker_yolo_venv\Scripts\pythonw.exe"
) else if exist "%TEMP%\sticker_yolo_venv\Scripts\python.exe" (
    set "PYTHON_EXE=%TEMP%\sticker_yolo_venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

start "" "%PYTHON_EXE%" "%~dp0desktop_app.py"
exit /b
