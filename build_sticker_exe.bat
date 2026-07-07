@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist "%TEMP%\sticker_yolo_venv\Scripts\python.exe" (
    set "PYTHON_EXE=%TEMP%\sticker_yolo_venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%

"%PYTHON_EXE%" -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo PyInstaller is missing. Installing PyInstaller...
    "%PYTHON_EXE%" -m pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building StickerInspection.exe in ONEDIR mode...
"%PYTHON_EXE%" -m PyInstaller --noconfirm --clean StickerInspection.spec
if errorlevel 1 (
    echo PyInstaller build failed.
    pause
    exit /b 1
)

echo Creating Desktop runtime folder...
"%PYTHON_EXE%" package_runtime.py
if errorlevel 1 (
    echo Runtime packaging failed.
    pause
    exit /b 1
)

echo.
echo Build complete.
echo Runtime folder: "%USERPROFILE%\Desktop\sticker app"
echo EXE: "%USERPROFILE%\Desktop\sticker app\StickerInspection.exe"
echo.
pause
