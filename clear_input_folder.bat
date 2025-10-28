@echo off
setlocal EnableExtensions

REM Resolve project-relative input directory.
set "PROJECT_DIR=%~dp0"
set "INPUT_DIR=%PROJECT_DIR%input"

if not exist "%INPUT_DIR%" (
    echo Input folder not found: "%INPUT_DIR%"
    pause
    exit /b 1
)

echo Clearing input folder: "%INPUT_DIR%"

REM Delete supported audio files and any empty sub-folders.
del /q "%INPUT_DIR%\*.wav" >nul 2>&1
del /q "%INPUT_DIR%\*.aif" >nul 2>&1
del /q "%INPUT_DIR%\*.aiff" >nul 2>&1

for /d %%D in ("%INPUT_DIR%\*") do (
    rd /s /q "%%~D" >nul 2>&1
)

echo Done. Add new files to the input folder when ready.
pause
