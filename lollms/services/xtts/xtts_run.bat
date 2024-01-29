@echo off
setlocal

:: Check if system-wide Conda is installed
where conda >nul 2>&1
if %errorlevel% == 0 (
    echo "Using system-wide Conda installation."
    call conda activate xtts
    python -m xtts_api_server -o %1 -sf %2
    exit /b
)

:: Use portable Conda if system-wide installation is not found
echo "No system-wide Conda found. Using portable Conda installation."
set CONDA_BASE=installer_files\miniconda3
call "%CONDA_BASE%\Scripts\activate.bat" xtts
call "%CONDA_BASE%\python.exe" -m xtts_api_server -o %1 -sf %2
call "%CONDA_BASE%\Scripts\deactivate.bat"
exit /b
