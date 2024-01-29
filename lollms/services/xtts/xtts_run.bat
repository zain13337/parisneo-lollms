@echo off
setlocal
cd
:: Check if portable Conda is installed
IF EXIST ".\installer_files\miniconda3\Scripts\activate.bat" (
    echo "Using portable Conda installation."
    call .\installer_files\miniconda3\Scripts\activate.bat xtts
    call .\installer_files\miniconda3\python.exe -m xtts_api_server -o %1 -sf %2
    exit /b
) else (
    echo "No portable Conda found. Checking for system-wide Conda installation."
    where conda >nul 2>&1
    if %errorlevel% == 0 (
        echo "Using system-wide Conda installation."
        call conda activate xtts
        python -m xtts_api_server -o %1 -sf %2
        exit /b
    ) else (
        echo "No Conda installation found. Please install Conda."
        exit /b 1
    )
)
exit /b
