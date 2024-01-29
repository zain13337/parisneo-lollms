@echo off
setlocal

:: Check if portable Conda is installed
IF EXIST ".\installer_files\miniconda3\Scripts\activate.bat" (
    echo "Using portable Conda installation."
    set CONDA_BASE=installer_files\miniconda3
    call .\installer_files\miniconda3\Scripts\activate.bat
    call .\installer_files\miniconda3\condabin\conda.bat deactivate
    call .\installer_files\miniconda3\condabin\conda.bat info --envs | findstr /B /C:"xtts" >nul 2>&1
    if %errorlevel% == 0 (
        echo "Conda environment 'xtts' already exists. Deleting it."
        call .\installer_files\miniconda3\condabin\conda.bat env remove --name xtts --yes
    )
    call .\installer_files\miniconda3\condabin\conda.bat create --name xtts --yes
    call .\installer_files\miniconda3\Scripts\activate.bat xtts
    call .\installer_files\miniconda3\Scripts\pip.exe install xtts-api-server --user
) else (
    echo "No portable Conda found. Checking for system-wide Conda installation."
    where conda >nul 2>&1
    if %errorlevel% == 0 (
        echo "Using system-wide Conda installation."
        call conda deactivate
        call conda info --envs | findstr /B /C:"xtts" >nul 2>&1
        if %errorlevel% == 0 (
            echo "Conda environment 'xtts' already exists. Deleting it."
            call conda env remove --name xtts --yes
        )
        call conda create --name xtts --yes
        call conda activate xtts
        call pip install xtts-api-server --user
    ) else (
        echo "No Conda installation found. Please install Conda."
        exit /b 1
    )
)
exit /b
