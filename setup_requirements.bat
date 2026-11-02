@echo off
setlocal EnableExtensions

rem Install the pinned project dependencies in an external venv to avoid
rem Windows path length issues inside this repository.
set "PROJECT_DIR=%~dp0"
pushd "%PROJECT_DIR%" || (echo ERROR: Cannot access project folder: "%PROJECT_DIR%" & exit /b 1)

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_LAUNCHER=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python 3 was not found. Install Python 3.11 or newer and enable the PATH option.
        popd
        exit /b 1
    )
    set "PYTHON_LAUNCHER=python"
)

for /f "usebackq delims=" %%V in (`%PYTHON_LAUNCHER% -c "import sys; print(sys.version_info >= (3, 11))"`) do set "PYTHON_OK=%%V"
if not "%PYTHON_OK%"=="True" (
    echo ERROR: Python 3.11 or newer is required.
    popd
    exit /b 1
)

if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "VENV_DIR=%LOCALAPPDATA%\venvs\itri626"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment at "%VENV_DIR%"...
    %PYTHON_LAUNCHER% -m venv "%VENV_DIR%"
    if errorlevel 1 (echo ERROR: Could not create the virtual environment.& popd& exit /b 1)
)

echo Updating pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (echo ERROR: pip upgrade failed.& popd& exit /b 1)

echo Installing CPU PyTorch as documented for a laptop without CUDA...
"%VENV_PYTHON%" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (echo ERROR: PyTorch installation failed. Check your internet connection and Python version.& popd& exit /b 1)

echo Installing/checking the remaining pinned requirements...
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (echo ERROR: Requirements installation failed.& popd& exit /b 1)

echo.
echo Requirements are ready.
echo Virtual environment: "%VENV_DIR%"
echo To activate it in a terminal: call "%VENV_DIR%\Scripts\activate.bat"
popd
exit /b 0
