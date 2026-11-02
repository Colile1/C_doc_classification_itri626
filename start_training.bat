@echo off
setlocal EnableExtensions

rem Usage: start_training.bat [model] [config]
rem Defaults: baseline_cnn_v4 and configs\baseline_cnn_v4.yaml
set "PROJECT_DIR=%~dp0"
pushd "%PROJECT_DIR%" || (echo ERROR: Cannot access project folder: "%PROJECT_DIR%" & exit /b 1)

if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "VENV_PYTHON=%LOCALAPPDATA%\venvs\itri626\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo ERROR: Project environment not found.
    echo Run setup_requirements.bat first.
    popd
    exit /b 1
)

set "MODEL=%~1"
if not defined MODEL set "MODEL=baseline_cnn_v4"
set "CONFIG=%~2"
if not defined CONFIG set "CONFIG=configs\%MODEL%.yaml"
if not exist "%CONFIG%" (
    echo ERROR: Config not found: "%CONFIG%"
    echo Usage: start_training.bat [baseline_cnn_v4^|resnet50^|vit_b16] [config path]
    popd
    exit /b 1
)

if not exist "data\processed\working_set.csv" (
    echo ERROR: Training data is missing: data\processed\working_set.csv
    echo Prepare the dataset first with run_prepare_data.py.
    popd
    exit /b 1
)

if not exist "data\splits\train.csv" goto create_split
if not exist "data\splits\val.csv" goto create_split
if not exist "data\splits\test.csv" goto create_split
goto split_ready

:create_split
    echo No committed split files found. Building the grouped train/validation/test split...
    "%VENV_PYTHON%" run_split.py --config "%CONFIG%" --env local
    if errorlevel 1 (echo ERROR: Could not create the split.& popd& exit /b 1)

:split_ready

echo Starting %MODEL% training with config %CONFIG%...
echo Epoch progress and validation scores will appear below and be saved under outputs\logs.
echo Press Ctrl+C to stop training. Completed epochs are checkpointed.
echo.
"%VENV_PYTHON%" run_train.py --config "%CONFIG%" --env local --model "%MODEL%"
set "TRAIN_EXIT=%ERRORLEVEL%"
echo.
if "%TRAIN_EXIT%"=="0" (
    echo Training finished. Review outputs\metrics\history_%MODEL%.csv and outputs\logs\train_%MODEL%.log
) else (
    echo Training exited with code %TRAIN_EXIT%. Review outputs\logs\train_%MODEL%.log
)
popd
exit /b %TRAIN_EXIT%
