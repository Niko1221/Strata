@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Flint 40B MoE (Strata Native C++/CUDA Engine)

set "PORT=8081"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo.
echo ============================================================
echo   Flint 40B MoE (256 experts) - Native Strata Engine
echo ============================================================
echo   Модель:        flint-UD-Q3_K_XL.gguf
echo   Архитектура:   NVIDIA Blackwell sm_120 (CUDA 12.8)
echo   Движок:        strata-flint.exe (Нативный C++/CUDA)
echo   Скорость:      ~85-92 т/с decode, ~360-385 т/с prefill
echo   OpenAI API:    http://127.0.0.1:%PORT%/v1
echo   Web Chat:      http://127.0.0.1:%PORT%/
echo ============================================================
echo.

set "EXE=%ROOT%\build\Release\strata-flint.exe"
if not exist "%EXE%" set "EXE=%ROOT%\bin\strata-flint.exe"

if not exist "%EXE%" (
    echo [ERROR] strata-flint.exe не найден!
    echo Скомпилируйте его: cmake --build build --config Release --target strata-flint
    echo Или скачайте готовый бинарник из GitHub Releases.
    pause
    exit /b 1
)

set "MODEL=E:\AI-models\models\flint\flint-UD-Q3_K_XL.gguf"
if not "%~1"=="" set "MODEL=%~1"

echo   >> Запуск Strata Server...
set "PYTHONPATH=%ROOT%;%ROOT%\tools"
python "serve\server.py" --engine strata --config "strata-flint.json" --port %PORT% --open

if errorlevel 1 pause
