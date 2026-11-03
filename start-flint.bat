@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Flint 40B Preview MoE @ 256K (Strata Native Engine)

set "PORT=8081"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo.
echo ============================================================
echo   Flint 40B Preview MoE @ 256K (Strata Native Engine)
echo ============================================================
echo   Модель:       flint-UD-Q3_K_XL.gguf (18.5 GB)
echo   Архитектура:  MoE 40B (256 экспертов, top-6 + 2 shared, 49 слоев)
echo   Память:       3-Tier Ring (32K VRAM Ring + Host RAM + NVMe)
echo   Контекст:     До 256,000 токенов (256K)
echo   Видеокарта:   NVIDIA RTX 5060 Ti 16 GB VRAM
echo   Префилд:      Batched Tensor Core GEMM (~1,200+ tok/s)
echo   Декод:        CUDA Graph Replay (~89-90 tok/s)
echo   Порт API:     %PORT%
echo ------------------------------------------------------------
echo   [1] Chat-режим 32K Turbo (Pure VRAM, префилл ~1,238 t/s, декод ~90.4 t/s)
echo   [2] Multi-turn 65K (32K VRAM Ring + RAM, баланс для длинных диалогов)
echo   [3] 256K Context (Full 3-Tier Ring, 262,144 токенов, анализ гигантских документов)
echo   [4] Agentic 256K + Adaptive MoE (КОМБО: 256K Context + авто-своп экспертов) [По умолчанию]
echo ============================================================
echo.

set "MODE=%~1"
if "%MODE%"=="" set /p MODE="  Выберите вариант [1-4, Enter=4]: "
if "%MODE%"=="" set "MODE=4"

set "CFG=strata-flint.json"
set "DESC=Agentic 256K + Adaptive MoE (КОМБО: 256K Context + динамический своп экспертов)"

if "%MODE%"=="1" (
    echo   ^>^> Режим: Chat-режим 32K Turbo (Pure VRAM, 32,768 токенов)
    set "CFG=strata-flint-32k.json"
    set "DESC=Chat-режим 32K Turbo (Pure VRAM, 32,768 токенов)"
)
if "%MODE%"=="2" (
    echo   ^>^> Режим: Multi-turn 65K (Sliding Ring + RAM, 65,536 токенов)
    set "CFG=strata-flint-65k.json"
    set "DESC=Multi-turn 65K (Sliding Ring + RAM, 65,536 токенов)"
)
if "%MODE%"=="3" (
    echo   ^>^> Режим: 256K Context (Full 3-Tier Ring, 262,144 токена)
    set "CFG=strata-flint-256k.json"
    set "DESC=256K Context (Full 3-Tier Ring, 262,144 токена)"
)
if "%MODE%"=="4" (
    echo   ^>^> Режим: Agentic 256K + Adaptive MoE (КОМБО)
    set "CFG=strata-flint.json"
    set "DESC=Agentic 256K + Adaptive MoE (КОМБО: 256K Context + динамический своп экспертов)"
)

echo.
echo   ^>^> Запуск Strata Engine: %DESC%
echo   ^>^> Конфиг: %CFG%
echo   ^>^> OpenAI API: http://127.0.0.1:%PORT%/v1
echo   ^>^> Web Chat:   http://127.0.0.1:%PORT%/
echo   ^>^> Нажмите Ctrl+C для остановки.
echo.

cd /d "%ROOT%"
set "PYTHONPATH=%ROOT%;%ROOT%\tools"
python "serve\server.py" --engine strata --config "%CFG%" --port %PORT% --open

if errorlevel 1 pause
