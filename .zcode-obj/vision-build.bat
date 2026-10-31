@echo off
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set STRATA_GGML_DIR=F:\Strata\third_party\llama.cpp
call F:\Strata-dev\packaging\build_vision_windows.bat cuda13 F:\Strata-dev\.zcode-obj\engine-stage
exit /b %errorlevel%
