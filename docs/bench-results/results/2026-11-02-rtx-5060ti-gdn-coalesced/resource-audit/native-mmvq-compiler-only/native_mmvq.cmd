@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -std=c++20 -O2 -arch=sm_120 -Xptxas=-v -DNDEBUG -D_WINDOWS -DSTRATA_VERIFY_MAX_T=8 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_HAVE_DFLASH2=1 -Xcompiler /EHsc,/utf-8,/MD,/Ob2 -I "C:\Users\downl\Documents\ChatGPT\Strata\include" -I "C:\Users\downl\Documents\ChatGPT\Strata\third_party\ggml" -I "C:\Users\downl\Documents\ChatGPT\Strata\src\kernels\cuda" --use_fast_math -c "C:\Users\downl\Documents\ChatGPT\Strata\src\kernels\cuda\native_mmvq.cu" -o "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-audit\native_mmvq.obj"
exit /b %errorlevel%
