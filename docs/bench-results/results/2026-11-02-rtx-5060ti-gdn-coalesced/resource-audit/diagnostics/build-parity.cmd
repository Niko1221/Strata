@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -std=c++20 -O2 -arch=sm_120 -DNDEBUG -DSTRATA_VERIFY_MAX_T=8 -Xcompiler /EHsc,/utf-8,/MD,/Ob2 -I "C:\Users\downl\Documents\ChatGPT\Strata\include" -c "C:\Users\downl\Documents\ChatGPT\Strata\src\kernels\gdn_parity.cpp" -o "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\parity.obj"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\parity.obj" "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\verify_kernels.obj" "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\fused_gdn.obj" "H:\Strata\releases\hakua040-native-upstream-r1\Release\strata_kernels.lib" -o "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\verify_kernels_parity.exe"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\parity.obj" "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\verify_kernels_ungrouped_q.obj" "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\fused_gdn.obj" "H:\Strata\releases\hakua040-native-upstream-r1\Release\strata_kernels.lib" -o "C:\Users\downl\Documents\ChatGPT\Strata\build\gdn-resource-specialization\verify_kernels_ungrouped_q_parity.exe"
if errorlevel 1 exit /b %errorlevel%
