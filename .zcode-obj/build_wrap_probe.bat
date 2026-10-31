@echo off
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /EHsc /MD /std:c++17 /Zc:__cplusplus /I F:\Strata-dev\include /I F:\Strata-dev\third_party\nlohmann /I F:\Strata\third_party\llama.cpp\grammar F:\Strata-dev\.zcode-obj\wrap_probe.cpp F:\Strata-dev\src\grammar\grammar.cpp F:\Strata-dev\src\grammar\grammar_vocab.cpp /Fe:F:\Strata-dev\.zcode-obj\wrap_probe.exe 1>F:\Strata-dev\.zcode-obj\wrap_probe_build.log 2>&1
if errorlevel 1 exit /b 1
