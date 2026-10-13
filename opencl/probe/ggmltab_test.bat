@echo off
rem One-off: can a plain C++ TU read the ggml codebook tables?  Same hand-rolled MSVC env as the probes.
setlocal
set BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
set MSVC=%BT%\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKV=10.0.26100.0
set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKV%\x64;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64
cl.exe /nologo /std:c++17 /O2 /EHsc /I "G:\01_Projects\Strata\third_party\ggml" ^
  /Fe:"%~dp0build\ggmltab_test.exe" /Fo:"%~dp0build\\" "%~dp0ggmltab_test.cpp" || exit /b 1
"%~dp0build\ggmltab_test.exe"
