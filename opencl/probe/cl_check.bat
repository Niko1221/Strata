@echo off
rem Compile-only check of src/core/ocl_peer.cpp (the peer tier is C++ + OpenCL, no CUDA).
rem Same hand-rolled MSVC + Windows SDK environment as build_probe.bat.
setlocal
set BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
set MSVC=%BT%\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKV=10.0.26100.0
set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKV%\x64;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64
set CLINC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include
set ROOT=%~dp0..\..
if not exist "%~dp0build" mkdir "%~dp0build"
cl /nologo /c /std:c++17 /O2 /EHsc /W3 /wd4100 /wd4244 /wd4267 ^
   /I "%ROOT%\include" /I "%ROOT%\third_party\ggml" /I "%CLINC%" ^
   "%ROOT%\src\core\ocl_peer.cpp" /Fo:"%~dp0build\ocl_peer.obj" /Fd:"%~dp0build\ocl_peer.pdb" || exit /b 1
echo CL CHECK OK
