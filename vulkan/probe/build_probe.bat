@echo off
rem Build the Vulkan probe. Sets the MSVC + Windows SDK environment by hand
rem (no vcvars, which shells out to reg.exe).
setlocal
set BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
set MSVC=%BT%\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKV=10.0.26100.0
set VULKAN_SDK=C:\VulkanSDK\1.4.357.0
set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKV%\x64;%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64
set CMAKE=%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
"%CMAKE%" -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DVulkan_INCLUDE_DIR="%VULKAN_SDK%\Include" -DVulkan_LIBRARY="%VULKAN_SDK%\Lib\vulkan-1.lib" ^
  -DCMAKE_MAKE_PROGRAM="%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" || exit /b 1
"%CMAKE%" --build "%~dp0build" || exit /b 1
echo BUILD OK
