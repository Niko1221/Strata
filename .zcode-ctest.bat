@echo off
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul
set CMAKE=D:\VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
%CMAKE% --build F:\Strata-dev\build -j 16 1>F:\Strata-dev\.zcode-ctest-build.log 2>&1
if errorlevel 1 exit /b 1
cd /d F:\Strata-dev\build
ctest --output-on-failure 1>F:\Strata-dev\.zcode-ctest.log 2>&1
exit /b %errorlevel%
