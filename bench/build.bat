@echo off
REM Build the engine under a full MSVC environment.  Git Bash has no INCLUDE/LIB, so cl.exe cannot find
REM even cstdint; vcvars64.bat is what puts the standard library and the Windows SDK on those paths.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d "G:\01_Projects\Strata\build-cuda"
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" %*
