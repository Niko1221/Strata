@echo off
rem tools/remote-vision/strata-vision-remote.cmd
rem
rem The Strata config's `vision.exe`: the server starts it with subprocess.Popen(exe, stdin=PIPE,
rem stdout=PIPE) and speaks the encoder's line protocol on those pipes.  A .py cannot be that
rem process, so this shim runs it with the venv's Python.
rem
rem Why a .cmd and not a packed .exe: Popen can start a .cmd, and cmd.exe forwards stdin and stdout
rem untouched (verified: `Popen(["x.cmd", "--mmproj", "x"])` reaches the script with its arguments
rem and its pipes intact).  PyInstaller meant an 8 MB opaque artifact in the tree and a build step
rem before anything ran - for a 170-line Python script the server's own venv can execute.
rem
rem Set STRATA_VISION_PYTHON to use another interpreter.
setlocal
if not defined STRATA_VISION_PYTHON set "STRATA_VISION_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%STRATA_VISION_PYTHON%" set "STRATA_VISION_PYTHON=python"
"%STRATA_VISION_PYTHON%" "%~dp0strata_vision_remote.py" %*
