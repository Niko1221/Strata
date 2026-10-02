@echo off
rem Update Strata without starting the model (#475): the newest code (git pull, when this folder is a git clone), then
rem what START-HERE.bat does before a start - the engine (a new one when this version needs it), the Python packages,
rem each installed model's settings and draft subset. The model files are not touched. Start the model later with
rem START-HERE.bat. Options are passed on to setup.py.
setlocal
title Strata - update
cd /d "%~dp0"
rem All of it in one block: cmd reads a .bat file while it runs it, and the git pull can change this very file.
(
  if exist ".git" (
    where git >nul 2>nul || (
      echo  This folder is a git clone, but git is not on PATH: install Git for Windows ^(winget install Git.Git^)
      echo  or run "git pull" here yourself, then run UPDATE.bat again.
      pause
      exit /b 1
    )
    echo  Getting the newest Strata ^(git pull^) ...
    git pull --ff-only || (
      echo.
      echo  git pull did not succeed ^(the reason is above^): nothing was updated. Files you changed here can stop it:
      echo  "git status" lists them.
      pause
      exit /b 1
    )
  ) else (
    echo  This copy of Strata was not made with git, so it cannot fetch new files itself. Download the newest one:
    echo    https://github.com/Niko1221/Strata/archive/refs/heads/main.zip
    echo  unzip it anywhere and run START-HERE.bat ^(or UPDATE.bat^) in it: it finds the model files in Strata-data
    echo  and sets itself up the same way - nothing big is downloaded again.
    echo  Checking this copy's engine and settings meanwhile ...
  )
  call "%~dp0START-HERE.bat" --update %*
  if not errorlevel 1 pause
  exit /b
)
