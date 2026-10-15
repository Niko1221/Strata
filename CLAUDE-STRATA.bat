@echo off
setlocal
if not exist "%USERPROFILE%\.local\bin\claude.exe" (
  echo Claude Code was not found at "%USERPROFILE%\.local\bin\claude.exe".
  exit /b 1
)
"%USERPROFILE%\.local\bin\claude.exe" --settings "%~dp0claude-code\settings.local.json" %*
exit /b %ERRORLEVEL%
