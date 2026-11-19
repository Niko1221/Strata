Unicode true
!include MUI2.nsh
!include x64.nsh
!include LogicLib.nsh
!include WinVer.nsh
!ifndef PAYLOAD_ZIP
  !error "Pinned payload defines are required."
!endif
Name "Strata Windows Full Preview"
OutFile "${OUTPUT_EXE}"
InstallDir "$LOCALAPPDATA\Programs\Strata-0.5.5-local.1"
RequestExecutionLevel user
SetCompressor zlib
SetCompress off
ShowInstDetails show
Var PowerShellExe
Icon "${OWNED_ICON}"
!define MUI_ICON "${OWNED_ICON}"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "English"
VIProductVersion "0.3.2.0"
VIAddVersionKey /LANG=1033 "ProductName" "Strata Full Preview"
VIAddVersionKey /LANG=1033 "FileDescription" "Fresh-install GUI, engines and runtimes"
VIAddVersionKey /LANG=1033 "FileVersion" "0.5.5-local.1"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "This preview requires 64-bit Windows." /SD IDOK
    Abort
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_ICONSTOP "This preview requires Windows 10 or later." /SD IDOK
    Abort
  ${EndIf}
  SetShellVarContext current
  ; NSIS is a 32-bit carrier. SysNative selects 64-bit PowerShell/registry.
  StrCpy $PowerShellExe "$WINDIR\SysNative\WindowsPowerShell\v1.0\powershell.exe"
FunctionEnd

Section "Strata GUI, engines and runtimes" Main
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File /oname=bootstrap.ps1 "${BOOTSTRAP_PS1}"
  DetailPrint "Checking space for the full installation..."
  nsExec::ExecToLog '"$PowerShellExe" -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\bootstrap.ps1" -Mode Check -TempRoot "$PLUGINSDIR" -ProgramsRoot "$INSTDIR" -ZipSha256 ${ZIP_SHA256} -ManifestSha256 ${MANIFEST_SHA256} -ZipBytes ${ZIP_BYTES} -UnpackBytes ${UNPACK_BYTES} -ProgramBytes ${PROGRAM_BYTES} -ComponentBytes ${COMPONENT_BYTES}'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP "Setup checks failed. See the details above. Existing installations were preserved." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
  DetailPrint "Unpacking the included full package..."
  File /oname=full-package.zip "${PAYLOAD_ZIP}"
  nsExec::ExecToLog '"$PowerShellExe" -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\bootstrap.ps1" -Mode Install -TempRoot "$PLUGINSDIR" -ProgramsRoot "$INSTDIR" -ZipSha256 ${ZIP_SHA256} -ManifestSha256 ${MANIFEST_SHA256} -ZipBytes ${ZIP_BYTES} -UnpackBytes ${UNPACK_BYTES} -ProgramBytes ${PROGRAM_BYTES} -ComponentBytes ${COMPONENT_BYTES}'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP "Installation did not finish. See the details above. Verified rollback records were retained." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
SectionEnd
