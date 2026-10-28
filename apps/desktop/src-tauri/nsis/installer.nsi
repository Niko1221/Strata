Unicode true
ManifestDPIAware true
; Add in `dpiAwareness` `PerMonitorV2` to manifest for Windows 10 1607+ (note this should not affect lower versions since they should be able to ignore this and pick up `dpiAware` `true` set by `ManifestDPIAware true`)
; Currently undocumented on NSIS's website but is in the Docs folder of source tree, see
; https://github.com/kichik/nsis/blob/5fc0b87b819a9eec006df4967d08e522ddd651c9/Docs/src/attributes.but#L286-L300
; https://github.com/tauri-apps/tauri/pull/10106
ManifestDPIAwareness PerMonitorV2

!if "{{compression}}" == "none"
  SetCompress off
!else
  ; Set the compression algorithm. We default to LZMA.
  SetCompressor /SOLID "{{compression}}"
!endif

; Keep above !include to stay ahead of any plugin command
; see https://github.com/tauri-apps/tauri/pull/15422#discussion_r3289239624
{{#if signed_plugins_path}}
!addplugindir "{{signed_plugins_path}}"
{{/if}}

!include MUI2.nsh
!include FileFunc.nsh
!include x64.nsh
!include WordFunc.nsh
!include "utils.nsh"
!include "FileAssociation.nsh"
!include "Win\COM.nsh"
!include "Win\Propkey.nsh"
!include "Win\RestartManager.nsh"
!include "StrFunc.nsh"
${StrCase}
${StrLoc}

{{#if installer_hooks}}
!include "{{installer_hooks}}"
{{/if}}

!define WEBVIEW2APPGUID "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

!define MANUFACTURER "{{manufacturer}}"
!define PRODUCTNAME "{{product_name}}"
!define VERSION "{{version}}"
!define VERSIONWITHBUILD "{{version_with_build}}"
!define HOMEPAGE "{{homepage}}"
!define INSTALLMODE "{{install_mode}}"
!define LICENSE "{{license}}"
!define INSTALLERICON "{{installer_icon}}"
!define SIDEBARIMAGE "{{sidebar_image}}"
!define HEADERIMAGE "{{header_image}}"
!define UNINSTALLERICON "{{uninstaller_icon}}"
!define UNINSTALLERHEADERIMAGE "{{uninstaller_header_image}}"
!define MAINBINARYNAME "{{main_binary_name}}"
!define MAINBINARYSRCPATH "{{main_binary_path}}"
!define BUNDLEID "{{bundle_id}}"
!define COPYRIGHT "{{copyright}}"
!define OUTFILE "{{out_file}}"
!define ARCH "{{arch}}"
!define ADDITIONALPLUGINSPATH "{{additional_plugins_path}}"
!define ALLOWDOWNGRADES "{{allow_downgrades}}"
!define DISPLAYLANGUAGESELECTOR "{{display_language_selector}}"
!define INSTALLWEBVIEW2MODE "{{install_webview2_mode}}"
!define WEBVIEW2INSTALLERARGS "{{webview2_installer_args}}"
!define WEBVIEW2BOOTSTRAPPERPATH "{{webview2_bootstrapper_path}}"
!define WEBVIEW2INSTALLERPATH "{{webview2_installer_path}}"
!define MINIMUMWEBVIEW2VERSION "{{minimum_webview2_version}}"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCTNAME}"
!define MANUKEY "Software\${MANUFACTURER}"
!define MANUPRODUCTKEY "${MANUKEY}\${PRODUCTNAME}"
!define UNINSTALLERSIGNCOMMAND "{{uninstaller_sign_cmd}}"
!define ESTIMATEDSIZE "{{estimated_size}}"
!define STARTMENUFOLDER "{{start_menu_folder}}"

Var PassiveMode
Var UpdateMode
Var NoShortcutMode
Var WixMode
Var OldMainBinaryName

Name "${PRODUCTNAME}"
BrandingText "${COPYRIGHT}"
OutFile "${OUTFILE}"

; We don't actually use this value as default install path,
; it's just for nsis to append the product name folder in the directory selector
; https://nsis.sourceforge.io/Reference/InstallDir
!define PLACEHOLDER_INSTALL_DIR "placeholder\${PRODUCTNAME}"
InstallDir "${PLACEHOLDER_INSTALL_DIR}"

VIProductVersion "${VERSIONWITHBUILD}"
VIAddVersionKey "ProductName" "${PRODUCTNAME}"
VIAddVersionKey "FileDescription" "${PRODUCTNAME}"
VIAddVersionKey "LegalCopyright" "${COPYRIGHT}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"

# additional plugins
!addplugindir "${ADDITIONALPLUGINSPATH}"

; Uninstaller signing command
!if "${UNINSTALLERSIGNCOMMAND}" != ""
  !uninstfinalize '${UNINSTALLERSIGNCOMMAND}'
!endif

; Handle install mode, `perUser`, `perMachine` or `both`
!if "${INSTALLMODE}" == "perMachine"
  RequestExecutionLevel admin
!endif

!if "${INSTALLMODE}" == "currentUser"
  RequestExecutionLevel user
!endif

!if "${INSTALLMODE}" == "both"
  !define MULTIUSER_MUI
  !define MULTIUSER_INSTALLMODE_INSTDIR "${PRODUCTNAME}"
  !define MULTIUSER_INSTALLMODE_COMMANDLINE
  !if "${ARCH}" == "x64"
    !define MULTIUSER_USE_PROGRAMFILES64
  !else if "${ARCH}" == "arm64"
    !define MULTIUSER_USE_PROGRAMFILES64
  !endif
  !define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_KEY "${UNINSTKEY}"
  !define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_VALUENAME "CurrentUser"
  !define MULTIUSER_INSTALLMODEPAGE_SHOWUSERNAME
  !define MULTIUSER_INSTALLMODE_FUNCTION RestorePreviousInstallLocation
  !define MULTIUSER_EXECUTIONLEVEL Highest
  !include MultiUser.nsh
!endif

; Installer icon
!if "${INSTALLERICON}" != ""
  !define MUI_ICON "${INSTALLERICON}"
!endif

; Installer sidebar image
!if "${SIDEBARIMAGE}" != ""
  !define MUI_WELCOMEFINISHPAGE_BITMAP "${SIDEBARIMAGE}"
!endif

; Enable header images for installer and uninstaller pages when either image is configured.
!if "${HEADERIMAGE}" != ""
  !define MUI_HEADERIMAGE
!else if "${UNINSTALLERHEADERIMAGE}" != ""
  !define MUI_HEADERIMAGE
!endif

; Installer header image
!if "${HEADERIMAGE}" != ""
  !define MUI_HEADERIMAGE_BITMAP "${HEADERIMAGE}"
!endif

; Uninstaller header image
!if "${UNINSTALLERHEADERIMAGE}" != ""
  !define MUI_HEADERIMAGE_UNBITMAP "${UNINSTALLERHEADERIMAGE}"
!endif

; Uninstaller icon
!if "${UNINSTALLERICON}" != ""
  !define MUI_UNICON "${UNINSTALLERICON}"
!else if "${INSTALLERICON}" != ""
  ; Tauri leaves the uninstaller icon empty unless asked; use the Strata icon so the uninstaller is not the NSIS default
  !define MUI_UNICON "${INSTALLERICON}"
!endif

; Define registry key to store installer language
!define MUI_LANGDLL_REGISTRY_ROOT "HKCU"
!define MUI_LANGDLL_REGISTRY_KEY "${MANUPRODUCTKEY}"
!define MUI_LANGDLL_REGISTRY_VALUENAME "Installer Language"

; Strata's look and words: the app's light surface and ink, short copy. The install folder page is shown on a first or manual install (#3186 #3190: C: can be short on space) and
; skipped in an update (passive), which goes on into the folder the earlier install wrote to the registry.
!define MUI_BGCOLOR "F9FAFB"
!define MUI_TEXTCOLOR "171B1E"
!define MUI_INSTFILESPAGE_PROGRESSBAR "smooth"
; The words are language strings (nsis/strata-languages.nsh), so the installer speaks the language of the person's
; Windows, like the app does. English stays the first language and the fallback.
!define MUI_WELCOMEPAGE_TITLE "$(StrataWelcomeTitle)"
!define MUI_WELCOMEPAGE_TEXT "$(StrataWelcomeText)"
!define MUI_FINISHPAGE_TITLE "$(StrataFinishTitle)"
!define MUI_FINISHPAGE_TEXT "$(StrataFinishText)"
!define MUI_FINISHPAGE_RUN_TEXT "$(StrataOpen)"
!define MUI_PAGE_HEADER_TEXT "$(StrataInstallHeader)"
!define MUI_PAGE_HEADER_SUBTEXT "$(StrataInstallSubtext)"
BrandingText "Strata"
ShowInstDetails nevershow

; Installer pages, must be ordered as they appear
; 1. Welcome Page
!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipIfPassive
!insertmacro MUI_PAGE_WELCOME

; 2. License Page (if defined)
!if "${LICENSE}" != ""
  !define MUI_PAGE_CUSTOMFUNCTION_PRE SkipIfPassive
  !insertmacro MUI_PAGE_LICENSE "${LICENSE}"
!endif

; 3. Install mode (if it is set to `both`)
!if "${INSTALLMODE}" == "both"
  !define MUI_PAGE_CUSTOMFUNCTION_PRE SkipIfPassive
  !insertmacro MULTIUSER_PAGE_INSTALLMODE
!endif

; 4. Custom page to ask user if he wants to reinstall/uninstall
;    only if a previous installation was detected
Var ReinstallPageCheck
; 1 when this installer was started on an installed copy of the SAME version and the person chose Uninstall: after it ran the
; uninstaller this installer ends. (Stock Tauri goes on to the install pages, so Uninstall reinstalled Strata at once; tester report 2026-10-17.)
Var UninstallOnly
Page custom PageReinstall PageLeaveReinstall
Function PageReinstall
  ; Uninstall previous WiX installation if exists.
  ;
  ; A WiX installer stores the installation info in registry
  ; using a UUID and so we have to loop through all keys under
  ; `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall`
  ; and check if `DisplayName` and `Publisher` keys match ${PRODUCTNAME} and ${MANUFACTURER}
  ;
  ; This has a potential issue that there maybe another installation that matches
  ; our ${PRODUCTNAME} and ${MANUFACTURER} but wasn't installed by our WiX installer,
  ; however, this should be fine since the user will have to confirm the uninstallation
  ; and they can chose to abort it if doesn't make sense.
  StrCpy $0 0
  wix_loop:
    EnumRegKey $1 HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" $0
    StrCmp $1 "" wix_loop_done ; Exit loop if there is no more keys to loop on
    IntOp $0 $0 + 1
    ReadRegStr $R0 HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$1" "DisplayName"
    ReadRegStr $R1 HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$1" "Publisher"
    StrCmp "$R0$R1" "${PRODUCTNAME}${MANUFACTURER}" 0 wix_loop
    ReadRegStr $R0 HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$1" "UninstallString"
    ${StrCase} $R1 $R0 "L"
    ${StrLoc} $R0 $R1 "msiexec" ">"
    StrCmp $R0 0 0 wix_loop_done
    StrCpy $WixMode 1
    StrCpy $R6 "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$1"
    Goto compare_version
  wix_loop_done:

  ; Check if there is an existing installation, if not, abort the reinstall page
  ReadRegStr $R0 SHCTX "${UNINSTKEY}" ""
  ReadRegStr $R1 SHCTX "${UNINSTKEY}" "UninstallString"
  ${IfThen} "$R0$R1" == "" ${|} Abort ${|}

  ; Compare this installar version with the existing installation
  ; and modify the messages presented to the user accordingly
  compare_version:
  StrCpy $R4 "$(older)"
  ${If} $WixMode = 1
    ReadRegStr $R0 HKLM "$R6" "DisplayVersion"
  ${Else}
    ReadRegStr $R0 SHCTX "${UNINSTKEY}" "DisplayVersion"
  ${EndIf}
  ${IfThen} $R0 == "" ${|} StrCpy $R4 "$(unknown)" ${|}

  nsis_tauri_utils::SemverCompare "${VERSION}" $R0
  Pop $R0
  ; Reinstalling the same version
  ${If} $R0 = 0
    StrCpy $R1 "$(alreadyInstalledLong)"
    StrCpy $R2 "$(addOrReinstall)"
    StrCpy $R3 "$(uninstallApp)"
    !insertmacro MUI_HEADER_TEXT "$(alreadyInstalled)" "$(chooseMaintenanceOption)"
  ; Upgrading
  ${ElseIf} $R0 = 1
    StrCpy $R1 "$(olderOrUnknownVersionInstalled)"
    StrCpy $R2 "$(uninstallBeforeInstalling)"
    StrCpy $R3 "$(dontUninstall)"
    !insertmacro MUI_HEADER_TEXT "$(alreadyInstalled)" "$(choowHowToInstall)"
  ; Downgrading
  ${ElseIf} $R0 = -1
    StrCpy $R1 "$(newerVersionInstalled)"
    StrCpy $R2 "$(uninstallBeforeInstalling)"
    !if "${ALLOWDOWNGRADES}" == "true"
      StrCpy $R3 "$(dontUninstall)"
    !else
      StrCpy $R3 "$(dontUninstallDowngrade)"
    !endif
    !insertmacro MUI_HEADER_TEXT "$(alreadyInstalled)" "$(choowHowToInstall)"
  ${Else}
    Abort
  ${EndIf}

  ; Skip showing the page if passive
  ;
  ; Note that we don't call this earlier at the beginning
  ; of this function because we need to populate some variables
  ; related to current installed version if detected and whether
  ; we are downgrading or not.
  ${If} $PassiveMode = 1
    Call PageLeaveReinstall
  ${Else}
    nsDialogs::Create 1018
    Pop $R4
    ${IfThen} $(^RTL) = 1 ${|} nsDialogs::SetRTL $(^RTL) ${|}

    ${NSD_CreateLabel} 0 0 100% 24u $R1
    Pop $R1

    ${NSD_CreateRadioButton} 30u 50u -30u 8u $R2
    Pop $R2
    ${NSD_OnClick} $R2 PageReinstallUpdateSelection

    ${NSD_CreateRadioButton} 30u 70u -30u 8u $R3
    Pop $R3
    ; Disable this radio button if downgrading and downgrades are disabled
    !if "${ALLOWDOWNGRADES}" == "false"
      ${IfThen} $R0 = -1 ${|} EnableWindow $R3 0 ${|}
    !endif
    ${NSD_OnClick} $R3 PageReinstallUpdateSelection

    ; Check the first radio button if this the first time
    ; we enter this page or if the second button wasn't
    ; selected the last time we were on this page
    ${If} $ReinstallPageCheck <> 2
      SendMessage $R2 ${BM_SETCHECK} ${BST_CHECKED} 0
    ${Else}
      SendMessage $R3 ${BM_SETCHECK} ${BST_CHECKED} 0
    ${EndIf}

    ${NSD_SetFocus} $R2
    nsDialogs::Show
  ${EndIf}
FunctionEnd
Function PageReinstallUpdateSelection
  ${NSD_GetState} $R2 $R1
  ${If} $R1 == ${BST_CHECKED}
    StrCpy $ReinstallPageCheck 1
  ${Else}
    StrCpy $ReinstallPageCheck 2
  ${EndIf}
FunctionEnd
Function PageLeaveReinstall
  ${NSD_GetState} $R2 $R1

  ; If migrating from Wix, always uninstall
  ${If} $WixMode = 1
    Goto reinst_uninstall
  ${EndIf}

  ; In update mode, always proceeds without uninstalling
  ${If} $UpdateMode = 1
    Goto reinst_done
  ${EndIf}

  ; $R0 holds whether same(0)/upgrading(1)/downgrading(-1) version
  ; $R1 holds the radio buttons state:
  ;   1 => first choice was selected
  ;   0 => second choice was selected
  ${If} $R0 = 0 ; Same version, proceed
    ${If} $R1 = 1              ; User chose to add/reinstall
      Goto reinst_done
    ${Else}                    ; User chose to uninstall
      StrCpy $UninstallOnly 1
      Goto reinst_uninstall
    ${EndIf}
  ${ElseIf} $R0 = 1 ; Upgrading
    ${If} $R1 = 1              ; User chose to uninstall
      Goto reinst_uninstall
    ${Else}
      Goto reinst_done         ; User chose NOT to uninstall
    ${EndIf}
  ${ElseIf} $R0 = -1 ; Downgrading
    ${If} $R1 = 1              ; User chose to uninstall
      Goto reinst_uninstall
    ${Else}
      Goto reinst_done         ; User chose NOT to uninstall
    ${EndIf}
  ${EndIf}

  reinst_uninstall:
    HideWindow
    ClearErrors

    ${If} $WixMode = 1
      ReadRegStr $R1 HKLM "$R6" "UninstallString"
      ExecWait '$R1' $0
    ${Else}
      ReadRegStr $4 SHCTX "${MANUPRODUCTKEY}" ""
      ReadRegStr $R1 SHCTX "${UNINSTKEY}" "UninstallString"
      ${IfThen} $UpdateMode = 1 ${|} StrCpy $R1 "$R1 /UPDATE" ${|} ; append /UPDATE
      ${IfThen} $PassiveMode = 1 ${|} StrCpy $R1 "$R1 /P" ${|} ; append /P
      StrCpy $R1 "$R1 _?=$4" ; append uninstall directory
      ExecWait '$R1' $0
    ${EndIf}

    BringToFront

    ${IfThen} ${Errors} ${|} StrCpy $0 2 ${|} ; ExecWait failed, set fake exit code

    ${If} $0 <> 0
    ${OrIf} ${FileExists} "$INSTDIR\${MAINBINARYNAME}.exe"
      ; User cancelled wix uninstaller? return to select un/reinstall page
      ${If} $WixMode = 1
      ${AndIf} $0 = 1602
        Abort
      ${EndIf}

      ; User cancelled NSIS uninstaller? return to select un/reinstall page
      ${If} $0 = 1
        Abort
      ${EndIf}

      ; Other errors? show generic error message and return to select un/reinstall page
      MessageBox MB_ICONEXCLAMATION "$(unableToUninstall)"
      Abort
    ${EndIf}

    ; The NSIS uninstaller was run in place (`_?=`), so it cannot delete itself
    ; and the install directory, clean them up here once the uninstall succeeded
    ; https://nsis.sourceforge.io/Docs/Chapter3.html#:~:text=path%20contains%20spaces.-,3.2.3%20Examples,-installer.exe%20/NCRC
    ${If} $WixMode = 0
    ${AndIf} $4 != ""
      Delete "$4\uninstall.exe"
      RMDir "$4"
    ${EndIf}

    ; Uninstall means uninstall: do not go on to the install pages.
    ${If} $UninstallOnly = 1
      MessageBox MB_OK|MB_ICONINFORMATION "Strata was uninstalled."
      Quit
    ${EndIf}
  reinst_done:
FunctionEnd

; 5. Choose install directory page. The app is installed for this user only (no administrator), so the folder must be one the
; user can write to: another drive (D:\Strata) or a folder in the profile, not Program Files. An update (passive: the app's own
; updater) and a silent install skip it and keep the earlier folder (RestorePreviousInstallLocation).
!ifdef MUI_PAGE_HEADER_TEXT
  !undef MUI_PAGE_HEADER_TEXT
!endif
!ifdef MUI_PAGE_HEADER_SUBTEXT
  !undef MUI_PAGE_HEADER_SUBTEXT
!endif
!define MUI_PAGE_HEADER_TEXT "$(StrataDirHeader)"
!define MUI_PAGE_HEADER_SUBTEXT "$(StrataDirSubtext)"
!define MUI_DIRECTORYPAGE_TEXT_TOP "$(StrataDirTop)"
!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipIfPassive
!insertmacro MUI_PAGE_DIRECTORY

; 6. Start menu shortcut page
Var AppStartMenuFolder
!if "${STARTMENUFOLDER}" != ""
  !define MUI_PAGE_CUSTOMFUNCTION_PRE SkipIfPassive
  !define MUI_STARTMENUPAGE_DEFAULTFOLDER "${STARTMENUFOLDER}"
!else
  !define MUI_PAGE_CUSTOMFUNCTION_PRE Skip
!endif
!insertmacro MUI_PAGE_STARTMENU Application $AppStartMenuFolder

; 7. Installation page (a page macro uses up the header texts, so they are set again here)
!ifdef MUI_PAGE_HEADER_TEXT
  !undef MUI_PAGE_HEADER_TEXT
!endif
!ifdef MUI_PAGE_HEADER_SUBTEXT
  !undef MUI_PAGE_HEADER_SUBTEXT
!endif
!define MUI_PAGE_HEADER_TEXT "$(StrataInstallHeader)"
!define MUI_PAGE_HEADER_SUBTEXT "$(StrataInstallSubtext)"
!insertmacro MUI_PAGE_INSTFILES

; 8. Finish page
;
; Don't auto jump to finish page after installation page,
; because the installation page has useful info that can be used debug any issues with the installer.
; (no NOAUTOCLOSE: welcome, progress, finish)
; Use show readme button in the finish page as a button create a desktop shortcut
!define MUI_FINISHPAGE_SHOWREADME
!define MUI_FINISHPAGE_SHOWREADME_TEXT "$(createDesktop)"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION CreateOrUpdateDesktopShortcut
; Show run app after installation.
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION RunMainBinary
!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipIfPassive
!insertmacro MUI_PAGE_FINISH

Function RunMainBinary
  nsis_tauri_utils::RunAsUser "$INSTDIR\${MAINBINARYNAME}.exe" ""
FunctionEnd

; Uninstaller Pages
; 1. Confirm uninstall page
Var DeleteAppDataCheckbox
Var DeleteAppDataCheckboxState
!define MUI_PAGE_CUSTOMFUNCTION_PRE un.SkipIfPassive
!insertmacro MUI_UNPAGE_CONFIRM

; 2. "Also remove my Strata data": unticked by default, and it says exactly what it deletes. Skipped when updating or in passive mode
;    (an update never deletes anything). Model folders the person chose themselves are never touched.
UninstPage custom un.DataPage un.DataPageLeave
Function un.DataPage
  ${If} $PassiveMode = 1
  ${OrIf} $UpdateMode = 1
    Abort
  ${EndIf}
  !insertmacro MUI_HEADER_TEXT "$(StrataRemoveTitle)" "$(StrataRemoveSubtext)"
  nsDialogs::Create 1018
  Pop $0
  ${IfThen} $(^RTL) = 1 ${|} nsDialogs::SetRTL $(^RTL) ${|}
  ${NSD_CreateCheckBox} 0 0 100% 12u "$(StrataRemoveCheck)"
  Pop $DeleteAppDataCheckbox
  ; The paths are the same in every language, so only the words around them are translated.
  StrCpy $0 "$(StrataRemoveText)$\r$\n$\r$\n$LOCALAPPDATA\Strata   ($(StrataDataEngine))$\r$\n$APPDATA\Strata   ($(StrataDataSettings))$\r$\n$PROFILE\.strata   ($(StrataDataChats))$\r$\n$DOCUMENTS\Strata\Models   ($(StrataDataModels))$\r$\n$\r$\n$(StrataRemoveNote)"
  ${NSD_CreateLabel} 12u 18u -12u 96u "$0"
  Pop $0
  ${NSD_SetFocus} $DeleteAppDataCheckbox
  nsDialogs::Show
FunctionEnd
Function un.DataPageLeave
  ${NSD_GetState} $DeleteAppDataCheckbox $DeleteAppDataCheckboxState
FunctionEnd

; 3. Uninstalling Page
!insertmacro MUI_UNPAGE_INSTFILES

;Languages
{{#each languages}}
!insertmacro MUI_LANGUAGE "{{this}}"
{{/each}}
!insertmacro MUI_RESERVEFILE_LANGDLL
{{#each language_files}}
  !include "{{this}}"
{{/each}}
; The installer speaks the nine languages the app speaks (apps/host/src/locales). Windows picks the language of
; the person's own system; English is first in the list and the fallback. The paths in the uninstall page are the
; same in every language, so they live in the page itself and only the words around them are translated here.

LangString StrataWelcomeTitle ${LANG_ENGLISH} "State of the art. At home."
LangString StrataWelcomeTitle ${LANG_SIMPCHINESE} "顶尖技术，就在家里。"
LangString StrataWelcomeTitle ${LANG_JAPANESE} "最先端を、自宅で。"
LangString StrataWelcomeTitle ${LANG_GERMAN} "Technik auf höchstem Stand. Bei Ihnen zu Hause."
LangString StrataWelcomeTitle ${LANG_SPANISH} "La última tecnología. En casa."
LangString StrataWelcomeTitle ${LANG_RUSSIAN} "Передовая технология. У вас дома."
LangString StrataWelcomeTitle ${LANG_PORTUGUESEBR} "Tecnologia de ponta. Na sua casa."
LangString StrataWelcomeTitle ${LANG_KOREAN} "최고의 기술을, 집에서."
LangString StrataWelcomeTitle ${LANG_FRENCH} "La pointe de la technologie. Chez vous."

LangString StrataWelcomeText ${LANG_ENGLISH} "Strata runs AI on your own computer.$\r$\n$\r$\nThis takes about a minute and needs no administrator. You choose your first model inside Strata."
LangString StrataWelcomeText ${LANG_SIMPCHINESE} "Strata 在你自己的电脑上运行 AI。$\r$\n$\r$\n大约需要一分钟，不需要管理员权限。第一个模型在 Strata 里选择。"
LangString StrataWelcomeText ${LANG_JAPANESE} "Strata は自分のパソコンで AI を動かします。$\r$\n$\r$\n約 1 分で終わり、管理者権限は不要です。最初のモデルは Strata の中で選びます。"
LangString StrataWelcomeText ${LANG_GERMAN} "Strata betreibt KI auf Ihrem eigenen Computer.$\r$\n$\r$\nDas dauert etwa eine Minute und braucht keine Administratorrechte. Das erste Modell wählen Sie in Strata."
LangString StrataWelcomeText ${LANG_SPANISH} "Strata ejecuta IA en tu propio ordenador.$\r$\n$\r$\nTarda alrededor de un minuto y no necesita administrador. El primer modelo lo eliges dentro de Strata."
LangString StrataWelcomeText ${LANG_RUSSIAN} "Strata запускает ИИ на вашем компьютере.$\r$\n$\r$\nЭто занимает около минуты и не требует прав администратора. Первую модель вы выберете в Strata."
LangString StrataWelcomeText ${LANG_PORTUGUESEBR} "O Strata roda IA no seu próprio computador.$\r$\n$\r$\nLeva cerca de um minuto e não precisa de administrador. Você escolhe o primeiro modelo dentro do Strata."
LangString StrataWelcomeText ${LANG_KOREAN} "Strata는 내 컴퓨터에서 AI를 실행합니다.$\r$\n$\r$\n약 1분이면 되고 관리자 권한이 필요 없습니다. 첫 모델은 Strata 안에서 고릅니다."
LangString StrataWelcomeText ${LANG_FRENCH} "Strata fait tourner l’IA sur votre propre ordinateur.$\r$\n$\r$\nCela prend environ une minute et ne demande pas de droits d’administrateur. Vous choisissez le premier modèle dans Strata."

LangString StrataFinishTitle ${LANG_ENGLISH} "Strata is ready."
LangString StrataFinishTitle ${LANG_SIMPCHINESE} "Strata 已就绪。"
LangString StrataFinishTitle ${LANG_JAPANESE} "Strata の準備ができました。"
LangString StrataFinishTitle ${LANG_GERMAN} "Strata ist bereit."
LangString StrataFinishTitle ${LANG_SPANISH} "Strata está listo."
LangString StrataFinishTitle ${LANG_RUSSIAN} "Strata готов."
LangString StrataFinishTitle ${LANG_PORTUGUESEBR} "O Strata está pronto."
LangString StrataFinishTitle ${LANG_KOREAN} "Strata가 준비되었습니다."
LangString StrataFinishTitle ${LANG_FRENCH} "Strata est prêt."

LangString StrataFinishText ${LANG_ENGLISH} "Open it to pick your first model. Strata downloads it for you and gets it ready."
LangString StrataFinishText ${LANG_SIMPCHINESE} "打开它选择第一个模型。Strata 会为你下载并准备好。"
LangString StrataFinishText ${LANG_JAPANESE} "開いて最初のモデルを選んでください。Strata がダウンロードして準備します。"
LangString StrataFinishText ${LANG_GERMAN} "Öffnen Sie es, um Ihr erstes Modell zu wählen. Strata lädt es herunter und richtet es ein."
LangString StrataFinishText ${LANG_SPANISH} "Ábrelo para elegir tu primer modelo. Strata lo descarga y lo prepara."
LangString StrataFinishText ${LANG_RUSSIAN} "Откройте, чтобы выбрать первую модель. Strata скачает её и подготовит."
LangString StrataFinishText ${LANG_PORTUGUESEBR} "Abra para escolher seu primeiro modelo. O Strata baixa e prepara tudo para você."
LangString StrataFinishText ${LANG_KOREAN} "열어서 첫 모델을 고르세요. Strata가 내려받아 준비합니다."
LangString StrataFinishText ${LANG_FRENCH} "Ouvrez-le pour choisir votre premier modèle. Strata le télécharge et le prépare."

LangString StrataOpen ${LANG_ENGLISH} "Open Strata"
LangString StrataOpen ${LANG_SIMPCHINESE} "打开 Strata"
LangString StrataOpen ${LANG_JAPANESE} "Strata を開く"
LangString StrataOpen ${LANG_GERMAN} "Strata öffnen"
LangString StrataOpen ${LANG_SPANISH} "Abrir Strata"
LangString StrataOpen ${LANG_RUSSIAN} "Открыть Strata"
LangString StrataOpen ${LANG_PORTUGUESEBR} "Abrir o Strata"
LangString StrataOpen ${LANG_KOREAN} "Strata 열기"
LangString StrataOpen ${LANG_FRENCH} "Ouvrir Strata"

LangString StrataInstallHeader ${LANG_ENGLISH} "Installing Strata"
LangString StrataInstallHeader ${LANG_SIMPCHINESE} "正在安装 Strata"
LangString StrataInstallHeader ${LANG_JAPANESE} "Strata をインストールしています"
LangString StrataInstallHeader ${LANG_GERMAN} "Strata wird installiert"
LangString StrataInstallHeader ${LANG_SPANISH} "Instalando Strata"
LangString StrataInstallHeader ${LANG_RUSSIAN} "Установка Strata"
LangString StrataInstallHeader ${LANG_PORTUGUESEBR} "Instalando o Strata"
LangString StrataInstallHeader ${LANG_KOREAN} "Strata 설치 중"
LangString StrataInstallHeader ${LANG_FRENCH} "Installation de Strata"

LangString StrataInstallSubtext ${LANG_ENGLISH} "Copying the app. You can keep working meanwhile."
LangString StrataInstallSubtext ${LANG_SIMPCHINESE} "正在复制应用，这期间你可以继续工作。"
LangString StrataInstallSubtext ${LANG_JAPANESE} "アプリをコピーしています。その間も作業を続けられます。"
LangString StrataInstallSubtext ${LANG_GERMAN} "Die App wird kopiert. Sie können in der Zwischenzeit weiterarbeiten."
LangString StrataInstallSubtext ${LANG_SPANISH} "Copiando la aplicación. Puedes seguir trabajando mientras tanto."
LangString StrataInstallSubtext ${LANG_RUSSIAN} "Копирование приложения. Пока это идёт, можно продолжать работу."
LangString StrataInstallSubtext ${LANG_PORTUGUESEBR} "Copiando o aplicativo. Você pode continuar trabalhando enquanto isso."
LangString StrataInstallSubtext ${LANG_KOREAN} "앱을 복사하는 중입니다. 그동안 다른 작업을 계속하셔도 됩니다."
LangString StrataInstallSubtext ${LANG_FRENCH} "Copie de l’application. Vous pouvez continuer à travailler pendant ce temps."

LangString StrataDirHeader ${LANG_ENGLISH} "Choose where to install Strata"
LangString StrataDirHeader ${LANG_SIMPCHINESE} "选择 Strata 的安装位置"
LangString StrataDirHeader ${LANG_JAPANESE} "Strata のインストール先を選択"
LangString StrataDirHeader ${LANG_GERMAN} "Wählen Sie, wohin Strata installiert wird"
LangString StrataDirHeader ${LANG_SPANISH} "Elige dónde instalar Strata"
LangString StrataDirHeader ${LANG_RUSSIAN} "Выберите папку для установки Strata"
LangString StrataDirHeader ${LANG_PORTUGUESEBR} "Escolha onde instalar o Strata"
LangString StrataDirHeader ${LANG_KOREAN} "Strata를 설치할 위치 선택"
LangString StrataDirHeader ${LANG_FRENCH} "Choisissez où installer Strata"

LangString StrataDirSubtext ${LANG_ENGLISH} "Pick another drive if C: is short on space."
LangString StrataDirSubtext ${LANG_SIMPCHINESE} "C 盘空间不足时，可以换一个磁盘。"
LangString StrataDirSubtext ${LANG_JAPANESE} "C: の空きが少ない場合は別のドライブを選んでください。"
LangString StrataDirSubtext ${LANG_GERMAN} "Wählen Sie ein anderes Laufwerk, wenn C: wenig Platz hat."
LangString StrataDirSubtext ${LANG_SPANISH} "Elige otro disco si a C: le queda poco espacio."
LangString StrataDirSubtext ${LANG_RUSSIAN} "Выберите другой диск, если на C: мало места."
LangString StrataDirSubtext ${LANG_PORTUGUESEBR} "Escolha outro disco se o C: estiver com pouco espaço."
LangString StrataDirSubtext ${LANG_KOREAN} "C: 드라이브 공간이 부족하면 다른 드라이브를 고르세요."
LangString StrataDirSubtext ${LANG_FRENCH} "Choisissez un autre disque si C: manque de place."

LangString StrataDirTop ${LANG_ENGLISH} "Strata is installed into this folder, for your user only (no administrator needed). Models and chats are stored separately, in the data folder you choose inside Strata."
LangString StrataDirTop ${LANG_SIMPCHINESE} "Strata 会安装到这个文件夹，只对当前用户生效（不需要管理员权限）。模型和对话另外存放，位置在 Strata 里选择。"
LangString StrataDirTop ${LANG_JAPANESE} "Strata はこのフォルダーに、このユーザーだけ向けにインストールされます（管理者権限は不要）。モデルとチャットは別に保存され、保存先は Strata の中で選びます。"
LangString StrataDirTop ${LANG_GERMAN} "Strata wird in diesen Ordner installiert, nur für Ihren Benutzer (keine Administratorrechte nötig). Modelle und Chats liegen getrennt davon in dem Datenordner, den Sie in Strata wählen."
LangString StrataDirTop ${LANG_SPANISH} "Strata se instala en esta carpeta, solo para tu usuario (no hace falta administrador). Los modelos y los chats se guardan aparte, en la carpeta de datos que elijas dentro de Strata."
LangString StrataDirTop ${LANG_RUSSIAN} "Strata установится в эту папку, только для вашего пользователя (права администратора не нужны). Модели и чаты хранятся отдельно, в папке данных, которую вы выберете в Strata."
LangString StrataDirTop ${LANG_PORTUGUESEBR} "O Strata é instalado nesta pasta, só para o seu usuário (sem precisar de administrador). Modelos e conversas ficam separados, na pasta de dados que você escolher dentro do Strata."
LangString StrataDirTop ${LANG_KOREAN} "Strata는 이 폴더에 현재 사용자용으로 설치됩니다(관리자 권한 필요 없음). 모델과 대화는 따로 저장되며, 위치는 Strata 안에서 고릅니다."
LangString StrataDirTop ${LANG_FRENCH} "Strata s’installe dans ce dossier, pour votre utilisateur uniquement (aucun droit d’administrateur nécessaire). Les modèles et les conversations sont stockés à part, dans le dossier de données que vous choisissez dans Strata."

LangString StrataRemoveTitle ${LANG_ENGLISH} "Remove your Strata data too?"
LangString StrataRemoveTitle ${LANG_SIMPCHINESE} "同时删除你的 Strata 数据吗？"
LangString StrataRemoveTitle ${LANG_JAPANESE} "Strata のデータも削除しますか？"
LangString StrataRemoveTitle ${LANG_GERMAN} "Auch Ihre Strata-Daten löschen?"
LangString StrataRemoveTitle ${LANG_SPANISH} "¿Eliminar también tus datos de Strata?"
LangString StrataRemoveTitle ${LANG_RUSSIAN} "Удалить и данные Strata?"
LangString StrataRemoveTitle ${LANG_PORTUGUESEBR} "Remover também seus dados do Strata?"
LangString StrataRemoveTitle ${LANG_KOREAN} "Strata 데이터도 삭제할까요?"
LangString StrataRemoveTitle ${LANG_FRENCH} "Supprimer aussi vos données Strata ?"

LangString StrataRemoveSubtext ${LANG_ENGLISH} "Your models, chats and settings stay on this PC unless you tick the box."
LangString StrataRemoveSubtext ${LANG_SIMPCHINESE} "不勾选的话，你的模型、对话和设置会留在这台电脑上。"
LangString StrataRemoveSubtext ${LANG_JAPANESE} "チェックしない限り、モデル・チャット・設定はこの PC に残ります。"
LangString StrataRemoveSubtext ${LANG_GERMAN} "Ohne Häkchen bleiben Ihre Modelle, Chats und Einstellungen auf diesem PC."
LangString StrataRemoveSubtext ${LANG_SPANISH} "Tus modelos, chats y ajustes se quedan en este PC salvo que marques la casilla."
LangString StrataRemoveSubtext ${LANG_RUSSIAN} "Ваши модели, чаты и настройки останутся на этом ПК, если не поставить галочку."
LangString StrataRemoveSubtext ${LANG_PORTUGUESEBR} "Seus modelos, conversas e ajustes ficam neste PC a menos que você marque a caixa."
LangString StrataRemoveSubtext ${LANG_KOREAN} "체크하지 않으면 모델, 대화, 설정은 이 PC에 남습니다."
LangString StrataRemoveSubtext ${LANG_FRENCH} "Vos modèles, conversations et réglages restent sur ce PC sauf si vous cochez la case."

LangString StrataRemoveCheck ${LANG_ENGLISH} "Also remove my Strata data (models, chats, settings)"
LangString StrataRemoveCheck ${LANG_SIMPCHINESE} "同时删除我的 Strata 数据（模型、对话、设置）"
LangString StrataRemoveCheck ${LANG_JAPANESE} "Strata のデータも削除する（モデル、チャット、設定）"
LangString StrataRemoveCheck ${LANG_GERMAN} "Auch meine Strata-Daten löschen (Modelle, Chats, Einstellungen)"
LangString StrataRemoveCheck ${LANG_SPANISH} "Eliminar también mis datos de Strata (modelos, chats, ajustes)"
LangString StrataRemoveCheck ${LANG_RUSSIAN} "Удалить и мои данные Strata (модели, чаты, настройки)"
LangString StrataRemoveCheck ${LANG_PORTUGUESEBR} "Remover também meus dados do Strata (modelos, conversas, ajustes)"
LangString StrataRemoveCheck ${LANG_KOREAN} "내 Strata 데이터도 삭제(모델, 대화, 설정)"
LangString StrataRemoveCheck ${LANG_FRENCH} "Supprimer aussi mes données Strata (modèles, conversations, réglages)"

LangString StrataRemoveText ${LANG_ENGLISH} "If you tick it, these folders are deleted, and nothing else:"
LangString StrataRemoveText ${LANG_SIMPCHINESE} "勾选后只会删除这些文件夹，不会动其他内容："
LangString StrataRemoveText ${LANG_JAPANESE} "チェックすると、次のフォルダーだけが削除されます:"
LangString StrataRemoveText ${LANG_GERMAN} "Mit Häkchen werden nur diese Ordner gelöscht:"
LangString StrataRemoveText ${LANG_SPANISH} "Si la marcas, solo se eliminan estas carpetas:"
LangString StrataRemoveText ${LANG_RUSSIAN} "Если поставить галочку, удалятся только эти папки:"
LangString StrataRemoveText ${LANG_PORTUGUESEBR} "Se marcar, apenas estas pastas são removidas:"
LangString StrataRemoveText ${LANG_KOREAN} "체크하면 다음 폴더만 삭제됩니다:"
LangString StrataRemoveText ${LANG_FRENCH} "Si vous cochez, seuls ces dossiers sont supprimés :"

LangString StrataDataEngine ${LANG_ENGLISH} "engine, installed copy and tunings"
LangString StrataDataEngine ${LANG_SIMPCHINESE} "引擎、已安装的副本和调校"
LangString StrataDataEngine ${LANG_JAPANESE} "エンジン、インストール済みのコピー、調整値"
LangString StrataDataEngine ${LANG_GERMAN} "Engine, installierte Kopie und Abstimmungen"
LangString StrataDataEngine ${LANG_SPANISH} "motor, copia instalada y ajustes"
LangString StrataDataEngine ${LANG_RUSSIAN} "движок, установленная копия и настройки"
LangString StrataDataEngine ${LANG_PORTUGUESEBR} "motor, cópia instalada e ajustes"
LangString StrataDataEngine ${LANG_KOREAN} "엔진, 설치된 복사본, 튜닝"
LangString StrataDataEngine ${LANG_FRENCH} "moteur, copie installée et réglages"

LangString StrataDataSettings ${LANG_ENGLISH} "settings"
LangString StrataDataSettings ${LANG_SIMPCHINESE} "设置"
LangString StrataDataSettings ${LANG_JAPANESE} "設定"
LangString StrataDataSettings ${LANG_GERMAN} "Einstellungen"
LangString StrataDataSettings ${LANG_SPANISH} "ajustes"
LangString StrataDataSettings ${LANG_RUSSIAN} "настройки"
LangString StrataDataSettings ${LANG_PORTUGUESEBR} "ajustes"
LangString StrataDataSettings ${LANG_KOREAN} "설정"
LangString StrataDataSettings ${LANG_FRENCH} "réglages"

LangString StrataDataChats ${LANG_ENGLISH} "your chats and app settings"
LangString StrataDataChats ${LANG_SIMPCHINESE} "你的对话和应用设置"
LangString StrataDataChats ${LANG_JAPANESE} "チャットとアプリの設定"
LangString StrataDataChats ${LANG_GERMAN} "Ihre Chats und App-Einstellungen"
LangString StrataDataChats ${LANG_SPANISH} "tus chats y ajustes de la aplicación"
LangString StrataDataChats ${LANG_RUSSIAN} "ваши чаты и настройки приложения"
LangString StrataDataChats ${LANG_PORTUGUESEBR} "suas conversas e ajustes do aplicativo"
LangString StrataDataChats ${LANG_KOREAN} "대화와 앱 설정"
LangString StrataDataChats ${LANG_FRENCH} "vos conversations et réglages de l’application"

LangString StrataDataModels ${LANG_ENGLISH} "models Strata downloaded into its own folder"
LangString StrataDataModels ${LANG_SIMPCHINESE} "Strata 下载到自己文件夹里的模型"
LangString StrataDataModels ${LANG_JAPANESE} "Strata が自分のフォルダーにダウンロードしたモデル"
LangString StrataDataModels ${LANG_GERMAN} "Modelle, die Strata in seinen eigenen Ordner geladen hat"
LangString StrataDataModels ${LANG_SPANISH} "modelos que Strata descargó en su propia carpeta"
LangString StrataDataModels ${LANG_RUSSIAN} "модели, скачанные Strata в свою папку"
LangString StrataDataModels ${LANG_PORTUGUESEBR} "modelos que o Strata baixou na própria pasta"
LangString StrataDataModels ${LANG_KOREAN} "Strata가 자기 폴더에 내려받은 모델"
LangString StrataDataModels ${LANG_FRENCH} "modèles que Strata a téléchargés dans son propre dossier"

LangString StrataRemoveNote ${LANG_ENGLISH} "A model folder you chose yourself is never deleted."
LangString StrataRemoveNote ${LANG_SIMPCHINESE} "你自己选择的模型文件夹永远不会被删除。"
LangString StrataRemoveNote ${LANG_JAPANESE} "自分で選んだモデルのフォルダーは削除されません。"
LangString StrataRemoveNote ${LANG_GERMAN} "Ein Modellordner, den Sie selbst gewählt haben, wird nie gelöscht."
LangString StrataRemoveNote ${LANG_SPANISH} "Una carpeta de modelos que hayas elegido tú nunca se elimina."
LangString StrataRemoveNote ${LANG_RUSSIAN} "Папка моделей, которую вы выбрали сами, не удаляется никогда."
LangString StrataRemoveNote ${LANG_PORTUGUESEBR} "Uma pasta de modelos escolhida por você nunca é removida."
LangString StrataRemoveNote ${LANG_KOREAN} "직접 고른 모델 폴더는 절대 삭제되지 않습니다."
LangString StrataRemoveNote ${LANG_FRENCH} "Un dossier de modèles que vous avez choisi vous-même n’est jamais supprimé."

Function .onInit
  ${GetOptions} $CMDLINE "/P" $PassiveMode
  ${IfNot} ${Errors}
    StrCpy $PassiveMode 1
  ${EndIf}

  ${GetOptions} $CMDLINE "/NS" $NoShortcutMode
  ${IfNot} ${Errors}
    StrCpy $NoShortcutMode 1
  ${EndIf}

  ${GetOptions} $CMDLINE "/UPDATE" $UpdateMode
  ${IfNot} ${Errors}
    StrCpy $UpdateMode 1
  ${EndIf}

  !if "${DISPLAYLANGUAGESELECTOR}" == "true"
    !insertmacro MUI_LANGDLL_DISPLAY
  !endif

  !insertmacro SetContext

  ${If} $INSTDIR == "${PLACEHOLDER_INSTALL_DIR}"
    ; Set default install location
    !if "${INSTALLMODE}" == "perMachine"
      ${If} ${RunningX64}
        !if "${ARCH}" == "x64"
          StrCpy $INSTDIR "$PROGRAMFILES64\${PRODUCTNAME}"
        !else if "${ARCH}" == "arm64"
          StrCpy $INSTDIR "$PROGRAMFILES64\${PRODUCTNAME}"
        !else
          StrCpy $INSTDIR "$PROGRAMFILES\${PRODUCTNAME}"
        !endif
      ${Else}
        StrCpy $INSTDIR "$PROGRAMFILES\${PRODUCTNAME}"
      ${EndIf}
    !else if "${INSTALLMODE}" == "currentUser"
      StrCpy $INSTDIR "$LOCALAPPDATA\Programs\${PRODUCTNAME}"
    !endif

    Call RestorePreviousInstallLocation
  ${EndIf}


  !if "${INSTALLMODE}" == "both"
    !insertmacro MULTIUSER_INIT
  !endif

  ; Abort downgrades if they are disabled, in the modes that skip the reinstall page choice:
  ; silent, passive and update
  !if "${ALLOWDOWNGRADES}" == "false"
  ${If} ${Silent}
  ${OrIf} $PassiveMode = 1
  ${OrIf} $UpdateMode = 1
    ReadRegStr $R0 SHCTX "${UNINSTKEY}" "DisplayVersion"
    nsis_tauri_utils::SemverCompare "${VERSION}" $R0
    Pop $R0
    ; If downgrading
    ${If} $R0 = -1
      ${If} ${Silent}
        System::Call 'kernel32::AttachConsole(i -1)i.r0'
        ${If} $0 <> 0
          System::Call 'kernel32::GetStdHandle(i -11)i.r0'
          System::call 'kernel32::SetConsoleTextAttribute(i r0, i 0x0004)' ; set red color
          FileWrite $0 "$(silentDowngrades)"
        ${EndIf}
      ${Else}
        MessageBox MB_ICONSTOP "$(silentDowngrades)"
      ${EndIf}
      Abort
    ${EndIf}
  ${EndIf}
  !endif
FunctionEnd

Section WebView2
  ; Check if Webview2 is already installed and skip this section
  ${If} ${RunningX64}
    ReadRegStr $4 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\${WEBVIEW2APPGUID}" "pv"
  ${Else}
    ReadRegStr $4 HKLM "SOFTWARE\Microsoft\EdgeUpdate\Clients\${WEBVIEW2APPGUID}" "pv"
  ${EndIf}
  ${If} $4 == ""
    ReadRegStr $4 HKCU "SOFTWARE\Microsoft\EdgeUpdate\Clients\${WEBVIEW2APPGUID}" "pv"
  ${EndIf}

  ${If} $4 == ""
    ; Webview2 installation
    ;
    ; Skip if updating
    ${If} $UpdateMode <> 1
      !if "${INSTALLWEBVIEW2MODE}" == "downloadBootstrapper"
        Delete "$TEMP\MicrosoftEdgeWebview2Setup.exe"
        DetailPrint "$(webview2Downloading)"
        NSISdl::download "https://go.microsoft.com/fwlink/p/?LinkId=2124703" "$TEMP\MicrosoftEdgeWebview2Setup.exe"
        Pop $0
        ${If} $0 == "success"
          DetailPrint "$(webview2DownloadSuccess)"
        ${Else}
          DetailPrint "$(webview2DownloadError)"
          Abort "$(webview2AbortError)"
        ${EndIf}
        StrCpy $6 "$TEMP\MicrosoftEdgeWebview2Setup.exe"
        Goto install_webview2
      !endif

      !if "${INSTALLWEBVIEW2MODE}" == "embedBootstrapper"
        Delete "$TEMP\MicrosoftEdgeWebview2Setup.exe"
        File "/oname=$TEMP\MicrosoftEdgeWebview2Setup.exe" "${WEBVIEW2BOOTSTRAPPERPATH}"
        DetailPrint "$(installingWebview2)"
        StrCpy $6 "$TEMP\MicrosoftEdgeWebview2Setup.exe"
        Goto install_webview2
      !endif

      !if "${INSTALLWEBVIEW2MODE}" == "offlineInstaller"
        Delete "$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe"
        File "/oname=$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe" "${WEBVIEW2INSTALLERPATH}"
        DetailPrint "$(installingWebview2)"
        StrCpy $6 "$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe"
        Goto install_webview2
      !endif

      Goto webview2_done

      install_webview2:
        DetailPrint "$(installingWebview2)"
        ; $6 holds the path to the webview2 installer
        ExecWait "$6 ${WEBVIEW2INSTALLERARGS} /install" $1
        ${If} $1 = 0
          DetailPrint "$(webview2InstallSuccess)"
        ${Else}
          DetailPrint "$(webview2InstallError)"
          Abort "$(webview2AbortError)"
        ${EndIf}
      webview2_done:
    ${EndIf}
  ${Else}
    !if "${MINIMUMWEBVIEW2VERSION}" != ""
      ${VersionCompare} "${MINIMUMWEBVIEW2VERSION}" "$4" $R0
      ${If} $R0 = 1
        update_webview:
          DetailPrint "$(installingWebview2)"
          ${If} ${RunningX64}
            ReadRegStr $R1 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate" "path"
          ${Else}
            ReadRegStr $R1 HKLM "SOFTWARE\Microsoft\EdgeUpdate" "path"
          ${EndIf}
          ${If} $R1 == ""
            ReadRegStr $R1 HKCU "SOFTWARE\Microsoft\EdgeUpdate" "path"
          ${EndIf}
          ${If} $R1 != ""
            ; Chromium updater docs: https://source.chromium.org/chromium/chromium/src/+/main:docs/updater/user_manual.md
            ; Modified from "HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Microsoft EdgeWebView\ModifyPath"
            ExecWait `"$R1" /install appguid=${WEBVIEW2APPGUID}&needsadmin=true` $1
            ${If} $1 = 0
              DetailPrint "$(webview2InstallSuccess)"
            ${Else}
              MessageBox MB_ICONEXCLAMATION|MB_ABORTRETRYIGNORE "$(webview2InstallError)" IDIGNORE ignore IDRETRY update_webview
              Quit
              ignore:
            ${EndIf}
          ${EndIf}
      ${EndIf}
    !endif
  ${EndIf}
SectionEnd

; STRATA-MIGRATE-BEGIN
; An earlier build installed the app into %LOCALAPPDATA%\Strata, the folder that also holds the user's data (components, install,
; procs, runtime, time-skip). Remove only the app files that build put there; everything else stays.
!macro StrataMigrateOldInstall OLDDIR NEWDIR
  StrCmp "${OLDDIR}" "${NEWDIR}" +8
  IfFileExists "${OLDDIR}\strata-desktop.exe" 0 +7
    Delete "${OLDDIR}\strata-desktop.exe"
    Delete "${OLDDIR}\uninstall.exe"
    RMDir /r "${OLDDIR}\host"
    RMDir /r "${OLDDIR}\node"
    RMDir /r "${OLDDIR}\control"
    RMDir "${OLDDIR}"
!macroend
; STRATA-MIGRATE-END

; STRATA-STOP-BEGIN
; Stop what Strata itself started from the install folder (the app, its host, the node runtime, the engine) before files are removed:
; a running copy keeps its files open, so the folder stayed behind after use. Only processes whose image path is inside INSTDIR are
; stopped; this uninstaller's own process and any program elsewhere are left alone.
!macro StrataStopOwnProcesses INSTALLDIR
  System::Call 'kernel32::SetEnvironmentVariableW(w "STRATA_INSTDIR", w "${INSTALLDIR}")'
  System::Call 'kernel32::GetCurrentProcessId()i.r9'
  System::Call 'kernel32::SetEnvironmentVariableW(w "STRATA_SELFPID", w "$9")'
  ; the installer is a 32-bit program, so plain powershell.exe is the 32-bit one, which cannot read the image path of a 64-bit process: use the 64-bit one
  StrCpy $8 "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
  IfFileExists $8 +2 0
    StrCpy $8 "powershell.exe"
  nsExec::Exec '"$8" -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$d = [IO.Path]::GetFullPath($$env:STRATA_INSTDIR).TrimEnd([char]92) + [char]92; $$ids = @(); $$ids += Get-Process -ErrorAction SilentlyContinue | Where-Object { $$_.Path -and [IO.Path]::GetFullPath($$_.Path).StartsWith($$d, [StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { $$_.Id }; try { $$ids += Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $$_.ExecutablePath -and [IO.Path]::GetFullPath($$_.ExecutablePath).StartsWith($$d, [StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { $$_.ProcessId } } catch {}; $$ids | Sort-Object -Unique | Where-Object { $$_ -ne [int]$$env:STRATA_SELFPID } | ForEach-Object { Stop-Process -Id $$_ -Force -ErrorAction SilentlyContinue }; Start-Sleep -Milliseconds 1500"'
  Pop $9
!macroend
; STRATA-STOP-END

; STRATA-CLEAN-BEGIN
; The app writes into its own folders while it runs (Python bytecode caches under control\, for one), and the file-by-file delete below does not know them,
; so those folders stayed behind with a few dozen files. These four folders hold only what the installer put there and what the app made there;
; user data is never in them (it lives in the data folder), and the install folder itself is still removed with a plain RMDir.
!macro StrataRemoveAppFolders INSTALLDIR
  RMDir /r "${INSTALLDIR}\control"
  RMDir /r "${INSTALLDIR}\host"
  RMDir /r "${INSTALLDIR}\node"
  RMDir /r "${INSTALLDIR}\harness"
!macroend
; STRATA-CLEAN-END

Section Install
  SetOutPath $INSTDIR

  !ifmacrodef NSIS_HOOK_PREINSTALL
    !insertmacro NSIS_HOOK_PREINSTALL
  !endif

  !insertmacro StrataMigrateOldInstall "$LOCALAPPDATA\${PRODUCTNAME}" "$INSTDIR"

  !insertmacro CheckIfAppIsRunning "$INSTDIR\${MAINBINARYNAME}.exe" "${PRODUCTNAME}"

  ; Copy main executable
  File "${MAINBINARYSRCPATH}"

  ; Copy resources
  {{#each resources_dirs}}
    CreateDirectory "$INSTDIR\\{{this}}"
  {{/each}}
  {{#each resources}}
    File /a "/oname={{this.[1]}}" "{{no-escape @key}}"
  {{/each}}

  ; Copy external binaries
  {{#each binaries}}
    File /a "/oname={{this}}" "{{no-escape @key}}"
  {{/each}}

  ; Create file associations
  {{#each file_associations as |association| ~}}
    {{#each association.ext as |ext| ~}}
       !insertmacro APP_ASSOCIATE "{{ext}}" "{{or association.name ext}}" "{{association-description association.description ext}}" "$INSTDIR\${MAINBINARYNAME}.exe,0" "Open with ${PRODUCTNAME}" "$INSTDIR\${MAINBINARYNAME}.exe $\"%1$\""
    {{/each}}
  {{/each}}

  ; Register deep links
  {{#each deep_link_protocols as |protocol| ~}}
    WriteRegStr SHCTX "Software\Classes\\{{protocol}}" "URL Protocol" ""
    WriteRegStr SHCTX "Software\Classes\\{{protocol}}" "" "URL:${BUNDLEID} protocol"
    WriteRegStr SHCTX "Software\Classes\\{{protocol}}\DefaultIcon" "" "$\"$INSTDIR\${MAINBINARYNAME}.exe$\",0"
    WriteRegStr SHCTX "Software\Classes\\{{protocol}}\shell\open\command" "" "$\"$INSTDIR\${MAINBINARYNAME}.exe$\" $\"%1$\""
  {{/each}}

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Save $INSTDIR in registry for future installations
  WriteRegStr SHCTX "${MANUPRODUCTKEY}" "" $INSTDIR

  !if "${INSTALLMODE}" == "both"
    ; Save install mode to be selected by default for the next installation such as updating
    ; or when uninstalling
    WriteRegStr SHCTX "${UNINSTKEY}" $MultiUser.InstallMode 1
  !endif

  ; Remove old main binary if it doesn't match new main binary name
  ReadRegStr $OldMainBinaryName SHCTX "${UNINSTKEY}" "MainBinaryName"
  ${If} $OldMainBinaryName != ""
  ${AndIf} $OldMainBinaryName != "${MAINBINARYNAME}.exe"
    Delete "$INSTDIR\$OldMainBinaryName"
  ${EndIf}

  ; Save current MAINBINARYNAME for future updates
  WriteRegStr SHCTX "${UNINSTKEY}" "MainBinaryName" "${MAINBINARYNAME}.exe"

  ; Registry information for add/remove programs
  WriteRegStr SHCTX "${UNINSTKEY}" "DisplayName" "${PRODUCTNAME}"
  WriteRegStr SHCTX "${UNINSTKEY}" "DisplayIcon" "$\"$INSTDIR\${MAINBINARYNAME}.exe$\""
  WriteRegStr SHCTX "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr SHCTX "${UNINSTKEY}" "Publisher" "${MANUFACTURER}"
  WriteRegStr SHCTX "${UNINSTKEY}" "InstallLocation" "$\"$INSTDIR$\""
  WriteRegStr SHCTX "${UNINSTKEY}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
  WriteRegDWORD SHCTX "${UNINSTKEY}" "NoModify" "1"
  WriteRegDWORD SHCTX "${UNINSTKEY}" "NoRepair" "1"

  ${GetSize} "$INSTDIR" "/M=uninstall.exe /S=0K /G=0" $0 $1 $2
  IntOp $0 $0 + ${ESTIMATEDSIZE}
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD SHCTX "${UNINSTKEY}" "EstimatedSize" "$0"

  !if "${HOMEPAGE}" != ""
    WriteRegStr SHCTX "${UNINSTKEY}" "URLInfoAbout" "${HOMEPAGE}"
    WriteRegStr SHCTX "${UNINSTKEY}" "URLUpdateInfo" "${HOMEPAGE}"
    WriteRegStr SHCTX "${UNINSTKEY}" "HelpLink" "${HOMEPAGE}"
  !endif

  ; Create start menu shortcut
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
    Call CreateOrUpdateStartMenuShortcut
  !insertmacro MUI_STARTMENU_WRITE_END

  ; Create desktop shortcut for silent and passive installers
  ; because finish page will be skipped
  ${If} $PassiveMode = 1
  ${OrIf} ${Silent}
    Call CreateOrUpdateDesktopShortcut
  ${EndIf}

  !ifmacrodef NSIS_HOOK_POSTINSTALL
    !insertmacro NSIS_HOOK_POSTINSTALL
  !endif

  ; Auto close this page for passive mode
  ${If} $PassiveMode = 1
    SetAutoClose true
  ${EndIf}
SectionEnd

Function .onInstSuccess
  ; Check for `/R` flag only in silent and passive installers because
  ; GUI installer has a toggle for the user to (re)start the app
  ${If} $PassiveMode = 1
  ${OrIf} ${Silent}
    ${GetOptions} $CMDLINE "/R" $R0
    ${IfNot} ${Errors}
      ${GetOptions} $CMDLINE "/ARGS" $R0
      nsis_tauri_utils::RunAsUser "$INSTDIR\${MAINBINARYNAME}.exe" "$R0"
    ${EndIf}
  ${EndIf}
FunctionEnd

Function un.onInit
  !insertmacro SetContext

  !if "${INSTALLMODE}" == "both"
    !insertmacro MULTIUSER_UNINIT
  !endif

  !insertmacro MUI_UNGETLANGUAGE

  ${GetOptions} $CMDLINE "/P" $PassiveMode
  ${IfNot} ${Errors}
    StrCpy $PassiveMode 1
  ${EndIf}

  ${GetOptions} $CMDLINE "/UPDATE" $UpdateMode
  ${IfNot} ${Errors}
    StrCpy $UpdateMode 1
  ${EndIf}
FunctionEnd

Section Uninstall

  !ifmacrodef NSIS_HOOK_PREUNINSTALL
    !insertmacro NSIS_HOOK_PREUNINSTALL
  !endif

  !insertmacro StrataStopOwnProcesses "$INSTDIR"
  !insertmacro CheckIfAppIsRunning "$INSTDIR\${MAINBINARYNAME}.exe" "${PRODUCTNAME}"

  ; Delete the app directory and its content from disk
  ; Copy main executable
  Delete "$INSTDIR\${MAINBINARYNAME}.exe"

  ; Delete resources
  {{#each resources}}
    Delete "$INSTDIR\\{{this.[1]}}"
  {{/each}}

  ; Delete external binaries
  {{#each binaries}}
    Delete "$INSTDIR\\{{this}}"
  {{/each}}

  ; Delete app associations
  {{#each file_associations as |association| ~}}
    {{#each association.ext as |ext| ~}}
      !insertmacro APP_UNASSOCIATE "{{ext}}" "{{or association.name ext}}"
    {{/each}}
  {{/each}}

  ; Delete deep links
  {{#each deep_link_protocols as |protocol| ~}}
    ReadRegStr $R7 SHCTX "Software\Classes\\{{protocol}}\shell\open\command" ""
    ${If} $R7 == "$\"$INSTDIR\${MAINBINARYNAME}.exe$\" $\"%1$\""
      DeleteRegKey SHCTX "Software\Classes\\{{protocol}}"
    ${EndIf}
  {{/each}}


  !insertmacro StrataRemoveAppFolders "$INSTDIR"

  ; Delete uninstaller
  Delete "$INSTDIR\uninstall.exe"

  {{#each resources_ancestors}}
  RMDir /REBOOTOK "$INSTDIR\\{{this}}"
  {{/each}}
  RMDir "$INSTDIR"

  ; Remove shortcuts if not updating
  ${If} $UpdateMode <> 1
    !insertmacro DeleteAppUserModelId

    ; Remove start menu shortcut
    !insertmacro MUI_STARTMENU_GETFOLDER Application $AppStartMenuFolder
    !insertmacro IsShortcutTarget "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    Pop $0
    ${If} $0 = 1
      !insertmacro UnpinShortcut "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk"
      Delete "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk"
      RMDir "$SMPROGRAMS\$AppStartMenuFolder"
    ${EndIf}
    !insertmacro IsShortcutTarget "$SMPROGRAMS\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    Pop $0
    ${If} $0 = 1
      !insertmacro UnpinShortcut "$SMPROGRAMS\${PRODUCTNAME}.lnk"
      Delete "$SMPROGRAMS\${PRODUCTNAME}.lnk"
    ${EndIf}

    ; Remove desktop shortcuts
    !insertmacro IsShortcutTarget "$DESKTOP\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    Pop $0
    ${If} $0 = 1
      !insertmacro UnpinShortcut "$DESKTOP\${PRODUCTNAME}.lnk"
      Delete "$DESKTOP\${PRODUCTNAME}.lnk"
    ${EndIf}
  ${EndIf}

  ; Remove registry information for add/remove programs
  !if "${INSTALLMODE}" == "both"
    DeleteRegKey SHCTX "${UNINSTKEY}"
  !else if "${INSTALLMODE}" == "perMachine"
    DeleteRegKey HKLM "${UNINSTKEY}"
  !else
    DeleteRegKey HKCU "${UNINSTKEY}"
  !endif

  ; Removes the Autostart entry for ${PRODUCTNAME} from the HKCU Run key if it exists.
  ; This ensures the program does not launch automatically after uninstallation if it exists.
  ; If it doesn't exist, it does nothing.
  ; We do this when not updating (to preserve the registry value on updates)
  ${If} $UpdateMode <> 1
    DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${PRODUCTNAME}"
  ${EndIf}

  ; Delete app data if the checkbox is selected
  ; and if not updating
  ${If} $DeleteAppDataCheckboxState = 1
  ${AndIf} $UpdateMode <> 1
    ; Clear the install location $INSTDIR from registry
    DeleteRegKey SHCTX "${MANUPRODUCTKEY}"
    DeleteRegKey /ifempty SHCTX "${MANUKEY}"

    ; Clear the install language from registry
    DeleteRegValue HKCU "${MANUPRODUCTKEY}" "Installer Language"
    DeleteRegKey /ifempty HKCU "${MANUPRODUCTKEY}"
    DeleteRegKey /ifempty HKCU "${MANUKEY}"

    SetShellVarContext current
    RmDir /r "$APPDATA\${BUNDLEID}"
    RmDir /r "$LOCALAPPDATA\${BUNDLEID}"

    ; Strata's own data, for a fresh start: the folders listed on the page, nothing else. Each path is checked first.
    ${If} "$LOCALAPPDATA" != ""
      RmDir /r "$LOCALAPPDATA\Strata"
    ${EndIf}
    ${If} "$APPDATA" != ""
      RmDir /r "$APPDATA\Strata"
    ${EndIf}
    ${If} "$PROFILE" != ""
      RmDir /r "$PROFILE\.strata"
    ${EndIf}
    ${If} "$DOCUMENTS" != ""
      RmDir /r "$DOCUMENTS\Strata\Models"
      RmDir "$DOCUMENTS\Strata"          ; only if it is empty now
    ${EndIf}
  ${EndIf}

  !ifmacrodef NSIS_HOOK_POSTUNINSTALL
    !insertmacro NSIS_HOOK_POSTUNINSTALL
  !endif

  ; Auto close if passive mode or updating
  ${If} $PassiveMode = 1
  ${OrIf} $UpdateMode = 1
    SetAutoClose true
  ${EndIf}
SectionEnd

; StrIsUnder: Push path, Push folder, Call StrIsUnder, Pop result ("1" when path is the folder or inside it; case-insensitive,
; whole path segments only).
Function StrIsUnder
  Exch $1                  ; folder
  Exch
  Exch $0                  ; path
  Push $2
  Push $3
  StrCpy $R9 0
  StrCmp $1 "" under_done 0
  StrLen $2 $1
  StrCpy $3 $0 $2
  StrCmp $3 $1 0 under_done
  StrCpy $3 $0 1 $2
  StrCmp $3 "" under_yes 0
  StrCmp $3 "\" under_yes under_done
  under_yes:
    StrCpy $R9 1
  under_done:
  Pop $3
  Pop $2
  Pop $0
  Pop $1
  Push $R9
FunctionEnd

; Strata: the install folder remembered in the registry is only reused when it is a real place for an app. A location under the
; temp folders (a test install, a build machine's scratch run) is ignored and the default per-user folder is used instead.
Function RestorePreviousInstallLocation
  ReadRegStr $4 SHCTX "${MANUPRODUCTKEY}" ""
  StrCmp $4 "" restore_done 0
  ; %LOCALAPPDATA%\Strata holds the user's engines, model configs and tuning: never an app folder
  StrCmp $4 "$LOCALAPPDATA\${PRODUCTNAME}" restore_done 0
  Push $4
  Push "$LOCALAPPDATA\Temp"
  Call StrIsUnder
  Pop $5
  StrCmp $5 1 restore_done 0
  Push $4
  Push "$TEMP"
  Call StrIsUnder
  Pop $5
  StrCmp $5 1 restore_done 0
  ReadEnvStr $6 TMP
  Push $4
  Push $6
  Call StrIsUnder
  Pop $5
  StrCmp $5 1 restore_done 0
  ReadEnvStr $6 TEMP
  Push $4
  Push $6
  Call StrIsUnder
  Pop $5
  StrCmp $5 1 restore_done 0
  StrCpy $INSTDIR $4
  restore_done:
FunctionEnd

Function Skip
  Abort
FunctionEnd

Function SkipIfPassive
  ${IfThen} $PassiveMode = 1  ${|} Abort ${|}
FunctionEnd
Function un.SkipIfPassive
  ${IfThen} $PassiveMode = 1  ${|} Abort ${|}
FunctionEnd

Function CreateOrUpdateStartMenuShortcut
  ; We used to use product name as MAINBINARYNAME
  ; migrate old shortcuts to target the new MAINBINARYNAME
  StrCpy $R0 0

  !insertmacro IsShortcutTarget "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk" "$INSTDIR\$OldMainBinaryName"
  Pop $0
  ${If} $0 = 1
    !insertmacro SetShortcutTarget "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    StrCpy $R0 1
  ${EndIf}

  !insertmacro IsShortcutTarget "$SMPROGRAMS\${PRODUCTNAME}.lnk" "$INSTDIR\$OldMainBinaryName"
  Pop $0
  ${If} $0 = 1
    !insertmacro SetShortcutTarget "$SMPROGRAMS\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    StrCpy $R0 1
  ${EndIf}

  ${If} $R0 = 1
    Return
  ${EndIf}

  ; Skip creating shortcut if in update mode or no shortcut mode
  ; but always create if migrating from wix
  ${If} $WixMode = 0
    ${If} $UpdateMode = 1
    ${OrIf} $NoShortcutMode = 1
      Return
    ${EndIf}
  ${EndIf}

  !if "${STARTMENUFOLDER}" != ""
    CreateDirectory "$SMPROGRAMS\$AppStartMenuFolder"
    CreateShortcut "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe" "" "$INSTDIR\${MAINBINARYNAME}.exe" 0
    !insertmacro SetLnkAppUserModelId "$SMPROGRAMS\$AppStartMenuFolder\${PRODUCTNAME}.lnk"
  !else
    CreateShortcut "$SMPROGRAMS\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe" "" "$INSTDIR\${MAINBINARYNAME}.exe" 0
    !insertmacro SetLnkAppUserModelId "$SMPROGRAMS\${PRODUCTNAME}.lnk"
  !endif
FunctionEnd

Function CreateOrUpdateDesktopShortcut
  ; We used to use product name as MAINBINARYNAME
  ; migrate old shortcuts to target the new MAINBINARYNAME
  !insertmacro IsShortcutTarget "$DESKTOP\${PRODUCTNAME}.lnk" "$INSTDIR\$OldMainBinaryName"
  Pop $0
  ${If} $0 = 1
    !insertmacro SetShortcutTarget "$DESKTOP\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    Return
  ${EndIf}

  ; Skip creating shortcut if in update mode or no shortcut mode
  ; but always create if migrating from wix
  ${If} $WixMode = 0
    ${If} $UpdateMode = 1
    ${OrIf} $NoShortcutMode = 1
      Return
    ${EndIf}
  ${EndIf}

  CreateShortcut "$DESKTOP\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe" "" "$INSTDIR\${MAINBINARYNAME}.exe" 0
  !insertmacro SetLnkAppUserModelId "$DESKTOP\${PRODUCTNAME}.lnk"
FunctionEnd
