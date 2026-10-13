#!/bin/bash
# Build the engine with an MSVC environment assembled BY HAND.
#
# vcvars64.bat is the normal way in, but it runs reg.exe, which this machine's command security policy
# blocks - so the same four variables are set here from paths that exist.  Nothing is guessed: every path
# below was listed from the installed BuildTools 14.44.35207 and Windows SDK 10.0.26100.0.
set -u
VCDIR="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
SDK="C:/Program Files (x86)/Windows Kits/10"
SDKVER="10.0.26100.0"
NINJA="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"

export INCLUDE="$VCDIR\\include;$SDK\\Include\\$SDKVER\\ucrt;$SDK\\Include\\$SDKVER\\um;$SDK\\Include\\$SDKVER\\shared;$SDK\\Include\\$SDKVER\\winrt"
export LIB="$VCDIR\\lib\\x64;$SDK\\Lib\\$SDKVER\\ucrt\\x64;$SDK\\Lib\\$SDKVER\\um\\x64"
export LIBPATH="$VCDIR\\lib\\x64;$SDK\\Lib\\$SDKVER\\ucrt\\x64;$SDK\\Lib\\$SDKVER\\um\\x64"
export PATH="$VCDIR/bin/Hostx64/x64:$SDK/bin/$SDKVER/x64:$PATH"

cd /g/01_Projects/Strata/build-cuda || exit 1
"$NINJA" "$@"
