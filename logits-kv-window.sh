#!/bin/bash
# Quantitative check of the FP16 sliding window + attention sinks: dump the logits of one fixed 128-token prompt
# with (a) exact FP16 KV, (b) --kv q4_0, (c) --kv q4_0 + the overlay. The overlay must move the logits TOWARD the
# FP16 reference - that is what the exact copy of the sinks and the last `window` cells buys.
set -u
ROOT=C:/Users/blsor/Strata
EXE="$ROOT/build/strata.exe"
IDS="1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127"
COMMON=(--pack "C:\\Users\\blsor\\Strata-data\\packs\\swift-iq3_xxs"
  --native "C:\\Users\\blsor\\Strata-data\\models\\swift-IQ3_XXS\\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
  --ple-gguf "C:\\Users\\blsor\\Strata-data\\models\\swift-IQ3_XXS\\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
  --expert-profile "C:\\Users\\blsor\\Strata\\data\\expert-profile.bin"
  --expert-cache auto --prefill auto --spec 4 --max-context 4096
  --tokens "$IDS" --logits-stride 127 --max-new 1)

stop_engine() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*serve/server.py*' -or \$_.Name -eq 'strata.exe' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  sleep 5
}

dump() {   # $1 = arm, rest = extra args
  local arm="$1"; shift
  stop_engine
  echo "=== $arm ==="
  "$EXE" "${COMMON[@]}" --kv "$KV" "$@" --dump-logits "C:\\Users\\blsor\\Strata\\logits-$arm.txt" 2>&1 | tail -3
  ls -la "$ROOT/logits-$arm.txt" 2>/dev/null | awk '{print $5, $9}'
}

schtasks /End /TN StrataServe >/dev/null 2>&1
KV=fp16        dump fp16
KV=q4_0        dump q4_0
KV=q4_0        dump q4_0win --kv-window 128 --kv-sink 8
schtasks /Run /TN StrataServe >/dev/null 2>&1
echo "=== done ==="