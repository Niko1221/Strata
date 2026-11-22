#!/usr/bin/env bash
# GPU-temp-linked chassis fan controller (Super I/O PWM, e.g. Nuvoton nct6779).
# Curve keys on the HOTTEST GPU junction across all cards; ramps up fast,
# down slow (hysteresis), and RE-CLAIMS manual PWM mode every cycle because
# board firmware reasserts smart-fan mode behind your back.
#
# CONFIG: edit the three values below, then run as root via the systemd unit.
PWMS=(pwm4 pwm5)   # which Super-I/O headers feed your GPU fans (dip-test them!)
GPU_COUNT=4        # number of GPUs whose junction temps feed the curve
GPU_DEVICE_ID=0x66a1   # PCI device id of the GPU (MI50 = 0x66a1)
# END CONFIG
#
# Verified mapping on our board (ASRock X399 Taichi, NCT6779):
#   pwm4 + pwm5 -> the two 80mm GPU-cooling fans (0.7A each, one per header:
#   two on one header via splitter = 1.4A on a 1A header — split them).
#   fan1 (CPU_FAN) reads 0: nothing on that header. Check YOUR header ratings.
set -u
HWMON_DIR=/sys/class/hwmon
INTERVAL=5
FLOOR=90    # ~35% — minimum audible-but-quiet
CEIL=255

find_nct() {
  # Resolve by chip NAME every time — the hwmon index shifts between boots
  # AND mid-session (the chip drops off the bus and re-registers).
  for h in $HWMON_DIR/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = nct6779 ] && { echo $h; return; }
  done
}

# Resolve GPU hwmons by PCI device id so a bus re-enumeration can't misfire.
gpu_dirs=()
for h in $HWMON_DIR/hwmon*; do
  [ "$(cat $h/name 2>/dev/null)" = amdgpu ] && \
    grep -q $GPU_DEVICE_ID $h/device/device 2>/dev/null && gpu_dirs+=("$h")
done
[ ${#gpu_dirs[@]} -eq $GPU_COUNT ] || { echo "expected $GPU_COUNT GPU hwmons, got ${#gpu_dirs[@]}"; exit 1; }
NCT=$(find_nct)
[ -n "$NCT" ] && [ -w "$NCT/${PWMS[0]}" ] || { echo "nct6779 not writable (modprobe nct6775? root?)"; exit 1; }

while :; do
  NCT=$(find_nct)
  if [ -z "$NCT" ]; then
    modprobe nct6775 2>/dev/null   # chip dormant at boot on some boards; persist via /etc/modules-load.d/
    sleep 3; continue
  fi

  max_j=0
  for g in "${gpu_dirs[@]}"; do
    for t in $g/temp*_label; do
      [ -f "$t" ] || continue
      if grep -q junction "$t"; then
        v=$(cat "${t%_label}_input" 2>/dev/null) || v=0
        (( v > max_j )) && max_j=$v
      fi
    done
  done
  j=$(( max_j / 1000 ))   # °C

  # Piecewise-linear curve on hottest junction:
  #  <=40C -> FLOOR | 40-60 -> FLOOR..140 | 60-75 -> 140..200 | 75-85 -> 200..CEIL | >=85 -> CEIL
  if   (( j <= 40 )); then pwm=$FLOOR
  elif (( j <= 60 )); then pwm=$(( FLOOR + (j-40)*(140-FLOOR)/20 ))
  elif (( j <= 75 )); then pwm=$(( 140 + (j-60)*60/15 ))
  elif (( j <= 85 )); then pwm=$(( 200 + (j-75)*(CEIL-200)/10 ))
  else pwm=$CEIL
  fi

  for P in "${PWMS[@]}"; do
    PWM=$NCT/$P; PWVEN=$NCT/${P}_enable
    # Hysteresis: step up fast, ease down max 15/cycle, deadband ±6.
    cur=$(cat "$PWM" 2>/dev/null) || cur=$FLOOR
    if (( pwm > cur + 6 )); then target=$pwm
    elif (( pwm < cur - 6 )); then target=$(( cur - 15 > pwm ? cur - 15 : pwm ))
    else target=$cur
    fi
    # Re-claim manual mode (firmware fights us: writes fail EBUSY otherwise)
    # and apply.
    echo 0 > "$PWVEN" 2>/dev/null
    echo "$target" > "$PWM" 2>/dev/null
  done

  sleep $INTERVAL
done
