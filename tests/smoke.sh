#!/usr/bin/env bash
# Emulator smoke test: install the APK, open problems through the launch
# intent, and check the screen contents and the log for crashes.
set -u
APK="$1"
PKG=com.androidmath.app
OUT=smoke-out
mkdir -p "$OUT"
fail=0

adb install -r "$APK" || exit 1
adb logcat -c

open() { # name equation x0 y0 tab text-to-wait-for
  adb shell am force-stop $PKG
  local extras="--es equation '$2' --ei tab $5"
  [ -n "$3" ] && extras="$extras --es x0 '$3' --es y0 '$4'"
  # Single quotes keep the device shell from expanding * and ^.
  adb shell "am start -W -n $PKG/.MainActivity $extras" >/dev/null
  # First launch extracts Python and imports SymPy, which is slow on an emulator.
  for i in $(seq 1 24); do
    sleep 5
    adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
    adb exec-out cat /sdcard/ui.xml > "$OUT/$1.xml" 2>/dev/null
    grep -q "$6" "$OUT/$1.xml" && break
  done
  sleep 6
  adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
  adb exec-out cat /sdcard/ui.xml > "$OUT/$1.xml"
  adb exec-out screencap -p > "$OUT/$1.png"
  echo "--- $1: text on screen"
  grep -o 'text="[^"]\+"' "$OUT/$1.xml" | sort -u | head -60
  echo "--- $1: foreground activity"
  adb shell dumpsys activity activities | grep -E "ResumedActivity|topResumedActivity" | head -3
  adb logcat -d > "$OUT/$1-logcat.txt"
  adb logcat -c
  if grep -qE "FATAL EXCEPTION|Fatal signal|ANR in $PKG|E Engine|E/Engine" "$OUT/$1-logcat.txt"; then
    echo "FAIL $1: crash or engine error in the log"
    grep -B2 -A40 -E "FATAL EXCEPTION|Fatal signal|ANR in $PKG|E Engine|E/Engine" "$OUT/$1-logcat.txt" | head -120
    fail=1
  else
    echo "ok   $1 log has no crash"
  fi
}

expect() { # name pattern
  if grep -q "$2" "$OUT/$1.xml"; then echo "ok   $1 has $2"; else echo "FAIL $1 lacks $2"; fail=1; fi
}

open steps "dy/dx=x*y" 0 1 0 'text="Steps"'
expect steps 'text="Solution"'
expect steps 'text="Separation of variables"'
expect steps 'text="Integrating factor (linear)"'
expect steps 'Integrate both sides'

open graph "dy/dx=x^2+y^2" "" "" 1 'text="Graph"'
expect graph 'text="Graph"'
expect graph 'Numerical (RK4)'

open input_error "dy/dx=z" "" "" 0 "Unknown symbol"
expect input_error 'Unknown symbol'

exit $fail
