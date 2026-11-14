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

open() { # name equation x0 y0 tab text-to-wait-for [extra am arguments]
  adb shell am force-stop $PKG
  local extras="--es equation '$2' --ei tab $5 ${7:-}"
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
  if ! adb shell dumpsys activity activities | grep -q "ResumedActivity.*$PKG"; then
    echo "FAIL $1: the app is not in the foreground; related log lines:"
    grep -iE "androidmath|chromium|render process|has died|Killing|lowmemorykiller|MathWebView|am_crash|Force finishing" "$OUT/$1-logcat.txt" | tail -80
    fail=1
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

open bernoulli "dy/dx+y=x*y^2" 0 1 0 'text="Steps"'
expect bernoulli 'text="Bernoulli substitution"'

# Second order, written with d2y/dx2 so the shell needs no nested quotes.
open second "d2y/dx2+3*dy/dx+2*y=e^x" 0 1 0 'text="Steps"' "--es mode second --es yp0 0"
expect second 'text="Undetermined coefficients"'
expect second 'text="Variation of parameters"'
# The third chip (Laplace transform) is scrolled off this narrow screen, so check the count.
expect second 'text="3 ways to solve this"'
expect second 'Guess the particular solution'

open second_graph "d2y/dx2+y=0" "" "" 1 'text="Graph"' "--es mode second"
expect second_graph 'Exact solution'
expect second_graph '= 1.00'

open numeric "dy/dx=x*y" 0 1 0 'text="Steps"' "--es mode numeric --es h 0.25 --es n 4"
expect numeric 'text="Approximation"'
expect numeric 'text="Euler"'
expect numeric 'All steps'
expect numeric '1.419434'

open numeric_graph "dy/dx=x*y" 0 1 1 'text="Graph"' "--es mode numeric --es h 0.25 --es n 4"
expect numeric_graph 'Exact solution'
expect numeric_graph 'text="Euler"'

open forced "d2y/dt2+y=u(t-2)" 0 0 0 'text="Steps"' "--es mode second --es yp0 0"
expect forced 'text="Laplace transform"'
expect forced 'Transform back'

open laplace "e^(2*t)*sin(3*t)" "" "" 0 'text="Steps"' "--es mode laplace"
expect laplace 'text="Transform"'
expect laplace 'text="Table and rules"'
expect laplace 'First shifting theorem'

open inverse "1/(s*(s+1))" "" "" 0 'text="Steps"' "--es mode inverse"
expect inverse 'text="Partial fractions and table"'
expect inverse 'Split into partial fractions'

open inverse_graph "e^(-2*s)/(s^2+1)" "" "" 1 'text="Graph"' "--es mode inverse"
expect inverse_graph 'Function of t'

open series "d2y/dx2-x*y=0" "" "" 0 'text="Steps"' "--es mode second"
expect series 'text="Series solution"'
expect series 'Find the recurrence relation'

open system "x+2*y" 1 0 0 'text="Steps"' "--es mode system --es equation2 '3*x+2*y'"
expect system 'text="Eigenvalue method"'
expect system 'text="Matrix exponential"'
expect system 'Find the eigenvectors'

open system_graph "x*(1-y)" "" "" 1 'text="Graph"' "--es mode system --es equation2 'y*(x-1)'"
expect system_graph 'Tap the phase plane'
expect system_graph 'Equilibria'

# Third order, written with d3y/dx3; the derivative values go in one comma-separated extra.
open third "d3y/dx3-6*d2y/dx2+11*dy/dx-6*y=0" 0 1 0 'text="Steps"' "--es mode second --es yp0 '0,0'"
expect third 'text="Characteristic equation"'
expect third 'Solve for the roots'

open fourier "x" pi "" 0 'text="Steps"' "--es mode fourier"
expect fourier 'text="Fourier series"'
expect fourier 'Compute the coefficients'

open bvp "d2y/dx2+y=0" "" "" 0 'text="Steps"' "--es mode boundary --es a 0 --es va 0 --es b pi/2 --es vb 1"
expect bvp 'text="Boundary value problem"'
expect bvp 'Impose the boundary conditions'

# The eigenvalue problem has no equation field; the equation extra only triggers the launch.
open eigen "0" pi "" 1 'text="Graph"' "--es mode eigen --es bc dd"
expect eigen 'text="Eigenvalue problem"'
expect eigen 'Eigenfunction 1'

open system_forced "x+2*y+e^t" 0 0 0 'text="Steps"' "--es mode system --es equation2 '3*x+2*y'"
expect system_forced 'Variation of parameters (systems)'
expect system_forced 'Find a fundamental matrix'

open system_three "x+y" 1 0 0 'text="Steps"' "--es mode system --es equation2 'y+z' --es equation3 '2*z' --es z0 1"
expect system_three 'text="Eigenvalue method"'
expect system_three 'Find the eigenvectors'

open heat "x*(1-x)" 1 1 0 'text="Steps"' "--es mode heat --es bc 00"
expect heat 'text="Heat equation"'
expect heat 'Separate the variables'
expect heat 'Long-time behaviour'

open wave "sin(pi*x)" 1 2 1 'text="Graph"' "--es mode wave"
expect wave 'text="Wave equation"'
expect wave 'text="Play"'
expect wave 'Initial profile'

# Laplace's equation has its own mode name: plain "laplace" is the transform section.
open laplace_eq "sin(pi*x)" 1 1 0 'text="Steps"' "--es mode laplace_eq --es bc top"
expect laplace_eq 'Solve in the other direction'

# x0 and y0 carry the coefficients a and b here.
open characteristics "e^(-x^2)" 2 1 0 'text="Steps"' "--es mode characteristics"
expect characteristics 'text="Characteristics"'
expect characteristics 'Solve for the characteristic curves'

# A, B, C = 1, 0, -4 (the wave equation); the equation extra is not used.
open classify "0" 1 0 0 'text="Steps"' "--es mode classify --es t1 '-4'"
expect classify 'text="Classification"'
expect classify 'Compute the discriminant'

open fd "sin(pi*x)" 1 1 1 'text="Graph"' "--es mode fd --es t1 4 --es t2 0.02 --es steps 10"
expect fd 'text="Finite differences"'
expect fd 'Grid values'

open disk "cos(t)" 2 1 1 'text="Graph"' "--es mode disk"
expect disk 'The disk has radius'

# The problems opened above must now be listed on the home screen.
adb shell am force-stop $PKG
adb shell "am start -W -n $PKG/.MainActivity" >/dev/null
sleep 8
# The list sits below the section cards and only on-screen rows are in the
# dump, so scroll a screenful at a time until the Recent header shows up.
for i in $(seq 1 15); do
  adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
  adb exec-out cat /sdcard/ui.xml > "$OUT/history.xml"
  grep -q 'Clear recent' "$OUT/history.xml" && break
  adb shell input swipe 160 520 160 180 400
  sleep 1
done
adb exec-out screencap -p > "$OUT/history.png"
expect history 'Clear recent'

# Math tools. The operation is passed as "op"; x0 and y0 are the tool's small boxes.
open linalg "2 1; 1 3" "5 10" "" 0 'text="Steps"' "--es mode linalg --es op solve"
expect linalg 'text="Linear system"'
expect linalg 'text="Gaussian elimination"'
expect linalg 'Write the augmented matrix'

open calculus "x*cos(x^2)" "" "" 0 'text="Steps"' "--es mode calculus --es op integral"
expect calculus 'text="Integral"'
expect calculus 'text="Substitution"'
expect calculus 'Substitute back'

open series "sin(x)" "" "" 1 'text="Graph"' "--es mode series --es op taylor"
expect series 'text="Taylor series"'
expect series 'Taylor polynomial, degree'

open complex "(1+2*i)/(3-i)" "" "" 0 'text="Steps"' "--es mode complex --es op evaluate"
expect complex 'text="Complex number"'
expect complex 'Multiply by the conjugate'

open special "3" "" "" 0 'text="Steps"' "--es mode special --es op legendre"
expect special 'text="Legendre polynomials"'
expect special 'Rodrigues'

open ftransform "e^(-x^2)" "" "" 0 'text="Steps"' "--es mode ftransform --es op forward"
expect ftransform 'text="Fourier transform"'
expect ftransform 'text="Defining integral"'

open recurrence "a(n+2)=a(n+1)+a(n)" "" "" 0 'text="Steps"' "--es mode recurrence --es op solve --es x0 '0,1'"
expect recurrence 'text="Difference equation"'
expect recurrence 'Write the characteristic equation'

open dynamics "y;-x-y" "" "" 0 'text="Steps"' "--es mode dynamics --es op lyapunov"
expect dynamics 'text="Lyapunov stability"'
expect dynamics 'Differentiate along solutions'

open bifurcation "m*x-x^3" "" "" 1 'text="Graph"' "--es mode dynamics --es op bifurcation"
expect bifurcation 'text="Bifurcation"'
expect bifurcation 'the parameter m'

open vector "-y;x" "cos(t);sin(t)" "0,2*pi" 0 'text="Steps"' "--es mode vector --es op line"
expect vector 'text="Line integral"'
expect vector 'Take the dot product'

open gradient "x^2-y^2" "" "" 0 'text="Steps"' "--es mode vector --es op gradient"
expect gradient 'text="Gradient"'
expect gradient 'Form the gradient'

# Physics scenarios: the scenario id travels as "equation", the question as "op".
open physics_fall "falling" "" "" 0 'text="Steps"' "--es mode physics --es preset skydiver --es op ground"
expect physics_fall 'text="Falling object"'
expect physics_fall 'Reaches the ground after'
expect physics_fall 'Set up the equation from Newton'

open physics_spring "spring" "" "" 1 'text="Graph"' "--es mode physics --es preset car --es op motion"
expect physics_spring 'text="Mass on a spring"'
expect physics_spring 'text="Play"'
expect physics_spring 'Underdamped'

open physics_rc "rc" "" "" 0 'text="Steps"' "--es mode physics --es preset timer --es op target"
expect physics_rc 'text="RC circuit"'
expect physics_rc 'The capacitor reaches'
expect physics_rc 'Kirchhoff'

exit $fail
