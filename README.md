# DiffEq Solver (Android)

Step-by-step differential equation solver for Android. Phase 1 covers
first-order equations that are separable or linear, with initial value
problems and an interactive slope field.

## Layout

| Path | What it is |
|---|---|
| `app/src/main/python/deengine/` | The solver engine: Python + SymPy, run on-device through Chaquopy. Produces the steps as JSON. |
| `app/src/main/assets/math/` | The steps page (KaTeX) shown in a WebView. |
| `app/src/main/kotlin/` | Jetpack Compose UI, the engine bridge and the graph. |
| `core/` | Pure Kotlin: expression evaluator, RK4, slope-field and plotting maths, JSON models. |
| `tests/python/` | Engine tests (solutions are checked against an independent numerical integration). |
| `tests/smoke.sh` | Emulator smoke test used by CI. |

## Build and test

```sh
pip install sympy==1.14.0 pytest
python -m pytest tests/python          # engine
./gradlew -PcoreOnly :core:test        # Kotlin core, no Android SDK needed
./gradlew :app:assembleDebug           # APK (needs the Android SDK and Python 3.12 on PATH)
```

Try the engine from a terminal:

```sh
python tests/python/show.py "y' + 2y = x" 0 1
```

## Adding a solution method

Write a function in `deengine/first_order.py` that takes `(prob, f, ic)` for
`y' = f(x, y)` and returns either `_not_applicable(...)` with the reason, or
`_result(...)` with its list of steps, then add it to `METHODS`. The app
shows every method that applies and explains why the others do not.
