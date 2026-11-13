# DiffEq Solver (Android)

Step-by-step differential equation solver for Android. Every problem is
solved by each method that applies, and the app explains why the others do not.

What it covers so far:

- **First order**: separable, linear (integrating factor), exact (including
  integrating factors), Bernoulli, homogeneous (y = vx) and Laplace transforms,
  with initial value problems and an interactive slope field.
- **Second order**: constant coefficients (characteristic equation),
  undetermined coefficients, variation of parameters, homogeneous
  Cauchy-Euler equations and Laplace transforms.
- **Series solutions**: power series at ordinary points and Frobenius series
  at regular singular points.
- **Laplace transforms**: forward and inverse transforms with the rules shown
  (shifting theorems, partial fractions, convolution), and step and impulse
  forcing in equations.
- **Systems**: eigenvalue method and matrix exponential for linear 2x2 systems,
  linearization of nonlinear systems, and phase portraits.
- **Numerical methods**: Euler, improved Euler, midpoint, RK4,
  Adams-Bashforth and adaptive RK45, with step tables and the error against
  the exact solution when one is known.

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
python tests/python/show.py "y'' + 3y' + 2y = e^x" 0 1 0
```

## Adding a solution method

Write a function in `deengine/first_order.py` that takes `(prob, f, ic)` for
`y' = f(x, y)` and returns either `_not_applicable(...)` with the reason, or
`_result(...)` with its list of steps, then add it to `METHODS`. Second-order
methods live in `deengine/second_order.py` and take `(eq, ic)`. The app shows
every method that applies and explains why the others do not.
