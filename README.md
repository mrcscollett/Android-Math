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
- **Third and fourth order**: constant coefficients by the characteristic
  equation and undetermined coefficients.
- **Also**: reduction of order, variation of parameters for Cauchy-Euler and
  other variable-coefficient equations, a phase line for autonomous equations,
  and the integration technique (substitution, parts, partial fractions) shown
  inside the steps.
- **Series solutions**: power series at ordinary points and Frobenius series
  at regular singular points.
- **Laplace transforms**: forward and inverse transforms with the rules shown
  (shifting theorems, partial fractions, convolution), and step and impulse
  forcing in equations.
- **Fourier series**: full series on (-L, L) and half-range sine and cosine
  series, with the coefficient integrals and a graph of the partial sums.
- **Boundary value problems**: two-point conditions on y or y' (one solution,
  none, or infinitely many), and the eigenvalue problem y'' + lambda y = 0.
- **Systems**: eigenvalue method and matrix exponential for linear 2x2 and 3x3
  systems, forced 2x2 systems (shifted equilibrium, variation of parameters),
  linearization of nonlinear systems, and phase portraits.
- **History and favorites**: solved problems are listed on the home screen and
  can be reopened or starred.
- **Partial differential equations**: the heat equation on a rod (five kinds
  of boundary conditions), the wave equation on a string with fixed ends
  (series and d'Alembert's formula) and Laplace's equation on a rectangle, by
  separation of variables, with an animated graph or a heat map. Also
  Laplace's equation on a disk, the method of characteristics for first-order
  equations, classification of second-order equations, and the explicit and
  Crank-Nicolson finite-difference schemes for the heat equation.
- **Stability and dynamics**: Lyapunov's direct method (quadratic candidates
  or your own function), conservative systems and their energy, limit cycles
  (Bendixson, polar coordinates, Lienard), bifurcations of a one-parameter
  equation with a diagram, and the Hopf bifurcation.
- **Vector calculus**: gradient, directional derivative, Laplacian, divergence,
  curl and potential functions; line integrals; double integrals in Cartesian
  and polar coordinates; Green's theorem, the divergence theorem and Stokes'
  theorem on standard regions, each computed both ways.
- **Numerical methods**: Euler, improved Euler, midpoint, RK4,
  Adams-Bashforth and adaptive RK45, with step tables and the error against
  the exact solution when one is known.
- **Linear algebra**: row reduction, linear systems (elimination, Cramer's
  rule, inverse), determinants, inverses, eigenvalues with diagonalization or
  Jordan form, rank and null space, independence, the Wronskian.
- **Calculus**: integrals with the technique shown, definite and improper
  integrals, partial fractions, derivatives.
- **Series**: Taylor series, convergence tests for numerical series, radius
  and interval of convergence.
- **Complex numbers**: arithmetic, polar and exponential form, De Moivre's
  theorem, n-th roots.
- **Special functions**: Bessel functions (series, zeros, half-integer closed
  forms), Legendre, Hermite, Laguerre and Chebyshev polynomials, the
  Sturm-Liouville (self-adjoint) form of an equation, Legendre expansions.
- **Fourier transforms**: transforms and inverses with the convention
  F(w) = integral of f(x) e^(-iwx) dx, sine and cosine transforms, and the
  heat equation on the whole line through the heat kernel.
- **Difference equations**: linear recurrences with constant coefficients up
  to third order, by the characteristic equation and undetermined
  coefficients, with starting values.

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
