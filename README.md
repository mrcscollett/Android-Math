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

- **Physics**: real situations set up from the physical law and solved as
  differential equations. Each model is a list of variables with units: type
  in the ones you know, leave the rest empty, and every empty one that your
  values determine is worked out step by step (any variable can be the
  unknown, including a time, a starting value or a constant such as a drag
  coefficient). Optional effects are switches that add their own variables
  and terms (air resistance, buoyancy, wind, damping, a driving force, a
  heat source, leakage, internal resistance, an AC source, ...). If something
  cannot be found the result says which other value is needed; values that
  contradict each other are reported with the relation they break. Models
  so far: falling object, projectile, mass on a spring, pendulum, cooling
  and heating, exponential decay and growth, RC circuit, RL circuit.

## Layout

| Path | What it is |
|---|---|
| `app/src/main/python/deengine/` | The solver engine: Python + SymPy, run on-device through Chaquopy. Produces the steps as JSON. Physics models and their knowns-to-unknowns solver are in `scenarios/`. |
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

## Adding a physics model

A model is one module in `app/src/main/python/deengine/scenarios/`. It
declares its variables and the relations between them; the generic solver in
`scenarios/solver.py` does the rest, and the app builds its form from the
declaration, so no app code changes are needed.

1. Declare `MODEL` with the constructors in `scenarios/spec.py`: `field(...)`
   for every variable (its quantity picks the unit choices from `units.py`;
   its group is `setup`, `initial`, `effect`, `moment`, `moment2` or
   `derived`; `typical` gives the solver a sense of scale), `switch(...)` or
   `choice(...)` for each optional effect and the variables it adds, and
   `preset(...)` for real examples: the known values, with interesting ones
   left blank.
2. Write `relations(effects)`: a list of `Rel(id, name, latex, variables,
   fn)`, where `fn(values)` returns the two sides of an equation among those
   variables in SI units. Algebraic links (`tau = R C`) and "the state at
   time t from the solution of the ODE" are both relations; the latter get
   `traj=True`. The solver finds any single unknown of a relation by a
   bracketed root search (which is also the shooting method when the unknown
   is a parameter of the ODE), and two unknowns shared by two relations
   together. Optional `forms={var: latex}` gives the rearranged formula
   shown in the steps.
3. Write `trajectory(effects, values)` (the solution: a closed form where
   one exists, otherwise `common.numeric(...)`, plus the right-hand side used
   for the independent check), `derivation(effects)` and `equation(effects)`
   (the law, the ODE and its solution for the steps and the live preview),
   `horizon(effects, values)` (a natural time span) and `graph(effects,
   values)` (the curve, the ideal curve and the marked moment).
4. Add the module to `MODELS` in `scenarios/__init__.py`.
5. Add cases to `PHYSICS_CASES` in `tests/python/test_engine.py`: each is
   checked against an independent integration of the equation written out in
   the test, and then every variable in turn is blanked and must be
   recovered from the others. Add the model to `_reference` there.

## Adding a solution method

Write a function in `deengine/first_order.py` that takes `(prob, f, ic)` for
`y' = f(x, y)` and returns either `_not_applicable(...)` with the reason, or
`_result(...)` with its list of steps, then add it to `METHODS`. Second-order
methods live in `deengine/second_order.py` and take `(eq, ic)`. The app shows
every method that applies and explains why the others do not.
