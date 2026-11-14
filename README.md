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

- **Physics**: five general systems, each one differential equation with
  its forces or components as switches. Nothing is a fixed word problem: a
  system opens with every box blank. Switch on what acts, type in the values
  you know (an initial velocity, a mass, a time, ...), leave the rest empty,
  and every empty one that your values determine is worked out step by step.
  Any variable can be the unknown, including a time, an initial value or a
  constant such as a drag coefficient. If something cannot be found the
  result says which other value is needed; values that contradict each other
  are reported with the relation they break. An **Examples** button lists
  real-world cases (skydiver, car braking, satellite orbit, camera flash,
  ...); picking one only sets the switches and fills in its known values.
  - **Motion along a line** (`m x'' = sum of forces`): gravity (vertical, or
    on a surface or incline), a constant applied force, sliding friction
    (which can hold the object still), linear or quadratic air resistance, a
    spring, buoyancy, a periodic driving force, gravity that weakens with
    height. With nothing switched on it is constant-acceleration kinematics.
    Also finds terminal speed, stopping time and distance, time and speed on
    reaching x = 0, natural frequency, damping ratio, steady-state amplitude
    and the energies.
  - **Motion in a plane**: uniform gravity, linear or quadratic drag with
    wind, a uniform electric field, a uniform magnetic field perpendicular
    to the plane, inverse-square attraction to the origin. Gravity alone is
    projectile motion (range, greatest height, flight time, landing speed and
    angle); the magnetic field alone gives the cyclotron radius and period;
    the central attraction alone gives orbits (period, eccentricity, closest
    and farthest distance, escape speed).
  - **Rotation and pendulums** (`I theta'' = sum of torques`): no gravity, a
    point mass on a rod or a rigid body; exact large angles, a torsion
    spring, damping, a constant torque. Constant angular acceleration when
    nothing else acts.
  - **First-order change** (`y' = -k (y - level) + source`): decay, approach
    to a level, or growth, labelled as temperature, amount, concentration,
    voltage, current, speed, population or a plain number; two moments.
  - **Series circuit**: resistor, inductor and capacitor switched in or out
    (RC, RL, LC, RLC), no source, a battery or AC, internal resistance,
    capacitor leakage; under-, critically and over-damped.

## Layout

| Path | What it is |
|---|---|
| `app/src/main/python/deengine/` | The solver engine: Python + SymPy, run on-device through Chaquopy. Produces the steps as JSON. Physics systems and their knowns-to-unknowns solver are in `scenarios/`. |
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

## Adding a force, a component or a physics system

A system is one module in `app/src/main/python/deengine/scenarios/`
(`line.py`, `plane.py`, `rotation.py`, `change.py`, `circuit.py`). It
declares its variables, its switches and the relations between the
variables; the generic solver in `scenarios/solver.py` does the rest, and
the app builds its form from the declaration, so no app code changes are
needed.

**A new force or component in an existing system** (say a second spring in
`line.py`):

1. Add its variables to `MODEL["fields"]` with `field(...)` and a
   `switch(...)` or `choice(...)` to `MODEL["effects"]` naming the variables
   it adds. Leave every default blank.
2. Add its term to the right-hand side used by `trajectory` (and to the
   closed form if there is one; otherwise the module falls back to
   `common.numeric(...)`), to `_params` (what the solution depends on), to
   `equation` (the live preview) and to `derivation` (the wording).
3. Add any relations it brings to `relations(effects)`.
4. Add a case to `PHYSICS_CASES` in `tests/python/test_engine.py` and its
   term to `_reference` there, which integrates the equation independently.
5. Optionally add a `preset(...)`: it appears under Examples.

**A new system**:

1. Declare `MODEL` with the constructors in `scenarios/spec.py`: `field(...)`
   for every variable (its quantity picks the unit choices from `units.py`;
   its group is `setup`, `initial`, `effect`, `moment`, `moment2` or
   `derived`; `typical` gives the solver a sense of scale; `show=` / `hide=`
   make a row depend on the switches; `optional=True` keeps an extra out of
   the "could not be found" list), `switch(...)` or `choice(...)` for each
   force or component and the variables it adds, and `preset(...)` for the
   optional examples: switches plus known values, with the interesting ones
   left blank. Fields have no default values: the form opens blank.
2. Write `relations(effects)`: a list of `Rel(id, name, latex, variables,
   fn)`, where `fn(values)` returns the two sides of an equation among those
   variables in SI units. Algebraic links (`tau = R C`) and "the state at
   time t from the solution of the ODE" are both relations; the latter get
   `traj=True`, and extras such as energies `late=True`. A relation is used
   only once all but one (or, in pairs, two) of its variables are known. The
   solver finds a single unknown by a bracketed root search (which is also
   the shooting method when the unknown is a parameter of the ODE).
   Optional `forms={var: latex}` gives the rearranged formula shown in the
   steps. When a side can be zero, add the same positive size to both sides
   so the comparison stays relative.
3. Write `trajectory(effects, values)` (the solution: a closed form where
   one exists, otherwise `common.numeric(...)`, which caches the integration
   while the solver searches, plus the right-hand side used for the
   independent check), `derivation(effects)` and `equation(effects)` (the
   law, the ODE and its solution for the steps and the live preview),
   `horizon(effects, values)` (a natural time span), `graph(effects,
   values)` (the curve, the ideal curve and the marked moment) and
   optionally `baseline(effects)` (the same switches without the losses, for
   the comparison step).
4. Add the module to `MODELS` in `scenarios/__init__.py`.
5. Add cases to `PHYSICS_CASES` in `tests/python/test_engine.py`: each is
   checked against an independent integration of the equation written out in
   the test, and then every variable in turn is blanked and must be
   recovered from the others. Add the system to `_reference` there.

## Adding a solution method

Write a function in `deengine/first_order.py` that takes `(prob, f, ic)` for
`y' = f(x, y)` and returns either `_not_applicable(...)` with the reason, or
`_result(...)` with its list of steps, then add it to `METHODS`. Second-order
methods live in `deengine/second_order.py` and take `(eq, ic)`. The app shows
every method that applies and explains why the others do not.
