# LEO Nanosatellite Orbital Dynamics Simulator (Stage 1)

Stage 1 numerical orbit propagator for Low Earth Orbit (LEO) nanosatellites utilizing `scipy.integrate.solve_ivp`.

## Architecture & Modularity (R1)

The codebase is organized into modular packages separating physics, orbital mechanics, numerical integration, visualization, and execution:

```
leo_simulator/
├── __init__.py                 # Top-level exports and versioning
├── constants.py                # WGS-84 geodetic parameters, standard mu, J2, atmosphere constants
├── models/
│   ├── __init__.py
│   ├── gravity.py              # Central gravity (mu/r^2) & J2 oblateness perturbation
│   ├── drag.py                 # Exponential & piecewise density, relative velocity & drag force
│   └── dynamics.py             # Equations of motion: d/dt [r, v] = [v, a_total]
├── orbit/
│   ├── __init__.py
│   ├── elements.py             # Classical Keplerian orbital elements (coe <-> rv), analytical rates
│   └── satellite.py            # Spacecraft physical properties (mass, drag area, Cd, ballistic coeff)
├── propagator.py               # Numerical propagator wrapping scipy.integrate.solve_ivp
├── visualization.py            # Headless plotting routines (3D trajectory, altitude, RAAN, elements)
├── main.py                     # Demo simulation and telemetry output
├── test_simulation.py          # Programmatic verification assertions (drag decay & J2 regression)
└── tests/
    ├── test_models.py          # Unit tests for acceleration models
    ├── test_elements.py        # Unit tests for Keplerian conversions and singularities
    ├── test_propagator.py      # Conservation laws & event handling tests
    └── test_simulation.py      # Mirrored integration tests
```

## Physics Models (R2)

1. **Central Two-Body Gravity**:
   $$\vec{a}_{grav} = -\frac{\mu}{\|\vec{r}\|^3} \vec{r}$$
   Utilizes standard Earth gravitational parameter $\mu = 3.986004418 \times 10^{14} \text{ m}^3/\text{s}^2$ (WGS-84).

2. **Earth $J_2$ Oblateness Perturbation**:
   $$\vec{a}_{J_2} = -\frac{3}{2} \frac{J_2 \mu R_E^2}{r^5} \begin{bmatrix} x (1 - 5 z^2 / r^2) \\ y (1 - 5 z^2 / r^2) \\ z (3 - 5 z^2 / r^2) \end{bmatrix}$$
   Captures secular nodal regression $\dot{\Omega}_{J2}$ and apsidal precession.

3. **Aerodynamic Atmospheric Drag**:
   $$\vec{a}_{drag} = -\frac{1}{2} \rho(h) \frac{C_D A}{m} \|\vec{v}_{rel}\| \vec{v}_{rel}$$
   - Accounts for atmospheric co-rotation: $\vec{v}_{rel} = \vec{v} - (\vec{\omega}_E \times \vec{r})$.
   - Implements both standard exponential density $\rho(h) = \rho_0 \exp(-(h - h_0)/H)$ and multi-layer piecewise exponential reference atmosphere (US Standard 1976 table up to 1,000 km).

## Quickstart

### Run Demonstration Simulation
```bash
python3 main.py
```
Exports telemetry summaries and plots to `output/`:
- `output/3d_trajectory.png`: 3D Earth and orbit trajectory.
- `output/altitude_decay.png`: Comparative altitude loss over time.
- `output/raan_regression.png`: Nodal precession compared to analytical $J_2$ secular rate.
- `output/elements_evolution.png`: Multi-panel Keplerian elements time history ($a, e, i, \Omega$).

### Run Verification Test Suite
```bash
python3 test_simulation.py
# or
pytest -v
```
All 30 unit and integration tests verify energy conservation, edge cases, drag decay, and $J_2$ nodal regression within analytical tolerances.
