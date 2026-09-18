# LEO Nanosatellite Orbital Dynamics Simulator

An interactive, high-fidelity Stage 2 numerical orbit propagator for Low Earth Orbit (LEO) nanosatellites. This suite utilizes `scipy.integrate.solve_ivp` for rigorous numerical integration, alongside a PyTorch Physics-Informed Neural Network (PINN) surrogate model, and a Three.js / FastAPI full-stack 3D interactive web visualization.

## Architecture & Modularity

The codebase is organized into decoupled packages separating backend physics, ML surrogate modeling, and frontend 3D rendering.

```text
leo_simulator/
├── constants.py                # WGS-84 geodetic parameters, standard mu, J2, atmosphere constants
├── models/
│   ├── gravity.py              # Central gravity, J2 oblateness, and Lunar Third-Body ephemeris
│   ├── drag.py                 # Exponential & piecewise density, relative velocity & drag force
│   └── dynamics.py             # Equations of motion: d/dt [r, v] = [v, a_total]
├── orbit/
│   ├── elements.py             # Classical Keplerian orbital elements conversions
│   └── satellite.py            # Spacecraft properties (mass, drag area, Cd, ballistic coeff)
├── propagator.py               # Numerical propagator wrapping scipy.integrate.solve_ivp
├── main.py                     # Headless demo simulation and trajectory JSON exporter
├── ai/
│   ├── pinn_model.py           # PyTorch OrbitalPINN Architecture
│   └── train.py                # Autograd physics-informed loss loop enforcing Newtonian gravity
└── visualization_3d/
    ├── app.py                  # FastAPI server containing /simulate, /save, /load endpoints
    ├── saved_sims/             # Directory for saved JSON multi-satellite configurations
    └── static/
        └── index.html          # Three.js 3D WebGL renderer with Lil-GUI satellite controls
```

## Physics Models

The propagation engine relies on cowell's formulation to directly integrate the Cartesian state vector subject to multiple perturbing accelerations.

1. **Central Two-Body Gravity**:
   $$\mathbf{a}_{grav} = -\frac{\mu}{\|\mathbf{r}\|^3} \mathbf{r}$$
   Utilizes standard Earth gravitational parameter $\mu = 3.986 \times 10^{14} \text{ m}^3/\text{s}^2$ (WGS-84).

2. **Earth $J_2$ Oblateness Perturbation**:
   $$\mathbf{a}_{J_2} = -\frac{3}{2} \frac{J_2 \mu R_\oplus^2}{r^5} \begin{bmatrix} x (1 - 5 z^2 / r^2) \\ y (1 - 5 z^2 / r^2) \\ z (3 - 5 z^2 / r^2) \end{bmatrix}$$
   Captures secular nodal regression $\dot{\Omega}$ (causing the orbital plane to precess) and apsidal precession due to the Earth's equatorial bulge.

3. **Aerodynamic Atmospheric Drag**:
   $$\mathbf{a}_{drag} = -\frac{1}{2} \rho(h) \left(\frac{C_D A}{m}\right) \|\mathbf{v}_{rel}\| \mathbf{v}_{rel}$$
   - Accounts for atmospheric co-rotation: $\mathbf{v}_{rel} = \mathbf{v} - (\boldsymbol{\omega}_\oplus \times \mathbf{r})$.
   - Implements exponential density $\rho(h) = \rho_0 \exp(-(h - h_0)/H)$ stripping orbital energy until re-entry.

4. **Lunar Third-Body Perturbation**:
   $$\mathbf{a}_{3rd} = \mu_{moon} \left( \frac{\mathbf{r}_{moon} - \mathbf{r}_{sat}}{\|\mathbf{r}_{moon} - \mathbf{r}_{sat}\|^3} - \frac{\mathbf{r}_{moon}}{\|\mathbf{r}_{moon}\|^3} \right)$$
   Calculates the gravitational pull of the moon (using a circular-orbit ephemeris approximation) acting on the satellite.

## The Numerical Propagator

The core calculations are processed in `propagator.py` via `scipy.integrate.solve_ivp`. 
- **Method**: The simulator uses the **DOP853** explicit Runge-Kutta method of order 8(5,3), which is highly accurate for non-stiff celestial mechanics ODEs.
- **Tolerances**: High precision relative and absolute tolerances (`rtol=1e-8, atol=1e-8`) are enforced.
- **Event Tracking**: An event function continuously tracks the altitude and safely terminates the integration if the spacecraft breaches atmospheric re-entry conditions ($h < 0$ km).

## Quickstart

### 1. Launch the 3D Interactive Web Simulator
The frontend provides a rich UI for dynamically simulating multi-satellite constellations.
```bash
cd visualization_3d
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- Navigate to `http://localhost:8000`.
- Use the **+ New Satellite** button to inject a new satellite into the physics engine and watch the trajectory spawn instantly.
- Save and load your multi-satellite scenarios dynamically via the top nav bar.

### 2. Train the Physics-Informed Neural Network (PINN)
```bash
cd ai
python3 train.py
```
This will train a PyTorch neural network to act as a surrogate model for the differential equations. The network's loss function utilizes `torch.autograd` to calculate gradients directly from the physical equations, ensuring it *learns* Newton's laws rather than just curve-fitting the trajectory data.

### 3. Run Headless Telemetry Tests
```bash
python3 main.py
pytest -v
```
Runs the 60+ unit tests and verifies physical limits (altitude decay rate and analytical $J_2$ right-ascension nodal drifts).
