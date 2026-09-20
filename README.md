# NanoDynamics: Real-Scale Earth-Moon Orbital Dynamics Simulator (v12-alpha)

An interactive, high-fidelity numerical orbit propagator and mission analysis suite for satellites orbiting the Earth and Moon. NanoDynamics combines high-order Runge-Kutta numerical integration (`scipy.integrate.solve_ivp`), a PyTorch Physics-Informed Neural Network (PINN) surrogate model, and a real-scale Three.js / FastAPI full-stack 3D interactive web application with rolling trajectory streaming and live maneuver simulation.

---

## 🚀 Key Features

- **Real-Scale Earth-Moon System:** True-to-life physical dimensions ($R_\oplus = 6,378.137\text{ km}$, $R_{Moon} = 1,737.4\text{ km}$) and real orbital separation ($384,400\text{ km}$) with realistic $27.32\text{ day}$ lunar propagation and $5.145^\circ$ orbital inclination.
- **Continuous Forward Time & Up to 10,000× Speedup:** Smooth, non-looping simulation driven by a true elapsed-time clock with speed multipliers from $1\times$ to $10,000\times$.
- **Rolling `/stream` Architecture:** Trajectories are streamed in rolling chunks asynchronously from the FastAPI backend via `asyncio.to_thread()`, keeping the interface lightweight and responsive indefinitely.
- **Multi-Body Satellite Placement:** Deploy satellites around **Earth** (LEO, MEO, GEO) or the **Moon** (Low Lunar Orbit) with dynamically guarded stable orbital parameter ranges.
- **In-Flight $\Delta v$ Orbit Maneuvers:** Perform orbital burns in the local RTN (Radial, Transverse/Prograde, Normal) reference frame to execute orbit raising, circularization, Hohmann transfers, and plane changes in real time.
- **3D Force Vector Overlays:** Interactive `THREE.ArrowHelper` vectors showing instantaneous acceleration components:
  - 🟢 **Central Gravity** ($-\frac{\mu}{r^3}\mathbf{r}$)
  - 🟡 **$J_2$ Earth Oblateness**
  - 🔴 **Atmospheric Drag** ($-\frac{1}{2}\rho \frac{C_D A}{m} v_{rel} \mathbf{v}_{rel}$)
- **Orbital Decay & Deorbit Lifetime Tracking:** Continuous altitude monitoring captures the exact deorbit epoch when atmospheric drag forces re-entry ($h \le 0\text{ km}$), recording satellite lifetime in live telemetry and downloadable CSV exports.
- **Individual Satellite Management:** Add, inspect, track, maneuver, and delete individual satellites on the fly.
- **Scenario Persistence:** Save and load multi-satellite configurations as JSON scenarios.

---

## 📁 Repository Structure & Hierarchy

```text
leo_simulator/
├── constants.py                 # WGS-84 geodetic parameters, lunar constants, atmospheric defaults
├── models/
│   ├── gravity.py               # Central two-body gravity, J2 oblateness, Lunar ephemeris & 3rd-body
│   ├── drag.py                  # Exponential & piecewise density models, co-rotating atmospheric drag
│   └── dynamics.py              # Equations of motion: d/dt [r, v] = [v, a_total] & force decomposition
├── orbit/
│   ├── elements.py              # Keplerian orbital elements conversions [a, e, i, Omega, omega, nu]
│   └── satellite.py             # Spacecraft specifications (mass, cross-section area, Cd, ballistic coeff)
├── propagator.py                # High-order numerical propagator (DOP853 Runge-Kutta via SciPy)
├── main.py                      # Headless verification run & trajectory generator
├── ai/
│   ├── pinn_model.py            # PyTorch Physics-Informed Neural Network (PINN) architecture
│   └── train.py                 # Physics-informed training loop enforcing Newtonian conservation
├── tests/
│   ├── test_app.py              # E2E API tests (validation, async offload, streaming, burns)
│   ├── test_challenger1_stress.py # Stress tests (concurrency, energy conservation, boundary knife-edges)
│   ├── test_elements.py         # Keplerian elements round-trip conversion tests
│   ├── test_models.py           # Gravity, J2, and atmospheric drag physical model tests
│   ├── test_propagator.py       # Numerical convergence & event termination tests
│   ├── test_simulation.py       # Physics assertions (drag decay & nodal precession rates)
│   ├── test_ui_adversarial.py   # Adversarial UI stress tests (speed sweeps, tab switching)
│   ├── test_ui_refactor.py      # Columnar payload and animation integrity tests
│   └── test_visualization.py    # Telemetry and plotting unit tests
└── visualization_3d/
    ├── app.py                   # FastAPI async server (/simulate, /stream, /burn, /save, /load)
    ├── saved_sims/              # Saved multi-satellite JSON scenarios
    ├── tests/                   # Backend & UI automation test suites
    └── static/
        └── index.html           # Three.js 3D WebGL real-scale Earth-Moon interactive simulator
```

---

## 📐 Mathematical Formulations

The numerical propagation engine integrates the state vector $\mathbf{y}(t) = [\mathbf{r}(t), \mathbf{v}(t)]^T$ in Cartesian coordinates:

$$\frac{d}{dt} \begin{bmatrix} \mathbf{r} \\ \mathbf{v} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{a}_{\text{central}} + \mathbf{a}_{J_2} + \mathbf{a}_{\text{drag}} + \mathbf{a}_{3\text{rd}} \end{bmatrix}$$

### 1. Central Two-Body Gravity
$$\mathbf{a}_{\text{central}} = -\frac{\mu}{\|\mathbf{r}\|^3} \mathbf{r}$$
- **Earth:** $\mu_\oplus = 3.986004418 \times 10^{14}\text{ m}^3/\text{s}^2$, $R_\oplus = 6,378,137.0\text{ m}$ (WGS-84).
- **Moon:** $\mu_{\text{Moon}} = 4.902800066 \times 10^{12}\text{ m}^3/\text{s}^2$, $R_{\text{Moon}} = 1,737,400.0\text{ m}$.

### 2. Earth $J_2$ Oblateness Perturbation
Accounts for the equatorial bulge causing nodal regression ($\dot{\Omega}$) and apsidal drift ($\dot{\omega}$):

$$\mathbf{a}_{J_2} = -\frac{3}{2} \frac{J_2 \mu R_\oplus^2}{r^5} \begin{bmatrix} x \left(1 - 5 \frac{z^2}{r^2}\right) \\ y \left(1 - 5 \frac{z^2}{r^2}\right) \\ z \left(3 - 5 \frac{z^2}{r^2}\right) \end{bmatrix}$$

where $J_{2,\oplus} = 1.08262668 \times 10^{-3}$.

### 3. Aerodynamic Atmospheric Drag
$$\mathbf{a}_{\text{drag}} = -\frac{1}{2} \rho(h) \left(\frac{C_D A}{m}\right) \|\mathbf{v}_{\text{rel}}\| \mathbf{v}_{\text{rel}}$$

- **Atmospheric Co-rotation:** $\mathbf{v}_{\text{rel}} = \mathbf{v} - (\boldsymbol{\omega}_\oplus \times \mathbf{r})$, where $\omega_\oplus = 7.2921150 \times 10^{-5}\text{ rad/s}$.
- **Exponential Density:** $\rho(h) = \rho_0 \exp\left(-\frac{h - h_0}{H}\right)$, where $h_0 = 350\text{ km}$, $\rho_0 = 9.5 \times 10^{-12}\text{ kg/m}^3$, and scale height $H = 53.2\text{ km}$.

### 4. Lunar Third-Body Perturbation
$$\mathbf{a}_{3\text{rd}} = \mu_{\text{moon}} \left( \frac{\mathbf{r}_{\text{moon}} - \mathbf{r}_{\text{sat}}}{\|\mathbf{r}_{\text{moon}} - \mathbf{r}_{\text{sat}}\|^3} - \frac{\mathbf{r}_{\text{moon}}}{\|\mathbf{r}_{\text{moon}}\|^3} \right)$$

### 5. In-Flight RTN Maneuver Formulation
Maneuver burns are computed in the instantaneous orbital frame:
- **Radial unit vector:** $\hat{\mathbf{u}}_r = \frac{\mathbf{r}}{\|\mathbf{r}\|}$
- **Normal unit vector:** $\hat{\mathbf{u}}_n = \frac{\mathbf{r} \times \mathbf{v}}{\|\mathbf{r} \times \mathbf{v}\|}$
- **Transverse / Prograde unit vector:** $\hat{\mathbf{u}}_t = \hat{\mathbf{u}}_n \times \hat{\mathbf{u}}_r$
- **New Velocity:** $\mathbf{v}_{\text{new}} = \mathbf{v} + \Delta v_r \hat{\mathbf{u}}_r + \Delta v_t \hat{\mathbf{u}}_t + \Delta v_n \hat{\mathbf{u}}_n$

---

## 🛠️ Quickstart Guide

### 1. Run the 3D Interactive Web Simulator
```bash
cd visualization_3d
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- Open `http://localhost:8000` in your browser.
- Click **+ New Satellite** to configure an Earth or Moon orbit.
- Select any satellite to inspect real-time telemetry, toggle 3D force vectors, or execute $\Delta v$ burns.
- Export all satellite trajectories and deorbit lifetimes with **Export CSV**.

### 2. Train the Physics-Informed Neural Network (PINN)
```bash
cd ai
python3 train.py
```
Trains a PyTorch neural network that evaluates physical residual losses directly via `torch.autograd`, learning Newtonian gravity constraints without pure data overfitting.

### 3. Run the Automated Test Suites
```bash
pytest -v
```
Runs 190+ comprehensive unit, integration, numerical sanity, and adversarial stress tests.

---

## 📜 Version History

- **`v12-alpha` (Current):** Real-scale isolated Earth-Moon system ($384,400\text{ km}$ separation), $10,000\times$ speedup, rolling `/stream` trajectory chunks, RTN burn maneuvers, 3D force vector overlays, and deorbit lifetime CSV tracking.
- **`v11-alpha`:** Multi-satellite dynamic FastAPI architecture, Pydantic bounds validation, async offloading, and initial WebGL fallback safeguards.
