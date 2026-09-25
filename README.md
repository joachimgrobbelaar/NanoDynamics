# NanoDynamics: Real-Scale Earth-Moon Orbital Dynamics & Experimentation Suite (v1.0-beta)

An interactive, high-fidelity numerical orbit propagator, mission analysis suite, and parametric experimentation engine for satellites orbiting the Earth and Moon. NanoDynamics combines high-order Runge-Kutta numerical integration (`scipy.integrate.solve_ivp`), an offline PyTorch Physics-Informed Neural Network (PINN) surrogate model, and a real-scale Three.js / FastAPI full-stack 3D interactive web application with dual-engine switching, rolling trajectory streaming, live maneuver simulation, single-orbit trailing fade, custom avatar icons, and automated deorbit lifetime sweeps.

---

## 🚀 Key Features

- **🤖 Dual-Engine Orbital Propagation (Numerical RK45 vs. ML PINN):** Switch propagation engine between classical numerical integration (`solve_ivp` RK45) and an offline PyTorch Physics-Informed Neural Network surrogate (`TransitionMLP`). Runs local tensor inference without external cloud compute dependencies.
- **🔄 Instant Comparison Cloning:** Clone any active satellite with one click using the alternate engine to visually compare numerical ground truth against neural surrogate rollouts side-by-side in the 3D viewport.
- **🏷️ Floating 3D Stat Box & Engine Badging:** Screen-projected 3D HUD boxes displaying satellite name, active engine badge (`[RK45]` / `[PINN]`), altitude, velocity, and orbital inclination directly above tracked spacecraft.
- **🌌 Satellite Constellation Generator:** Instantly spawn multi-satellite Walker constellations or local clusters (e.g. string-of-pearls) with automatic incremental offsets in Phase ($\nu$), RAAN ($\Omega$), altitude, and inclination.
- **☁️ Visual Atmospheric Layers & Aerodynamic Heating:** Real-time 3D shells representing the Troposphere (12km) through Exosphere (2000km). Live telemetry calculates aerodynamic stagnation heating temperature (°C) via the Sutton-Graves equation based on dynamic atmospheric density and velocity.
- **⚡ Adaptive High-Speed Streaming:** Backend API endpoints dynamically adapt `chunk_duration` and `dt_eval` interpolation resolution based on simulation playback speed, easily supporting swarms of satellites at $10,000\times$ speed without stutter or backend overload.
- **🧪 Server-Side Parametric Experimentation Lab:** Automated, parallelized (multi-core) parametric sweeps evaluating independent variables ($B = \frac{m}{C_D A}$, mass, drag area, $C_D$, altitude, eccentricity, atmospheric density scale). Record lifetime, orbital decay rates, max velocity, and min altitude across up to 1000 sweep points per batch. Supports quiet high-speed headless evaluation and visual trajectory inspection.
- **🎨 Custom Satellite Colors & Avatar Icons:** Full color picker support (`<input type="color">`) and customizable 3D billboard avatar glyphs (🛰️ **Satellite**, 🚀 **Rocket**, 👨‍🚀 **Astronaut**, 👽 **Alien**, 🛸 **UFO**, ⚪ **Pure Sphere**) with dynamic glowing halos matching chosen colors.
- **✨ Single-Orbit Dynamic Window & Trailing Fading Trail:** Trajectory display is automatically scoped to the active single-orbit time window ($[t - T_{\text{orbit}}, t + T_{\text{orbit}}]$) with a smooth alpha/brightness gradient that fades out cleanly behind the satellite as it propagates forward.
- **🌕 Moon Orbit Propagation & Multi-Body Mechanics:** Full orbital mechanics around both **Earth** ($R_\oplus = 6,378.137\text{ km}$, $\mu_\oplus = 3.986\times 10^{14}$) and the **Moon** ($R_{\text{Moon}} = 1,737.4\text{ km}$, $\mu_{\text{Moon}} = 4.9048\times 10^{12}$) with real $384,400\text{ km}$ distance, dynamic ephemeris, and local lunar coordinate frames.
- **Full Cislunar System & Trans-Lunar Injection (TLI):** Real-scale Earth-Moon dynamics spanning $100\text{ km}$ LEO up to $400,000\text{ km}$ cislunar space. Supports high-energy maneuvers ($\pm 5,000\text{ m/s}$ to $\pm 10,000\text{ m/s}$) for Trans-Lunar Injection ($\Delta v \approx 3,100\text{ m/s}$) and Lunar Orbit Insertion.
- **Seamless Trajectory Bleeding & Ghost Trail Fading:** When a burn maneuver is executed, the new trajectory branches seamlessly into the orbit path while the pre-burn ghost trail progressively fades out over time.
- **Geodetic Coordinates & Key Event Tracking:** Real-time calculation and display of WGS-84 sub-satellite latitude/longitude ($\phi, \lambda$), initial orbital insertion coordinates, re-entry interface coordinates ($120\text{ km}$), and ground collision points.
- **Continuous Forward Time & Up to 10,000× Speedup:** Smooth, non-looping simulation driven by a true elapsed-time clock with speed multipliers from $1\times$ to $10,000\times$.
- **Rolling `/stream` Architecture:** Trajectories are streamed in rolling chunks asynchronously from the FastAPI backend via `asyncio.to_thread()`, keeping the interface lightweight and responsive indefinitely.
- **Interactive $\Delta v$ Maneuver Nodes & Previews:** Right-click any point along an orbit trajectory to add a maneuver node, drag or tune prograde/radial/normal $\Delta v$ with live orbit preview and quick presets (TLI, LOI, Deorbit).
- **3D Force Vector Overlays:** Interactive `THREE.ArrowHelper` vectors showing instantaneous acceleration components:
  - 🟢 **Central Gravity** ($-\frac{\mu}{r^3}\mathbf{r}$)
  - 🟡 **$J_2$ Earth Oblateness**
  - 🔴 **Atmospheric Drag** ($-\frac{1}{2}\rho \frac{C_D A}{m} v_{\text{rel}} \mathbf{v}_{\text{rel}}$)
- **Orbital Decay & Deorbit Lifetime Tracking:** Continuous altitude monitoring captures the exact deorbit epoch when atmospheric drag forces re-entry ($h \le 0\text{ km}$), recording satellite lifetime in live telemetry and downloadable CSV exports.
- **Scenario Persistence:** Save and load multi-satellite configurations as JSON scenarios.

---

## 📁 Repository Structure & Hierarchy

```text
leo_simulator/
├── constants.py                 # WGS-84 geodetic parameters, lunar constants, atmospheric defaults
├── experiment.py                # Parametric sweep engine & deorbit lifetime analysis framework
├── models/
│   ├── gravity.py               # Central two-body gravity, J2 oblateness, Lunar ephemeris & 3rd-body
│   ├── drag.py                  # Exponential & piecewise US Standard 1976 density models, co-rotating drag
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
│   ├── test_experiment.py       # Parametric sweep and atmospheric density unit tests
│   ├── test_models.py           # Gravity, J2, and atmospheric drag physical model tests
│   ├── test_propagator.py       # Numerical convergence & event termination tests
│   ├── test_simulation.py       # Physics assertions (drag decay & nodal precession rates)
│   ├── test_ui_adversarial.py   # Adversarial UI stress tests (speed sweeps, tab switching)
│   ├── test_ui_refactor.py      # Columnar payload and animation integrity tests
│   └── test_visualization.py    # Telemetry and plotting unit tests
└── visualization_3d/
    ├── app.py                   # FastAPI async server (/simulate, /stream, /burn, /experiment, /save, /load)
    ├── saved_sims/              # Saved multi-satellite JSON scenarios
    ├── tests/                   # Backend & UI automation test suites
    └── static/
        └── index.html           # Three.js 3D WebGL real-scale Earth-Moon interactive simulator & Lab
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

### 3. Aerodynamic Atmospheric Drag & Density Layers
$$\mathbf{a}_{\text{drag}} = -\frac{1}{2} \rho(h) \left(\frac{C_D A}{m}\right) \|\mathbf{v}_{\text{rel}}\| \mathbf{v}_{\text{rel}}$$

- **Atmospheric Co-rotation:** $\mathbf{v}_{\text{rel}} = \mathbf{v} - (\boldsymbol{\omega}_\oplus \times \mathbf{r})$, where $\omega_\oplus = 7.2921150 \times 10^{-5}\text{ rad/s}$.
- **Piecewise US Standard 1976 Model:** Layered base density $\rho_{0,i}$ and localized scale height $H_i$ across altitudes from $0\text{ to }1,000\text{ km}$:
  $$\rho(h) = \rho_{0,i} \exp\left(-\frac{h - h_{0,i}}{H_i}\right)$$
- **Ballistic Coefficient:** $B = \frac{m}{C_D A}\text{ [kg/m}^2\text{]}$. Lifetime scales linearly with $B$ for circular orbits.

### 4. Lunar Third-Body Perturbation
$$\mathbf{a}_{3\text{rd}} = \mu_{\text{moon}} \left( \frac{\mathbf{r}_{\text{moon}} - \mathbf{r}_{\text{sat}}}{\|\mathbf{r}_{\text{moon}} - \mathbf{r}_{\text{sat}}\|^3} - \frac{\mathbf{r}_{\text{moon}}}{\|\mathbf{r}_{\text{moon}}\|^3} \right)$$

### 5. In-Flight RTN Maneuver Formulation
- **Radial unit vector:** $\hat{\mathbf{u}}_r = \frac{\mathbf{r}}{\|\mathbf{r}\|}$
- **Normal unit vector:** $\hat{\mathbf{u}}_n = \frac{\mathbf{r} \times \mathbf{v}}{\|\mathbf{r} \times \mathbf{v}\|}$
- **Transverse / Prograde unit vector:** $\hat{\mathbf{u}}_t = \hat{\mathbf{u}}_n \times \hat{\mathbf{u}}_r$
- **New Velocity:** $\mathbf{v}_{\text{new}} = \mathbf{v} + \Delta v_r \hat{\mathbf{u}}_r + \Delta v_t \hat{\mathbf{u}}_t + \Delta v_n \hat{\mathbf{u}}_n$

---

## 🛠️ Quickstart Guide

### 1. Run the 3D Interactive Web Simulator & Experiment Lab
```bash
cd visualization_3d
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- Open `http://localhost:8000` in your browser.
- Click **+ New Satellite** to configure an Earth or Moon orbit.
- Click **🧪 Experiment Lab** to run automated parametric sweeps over Ballistic Coefficient, Mass, Drag Area, or Altitude, viewing tabular results and downloading clean experiment CSV files.
- Select any satellite to inspect real-time telemetry, toggle 3D force vectors, or execute $\Delta v$ burns.
- Export all satellite trajectories and deorbit lifetimes with **Export CSV**.

### 2. Run Headless Parametric Sweeps via Python
```python
from leo_simulator.experiment import run_parametric_sweep

# Run a deorbit lifetime sweep over ballistic coefficient B from 10 to 100 kg/m^2
res = run_parametric_sweep(
    param_name="ballistic_coefficient",
    values=[10.0, 20.0, 40.0, 80.0],
    base_params={"mass": 4.0, "drag_area": 0.03, "cd": 2.2, "altitude_km": 250.0},
    atmosphere_type="piecewise",
    max_duration_seconds=30 * 86400.0,
    quiet=True,
    output_dir="experiments"
)

for r in res.results:
    print(f"B = {r.ballistic_coeff_kg_m2:5.1f} kg/m² -> Lifetime = {r.lifetime_hours:6.1f} h ({r.lifetime_days:4.2f} days)")
```

### 3. Train the Physics-Informed Neural Network (PINN)
```bash
cd ai
python3 train.py
```
Trains a PyTorch neural network that evaluates physical residual losses directly via `torch.autograd`, learning Newtonian gravity constraints without pure data overfitting.

### 4. Run the Automated Test Suites
```bash
pytest -v
```
Runs 195+ comprehensive unit, integration, numerical sanity, atmospheric consistency, and adversarial stress tests.

---

## 📜 Version History

- **`v1.0-beta` (First Beta Build - Current):** Dual-Engine Orbital Propagation (Numerical RK45 vs. offline ML PINN Surrogate), instant comparison cloning with color-coded trails, 3D floating stat box with active engine badging, PINN Moon-orbit boundary guards, and complete independence from external cloud compute services.
- **`v23-alpha`:** Satellite Constellation Generator, Sutton-Graves Aerodynamic Heating simulation, visual Atmospheric Layers (Troposphere→Exosphere), adaptive streaming for stable 10,000x playback, ML Surrogate multithreading, and parallelized 1000-point Parametric Sweeps.
- **`v22-alpha`:** Interactive "Drag to Add Orbit" UX via camera-facing plane projection and 3D Raycaster KSP-style Maneuver node dragging.
- **`v21-alpha`:** Floating 3D Stat Box HTML overlays and Radial View Lock (tracking camera faces radially inward toward the planet).
- **`v20-alpha`:** ML Surrogate training speed and quality upgrade (`torch.set_num_threads`, batch size 1024, `ReduceLROnPlateau`, Adam momentum state persistence).
- **`v19-alpha`:** Inter-satellite Kinematics tracking (Line of Sight, Range Rate, 3D distance).
- **`v18-alpha`:** Custom 3D billboard avatar glyphs (Satellite, Rocket, Astronaut, Alien, UFO) with dynamic glowing halos matching chosen colors.
- **`v17-alpha`:** Single-Orbit Dynamic Window & Trailing Fading Trail (alpha/brightness gradient fading out cleanly behind the satellite).
- **`v16-alpha`:** High-energy maneuvers for Trans-Lunar Injection and Lunar Orbit Insertion with seamless Trajectory Bleeding & Ghost Trail Fading.
- **`v15-alpha`:** Geodetic Coordinates & Key Event Tracking (WGS-84 sub-satellite lat/lon, initial insertion coordinates, re-entry interface).
- **`v14-alpha`:** Core UI refactor, structural optimizations, and multi-threaded processing foundations.
- **`v13-alpha`:** Server-side Parametric Experimentation Lab (`leo_simulator/experiment.py`), US Standard 1976 atmosphere model reconciliation, clean monotonic CSV export with Run_IDs, and lifetime vs ballistic coefficient data collection framework.
- **`v12-alpha`:** Real-scale isolated Earth-Moon system ($384,400\text{ km}$ separation), $10,000\times$ speedup, rolling `/stream` trajectory chunks, RTN burn maneuvers, 3D force vector overlays, and deorbit lifetime tracking.
- **`v11-alpha`:** Multi-satellite dynamic FastAPI architecture, Pydantic bounds validation, async offloading, and initial WebGL fallback safeguards.
