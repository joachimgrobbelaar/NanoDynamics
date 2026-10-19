# Project: LEO Satellite Simulation Refactor

## Architecture
- **Backend Service**: FastAPI web application (`visualization_3d/app.py`) providing `/simulate`, `/moon_track`, and project save/load endpoints.
- **Physics Core**: `leo_simulator` modular library (`propagator.py`, `orbit/elements.py`, `orbit/satellite.py`, `physics/forces.py`, `physics/atmosphere.py`).
- **Frontend Client**: Single-page Three.js application (`visualization_3d/static/index.html`) rendering 3D Earth, Moon, satellite orbits, real-time telemetry, and control GUI.
- **Test Infrastructure**: Pytest suite (`visualization_3d/tests/test_app.py`, `tests/test_app.py`) and Playwright headless browser automation (`test_ui_refactor.py`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1: Pydantic Field Validation | Add strict physical bounds (`gt=0`, `ge=0, lt=1`, etc.) to `SatelliteParams` returning HTTP 400/422 | M2 | ORIGINAL_REQUEST §R1 |
| 2 | R1: Clean Error Reporting | Register `RequestValidationError` handler formatting human-readable error messages | M2 | ORIGINAL_REQUEST §R1 |
| 3 | R2: Async Endpoint Execution | Refactor `simulate_satellite` to `async def` and offload propagation via `asyncio.to_thread` | M2 | ORIGINAL_REQUEST §R2 |
| 4 | R3: Columnar Payload Compression | Return trajectory as structural dictionary of arrays (`{"t": [...], "x": [...], ...}`) | M2 | ORIGINAL_REQUEST §R3 |
| 5 | R3: Frontend Payload Ingestion | Update `addSatelliteToScene`, `exportData`, and telemetry in `index.html` to parse columnar arrays | M3 | ORIGINAL_REQUEST §R3 |
| 6 | R3: Backward Compatibility | Support loading legacy Array-of-Objects saved simulation files (e.g. `Alpha.json`) | M3 | Survey Explorer 2 |
| 7 | R4: Frame-Independent Animation | Decouple animation from frame rate using `THREE.Clock` / `performance.now()` with time-based lerp | M3 | ORIGINAL_REQUEST §R4 |
| 8 | R4: Sidereal Rotation & Moon Lerp | Synchronize Earth rotation and Moon movement to physical simulation time | M3 | Survey Explorer 3 |
| 9 | AC1: Input Validation Test Suite | Programmatic tests asserting HTTP 400/422 on invalid parameters (`tests/test_app.py`) | M1 | ORIGINAL_REQUEST §AC1 |
| 10 | AC2: Payload Structure Test Suite | Programmatic tests asserting dictionary-of-arrays trajectory format (`tests/test_app.py`) | M1 | ORIGINAL_REQUEST §AC2 |
| 11 | AC3: Playwright UI Automation | Browser automation asserting zero JavaScript console errors during launch and animation | M1 | ORIGINAL_REQUEST §AC3 |
| 12 | AC4: Adversarial Hardening | Stress testing corner cases, concurrency, and telemetry accuracy | M4 | Final Milestone |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | E2E Testing Track | Test suite for input validation, payload structure, and Playwright UI automation | none | DONE |
| M2 | Backend Refactor (R1, R2, R3) | Pydantic validation, async offloading, columnar payload in `app.py` | none | DONE |
| M3 | Frontend Refactor (R3, R4) | Columnar parser, backward compatibility, frame-independent animation in `index.html` | M2 | IN_PROGRESS |
| M4 | Final Integration & Adversarial Verification | Full E2E test verification (Tiers 1-4) + adversarial hardening (Tier 5) + Forensic Audit | M1, M2, M3 | PLANNED |

## Interface Contracts
### Backend (`/simulate`) ↔ Frontend (`addSatelliteToScene`)
- **Request Method**: `POST /simulate`
- **Request Body (JSON)**:
  ```json
  {
    "name": "string (min_length=1)",
    "mass": "float (gt=0.0, le=100000.0)",
    "drag_area": "float (ge=0.0, le=10000.0)",
    "cd": "float (ge=0.0, le=20.0)",
    "altitude_km": "float (ge=100.0, le=2000.0)",
    "eccentricity": "float (ge=0.0, lt=1.0)",
    "inclination_deg": "float (ge=0.0, le=180.0)",
    "raan_deg": "float (ge=0.0, lt=360.0)"
  }
  ```
- **Error Response**: HTTP 422 (or 400) with JSON:
  ```json
  { "detail": "Descriptive error message specifying the invalid field and constraint" }
  ```
- **Success Response (HTTP 200)**:
  ```json
  {
    "name": "string",
    "params": { ... },
    "trajectory": {
      "t": [0.0, 60.0, ...],
      "x": [r_x0, r_x1, ...],
      "y": [r_y0, r_y1, ...],
      "z": [r_z0, r_z1, ...],
      "vx": [v_x0, v_x1, ...],
      "vy": [v_y0, v_y1, ...],
      "vz": [v_z0, v_z1, ...]
    }
  }
  ```

## Code Layout
- `visualization_3d/app.py`: FastAPI server, Pydantic models, route handlers.
- `visualization_3d/static/index.html`: Three.js scene, UI, animation loop, telemetry.
- `visualization_3d/tests/test_app.py`: Backend integration tests.
- `tests/test_app.py`: Root forwarder/extension test suite for acceptance criteria.
- `tests/test_ui_refactor.py`: Playwright UI automation test suite.
