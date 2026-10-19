# E2E Test Infra: LEO Satellite Simulation Refactor

## Test Philosophy
- Opaque-box, requirement-driven.
- Automated API validation tests and headless browser UI tests.
- Methodology: Category-Partition + Boundary Value Analysis + Combinatorial Testing.

## Feature Inventory & Test Mapping
| # | Feature | Requirement | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|-------------|:------:|:------:|:------:|:------:|
| 1 | Mass validation (> 0) | R1 | ✓ | ✓ | ✓ | ✓ |
| 2 | Eccentricity validation ([0, 1)) | R1 | ✓ | ✓ | ✓ | ✓ |
| 3 | Altitude validation (100-2000 km) | R1 | ✓ | ✓ | ✓ | ✓ |
| 4 | Clean 400/422 errors | R1 | ✓ | ✓ | ✓ | ✓ |
| 5 | Async non-blocking offloading | R2 | ✓ | ✓ | ✓ | ✓ |
| 6 | Columnar trajectory format | R3 | ✓ | ✓ | ✓ | ✓ |
| 7 | Frontend columnar parsing | R3 | ✓ | ✓ | ✓ | ✓ |
| 8 | Legacy simulation compatibility | R3 | ✓ | ✓ | ✓ | ✓ |
| 9 | Frame-independent animation | R4 | ✓ | ✓ | ✓ | ✓ |
| 10 | Sidereal Earth / Moon animation | R4 | ✓ | ✓ | ✓ | ✓ |

## Test Architecture
- **API Test Runner**: `pytest`
- **Location**: `tests/test_app.py` & `visualization_3d/tests/test_app.py`
- **UI Automation Runner**: Playwright (`python3 -m pytest tests/test_ui_refactor.py` or standalone runner)
- **Error Assertion**: Zero JavaScript runtime exceptions (`pageerror`, `console(error)`).
