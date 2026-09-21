# Original User Request

## 2026-09-21T07:10:03Z

<USER_REQUEST>
# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: The full autonomous team (multiple concurrent agents)

Implement serverless PyTorch ML Surrogate training via Firebase Cloud Functions, automate deployment to Google Cloud, and display recorded ML data pair metrics in the 3D Web UI.

Working directory: ~/teamwork_projects/leo_simulator
Integrity mode: benchmark

## Requirements

### R1. Cloud PyTorch ML Training
Extend the existing Firebase Python Cloud Function environment (`functions/main.py`) to support PyTorch training for the ML Surrogate PINN model, enabling serverless model fitting on recorded trajectory data.

### R2. ML Data Pair Telemetry
Update the FastAPI backend and Three.js frontend to dynamically track, query, and render the total count of recorded ML data pairs inside the Experiment Lab UI.

### R3. Automated Firebase Deployment
Create an automated deployment script or pipeline that successfully bundles the PyTorch dependencies, pushes the Python Cloud Functions, updates Firestore rules, and deploys the static frontend to Firebase Hosting.

## Acceptance Criteria

### Testing & Verification
- [ ] **Programmatic Test:** A pytest script successfully hits the Cloud Function ML training endpoint (via emulator or live) and receives a structured training loss response.
- [ ] **UI Integrity:** A headless browser or HTTP test verifies the `/simulate` endpoint correctly increments the data pair count, and the HTML UI element `#ml-pairs-val` correctly displays this integer.
- [ ] **Agent-as-Judge:** An independent agent reviewer executes the deployment pipeline script and verifies it returns a successful exit code (or explicitly handles CI authentication barriers).
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-21T09:10:03+02:00.
</ADDITIONAL_METADATA>
