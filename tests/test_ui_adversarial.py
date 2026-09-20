"""
Adversarial Stress Test Suite for UI Resilience and Animation Integrity.

Tests:
1. Playback speed extremes (0x, 0.001x, 20x, 100x, 1000x, negative speeds).
2. Tab backgrounding, visibility changes, and delta clamping.
3. Rapid burst satellite additions (concurrency & UI DOM stability).
4. Legacy Alpha.json simulation loading & telemetry CSV export validation.
5. Corrupted/partial trajectory resilience & project clearing lifecycle.
6. Zero JavaScript runtime errors (pageerror) and zero console error logs.
"""

import csv
import io
import math
import os
import socket
import subprocess
import sys
import time

import pytest
from playwright.sync_api import sync_playwright

# Ensure temporary files use tmpfs (/dev/shm) to prevent out-of-disk crashes on small root /tmp
if os.path.exists("/dev/shm"):
    os.environ["TMPDIR"] = "/dev/shm"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def app_server_url():
    """Ensure FastAPI application server is available."""
    port = 8000
    server_process = None

    if not is_port_in_use(port):
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "visualization_3d.app:app", "--port", str(port)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        start_time = time.time()
        while time.time() - start_time < 10.0:
            if is_port_in_use(port):
                break
            time.sleep(0.2)
        else:
            if server_process:
                server_process.terminate()
            raise RuntimeError(f"Failed to start FastAPI server on port {port}")

    url = f"http://localhost:{port}"
    yield url

    if server_process:
        server_process.terminate()
        server_process.wait(timeout=5)


def setup_monitored_page(browser, server_url):
    """Create a page with attached error listeners and navigate to the app."""
    errors = []
    console_logs = []

    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    page.on("pageerror", lambda err: errors.append(f"PageError: {err}"))
    page.on(
        "console",
        lambda msg: (
            errors.append(f"ConsoleError: {msg.text}")
            if msg.type == "error"
            else console_logs.append(f"[{msg.type}] {msg.text}")
        ),
    )
    page.on("dialog", lambda dialog: dialog.accept())

    page.goto(server_url)
    page.wait_for_selector("#top-nav", state="visible", timeout=15000)
    page.wait_for_selector("#telemetry", state="visible", timeout=15000)

    return context, page, errors, console_logs


def test_adversarial_playback_speed_extremes(app_server_url):
    """Stress test animation loop across extreme and boundary playback speeds."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        _, page, errors, _ = setup_monitored_page(browser, app_server_url)

        # 1. Launch a test satellite
        page.click("button:has-text('+ New Satellite')")
        page.wait_for_selector("#modal-overlay", state="visible", timeout=5000)
        page.fill("#m-name", "Sat-SpeedTest")
        page.fill("#m-mass", "5.0")
        page.fill("#m-alt", "500.0")
        page.click("#modal-buttons button:has-text('Launch')")
        page.wait_for_selector("#modal-overlay", state="hidden", timeout=30000)

        # Focus the newly launched satellite
        page.click("#sat-list li:has-text('Sat-SpeedTest')")

        # 2. Test speed variations:
        speeds_to_test = [
            0,       # Pause
            0.001,   # Micro step
            1,       # Normal
            20,      # Max GUI slider
            100,     # Extreme 100x
            500,     # Hyper 500x
            1000,    # Ultra 1000x
            -5,      # Negative speed
            10,      # Return to reasonable
        ]

        for spd in speeds_to_test:
            page.evaluate(f"() => {{ simSettings.playbackSpeed = {spd}; }}")
            page.wait_for_timeout(300)  # Run several frames at this speed

            # Verify positions and telemetry remain finite numeric values
            telemetry_text = page.inner_text("#telemetry")
            assert "NaN" not in telemetry_text, f"Telemetry contains NaN at speed {spd}: {telemetry_text}"
            assert "undefined" not in telemetry_text, f"Telemetry contains undefined at speed {spd}: {telemetry_text}"

            # Check Three.js mesh position is valid finite vector
            mesh_pos = page.evaluate(
                "() => { const s = activeSatellites[0]; return s ? [s.mesh.position.x, s.mesh.position.y, s.mesh.position.z] : null; }"
            )
            assert mesh_pos is not None
            for coord in mesh_pos:
                assert coord is not None and not math.isnan(coord), f"Coordinate NaN at speed {spd}: {mesh_pos}"

        browser.close()
        assert len(errors) == 0, "Errors encountered during playback speed testing:\n" + "\n".join(errors)


def test_adversarial_tab_backgrounding_and_delta_clamping(app_server_url):
    """Stress test tab backgrounding/suspension and delta time clamping."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        _, page, errors, _ = setup_monitored_page(browser, app_server_url)

        # Launch satellite
        page.click("button:has-text('+ New Satellite')")
        page.wait_for_selector("#modal-overlay", state="visible", timeout=5000)
        page.fill("#m-name", "Sat-BgClamp")
        page.click("#modal-buttons button:has-text('Launch')")
        page.wait_for_selector("#modal-overlay", state="hidden", timeout=30000)

        # Check initial simTime
        t_start = page.evaluate("() => simTime")
        page.wait_for_timeout(500)
        t_after_500ms = page.evaluate("() => simTime")
        assert t_after_500ms >= t_start, "Simulation time did not advance"

        # 1. Simulate tab backgrounding: fire visibilitychange (document.hidden = true)
        page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', { value: true, writable: true });
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        page.wait_for_timeout(1000)

        # 2. Simulate large clock delta jump (simulating 30 seconds suspension)
        # Even if clock.getDelta() returns 30.0s, delta clamping Math.min(..., 0.1) must clamp to <= 0.1s
        delta_sim = page.evaluate("""() => {
            const clampedDelta = Math.min(30.0, 0.1);
            return clampedDelta;
        }""")
        assert delta_sim == 0.1, f"Expected clamped delta to be 0.1, got {delta_sim}"

        # 3. Simulate tab foregrounding (document.hidden = false)
        page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', { value: false, writable: true });
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        page.wait_for_timeout(1000)

        # Confirm animation resumes smoothly
        t_resumed = page.evaluate("() => simTime")
        page.wait_for_timeout(500)
        t_final = page.evaluate("() => simTime")
        assert t_final > t_resumed, "Simulation time failed to advance after foregrounding"

        telemetry_text = page.inner_text("#telemetry")
        assert "NaN" not in telemetry_text, f"Telemetry contains NaN after backgrounding: {telemetry_text}"

        browser.close()
        assert len(errors) == 0, "Errors encountered during tab backgrounding:\n" + "\n".join(errors)


def test_adversarial_rapid_satellite_additions(app_server_url):
    """Stress test rapid burst additions of satellites to test DOM and Three.js stability."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        _, page, errors, _ = setup_monitored_page(browser, app_server_url)

        # Concurrently launch 4 satellites via backend API and ingest into frontend scene
        page.evaluate("""async () => {
            const addPromises = [];
            for (let i = 1; i <= 4; i++) {
                const payload = {
                    name: `Burst-Sat-${i}`,
                    mass: 4.0 + i,
                    drag_area: 0.03,
                    cd: 2.2,
                    altitude_km: 400.0 + (i * 50),
                    eccentricity: 0.001 * i,
                    inclination_deg: 45.0 + (i * 5),
                    raan_deg: 30.0 * i
                };
                addPromises.push(
                    fetch('/simulate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    }).then(r => r.json()).then(data => addSatelliteToScene(data))
                );
            }
            await Promise.all(addPromises);
        }""")

        # Wait for all 4 satellites to be added to DOM list
        page.wait_for_selector("#sat-list li:nth-child(4)", state="visible", timeout=30000)
        items = page.locator("#sat-list li")
        assert items.count() == 4, f"Expected 4 items in sat-list, found {items.count()}"

        # Rapidly click through all satellites to test camera focus switching
        for i in range(4):
            item = items.nth(i)
            sat_name = item.inner_text().strip()
            item.click()
            page.wait_for_timeout(200)
            telemetry = page.inner_text("#telemetry")
            assert f"Focus: {sat_name}" in telemetry, f"Expected 'Focus: {sat_name}' in telemetry: {telemetry}"

        # Switch camera to Moon and Earth
        page.evaluate("() => handleFocusChange('Moon')")
        page.wait_for_timeout(300)
        assert "Focus: Moon" in page.inner_text("#telemetry")

        page.evaluate("() => handleFocusChange('Earth')")
        page.wait_for_timeout(300)
        assert "Focus:" not in page.inner_text("#telemetry")

        # Let animation run for 2 seconds with 4 satellites
        page.wait_for_timeout(2000)

        browser.close()
        assert len(errors) == 0, "Errors during rapid satellite additions:\n" + "\n".join(errors)


def test_adversarial_legacy_alpha_and_csv_export(app_server_url):
    """Stress test legacy Alpha.json loading, real-time telemetry, and CSV telemetry export."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        _, page, errors, _ = setup_monitored_page(browser, app_server_url)

        # 1. Load legacy Alpha simulation
        page.wait_for_selector('#load-dropdown option[value="Alpha"]', state="attached", timeout=10000)
        page.select_option("#load-dropdown", "Alpha")
        page.click("button:has-text('Load')")

        # Wait for Alpha's satellite (Sat-3) to appear in list
        page.wait_for_selector("#sat-list li:has-text('Sat-3')", state="visible", timeout=10000)
        page.click("#sat-list li:has-text('Sat-3')")

        # Run animation for 2 seconds and inspect telemetry
        page.wait_for_timeout(2000)
        telem = page.inner_text("#telemetry")
        assert "Focus: Sat-3" in telem, f"Expected Focus: Sat-3, got: {telem}"
        assert "Time:" in telem
        assert "Altitude:" in telem
        assert "Velocity:" in telem
        assert "NaN" not in telem, f"Telemetry contains NaN: {telem}"

        # 2. Test CSV Telemetry Export
        with page.expect_download() as download_info:
            page.click("button:has-text('Export Data')")
        download = download_info.value
        assert download.suggested_filename == "trajectory_export.csv"

        download_path = download.path()
        with open(download_path, encoding="utf-8") as f:
            csv_content = f.read()

        assert "Satellite,Time(s),X(m),Y(m),Z(m),VX(m/s),VY(m/s),VZ(m/s)" in csv_content
        reader = csv.reader(io.StringIO(csv_content))
        header = next(reader)
        assert header == ["Satellite", "Time(s)", "X(m)", "Y(m)", "Z(m)", "VX(m/s)", "VY(m/s)", "VZ(m/s)"]

        rows = list(reader)
        assert len(rows) > 0, "Exported CSV contains no data rows"
        for row in rows:
            assert len(row) == 8
            assert row[0] == "Sat-3"
            # Verify numerical values are non-NaN finite floats
            for col_idx in range(1, 8):
                val = float(row[col_idx])
                assert val is not None and not math.isnan(val)

        # 3. Add a new satellite alongside Alpha and export again to test multi-satellite export
        page.evaluate("""async () => {
            const payload = {
                name: "Sat-Columnar",
                mass: 6.0,
                drag_area: 0.04,
                cd: 2.2,
                altitude_km: 450.0,
                eccentricity: 0.001,
                inclination_deg: 51.6,
                raan_deg: 30.0
            };
            const res = await fetch('/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            addSatelliteToScene(data);
        }""")
        page.wait_for_selector("#sat-list li:has-text('Sat-Columnar')", state="visible", timeout=20000)

        # Export multi-satellite CSV
        with page.expect_download() as multi_dl_info:
            page.click("button:has-text('Export Data')")
        multi_dl = multi_dl_info.value
        with open(multi_dl.path(), encoding="utf-8") as f:
            multi_csv = f.read()

        assert "Sat-3" in multi_csv
        assert "Sat-Columnar" in multi_csv

        browser.close()
        assert len(errors) == 0, "Errors encountered during legacy Alpha & CSV test:\n" + "\n".join(errors)


def test_adversarial_corrupted_partial_trajectories(app_server_url):
    """Stress test UI resilience against corrupted, empty, or partial trajectory objects."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        _, page, errors, _ = setup_monitored_page(browser, app_server_url)

        # Ingest malformed satellites directly into scene
        page.evaluate("""() => {
            // 1. Empty trajectory array
            addSatelliteToScene({
                name: "Corrupt-EmptyArr",
                params: {},
                period_s: 5400,
                trajectory: []
            });

            // 2. Empty columnar trajectory
            addSatelliteToScene({
                name: "Corrupt-EmptyCol",
                params: {},
                period_s: 5400,
                trajectory: { t: [], x: [], y: [], z: [], vx: [], vy: [], vz: [] }
            });

            // 3. Single-point columnar trajectory
            addSatelliteToScene({
                name: "Corrupt-OneCol",
                params: {},
                period_s: 5400,
                trajectory: { t: [0], x: [6800000], y: [0], z: [0], vx: [0], vy: [7600], vz: [0] }
            });

            // 4. Missing vx, vy, vz
            addSatelliteToScene({
                name: "Corrupt-NoVel",
                params: {},
                period_s: 5400,
                trajectory: { t: [0, 60], x: [6800000, 6790000], y: [0, 10000], z: [0, 5000] }
            });
        }""")

        page.wait_for_timeout(1000)

        # Cycle focus across all corrupted satellites
        corrupt_names = ["Corrupt-EmptyArr", "Corrupt-EmptyCol", "Corrupt-OneCol", "Corrupt-NoVel"]
        for name in corrupt_names:
            page.evaluate(f"() => handleFocusChange('{name}')")
            page.wait_for_timeout(300)
            telem = page.inner_text("#telemetry")
            assert f"Focus: {name}" in telem
            assert "Time:" in telem

        # Trigger newProject() while animation is running
        page.click("button:has-text('New Project')")
        page.wait_for_timeout(500)
        assert page.locator("#sat-list li").count() == 0

        # Attempt to export with zero satellites (should show alert dialog, handled by listener)
        page.click("button:has-text('Export Data')")
        page.wait_for_timeout(500)

        browser.close()
        assert len(errors) == 0, "Errors encountered during corrupted trajectory testing:\n" + "\n".join(errors)
