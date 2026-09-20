"""
Automated Headless Playwright UI Test Suite.

Verifies:
- Zero JavaScript runtime exceptions (pageerror).
- Zero JavaScript console error logs (console.type == 'error').
- Full UI workflow:
  1. Navigate to single-page visualization application.
  2. Open '+ New Satellite' modal, fill parameters, and launch satellite.
  3. Inspect real-time telemetry panel (Time, Altitude, Velocity).
  4. Run frame animation loop for at least 3 seconds.
  5. Load legacy simulation 'Alpha.json' from saved simulations dropdown.
  6. Assert zero runtime errors occurred throughout entire session.
"""

import os
import socket
import subprocess
import sys
import time

import pytest
from playwright.sync_api import sync_playwright

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def is_port_in_use(port: int) -> bool:
    """Check if a local TCP port is already open and accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def app_server_url():
    """Ensure FastAPI application server is available for headless browser tests."""
    port = 8000
    server_process = None

    if not is_port_in_use(port):
        # Spawn uvicorn process if not already running
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "visualization_3d.app:app", "--port", str(port)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait up to 10 seconds for server to start
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


def run_ui_refactor_test(server_url: str):
    """Execute the end-to-end headless browser test workflow."""
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Listen for unhandled page exceptions
        page.on("pageerror", lambda err: errors.append(f"PageError: {err}"))

        # Listen for console error messages
        page.on(
            "console",
            lambda msg: errors.append(f"ConsoleError: {msg.text}") if msg.type == "error" else None,
        )

        # Handle modal alert dialogs automatically without blocking
        page.on("dialog", lambda dialog: dialog.accept())

        # Step 1: Navigate to the application
        page.goto(server_url)
        page.wait_for_selector("#top-nav", state="visible", timeout=15000)
        page.wait_for_selector("#telemetry", state="visible", timeout=15000)

        # Step 2: Open '+ New Satellite' modal and launch new satellite
        page.click("button:has-text('+ New Satellite')")
        page.wait_for_selector("#modal-overlay", state="visible", timeout=5000)

        page.fill("#m-name", "E2E-Playwright-Sat")
        page.fill("#m-mass", "6.0")
        page.fill("#m-area", "0.04")
        page.fill("#m-cd", "2.2")
        page.fill("#m-alt", "450.0")
        page.fill("#m-ecc", "0.001")
        page.fill("#m-inc", "51.6")
        page.fill("#m-raan", "30.0")

        # Launch satellite and wait for computation / modal hide
        page.click("#modal-buttons button:has-text('Launch')")
        page.wait_for_selector("#modal-overlay", state="hidden", timeout=30000)

        # Step 3: Inspect Telemetry panel
        telem_text = page.inner_text("#telemetry")
        assert "Time:" in telem_text, f"Telemetry missing 'Time:' indicator: {telem_text}"
        assert "Altitude:" in telem_text, f"Telemetry missing 'Altitude:' indicator: {telem_text}"
        assert "Velocity:" in telem_text, f"Telemetry missing 'Velocity:' indicator: {telem_text}"

        # Step 4: Run animation loop for at least 3 seconds
        page.wait_for_timeout(3000)

        # Step 5: Test loading legacy simulation 'Alpha.json'
        # Wait for option to be attached to DOM
        page.wait_for_selector('#load-dropdown option[value="Alpha"]', state="attached", timeout=10000)
        page.select_option("#load-dropdown", "Alpha")
        page.click("button:has-text('Load')")

        # Give 2 seconds for scene reconstruction and animation update
        page.wait_for_timeout(2000)

        # Verify active satellites list contains loaded items
        sat_items = page.locator("#sat-list li")
        assert sat_items.count() > 0, "Expected at least one satellite in #sat-list after loading Alpha.json"

        browser.close()

    # Step 6: Assert zero runtime or console errors recorded
    assert len(errors) == 0, "Headless Playwright detected JavaScript errors:\n" + "\n".join(errors)


def test_ui_refactor_headless(app_server_url):
    """Pytest wrapper for Playwright UI refactor automation."""
    run_ui_refactor_test(app_server_url)


if __name__ == "__main__":
    test_url = "http://localhost:8000"
    print(f"Running standalone Playwright UI test against {test_url}...")
    run_ui_refactor_test(test_url)
    print("Standalone UI test completed successfully with 0 errors!")
