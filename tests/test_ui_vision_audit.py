"""Automated UI Contrast, Geospatial & Spotlight Mission Guide Validation (WCAG AAA Standards).

Verifies:
1. WCAG AAA color contrast ratios for SpaceX dark telemetry console tokens:
   - Background obsidian (#05070B to #090D14) vs primary text (#E2E8F0) >= 12:1
   - Nominal Cyan (#00E5FF) and Orbit Green (#00FF66) >= 7:1
   - Hazard Amber (#FF9100) and Burn/Abort Red (#FF1744) high-visibility alert states
2. Standalone CSS design tokens (visualization_3d/static/css/spacex_tokens.css) and <200-line
   Mission Guide engine (visualization_3d/static/js/mission_guide.js).
3. Headless Playwright layout & spotlight tour audit at 1280x720:
   - 36px #top-nav Command Deck header & 290px <-> 24px Collapsible Cyber-Drawer (#sidebar)
   - Clicking [? GUIDE] renders the SVG clip-path spotlight mask and steps through all 7 tour nodes
     with zero console errors
   - 3D THREE.InstancedMesh urban building clusters & Johannesburg POI presence
"""

import os
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest


def _rel_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(hex1: str, hex2: str) -> float:
    l1, l2 = _rel_luminance(hex1), _rel_luminance(hex2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def test_wcag_aaa_telemetry_contrast_ratios():
    """Verify WCAG AAA contrast ratios for SpaceX dark telemetry console color palette."""
    for bg in ("#05070B", "#090D14"):
        # Primary text >= 12:1
        assert contrast_ratio(bg, "#E2E8F0") >= 12.0
        # Nominal Cyan (#00E5FF) & Orbit Green (#00FF66) >= 7:1
        assert contrast_ratio(bg, "#00E5FF") >= 7.0
        assert contrast_ratio(bg, "#00FF66") >= 7.0
        # Hazard Amber (#FF9100) and Burn/Abort Red (#FF1744) high-visibility alert validation
        assert contrast_ratio(bg, "#FF9100") >= 7.0
        assert contrast_ratio(bg, "#FF1744") >= 4.5


def test_spacex_tokens_and_mission_guide_line_budget():
    """Verify spacex_tokens.css and mission_guide.js (< 200 lines) exist and meet specifications."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(root, "visualization_3d", "static", "css", "spacex_tokens.css")
    js_path = os.path.join(root, "visualization_3d", "static", "js", "mission_guide.js")

    assert os.path.exists(css_path)
    assert os.path.exists(js_path)

    with open(css_path, "r", encoding="utf-8") as f:
        css_text = f.read()
    assert "backdrop-filter: blur(16px)" in css_text
    assert "cubic-bezier(0.16, 1, 0.3, 1)" in css_text

    with open(js_path, "r", encoding="utf-8") as f:
        js_lines = f.readlines()
    assert len(js_lines) < 200, f"mission_guide.js must be < 200 lines, got {len(js_lines)}"


@pytest.fixture(scope="module")
def live_server_url():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "visualization_3d.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            with urlopen(url + "/api/pois", timeout=0.5) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.2)
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def test_playwright_1280x720_layout_and_7_step_mission_guide(live_server_url):
    """Run headless Playwright audit at 1280x720 verifying Command Deck, Cyber-Drawer, and 7-step Spotlight Tour."""
    playwright = pytest.importorskip("playwright.sync_api")
    console_errors = []

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()
        page.on("pageerror", lambda err: console_errors.append(str(err)))

        page.goto(live_server_url, wait_until="domcontentloaded")
        # Enter mission mode from splash screen
        page.evaluate("if (typeof enterMissionMode === 'function') enterMissionMode(true);")
        page.wait_for_timeout(300)

        # 1. Verify 36px #top-nav Command Deck header & non-overlapping bounding box at 1280x720
        top_nav_box = page.locator("#top-nav").bounding_box()
        assert top_nav_box is not None
        assert abs(top_nav_box["height"] - 36.0) <= 2.0
        assert top_nav_box["width"] <= 1280.0

        # 2. Verify Collapsible Cyber-Drawer (#sidebar): 290px expanded <-> 24px collapsed
        sidebar_box = page.locator("#sidebar").bounding_box()
        assert sidebar_box is not None
        assert abs(sidebar_box["width"] - 290.0) <= 2.0
        assert sidebar_box["y"] >= 35.0  # Does not overlap 36px #top-nav

        page.evaluate("toggleSidebarCollapse()")
        page.wait_for_timeout(260)
        collapsed_box = page.locator("#sidebar").bounding_box()
        assert collapsed_box is not None
        assert abs(collapsed_box["width"] - 24.0) <= 2.0

        page.evaluate("toggleSidebarCollapse()")
        page.wait_for_timeout(260)

        # 3. Verify 3D InstancedMesh urban buildings & Johannesburg POI
        urban_check = page.evaluate("""() => ({
            hasInstancedMesh: Boolean(typeof urbanInstancedMesh !== 'undefined' && urbanInstancedMesh && urbanInstancedMesh.count > 100),
            hasJohannesburg: Boolean(typeof earthCityObjects !== 'undefined' && earthCityObjects.some(c => c.data.name === 'Johannesburg')),
            hasRayleighAtmosphere: Boolean(typeof rayleighMieAtmosphere !== 'undefined' && rayleighMieAtmosphere)
        })""")
        assert urban_check["hasInstancedMesh"] is True
        assert urban_check["hasJohannesburg"] is True
        assert urban_check["hasRayleighAtmosphere"] is True

        # 4. Click [? GUIDE] and step through all 7 tour nodes without console errors
        page.locator("#cmd-guide-btn").click()
        page.wait_for_timeout(150)

        expected_badges = [
            "01 / 07 // COMMAND DECK & TIME WARP",
            "02 / 07 // CAMERA FOCUS & RADIAL LOCKING",
            "03 / 07 // TELEMETRY & AEROTHERMAL MONITORING",
            "04 / 07 // VNC THRUSTER BURN NODES",
            "05 / 07 // ONBOARD OPTICAL PIP CAMERA",
            "06 / 07 // ISL LASER COMMUNICATION MESH",
            "07 / 07 // EDGE TINYML HARDWARE",
        ]

        for idx, expected_badge in enumerate(expected_badges):
            badge_text = page.locator("#mg-step-badge").inner_text()
            assert badge_text == expected_badge, f"Step {idx+1} mismatch: {badge_text}"
            clip_val = page.evaluate("document.getElementById('mission-guide-mask').style.clipPath")
            assert "path(" in clip_val
            if idx < len(expected_badges) - 1:
                page.locator("#mg-next-btn").click()
                page.wait_for_timeout(80)

        # Exit guide via [✕ EXIT]
        page.locator("#mg-exit-btn").click()
        page.wait_for_timeout(80)
        is_guide_active = page.evaluate("window.MissionGuide.isActive()")
        assert is_guide_active is False

        assert len(console_errors) == 0, f"Unexpected page errors: {console_errors}"
        browser.close()
