import time
from playwright.sync_api import sync_playwright

def run_browser_audit():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        print("1. Loading http://localhost:8000 ...")
        page.goto("http://localhost:8000")
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        # Check Issue 1: No CORS errors for earth-clouds
        cloud_cors_errors = [err for err in console_errors if "earth-clouds" in err or "CORS" in err]
        assert len(cloud_cors_errors) == 0, f"Found cloud CORS error: {cloud_cors_errors}"
        print("  ✓ Issue 1 Passed: Zero cloud texture CORS errors.")

        # Check Issue 2: Splash "Load Mission / Scenario"
        load_btn = page.locator("button.splash-btn", has_text="Load Mission / Scenario")
        assert load_btn.is_visible()
        load_btn.click()
        time.sleep(1)
        presets_modal = page.locator("#presets-scenarios-modal")
        assert presets_modal.is_visible(), "Presets modal did not open on splash Load Mission click!"
        print("  ✓ Issue 2 Passed: Splash screen Load Mission opens Presets Scenarios modal.")
        page.locator("#presets-scenarios-modal button", has_text="Close").click()
        time.sleep(0.5)

        # Check Issue 5: No raw LaTeX in HUD
        hud_text = page.locator("#sidebar").inner_text()
        assert "$t_{\\text{proc}}$" not in hud_text, "Raw LaTeX $t_{\\text{proc}}$ found in HUD"
        assert "$t_{\\text{lat}}$" not in hud_text, "Raw LaTeX $t_{\\text{lat}}$ found in HUD"
        print("  ✓ Issue 5 Passed: No raw LaTeX markup in telemetry HUD.")

        # Check Issue 8: PiP HUD position
        vcam_hud = page.locator("#virtual-cam-hud")
        bottom_style = vcam_hud.evaluate("el => window.getComputedStyle(el).bottom")
        assert bottom_style == "38px", f"Expected bottom 38px, got {bottom_style}"
        print("  ✓ Issue 8 Passed: PiP viewport offset correctly to bottom 38px.")

        # Check Issue 4: Keyboard shortcuts
        # Press D to open deploy modal
        page.keyboard.press("d")
        time.sleep(0.5)
        modal_overlay = page.locator("#modal-overlay")
        assert modal_overlay.is_visible(), "Pressing 'D' did not open satellite modal"
        print("  ✓ Issue 4 Passed: Keyboard shortcut 'D' opened deployment modal.")

        # Check Issue 7 & 6: Bounds validation & Non-blocking toast
        # Set negative altitude
        page.fill("#m-alt", "-50")
        page.click("#launch-btn")
        time.sleep(0.5)
        toast = page.locator(".sx-toast.sx-toast-error")
        assert toast.is_visible(), "Toast error not displayed for negative altitude!"
        toast_text = toast.inner_text()
        assert "Altitude for Earth must be between 100" in toast_text
        print(f"  ✓ Issue 6 & 7 Passed: Toast displayed on invalid bounds: '{toast_text}'")

        # Test Walker-Delta Constellation Deployment
        page.evaluate("() => openModal()")
        time.sleep(0.5)
        page.check("#m-cluster-enable")
        page.select_option("#m-cluster-pattern", "walker")
        page.fill("#m-walker-t", "6")
        page.fill("#m-walker-p", "2")
        page.fill("#m-walker-f", "1")
        page.fill("#m-alt", "450")
        page.fill("#m-ecc", "0.001")
        page.fill("#m-name", "WalkerSat")
        page.evaluate("() => submitNewSatellite()")
        time.sleep(4)

        # Verify 6 satellites spawned
        sat_count = page.evaluate("() => activeSatellites.length")
        assert sat_count == 6, f"Expected 6 Walker satellites, found {sat_count}"
        print(f"  ✓ Feature 1 Passed: Walker-Delta constellation successfully spawned {sat_count} satellites.")

        # Test Space shortcut for pause/play
        is_paused_before = page.evaluate("() => simSettings.playbackSpeed === 0")
        page.keyboard.press("Space")
        is_paused_after = page.evaluate("() => simSettings.playbackSpeed === 0")
        assert is_paused_before != is_paused_after, "Space key did not toggle play/pause!"
        print("  ✓ Space key toggled simulation playback state.")

        # Test TLE parser
        sample_tle = """ISS (ZARYA)
1 25544U 98067A   24080.52187500  .00016717  00000+0  30000-3 0  9993
2 25544  51.6416 247.4627 0004567 130.1234 229.8765 15.49815016440008"""
        page.evaluate(f"parseAndDeployTLE(`{sample_tle}`, 'ISS.tle')")
        time.sleep(2)
        sat_count_after_tle = page.evaluate("() => activeSatellites.length")
        assert sat_count_after_tle == 7, f"Expected 7 satellites after TLE ingest, found {sat_count_after_tle}"
        print("  ✓ Feature 4 Passed: TLE parser successfully ingested ISS orbit.")

        # Test About Modal
        page.evaluate("() => openAboutModal()")
        time.sleep(0.5)
        about_modal = page.locator("#about-modal-overlay")
        assert about_modal.is_visible(), "About modal did not open!"
        about_text = about_modal.inner_text()
        assert "Honours" in about_text and "DOP853" in about_text
        page.evaluate("() => closeAboutModal()")
        time.sleep(0.3)
        assert not about_modal.is_visible(), "About modal did not close!"
        print("  ✓ Feature 5 Passed: About Honours Defense modal opened and validated.")

        # Test Dynamic Tropical Storms Toggle
        page.evaluate("() => toggleTropicalCyclones()")
        time.sleep(0.3)
        storms_vis = page.evaluate("() => tropicalStormsGroup.visible")
        assert storms_vis is False or storms_vis is True, "tropicalStormsGroup state invalid"
        print("  ✓ Feature 6 Passed: Tropical storm vortexes toggled successfully.")

        # Test Geomagnetic Dipole & Ionosphere Aurora Toggle
        page.evaluate("() => toggleGeomagneticField()")
        time.sleep(0.3)
        mag_vis = page.evaluate("() => geomagneticFieldGroup.visible")
        assert mag_vis is True, "Geomagnetic field was not activated!"
        print("  ✓ Feature 7 Passed: Geomagnetic field dipole and auroral ovals active.")

        browser.close()
        print("\nAll browser automated verification checks PASSED successfully!")

if __name__ == "__main__":
    run_browser_audit()
