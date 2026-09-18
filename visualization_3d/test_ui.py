from playwright.sync_api import sync_playwright
import time
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("Navigating to http://localhost:8000...")
        page.goto("http://localhost:8000")
        time.sleep(2)  # wait for three.js to load
        
        os.makedirs("screenshots", exist_ok=True)
        
        # 1. Initial State Screenshot
        page.screenshot(path="screenshots/1_initial.png")
        print("Captured initial state.")
        
        # 2. Click '+ New Satellite'
        print("Opening New Satellite Modal...")
        page.click("text=+ New Satellite")
        time.sleep(1)
        page.screenshot(path="screenshots/2_modal_open.png")
        
        # 3. Add Satellite parameters and Launch
        print("Launching Satellite...")
        page.fill("#m-name", "TestSat-Alpha")
        page.fill("#m-alt", "450")
        page.click("text=Launch")
        
        # Wait for simulation to compute and appear
        time.sleep(3)
        page.screenshot(path="screenshots/3_satellite_launched.png")
        
        # 4. Save Simulation
        print("Saving Simulation...")
        page.fill("#save-name", "alpha_sim")
        page.click("text=Save")
        time.sleep(1)
        # Handle the alert ("Saved!")
        # Playwright auto-dismisses alerts, or we might need an alert handler. 
        # Actually, let's setup an alert handler just in case.
        
        # 5. Load Simulation (Clear and Load)
        print("Loading Simulation...")
        # Select from dropdown
        page.select_option("#load-dropdown", "alpha_sim")
        page.click("text=Load")
        time.sleep(2)
        page.screenshot(path="screenshots/4_simulation_loaded.png")
        
        browser.close()
        print("Browser automation complete.")

if __name__ == "__main__":
    run()
