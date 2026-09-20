from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Browser Page Error: {err}"))
        page.on("response", lambda res: print(f"Network {res.status}: {res.url}") if "/simulate" in res.url else None)

        print("Navigating to http://localhost:8000...")
        page.goto("http://localhost:8000")
        time.sleep(1)
        
        print("Opening New Satellite Modal...")
        page.click("text=+ New Satellite")
        time.sleep(1)
        
        print("Launching Satellite...")
        page.fill("#m-name", "TestSat-Alpha")
        page.fill("#m-alt", "450")
        page.click("text=Launch")
        
        time.sleep(4)
        
        browser.close()

if __name__ == "__main__":
    run()
