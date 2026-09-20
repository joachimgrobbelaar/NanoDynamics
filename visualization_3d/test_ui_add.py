from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.on("console", lambda msg: errors.append(f"Console: {msg.text}") if msg.type == "error" else None)
        
        page.goto("http://localhost:8000")
        page.click("text=+ New Satellite")
        page.click("text=Launch")
        
        time.sleep(2)
        
        if errors:
            print("Errors detected:")
            for e in errors:
                print(e)
        else:
            print("No errors detected.")
            
        browser.close()

if __name__ == "__main__":
    run()
