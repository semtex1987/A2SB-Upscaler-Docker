from playwright.sync_api import Page, expect, sync_playwright

def test_gradio_info(page: Page):
  page.goto("http://localhost:7860")

  # Wait for gradio elements to be visible
  page.wait_for_selector(".gradio-container")
  # wait for elements to fully load, maybe sleep a bit
  page.wait_for_timeout(2000)

  # Take a full page screenshot to verify the UI changes
  page.screenshot(path="/home/jules/verification/verification.png", full_page=True)

if __name__ == "__main__":
  import os
  os.makedirs("/home/jules/verification", exist_ok=True)
  with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    try:
      test_gradio_info(page)
    finally:
      browser.close()