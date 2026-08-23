# Skip any test that relies on playwright if it's not available.
try:
    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    try:
        _b = _pw.chromium.launch(headless=True, args=["--no-sandbox"])
        _b.close()
    finally:
        _pw.stop()
    has_playwright = True
except Exception:
    has_playwright = False

# Separately check for a working Playwright Firefox install, since it may not
# be present even when Chromium is (e.g. `playwright install chromium` only).
try:
    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    try:
        _b = _pw.firefox.launch(headless=True)
        _b.close()
    finally:
        _pw.stop()
    has_playwright_firefox = True
except Exception:
    has_playwright_firefox = False
