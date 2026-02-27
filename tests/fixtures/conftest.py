"""Playwright fixtures for screenshots."""

import pytest
from pathlib import Path
from playwright.async_api import async_playwright


@pytest.fixture(scope="session")
async def browser():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        yield context
        await context.close()
        await browser.close()


@pytest.fixture
def screenshot_dir():
    """Erstellt Verzeichnis für Screenshots."""
    dir_path = Path("tests/screenshots")
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def take_screenshot(browser, screenshot_dir):
    """Fixture für Screenshots."""

    async def _screenshot(name: str):
        page = await browser.new_page()
        await page.goto("http://localhost:8000")

        path = screenshot_dir / f"{name}.png"
        await page.screenshot(path=str(path), full_page=True)

        await page.close()
        return path

    return _screenshot


@pytest.fixture
def save_cli_screenshot(screenshot_dir):
    """Speichert CLI-Ausgabe als Screenshot."""
    import subprocess
    from datetime import datetime

    def _save(text: str, name: str):
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: monospace;
                    background: #1a1a1a;
                    color: #f8f8f2;
                    padding: 20px;
                    white-space: pre-wrap;
                }}
            </style>
        </head>
        <body>
            <h3>{name}</h3>
            <pre>{text}</pre>
            <p><small>{datetime.now()}</small></p>
        </body>
        </html>
        """

        path = screenshot_dir / f"{name}.html"
        path.write_text(html)
        return path

    return _save


# Import LangGraph fixtures
from tests.fixtures.langgraph_conftest import tmp_path, test_config_path, mock_provider, mock_bus
