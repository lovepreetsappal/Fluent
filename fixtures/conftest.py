import os

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from pages.editor_page import EditorPage


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.getenv("APP_URL", "http://127.0.0.1:3999")


@pytest.fixture(scope="session")
def headless() -> bool:
    return os.getenv("HEADLESS", "true").lower() != "false"


@pytest.fixture(scope="session")
def playwright_instance() -> Playwright:
    with sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="session")
def browser(playwright_instance: Playwright, headless: bool) -> Browser:
    browser = playwright_instance.chromium.launch(headless=headless)
    yield browser
    browser.close()


@pytest.fixture()
def context(browser: Browser, base_url: str) -> BrowserContext:
    context = browser.new_context(base_url=base_url)
    context.set_default_timeout(10_000)
    yield context
    context.close()


@pytest.fixture()
def page(context: BrowserContext) -> Page:
    page = context.new_page()
    yield page
    page.close()


@pytest.fixture()
def editor_page(page: Page, base_url: str) -> EditorPage:
    return EditorPage(page=page, base_url=base_url)

