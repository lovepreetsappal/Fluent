from __future__ import annotations

from typing import Iterable

from playwright.sync_api import Page, expect


class EditorPage:
    def __init__(self, page: Page, base_url: str) -> None:
        self.page = page
        self.base_url = base_url

    @property
    def update_toolbar_button(self):
        return self.page.locator("#update_button")

    @property
    def preferences_button(self):
        return self.page.locator("#preferences")

    @property
    def active_learning_button(self):
        return self.page.locator("#active_learning")

    @property
    def preferences_modal(self):
        return self.page.locator("#preferences_modal")

    @property
    def active_learning_modal(self):
        return self.page.locator("#al_modal")

    @property
    def editor(self):
        # Summernote renders the editable area dynamically after page load.
        return self.page.locator(".note-editable")

    @property
    def hard_words(self):
        return self.page.locator("span.hard_word")

    @property
    def easy_words_input(self):
        return self.page.locator("#easy_words")

    @property
    def difficult_words_input(self):
        return self.page.locator("#diff_words")

    @property
    def threshold_slider(self):
        return self.page.locator("#threshold")

    @property
    def threshold_value(self):
        return self.page.locator("#coeff_slider_val")

    @property
    def update_preferences_button(self):
        return self.page.locator("#update")

    @property
    def active_learning_word(self):
        return self.page.locator("#al_word")

    @property
    def yes_button(self):
        return self.page.locator("#yes")

    @property
    def no_button(self):
        return self.page.locator("#no")

    @property
    def popover(self):
        return self.page.locator(".popover")

    def goto(self) -> None:
        self.page.goto(self.base_url, wait_until="domcontentloaded")
        expect(self.page.locator("#title")).to_have_text("Fluent")
        expect(self.editor).to_be_visible()

    def set_editor_text(self, text: str) -> None:
        # Setting innerText keeps the editor content deterministic for automation.
        self.editor.evaluate("(element, value) => { element.innerText = value; }", text)

    def editor_text(self) -> str:
        return self.editor.inner_text()

    def open_preferences(self) -> None:
        self.preferences_button.click()
        expect(self.preferences_modal).to_be_visible()

    def update_preferences(self, easy_words: str, difficult_words: str, threshold: int) -> None:
        self.easy_words_input.fill(easy_words)
        self.difficult_words_input.fill(difficult_words)
        # Range input needs an input event so the UI label updates.
        self.threshold_slider.evaluate(
            "(element, value) => { element.value = String(value); element.dispatchEvent(new Event('input', { bubbles: true })); }",
            threshold,
        )
        self.update_preferences_button.click()

    def open_active_learning(self) -> None:
        self.active_learning_button.click()
        expect(self.active_learning_modal).to_be_visible()

    def choose_active_learning_feedback(self, difficult: bool) -> None:
        target = self.yes_button if difficult else self.no_button
        target.click()

    def highlighted_words_text(self) -> list[str]:
        return [text.strip() for text in self.hard_words.all_inner_texts()]

    def hover_hard_word(self, word: str) -> None:
        self.page.locator("span.hard_word", has_text=word).first.hover()
        expect(self.popover).to_be_visible()

    def popover_options(self) -> list[str]:
        return [text.strip() for text in self.popover.locator("li").all_inner_texts()]

    def choose_popover_option(self, option_text: str) -> None:
        self.popover.get_by_text(option_text, exact=True).click()

    def expect_editor_contains(self, snippets: Iterable[str]) -> None:
        for snippet in snippets:
            expect(self.editor).to_contain_text(snippet)

