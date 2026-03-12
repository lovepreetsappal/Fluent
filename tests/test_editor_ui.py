from playwright.sync_api import expect

from pages.editor_page import EditorPage
from utils.helpers import mock_default_content, mock_update


def test_home_page_loads_default_content_and_editor(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "This is the default script.")
    mock_update(editor_page.page, hard_words=[], next_word="group")

    editor_page.goto()

    expect(editor_page.editor).to_contain_text("This is the default script.")
    expect(editor_page.active_learning_word).to_have_text("group")


def test_preferences_modal_updates_threshold_and_applies_highlighting(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "Graph theory helps group presentations.")
    update_requests = mock_update(
        editor_page.page,
        hard_words=[["graph", 0.92, ""], ["group", 0.85, ""]],
        next_word="printer",
    )

    editor_page.goto()
    editor_page.open_preferences()
    editor_page.update_preferences(
        easy_words="cat, mat, table",
        difficult_words="graph, group, printer",
        threshold=85,
    )

    expect(editor_page.threshold_value).to_have_text("85%")
    expect(editor_page.preferences_modal).not_to_be_visible()
    assert update_requests[-1]["easy"] == "cat, mat, table"
    assert update_requests[-1]["diff"] == "graph, group, printer"
    assert update_requests[-1]["thresh"] == "85"
    assert set(editor_page.highlighted_words_text()) == {"graph", "group"}
    expect(editor_page.active_learning_word).to_have_text("printer")


def test_update_button_reapplies_highlighting_to_current_editor_text(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "placeholder")
    update_requests = mock_update(
        editor_page.page,
        hard_words=[["crisis", 0.88, ""], ["grave", 0.9, ""]],
        next_word="green",
    )

    editor_page.goto()
    editor_page.set_editor_text("A crisis can feel grave.")
    editor_page.update_toolbar_button.click()

    expect(editor_page.editor).to_contain_text("A crisis can feel grave.")
    assert set(editor_page.highlighted_words_text()) == {"crisis", "grave"}
    assert "a crisis can feel grave." in update_requests[-1]["text"]

