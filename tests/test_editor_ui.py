from playwright.sync_api import expect

from pages.editor_page import EditorPage
from utils.helpers import mock_default_content, mock_update


def test_opening_fluent_loads_the_default_script_and_editor(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "Welcome to your practice script.")
    mock_update(editor_page.page, hard_words=[], next_word="group")

    editor_page.goto()

    expect(editor_page.editor).to_contain_text("Welcome to your practice script.")
    expect(editor_page.active_learning_word).to_have_text("group")


def test_raising_the_threshold_updates_preferences_and_reduces_review_to_confident_matches(
    editor_page: EditorPage,
) -> None:
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


def test_updating_a_revised_sentence_reapplies_highlighting_to_the_latest_draft(editor_page: EditorPage) -> None:
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


def test_reviewing_a_longer_practice_paragraph_keeps_multiple_highlights_visible(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "placeholder")
    mock_update(
        editor_page.page,
        hard_words=[["graph", 0.92, ""], ["group", 0.87, ""], ["grave", 0.9, ""]],
        next_word="printer",
    )

    editor_page.goto()
    editor_page.set_editor_text(
        "Graph ideas help the group stay calm. A grave tone can make the opening feel harder."
    )
    editor_page.update_toolbar_button.click()

    expect(editor_page.editor).to_contain_text("Graph ideas help the group stay calm.")
    assert set(editor_page.highlighted_words_text()) == {"graph", "group", "grave"}

