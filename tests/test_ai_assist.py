from playwright.sync_api import expect

from pages.editor_page import EditorPage
from utils.helpers import mock_datamuse, mock_default_content, mock_difficult_check, mock_update


def test_accepting_a_suggestion_replaces_the_selected_hard_word(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "Graph ideas improve confidence.")
    mock_update(
        editor_page.page,
        hard_words=[["graph", 0.96, ""]],
        next_word="group",
    )
    mock_datamuse(
        editor_page.page,
        synonyms=[
            {"word": "chart"},
            {"word": "diagram"},
            {"word": "shape"},
        ],
    )
    difficult_requests = mock_difficult_check(
        editor_page.page,
        alternatives=[
            ["chart", 0.12],
            ["diagram", 0.17],
        ],
    )

    editor_page.goto()
    editor_page.hover_hard_word("graph")

    assert "chart" in editor_page.popover_options()
    editor_page.choose_popover_option("chart")

    editor_page.expect_editor_contains(["chart", "ideas improve confidence"])
    expect(editor_page.difficult_words_input).to_contain_text("graph")
    expect(editor_page.easy_words_input).to_contain_text("chart")
    assert difficult_requests[-1]["thresh"] == "70"


def test_ignoring_a_suggestion_keeps_the_original_word_and_marks_it_easy(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "Graph ideas improve confidence.")
    mock_update(
        editor_page.page,
        hard_words=[["graph", 0.96, ""]],
        next_word="group",
    )
    mock_datamuse(editor_page.page, synonyms=[{"word": "chart"}])
    mock_difficult_check(editor_page.page, alternatives=[["chart", 0.12]])

    editor_page.goto()
    editor_page.hover_hard_word("graph")
    editor_page.choose_popover_option("Ignore")

    editor_page.expect_editor_contains(["Graph", "ideas improve confidence"])
    expect(editor_page.easy_words_input).to_contain_text("graph")


def test_active_learning_yes_adds_current_word_to_difficult_list(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "Group presentations can be stressful.")
    update_requests = mock_update(
        editor_page.page,
        hard_words=[["group", 0.9, ""]],
        next_word="group",
    )

    editor_page.goto()
    editor_page.open_active_learning()
    expect(editor_page.active_learning_word).to_have_text("group")
    editor_page.choose_active_learning_feedback(difficult=True)

    expect(editor_page.difficult_words_input).to_contain_text("group")
    assert update_requests, "Expected the Yes action to trigger a model refresh request."


def test_active_learning_no_adds_current_word_to_easy_list(editor_page: EditorPage) -> None:
    mock_default_content(editor_page.page, "Green rooms can still feel tense.")
    mock_update(
        editor_page.page,
        hard_words=[["green", 0.73, ""]],
        next_word="green",
    )

    editor_page.goto()
    editor_page.open_active_learning()
    expect(editor_page.active_learning_word).to_have_text("green")
    editor_page.choose_active_learning_feedback(difficult=False)

    expect(editor_page.easy_words_input).to_contain_text("green")
