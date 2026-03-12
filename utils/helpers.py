from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, Route


JsonDict = dict[str, Any]


def _json_route(route: Route, payload: Any, status: int = 200) -> None:
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(payload),
    )


def mock_default_content(page: Page, content: str) -> None:
    page.route("**/get_default_content", lambda route: route.fulfill(status=200, body=content))


def mock_update(page: Page, hard_words: list[list[Any]], next_word: str = "group") -> list[JsonDict]:
    recorded_requests: list[JsonDict] = []

    def handler(route: Route) -> None:
        parsed = urlparse(route.request.url)
        recorded_requests.append(parse_query_params(parsed.query))
        _json_route(route, {"hard_words": hard_words, "next_word": next_word})

    page.route("**/update*", handler)
    return recorded_requests


def mock_datamuse(page: Page, synonyms: list[dict[str, str]]) -> None:
    page.route(
        "https://api.datamuse.com/words?ml=*",
        lambda route: _json_route(route, synonyms),
    )


def mock_difficult_check(page: Page, alternatives: list[list[Any]]) -> list[JsonDict]:
    recorded_requests: list[JsonDict] = []

    def handler(route: Route) -> None:
        parsed = urlparse(route.request.url)
        recorded_requests.append(parse_query_params(parsed.query))
        _json_route(route, alternatives)

    page.route("**/check_if_word_difficult*", handler)
    return recorded_requests


def mock_static_json(page: Page, url_pattern: str, payload_factory: Callable[[], Any]) -> None:
    page.route(url_pattern, lambda route: _json_route(route, payload_factory()))


def parse_query_params(query: str) -> JsonDict:
    parsed = parse_qs(query, keep_blank_values=True)
    normalized: JsonDict = {}
    for key, value in parsed.items():
        normalized[key] = value[0] if len(value) == 1 else value
    return normalized

