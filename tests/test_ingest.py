"""Month discovery: only a 404 means "not published"; other HTTP errors must surface."""

from __future__ import annotations

import datetime as dt
import urllib.error

import pytest

from delay_contagion import ingest


class _Ok:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_head(published: set[tuple[int, int]], fail: dict[tuple[int, int], int] | None = None):
    """urlopen stand-in: 200 for published months, `fail` codes, 404 otherwise."""
    codes = {ingest._url(m): 200 for m in published}
    codes |= {ingest._url(m): c for m, c in (fail or {}).items()}

    def urlopen(request, timeout):
        code = codes.get(request.full_url, 404)
        if code != 200:
            raise urllib.error.HTTPError(request.full_url, code, "x", {}, None)
        return _Ok()

    return urlopen


def test_window_ends_at_latest_published_month(monkeypatch):
    published = {(2026, m) for m in range(1, 8)} | {(2025, m) for m in range(1, 13)}
    monkeypatch.setattr(ingest.urllib.request, "urlopen", _fake_head(published))
    window = ingest.latest_months(12, today=dt.date(2026, 10, 2))
    assert window[0] == (2025, 8)
    assert window[-1] == (2026, 7)
    assert len(window) == 12


def test_rate_limit_is_not_read_as_unpublished(monkeypatch):
    published = {(2026, m) for m in range(1, 8)} | {(2025, m) for m in range(1, 13)}
    fake = _fake_head(published, fail={(2026, 7): 429})
    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake)
    with pytest.raises(urllib.error.HTTPError):
        ingest.latest_months(12, today=dt.date(2026, 7, 15))


def test_gap_inside_the_window_fails_before_download(monkeypatch):
    published = {(2026, m) for m in range(1, 8)} | {(2025, m) for m in range(1, 13)}
    published.discard((2026, 2))
    monkeypatch.setattr(ingest.urllib.request, "urlopen", _fake_head(published))
    with pytest.raises(RuntimeError, match="missing"):
        ingest.latest_months(12, today=dt.date(2026, 10, 2))
